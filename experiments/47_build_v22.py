"""
Patch v20 into v22: replace the inherited blend weights with the first fitted, cross-validated ones.

Evidence (findings.md §15): the per-member OOF harness (`pikaaa/name-that-llm-oof-harness`) dumps
every member's log-prob on a held-out pool sample, so blend weights become a fitting problem
instead of inherited constants. Softplus log-loss fit with prompt-grouped 3-fold CV:

    v20 weights                  out-of-fold BA 0.7286
    variant C (shipped here)                    0.7497   +0.021

The gain is concentrated on hard rows (+0.043 at blend confidence 0.50-0.70), is zero on easy rows,
and every member's fitted weight moves by <=1.5x across independent folds (n14b: 0.169/0.169/0.171).

Variant C changes ONLY the `w` and `gauss` scalars. `twin` stays 0.35 on every member, so the
effective centring weight stays 0.35/1.35 = 0.259, which is the measured optimum from §13.3.
Nothing about clustering, Sinkhorn, ATLAS or the comp stack changes.

v20 gave n14b weight 2.5 (20% of the blend) although it is the weakest LLM member by 0.14 BA, and
gave ll8v5 weight 1.0 although it is the second strongest. That inversion is what this fixes.
"""
import json
import pathlib
import sys

SRC = pathlib.Path('notebooks/v20/name-that-llm-v20.ipynb')
DST = pathlib.Path('notebooks/v22/name-that-llm-v22.ipynb')
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.load(open(SRC))


def join(cell):
    return ''.join(cell['source'])


def patch(old, new, label):
    hits = [(i, join(c)) for i, c in enumerate(nb['cells']) if old in join(c)]
    if len(hits) != 1:
        sys.exit(f'FAIL {label}: {len(hits)} occurrences')
    i, src = hits[0]
    nb['cells'][i]['source'] = src.replace(old, new).splitlines(keepends=True)
    print(f'ok {label}')


# name -> (v20 w, fitted w, v20 gauss, fitted gauss)
# The fit ran on LLM members only, so its absolute scale is arbitrary and must NOT be applied
# directly: the TF-IDF block (tfidf 0.25, svc4 1.5, svc4c 1.0, distort 0.10 = 2.85) is unchanged,
# and the raw fitted vector halves the LLM block from 14.965 to 7.643, which would silently double
# TF-IDF's share to 27% - something the harness never measured. Rescale by 14.965/7.643 = 1.9581 so
# the LLM:TF-IDF balance is identical to v20 and only the within-block ratios change.
CHANGES = {
    'q14v5':  (3.0, 3.668, 0.70, 0.914),
    'q8v5':   (2.0, 1.863, 0.44, 0.599),
    'll8v5':  (1.0, 3.602, None, None),
    'n14b':   (2.5, 0.332, None, None),
    'q25_14': (1.0, 0.246, None, None),
    'deb':    (1.0, 0.344, None, None),
}

for name, (w0, w1, g0, g1) in CHANGES.items():
    # each spec line is unique because of its name
    old = f'"name": "{name}",'
    hits = [i for i, c in enumerate(nb['cells']) if old in join(c)]
    if len(hits) != 1:
        sys.exit(f'FAIL {name}: {len(hits)} cells contain the spec')
    i = hits[0]
    src = join(nb['cells'][i])
    line_start = src.rfind('\n', 0, src.find(old)) + 1
    line_end = src.find('\n', src.find(old))
    line = src[line_start:line_end]
    if f'"w": {w0}' not in line:
        sys.exit(f'FAIL {name}: expected "w": {w0} in\n{line}')
    new_line = line.replace(f'"w": {w0}', f'"w": {w1}', 1)
    if g0 is not None:
        if f'"gauss": {g0}' not in new_line:
            sys.exit(f'FAIL {name}: expected "gauss": {g0} in\n{new_line}')
        new_line = new_line.replace(f'"gauss": {g0}', f'"gauss": {g1}', 1)
    nb['cells'][i]['source'] = (src[:line_start] + new_line + src[line_end:]).splitlines(keepends=True)
    print(f'ok {name}: w {w0} -> {w1}' + (f', gauss {g0} -> {g1}' if g0 is not None else ''))

# Also dump the per-member log-probs from the real test run. v20 did not, which is why every weight
# experiment so far costs a full ~8 h GPU rerun. With the dump published, a future weight change is a
# CPU-only re-blend (unweight each member from weights.json, re-blend, re-apply Sinkhorn).
patch("'--atlas_base', str(B8), '--atlas_w', '0.20',",
      "'--atlas_base', str(B8), '--atlas_w', '0.20', '--dump_lp_dir', '/kaggle/working/dump_test',",
      'dump per-member LP on the real test run')

json.dump(nb, open(DST, 'w'), indent=1)

meta = json.load(open('notebooks/v20/kernel-metadata.json'))
meta['id'] = 'pikaaa/name-that-llm-v22-fitted-weights'
meta['title'] = 'name that llm v22 fitted weights'
meta['code_file'] = 'name-that-llm-v22.ipynb'
json.dump(meta, open(DST.parent / 'kernel-metadata.json', 'w'), indent=2)
print('written', DST)
