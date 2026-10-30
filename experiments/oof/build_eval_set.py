"""Build the held-out evaluation set for the OOF harness.

Why a pool sample and not a train fold: the frozen members (q14v5, q8v5, ll8v5, n14b, q25_14, deb)
were fit on all 2,699 rows of train.csv, so any held-out train fold is in-sample for them. Their
accuracy there would be meaningless. The external pool is the only labelled data they were not fit
on (beyond overlap, which infer.py strips for --extra and --pool, and which we check empirically
after the first run).

Construction:
  * clean rows only (drop the three leakage flags)
  * sample whole `prompt_group`s, not rows, so cluster recovery is exercised the way it is on test
  * stratified by model so every class is represented, mirroring balanced-accuracy weighting
  * ~4,000 rows, the size of the hidden test

Output: experiments/oof/pool_eval.parquet with columns id, text, model, prompt_group, source_dataset
"""
import numpy as np
import pandas as pd

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])

TARGET = 4000
PER_MODEL = TARGET // len(LABELS)

p = pd.read_parquet('/tmp/pool/ntl_open_pool.parquet')
clean = p[~p.flag_near_train_prompt & ~p.flag_train_prompt_regen & ~p.flag_gemini37_routing_risk]
clean = clean[clean.model.isin(LABELS)].drop_duplicates('text').copy()
print(f'clean pool {len(clean)} rows, {clean.prompt_group.nunique()} prompt groups')

rng = np.random.default_rng(42)
chosen, used_groups = [], set()
for m in LABELS:
    sub = clean[clean.model == m]
    got, tries = [], 0
    while len(got) < PER_MODEL and tries < 200:
        tries += 1
        g = sub.prompt_group.values[rng.integers(len(sub))]
        if g in used_groups:
            continue
        row = sub[sub.prompt_group == g]
        if len(row) > 18:
            row = row.sample(min(len(row), 18), random_state=int(rng.integers(1 << 30)))
        got.append(row)
        used_groups.add(g)
    if got:
        chosen.append(pd.concat(got))
    print(f'  {m:24s} {sum(len(x) for x in got):4d}')

ev = pd.concat(chosen, ignore_index=True)
ev = ev[['id', 'text', 'model', 'prompt_group', 'source_dataset']].copy()
ev['id'] = ev.id.astype(str)
ev = ev.drop_duplicates('id').reset_index(drop=True)

print(f'\neval set {len(ev)} rows, {ev.prompt_group.nunique()} prompt groups, '
      f'{ev.model.nunique()} models')
print('rows per model:', ev.model.value_counts().min(), '-', ev.model.value_counts().max())
sz = ev.groupby('prompt_group').size()
print(f'group sizes: mean {sz.mean():.2f}, median {sz.median():.0f}, max {sz.max()}')
print('sources:', ev.source_dataset.nunique())
assert ev.model.isin(LABELS).all() and ev.text.notna().all() and ev.id.is_unique

ev.to_parquet('experiments/oof/pool_eval.parquet', index=False)
print('\nwrote experiments/oof/pool_eval.parquet')

# ---------------- dense variant: groups of 8-18, to exercise clustering / centring / Sinkhorn
# The natural pool density is 1.27 rows/group, which leaves the grouping paths nearly unused.
# The hidden test is ~4,000 rows over unseen prompts; if that is ~222 prompts it is dense, so a
# dense eval set is the right instrument for the knobs that act on clusters.
sz_all = clean.groupby('prompt_group').size()
dense_groups = sz_all[(sz_all >= 8) & (sz_all <= 18)].index
dense = clean[clean.prompt_group.isin(dense_groups)]
print(f'\ndense candidate {len(dense)} rows over {dense.prompt_group.nunique()} groups')

rng2 = np.random.default_rng(7)
pick, used = [], set()
for m in LABELS:
    need, tries = 220, 0
    have = 0
    while have < need and tries < 400:
        tries += 1
        g = dense.prompt_group.values[rng2.integers(len(dense))]
        if g in used:
            continue
        row = dense[dense.prompt_group == g]
        if not (row.model == m).any():
            continue
        used.add(g)
        pick.append(row)
        have += len(row)
dv = pd.concat(pick, ignore_index=True)
dv = dv[['id', 'text', 'model', 'prompt_group', 'source_dataset']].copy()
dv['id'] = dv.id.astype(str)
dv = dv.drop_duplicates('id').reset_index(drop=True)
dsz = dv.groupby('prompt_group').size()
print(f'dense eval {len(dv)} rows, {dv.prompt_group.nunique()} groups, '
      f'mean group size {dsz.mean():.2f}, models {dv.model.nunique()}')
print('rows per model:', dv.model.value_counts().min(), '-', dv.model.value_counts().max())
assert dv.model.isin(LABELS).all() and dv.id.is_unique
dv.to_parquet('experiments/oof/pool_eval_dense.parquet', index=False)
print('wrote experiments/oof/pool_eval_dense.parquet')
