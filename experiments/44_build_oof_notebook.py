"""Build the OOF harness kernel from v20.

The harness answers the question this project could never answer before: *does configuration A
beat configuration B*, without spending a leaderboard submission. It does that by running the real
v20 member stack over two held-out pool samples and dumping every per-member log-prob, so blend
weights and grouping knobs can then be refitted offline on CPU.

Why the pool and not a train fold: the frozen members were fit on all of train.csv. `infer.py
--sim k` holds out a train fold, but (a) the fold assignment is infer.py's own
`RandomState(0)`, `i % 5`, which does not match the fold assignment the adapter authors used for
`q14v5_f{k}`, and (b) `l24_predict` averages all five fold adapters plus the full adapter, so the
held-out fold is in-sample either way. The pool is the only labelled data outside every member's
fit. Contamination is then *measured* (per-member accuracy is logged) rather than assumed away.

Two runs, one GPU pass each:
  pool_eval_dense.parquet  4,099 rows / 293 prompts / mean 14 answers  -> grouping knobs
  pool_eval.parquet        3,860 rows / 3,029 prompts / mean 1.3       -> weight fitting

ATLAS and the comp stack are switched off (`--atlas_w 0 --comp_w 0`): ATLAS is budgeted at 8.3 h and
would blow the runtime for a component whose weight we are not yet refitting.
"""
import json
import pathlib
import shutil

SRC_NB = pathlib.Path('notebooks/v20/name-that-llm-v20.ipynb')
OUT = pathlib.Path('notebooks/oof')
OUT.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NB))
patched_infer = pathlib.Path('experiments/oof/infer.py').read_text()


def src(cell):
    return ''.join(cell['source'])


def cell_of(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.splitlines(keepends=True)}


cells = []

cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': """# Name That LLM — OOF harness (evaluation + weight fitting)

Runs the **unmodified v20 member stack** over two held-out pool samples and dumps every
per-member log-prob, so blend weights and grouping knobs can be refitted offline without spending
leaderboard submissions.

This exists because no honest OOF existed for any frozen member: they were all fit on the full
`train.csv`, so `infer.py --sim k` (a held-out train fold) is in-sample for them, and
`l24_predict` averages all five per-fold adapters plus the full adapter regardless.

The pool is the only labelled data outside every member's fit. Per-member accuracy is logged on
every run, which is also the **contamination check**: a member scoring far above the blend's
0.661 is in-sample and must be excluded from weight fitting.

Outputs (saved to `/kaggle/working`, then published as a dataset):
- `dump_dense/member_lp.npz` + `weights.json`
- `dump_nat/member_lp.npz` + `weights.json`
""".splitlines(keepends=True)})

# reuse v20's path-setup cell, minus the 169k-row pool union: the LLM-only harness passes
# --pool '' so the union is never read, and building it costs RAM and ~200s
_setup = json.loads(json.dumps(nb['cells'][3]))
_setup_src = ''.join(_setup['source']).replace(
    "_OPEN = glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)",
    "_OPEN = []  # harness: LLM-only, pool member disabled")
_setup['source'] = _setup_src.splitlines(keepends=True)
cells.append(_setup)

# gauss.py verbatim
cells.append(nb['cells'][4])

# patched infer.py
cells.append(cell_of('%%writefile infer.py\n' + patched_infer))

cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': """## Run the harness

Both eval sets are scored with the identical member stack and CLI flags v20 uses in the rerun,
except `--atlas_w 0 --comp_w 0` (ATLAS is budgeted at 8.3 h) and `--sim_pool` instead of a real
test file. `--dump_lp_dir` writes every `LP[key]` matrix, the cluster labels `Gte`, the pre-typography
blend, the true labels and the prompt ids.
""".splitlines(keepends=True)})

cells.append(cell_of('''import subprocess, sys, glob, os, json
EVAL_DIR = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/pool_eval_dense.parquet', recursive=True))))
print('eval dir', EVAL_DIR)

W_N14, W_N8 = 1.0, 0.5
L = [2, 4, 6, 8, 12, 16, 20, 24]
MODELS = [
  {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70},
  {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44},
  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
  {"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 1.0, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C['eos_id']},
  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},
]
print(json.dumps([{k: v for k, v in m.items() if k in ('name','type','w','keep')} for m in MODELS], indent=1))

for tag, fname in [('dense', 'pool_eval_dense.parquet'), ('nat', 'pool_eval.parquet')]:
    ev = os.path.join(EVAL_DIR, fname)
    dump = f'/kaggle/working/dump_{tag}'
    os.makedirs(dump, exist_ok=True)
    cmd = [sys.executable, 'infer.py', '--data', str(DATA), '--models', json.dumps(MODELS),
           '--perpassage', '1', '--tfidf_w', '0', '--distort_w', '0', '--svc_w', '0',
           '--pool', '', '--adapt_w', '1.0', '--adapt_thr', '0.15',
           '--sinkhorn', '0.45', '--sinkhorn_target_mix', '0.35', '--post', '0',
           '--atlas_dir', '', '--atlas_w', '0', '--svc_cent_w', '0', '--comp_w', '0',
           '--sim_pool', ev, '--sim_pool_n', '1500', '--dump_lp_dir', dump,
           '--out', f'/kaggle/working/sub_{tag}.parquet', '--cache', f'/kaggle/working/cache_{tag}']
    print('=' * 100, '\\nRUN', tag, flush=True)
    import resource as _rs
    def _rss():
        return f'{_rs.getrusage(_rs.RUSAGE_CHILDREN).ru_maxrss / 1e6:.2f} GB peak child RSS'
    print('before:', _rss(), flush=True)
    r = subprocess.run(cmd)
    print('exit', r.returncode, tag, '|', _rss(), flush=True)

print('\\nfiles:')
for p in sorted(glob.glob('/kaggle/working/dump_*/*')):
    print(' ', p, os.path.getsize(p))
'''))

nb_out = {'cells': cells, 'metadata': nb['metadata'], 'nbformat': nb['nbformat'],
          'nbformat_minor': nb['nbformat_minor']}
json.dump(nb_out, open(OUT / 'name-that-llm-oof.ipynb', 'w'), indent=1)

meta = json.load(open('notebooks/v20/kernel-metadata.json'))
meta['id'] = 'pikaaa/name-that-llm-oof-harness'
meta['title'] = 'name that llm oof harness'
meta['code_file'] = 'name-that-llm-oof.ipynb'
meta['dataset_sources'] = list(dict.fromkeys(meta['dataset_sources'] + ['pikaaa/ntl-oof-eval-sets']))
# the harness needs no competition submission output; it only needs the train.csv splitter
json.dump(meta, open(OUT / 'kernel-metadata.json', 'w'), indent=2)
print('written', OUT / 'name-that-llm-oof.ipynb')
print('datasets:', meta['dataset_sources'])
