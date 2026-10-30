"""Patch the v20 infer.py into an OOF/eval harness.

Two additions, both needed because the frozen members were trained on *all* of train.csv:

1. `--sim_pool <parquet>` — like `--sim`, but the fake test comes from a labelled external pool
   sample instead of a held-out train fold. A held-out train fold is in-sample for every frozen
   member, so `--sim` cannot measure them; the pool can. Requires columns `id`, `text`, `model`,
   `prompt_group`.

2. `--dump_lp_dir <dir>` — saves every per-member `LP[key]` matrix, the cluster labels `Gte`, the
   pre-typography blend `tot`, the true labels, and a `weights.json` recording the multiplier
   already baked into each key. That lets the blend be refitted offline on CPU with no GPU and no
   new submission: unweight each component, refit, re-blend.
"""
import pathlib
import sys

SRC = pathlib.Path('experiments/oof/infer.py')
s = SRC.read_text()


def patch(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f'FAIL {label}: {n} occurrences')
    s = s.replace(old, new)
    print(f'ok {label}')


# ---------------- 1. new args
patch("""p.add_argument('--sim_k', default='', help="sim only: keep 'lo-hi' random passages per held-out prompt")""",
      """p.add_argument('--sim_k', default='', help="sim only: keep 'lo-hi' random passages per held-out prompt")
p.add_argument('--sim_pool', default='', help='evaluate on a labelled external pool sample (honest for frozen members): parquet with id,text,model,prompt_group')
p.add_argument('--sim_pool_n', type=int, default=0, help='subsample the pool eval set to N rows (0 = all)')
p.add_argument('--dump_lp_dir', default='', help='save per-member LP matrices + Gte + blend for offline weight fitting')""",
      'new args')

# ---------------- 2. pool eval branch, placed before the train-fold sim branch
patch("""train = pd.read_csv(os.path.join(a.data, 'train.csv'))
if a.sim:""",
      """train = pd.read_csv(os.path.join(a.data, 'train.csv'))
if a.sim_pool:
    # Honest held-out evaluation: the frozen members were fit on all of train.csv, so a held-out
    # train fold is in-sample for them. The external pool is not (beyond the overlap we strip).
    _ep = pd.read_parquet(a.sim_pool)
    for _c in ('id', 'text', 'model', 'prompt_group'):
        assert _c in _ep.columns, f'sim_pool needs column {_c}'
    if a.sim_pool_n:
        _ep = _ep.sample(min(len(_ep), a.sim_pool_n), random_state=0)
    _ep = _ep.reset_index(drop=True)
    test = _ep[['id', 'text']].copy()
    test_truth = _ep[['id', 'text', 'model', 'prompt_group']].copy()
    test_truth = test_truth.rename(columns={'prompt_group': 'prompt_id'})
    hold_any = np.zeros(len(train), bool)          # train stays whole: the heads were fit on it
    a.sim = 'pool'                                  # reuse the sim reporting + guards downstream
    log('sim_pool', len(test), 'eval rows from', a.sim_pool)
elif a.sim:""",
      'sim_pool data branch')

# ---------------- 3. dump per-member LP after the blend is final, before typography
patch("""_v38_before_hard = tot.copy()""",
      """_v38_before_hard = tot.copy()
if a.dump_lp_dir:
    import pathlib as _dp
    _dp.Path(a.dump_lp_dir).mkdir(parents=True, exist_ok=True)
    _wt = {}
    for _k in LP:
        if _k.endswith('_gauss'):
            _wt[_k] = next((float(sp.get('gauss', 0)) for sp in specs if sp['name'] + '_gauss' == _k), 1.0)
        elif _k.endswith('_twin'):
            _base = _k[:-5]
            _sp = next((sp for sp in specs if sp['name'] == _base), None)
            _wt[_k] = (a.adapt_w * float(_sp.get('w', 1.0)) * float(_sp.get('twin', 1.0))) if _sp else 1.0
        elif any(sp['name'] == _k for sp in specs):
            _wt[_k] = float(next(sp for sp in specs if sp['name'] == _k).get('w', 1.0))
        else:
            _wt[_k] = float({'tfidf': a.tfidf_w, 'distort': a.distort_w,
                             'svc4': a.svc_w, 'svc4c': a.svc_cent_w}.get(_k, 1.0))
    np.savez_compressed(os.path.join(a.dump_lp_dir, 'member_lp.npz'),
                        **{f'lp__{_k}': np.asarray(_v, np.float32) for _k, _v in LP.items()},
                        Gte=Gte.astype(np.int32), tot=tot.astype(np.float32),
                        y=(test_truth.model.map(L2I).values.astype(np.int32) if a.sim_pool
                           else np.full(len(test), -1, np.int32)),
                        prompt=(test_truth.prompt_id.values if a.sim_pool
                                else np.array([''] * len(test), dtype=object)),
                        ids=test.id.astype(str).values)
    json.dump({'weights': _wt, 'adapt_w': float(a.adapt_w), 'adapt_thr': float(a.adapt_thr),
               'sinkhorn': float(a.sinkhorn), 'sinkhorn_target_mix': float(a.sinkhorn_target_mix),
               'atlas_w': str(a.atlas_w), 'comp_w': float(a.comp_w), 'sk_div': int(a.sk_div),
               'perpassage': int(a.perpassage), 'post': int(a.post)},
              open(os.path.join(a.dump_lp_dir, 'weights.json'), 'w'), indent=1)
    log('dumped per-member LP:', len(LP), 'keys ->', a.dump_lp_dir)""",
      'dump per-member LP')

SRC.write_text(s)
print('patched', SRC)
