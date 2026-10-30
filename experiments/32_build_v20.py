"""Patch the 0.660 public notebook into v20.

Three changes, each with published or measured evidence (docs/research/frontier-2026-10-29.md):

1. POOL union: nlm_pool_v5 (120k) + the 245k open pool, its own leakage flags dropped,
   de-duplicated by normalised text, capped at 11k rows per model. Feeds svc4.
2. A new member `svc4c`: the 4-view LinearSVC trained on *prompt-centred* official + open-pool
   rows, scored on test rows in inferred clusters of 2..18. Measured on grouped CV at
   member level: plain pool SVC 0.4534, official-only centred twin 0.3987, this one 0.5060.
3. ATLAS 0.10 -> 0.20 (omerfarukmerey measured 0.660 -> 0.663 on this exact base).

Everything is guarded: if the open pool is missing or the centred member raises, the notebook
falls back to the unmodified 0.660 behaviour instead of crashing the rerun.
"""
import json
import pathlib
import sys

P = pathlib.Path("notebooks/v20/name-that-llm-v20.ipynb")
nb = json.load(open(P))


def join(cell):
    return "".join(cell["source"])


def split(text):
    return text.splitlines(keepends=True)


def patch_cell(idx, old, new, label):
    cell = nb["cells"][idx]
    src = join(cell)
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAIL {label}: found {n} occurrences")
    cell["source"] = split(src.replace(old, new))
    print(f"ok {label}")


# ---------------- 1. pool union (cell 3, right after POOL resolution)
OLD_POOL = "POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))\n"
NEW_POOL = OLD_POOL + """# ---- v20: pool union. nlm_pool_v5 (120k) plus the 245k open pool (crtproduction/name-that-llm-open-pool),
# its own leakage flags dropped, de-duplicated by normalised text, capped at 11k rows per model so the
# svc4 fit stays inside the time budget. Read by svc4 and by the new prompt-centred member svc4c.
_OPEN = glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)
if _OPEN:
    import pandas as _pdp
    _a = _pdp.read_parquet(POOL).copy()
    _b = _pdp.read_parquet(_OPEN[0]).copy()
    _fl = [c for c in ('flag_near_train_prompt', 'flag_train_prompt_regen', 'flag_gemini37_routing_risk') if c in _b.columns]
    if _fl:
        _b = _b[~_b[_fl].any(axis=1)]
    _a['_grp'] = ('nl_' + _a['prompt_id'].astype(str)) if 'prompt_id' in _a.columns else ''
    _b['_grp'] = 'op_' + _b['prompt_group'].astype(str)
    _u = _pdp.concat([_a[['text', 'model', '_grp']], _b[['text', 'model', '_grp']]], ignore_index=True)
    _u['text'] = _u.text.fillna('').astype(str)
    _u['_k'] = _u.text.str.lower().str.replace(r'\\s+', ' ', regex=True)
    _u = _u[_u._k.str.len() > 0].drop_duplicates('_k')
    _u = _u.groupby('model', group_keys=False).head(11000)[['text', 'model', '_grp']].reset_index(drop=True)
    _u.to_parquet('/kaggle/working/pool_union.parquet', index=False)
    _cnt = _u.groupby('model').size()
    print(f'v20 pool union {len(_u)} rows ({_cnt.min()}-{_cnt.max()} per model), {int((_u._grp != "").sum())} grouped')
    POOL = '/kaggle/working/pool_union.parquet'
else:
    print('v20: open pool not mounted; keeping', POOL)
"""
patch_cell(3, OLD_POOL, NEW_POOL, "pool union")

# ---------------- 2a. new CLI flag
OLD_ARG = "p.add_argument('--svc_w', type=float, default=0.0, help='4-view n-gram LinearSVC on train + --pool')"
NEW_ARG = OLD_ARG + "\np.add_argument('--svc_cent_w', type=float, default=0.0, help='prompt-centred 4-view LinearSVC trained on official + grouped pool rows, scored on clustered test rows')"
patch_cell(5, OLD_ARG, NEW_ARG, "svc_cent_w arg")

# ---------------- 2b. group column on the official rows of the svc4 fit set
OLD_OFF = """    _official['_srcw'] = 1.0
    _parts.append(_official)"""
NEW_OFF = """    _official['_srcw'] = 1.0
    _official['_grp'] = 'of_' + train.prompt_id.astype(str)
    _parts.append(_official)"""
patch_cell(5, OLD_OFF, NEW_OFF, "official groups")

# ---------------- 2c. group column on the extra rows (singletons: excluded from the centred fit)
OLD_EXTRA = """        _e = EXTRA_DF[['text', 'model']].copy(); _e['_srcw'] = float(a.extra_w)"""
NEW_EXTRA = """        _e = EXTRA_DF[['text', 'model']].copy(); _e['_srcw'] = float(a.extra_w); _e['_grp'] = ''"""
patch_cell(5, OLD_EXTRA, NEW_EXTRA, "extra groups")

# ---------------- 2d. group column on the pool rows
OLD_POOLROWS = """        _P = _P[_P.model.isin(LABELS)][['text', 'model']].copy()
        _P['text'] = _P.text.fillna('').astype(str); _P['_srcw'] = 0.35"""
NEW_POOLROWS = """        _P['_grp'] = _P['_grp'].fillna('').astype(str) if '_grp' in _P.columns else ''
        _P = _P[_P.model.isin(LABELS)][['text', 'model', '_grp']].copy()
        _P['text'] = _P.text.fillna('').astype(str); _P['_srcw'] = 0.35"""
patch_cell(5, OLD_POOLROWS, NEW_POOLROWS, "pool groups")

# ---------------- 2e. the centred member itself
OLD_FIT = """    _svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
    _svc.fit(_hs(_A, format='csr'), _fy, sample_weight=_sw)
    _d = _svc.decision_function(_hs(_B, format='csr'))
    LP['svc4'] = (_d - _d.max(1, keepdims=True) - np.log(np.exp(_d - _d.max(1, keepdims=True)).sum(1, keepdims=True))) * a.svc_w
    del _A, _B, _parts, _fitdf, _svc; gc.collect(); log('svc4 done')"""
NEW_FIT = """    _svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
    _svc.fit(_hs(_A, format='csr'), _fy, sample_weight=_sw)
    _d = _svc.decision_function(_hs(_B, format='csr'))
    LP['svc4'] = (_d - _d.max(1, keepdims=True) - np.log(np.exp(_d - _d.max(1, keepdims=True)).sum(1, keepdims=True))) * a.svc_w
    if a.svc_cent_w > 0:   # v20: the same views, prompt-centred, trained on grouped official + open-pool rows
        try:
            from scipy.sparse import csr_matrix as _csr, diags as _diags

            def _group_centre(X, groups, lo=2, hi=10 ** 9):
                X = _csr(X)
                _, _inv, _cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
                _kg = np.flatnonzero((_cnt >= lo) & (_cnt <= hi))
                if not len(_kg):
                    return X.copy()
                _col = np.full(len(_cnt), -1); _col[_kg] = np.arange(len(_kg))
                _rw = np.flatnonzero(_col[_inv] >= 0)
                _M = _csr((np.ones(len(_rw)), (_rw, _col[_inv][_rw])), shape=(X.shape[0], len(_kg)))
                return (X - _M @ (_diags(1.0 / _cnt[_kg]) @ (_M.T @ X))).tocsr()

            _grp = _fitdf['_grp'].fillna('').astype(str).to_numpy()
            _idx = np.flatnonzero(_grp != '')
            if len(_idx) > 60000:
                _idx = np.random.default_rng(0).choice(_idx, 60000, replace=False)
            if len(_idx) >= 1000:
                _Xc = _group_centre(_hs([_A[v][_idx] for v in range(len(_A))], format='csr'), _grp[_idx])
                _svcc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced')
                _svcc.fit(_Xc, _fy[_idx], sample_weight=_sw[_idx])
                del _Xc; gc.collect()
                _sz = np.bincount(np.unique(Gte, return_inverse=True)[1])
                _cl = np.isin(Gte, np.flatnonzero((_sz >= 2) & (_sz <= 18)))
                _out = np.zeros((len(test), K), dtype=np.float64)
                if _cl.any():
                    _Bc = _group_centre(_hs(_B, format='csr'), Gte)[_cl]
                    _dc = _svcc.decision_function(_Bc)
                    _dc = _dc - _dc.max(1, keepdims=True)
                    _dc = _dc - np.log(np.exp(_dc).sum(1, keepdims=True))
                    _out[np.flatnonzero(_cl)[:, None], _svcc.classes_[None, :]] = _dc
                LP['svc4c'] = _out * a.svc_cent_w
                log(f"svc4c centred member: fit rows {len(_idx)}, scored {int(_cl.sum())}/{len(test)} clustered rows, weight {a.svc_cent_w}")
                del _svcc, _out; gc.collect()
            else:
                log(f'svc4c skipped: only {len(_idx)} grouped fit rows')
        except Exception as _e:
            log('svc4c FAILED, continuing without it:', repr(_e))
    del _A, _B, _parts, _fitdf, _svc; gc.collect(); log('svc4 done')"""
patch_cell(5, OLD_FIT, NEW_FIT, "centred member")

# ---------------- 3. ATLAS 20% + pass the new weight
OLD_CMD = """'--atlas_w', '0.10', '--atlas_margin', '0.3',"""
NEW_CMD = """'--atlas_w', '0.20', '--atlas_margin', '0.3', '--svc_cent_w', '0.5',"""
patch_cell(8, OLD_CMD, NEW_CMD, "atlas 0.20 + svc_cent_w")

json.dump(nb, open(P, "w"), indent=1, ensure_ascii=False)
print("written", P)
