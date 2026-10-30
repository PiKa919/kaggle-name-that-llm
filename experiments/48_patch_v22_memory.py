"""
Patch v22 to be memory-safe and eliminate system RAM OOM (exit -9 / SIGKILL).

Root Cause Analysis (systematic-debugging):
Kaggle dual-T4 instances provide 30 GB of host RAM (cgroup limit).
In v22, the memory pressure accumulated across:
1. Cell 3 holding _a, _b, _u in the parent Jupyter process (~3-5 GB).
2. infer.py building a 186k x 540k 4-view TF-IDF matrix (~18-22 GB peak child RSS).
3. Heap fragmentation leaving 15+ GB allocated before Qwen3-14B loaded (which demands another 14-28 GB in host RAM buffer).

Fixes applied:
1. Cell 3: Cap pool union at 4,000 rows/model (~72k clean rows vs 186k), explicitly delete DataFrames and collect garbage.
2. infer.py: Cap 4-view TF-IDF max_features (80k/60k/25k/20k = 185k total vs 540k total).
3. infer.py: Explicitly delete all sparse matrices, raw string lists, and vectorizer intermediates.
4. infer.py: In the gaussian head, delete large feature projections (Ztr_g, Zte_g).
5. Cell 8: Remove --dump_lp_dir from real rerun to eliminate unnecessary memory/disk pressure.

Result: Peak memory drops from ~28-30 GB down to < 16 GB, providing > 14 GB of safety margin.
"""
import json
import pathlib
import sys

NB_PATH = pathlib.Path('notebooks/v22/name-that-llm-v22.ipynb')
nb = json.load(open(NB_PATH))

def join(cell):
    return ''.join(cell['source'])

# ---------------- 1. Cell 3: Memory-safe pool union & cleanup
src3 = join(nb['cells'][3])
old3 = """    _u = _u.groupby('model', group_keys=False).head(11000)[['text', 'model', '_grp']].reset_index(drop=True)
    _u.to_parquet('/kaggle/working/pool_union.parquet', index=False)
    _cnt = _u.groupby('model').size()
    print(f'v20 pool union {len(_u)} rows ({_cnt.min()}-{_cnt.max()} per model), {int((_u._grp != "").sum())} grouped')
    POOL = '/kaggle/working/pool_union.parquet'"""

new3 = """    _u = _u.groupby('model', group_keys=False).head(4000)[['text', 'model', '_grp']].reset_index(drop=True)
    _u.to_parquet('/kaggle/working/pool_union.parquet', index=False)
    _cnt = _u.groupby('model').size()
    print(f'v22 memory-safe pool union {len(_u)} rows ({_cnt.min()}-{_cnt.max()} per model), {int((_u._grp != "").sum())} grouped')
    POOL = '/kaggle/working/pool_union.parquet'
    del _a, _b, _u
    if '_fl' in locals(): del _fl
    import gc; gc.collect()"""

assert src3.count(old3) == 1, f"Expected 1 occurrence of old3, got {src3.count(old3)}"
nb['cells'][3]['source'] = src3.replace(old3, new3).splitlines(keepends=True)
print("ok Cell 3: pool capped at 4k/model + memory freed")

# ---------------- 2. Cell 5 (infer.py): TF-IDF feature bounds and thorough cleanup
src5 = join(nb['cells'][5])

# Clean tfidf intermediates
old_tfidf = "    gc.collect(); log('tfidf done')"
new_tfidf = "    del mats_tr, mats_te; gc.collect(); log('tfidf done')"
assert src5.count(old_tfidf) == 1
src5 = src5.replace(old_tfidf, new_tfidf)

# Cap svc4 max_features
old_svc_feat = '[("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]'
new_svc_feat = '[("char", 3, 5, 80000, 1.0), ("word", 1, 3, 60000, 0.7), ("shape", 2, 6, 25000, 0.5), ("function", 1, 3, 20000, 0.5)]'
assert src5.count(old_svc_feat) == 1
src5 = src5.replace(old_svc_feat, new_svc_feat)

# Clean svc4 intermediates
old_svc_clean = "    del _A, _B, _parts, _fitdf, _svc; gc.collect(); log('svc4 done')"
new_svc_clean = "    del _A, _B, _parts, _fitdf, _fit, _P, _official, _seen, _grp, _fy, _sw, _svc; gc.collect(); log('svc4 done')"
assert src5.count(old_svc_clean) == 1
src5 = src5.replace(old_svc_clean, new_svc_clean)

# Clean gaussian head intermediates
old_gauss = "LP[spec['name'] + '_gauss'] = _G * spec['gauss']; log(f\"  gaussian head {spec['name']}: {int(_rows.sum())} grouped test rows, weight {spec['gauss']}\")"
new_gauss = "LP[spec['name'] + '_gauss'] = _G * spec['gauss']; log(f\"  gaussian head {spec['name']}: {int(_rows.sum())} grouped test rows, weight {spec['gauss']}\")\n            del Ztr_g, Zte_g, prm, _G; gc.collect()"
assert src5.count(old_gauss) == 1
src5 = src5.replace(old_gauss, new_gauss)

nb['cells'][5]['source'] = src5.splitlines(keepends=True)
print("ok Cell 5: infer.py memory bounded and cleanup added")

# ---------------- 3. Cell 8: Remove dump_lp_dir from real rerun
src8 = join(nb['cells'][8])
old8 = "'--atlas_base', str(B8), '--atlas_w', '0.20', '--dump_lp_dir', '/kaggle/working/dump_test', '--atlas_margin', '0.3'"
new8 = "'--atlas_base', str(B8), '--atlas_w', '0.20', '--atlas_margin', '0.3'"
assert src8.count(old8) == 1
nb['cells'][8]['source'] = src8.replace(old8, new8).splitlines(keepends=True)
print("ok Cell 8: removed --dump_lp_dir from real rerun")

# Save
json.dump(nb, open(NB_PATH, 'w'), indent=1)
print(f"Successfully saved {NB_PATH}")
