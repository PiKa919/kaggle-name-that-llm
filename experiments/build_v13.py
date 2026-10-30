"""Build notebooks/v13/script.py from the scored v10 kernel.

Two zero-extra-compute changes:
  1. SVC: curated 80k pool, pool sample_weight 0.25, transductive TF-IDF vocabulary.
     Bench11 grouped-CV: 0.4667 vs v10's full-pool 0.414.
  2. lora24: capture the legacy readouts and re-weight the prompt-centred head at
     the top level (it is otherwise buried at 0.15 * 0.15 = 0.0225 effective).

No extra model forward passes, so runtime stays at v10's 11,331 s.
"""
import ast
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "v10" / "script.py"
OUT = ROOT / "notebooks" / "v13" / "script.py"
META_IN = ROOT / "notebooks" / "v10" / "kernel-metadata.json"
META_OUT = ROOT / "notebooks" / "v13" / "kernel-metadata.json"

src = SRC.read_text()

# ---------------------------------------------------------------- legacy capture
tree = ast.parse(src)
node = next(
    n for n in tree.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "_GAUSSIAN_SOURCE" for t in n.targets)
)
literal = ast.get_source_segment(src, node.value)
legacy = ast.literal_eval(literal)

OLD_ADD = '''def add_member(key, prob, rows=None, weight=None):
    """Add weight * log(prob) to the blend (all rows, or the given row subset)."""
    w = W[key] if weight is None else weight'''
NEW_ADD = '''LEGACY_MEMBERS = {}
def add_member(key, prob, rows=None, weight=None):
    """Add weight * log(prob) to the blend (all rows, or the given row subset)."""
    if key.startswith("lora24_"):
        LEGACY_MEMBERS[key] = (None if rows is None else np.asarray(rows).copy(), np.array(prob, dtype=np.float64, copy=True))
    w = W[key] if weight is None else weight'''
assert legacy.count(OLD_ADD) == 1, "legacy add_member not found exactly once"
legacy = legacy.replace(OLD_ADD, NEW_ADD)
assert "LEGACY_MEMBERS = {}" in legacy and "LEGACY_MEMBERS[key]" in legacy

# q25_7b has blend weight 0 and no Gaussian group head (the four measured views are
# q3_8b / glm9b / q3_4b / debv3l), so its train+test forward pass is dead weight.
# Dropping it is output-identical and buys ~10 min of the 4 h submission budget.
OLD_Q25 = "    ('q25_7b', 'qwen2', 3584, [(6, 'mean'), (20, 'mean')], [], 0.0025),\n"
assert legacy.count(OLD_Q25) == 1, "q25_7b LM head line not found exactly once"
legacy = legacy.replace(OLD_Q25, "")
assert "'q25_7b', 'qwen2'" not in legacy

src = src.replace(literal, repr(legacy), 1)

# ---------------------------------------------------------------- curated-pool SVC
OLD_POOL = 'pool = pd.read_parquet(find("pool_v1.parquet"))'
NEW_POOL = '''SVC_POOL_WEIGHT = 0.25
pool = pd.read_parquet(find("curated_pool.parquet"))
if "split" in pool:
    _drop = int((pool.split == "train_regen").sum())
    pool = pool[pool.split != "train_regen"]
    print(f"dropped train_regen rows (official train prompts): {_drop}", flush=True)
if "register_ok" in pool:
    pool = pool[pool.register_ok.astype(bool)]
pool = pool[["text", "model"]].copy()
pool["text"] = pool["text"].astype(str)
pool = pool[pool.model.isin(LABELS)]
pool["n"] = pool.text.str.split().str.len()
pool = pool[(pool.n >= 30) & (pool.n <= 300)].drop_duplicates("text").reset_index(drop=True)
print(f"curated pool {len(pool)} rows", flush=True)
print("pool class counts", np.bincount(pool.model.map({l: i for i, l in enumerate(LABELS)}).values, minlength=K).tolist(), flush=True)'''
assert src.count(OLD_POOL) == 1, "v10 pool line not found exactly once"
src = src.replace(OLD_POOL, NEW_POOL, 1)

OLD_FIT = '''fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]'''
NEW_FIT = '''fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]
fit_w = np.r_[np.ones(len(train)), np.full(len(pool), SVC_POOL_WEIGHT)]
vocab_texts = fit_texts + test.text.tolist()   # transductive vocabulary, labels only on fit rows'''
assert src.count(OLD_FIT) == 1, "v10 fit_texts block not found exactly once"
src = src.replace(OLD_FIT, NEW_FIT, 1)

OLD_VIEWS = '''    vec = TfidfVectorizer(**opts)
    A.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
    B.append(vec.transform([prep(t) for t in test.text]) * weight)
svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(hstack(A, format="csr"), fit_y)'''
NEW_VIEWS = '''    vec = TfidfVectorizer(**opts)
    vec.fit([prep(t) for t in vocab_texts])
    A.append(vec.transform([prep(t) for t in fit_texts]) * weight)
    B.append(vec.transform([prep(t) for t in test.text]) * weight)
svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(
    hstack(A, format="csr"), fit_y, sample_weight=fit_w)'''
assert src.count(OLD_VIEWS) == 1, "v10 SVC view block not found exactly once"
src = src.replace(OLD_VIEWS, NEW_VIEWS, 1)

# ---------------------------------------------------------------- top-level lora24 reweight
ANCHOR = ("print('BLEND two-stage .85 + original four-Gaussian stack .15; "
          "labels aligned; single final Sinkhorn', flush=True)")
assert src.count(ANCHOR) == 1, "blend print not found exactly once"
REWEIGHT = ANCHOR + '''

# ---- top-level lora24 re-weight -------------------------------------------------
# The legacy block computes the lora24 readouts but contributes only 15% of the blend;
# inside that, the centred twin is weighted 0.15, so its effective weight is 0.0225.
# The prompt-centred lora24 head is the strongest single readout measured (0.596
# prompt-grouped OOF), and it is nearly independent of the two-stage base.  Re-weight
# it here, where LOGIT is a normalised log-probability, so the OOF-fitted multipliers
# transfer directly.  No model forward pass is added: the probabilities already exist.
_LEGACY_MEMBERS = _old_namespace.get('LEGACY_MEMBERS', {})
EXTRA_LORA24 = {'lora24_public': 0.10, 'lora24_public_twin': 0.70,
                'lora24_seed7': 0.05, 'lora24_seed7_twin': 0.0,
                'lora24_claude': 0.05, 'lora24_claude_twin': 0.15}
for _key, _w in EXTRA_LORA24.items():
    if _key not in _LEGACY_MEMBERS or _w <= 0:
        continue
    _rows, _prob = _LEGACY_MEMBERS[_key]
    _prob = np.asarray(_prob)[:, _old_cols]   # legacy sorted-label columns -> production LABELS order
    _lp = np.log(np.clip(_prob, 1e-9, 1.0))
    if _rows is None:
        LOGIT += _w * _lp
    else:
        LOGIT[np.asarray(_rows)] += _w * _lp
    print(f'REWEIGHT {_key} +{_w:.2f} on {len(_prob)} rows', flush=True)
_member_payload = {'ids': test['id'].astype(str).to_numpy(), 'labels': np.array(LABELS),
                   'base': _mix_prob.astype(np.float32)}
for _key, (_rows, _prob) in _LEGACY_MEMBERS.items():
    _member_payload[_key + '_prob'] = np.asarray(_prob)[:, _old_cols].astype(np.float32)
    _member_payload[_key + '_rows'] = (np.arange(len(test)) if _rows is None else np.asarray(_rows)).astype(np.int32)
np.savez_compressed('members_test.npz', **_member_payload)
print(f'member matrix saved: {sorted(_member_payload)}', flush=True)'''
src = src.replace(ANCHOR, REWEIGHT, 1)

# ---------------------------------------------------------------- ATLAS payload
ATLAS_ANCHOR = "print(f'ATLAS predictions done ({time.time() - T0:.0f}s)', flush=True)"
assert src.count(ATLAS_ANCHOR) == 1, "ATLAS print not found exactly once"
src = src.replace(ATLAS_ANCHOR, ATLAS_ANCHOR + '''
_member_payload['atlas'] = ATLAS_PROB.astype(np.float32)
np.savez_compressed('members_test.npz', **_member_payload)
print(f'member matrix saved with atlas: {sorted(_member_payload)}', flush=True)''', 1)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(src)
compile(src, str(OUT), "exec")
print("V13_BUILD_OK", OUT, len(src))

# ---------------------------------------------------------------- metadata
meta = json.loads(META_IN.read_text())
meta["id"] = "pikaaa/name-that-llm-v13"
meta["title"] = "name that llm v13"
if "ankiteshtiwari/ntl-curated-pool" not in meta["dataset_sources"]:
    meta["dataset_sources"].append("ankiteshtiwari/ntl-curated-pool")
META_OUT.write_text(json.dumps(meta, indent=2) + "\n")
print("META_OK", meta["id"], len(meta["dataset_sources"]), "datasets")
print("datasets:", meta["dataset_sources"])
