"""
experiments/50_build_v24_gold_master.py

Synthesizes the v24 Gold Master Pipeline for 'Name That LLM'.
Combines:
1. 6 Diverse LLM Backbones: Qwen3-14B (q14v5), Qwen3-8B (q8v5), Llama-3.1-8B (ll8v5),
   OLMo-2-7B (olmo7), Mistral-7B (mistral7), DeBERTa-v3-large (deb).
2. DataParallel 8B sharding (dp: True) across dual T4 GPUs.
3. Memory safeguards: capped pool union (72k rows), capped sparse features (185k), gc.collect().
4. Roman Tamrazov's V40A Sibling-Style Residual (sibling_style.py).
5. Hierarchical Beta-Bernoulli Typography model (kappa=24.0, 37 markers).
6. 20% ATLAS autoregressive token likelihood with dynamic time budget.
7. LinearSVC 4-view calibrated classifier (pool weight 0.25, C=0.5).
8. Soft Sinkhorn-Knopp prior balancing (0.45, target mix 0.35).
"""
import os
import json
import pathlib
import shutil

BASE_V22_DIR = pathlib.Path("notebooks/v22")
DST_V24_DIR = pathlib.Path("notebooks/v24")
DST_V24_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load v22 notebook as starting base
with open(BASE_V22_DIR / "name-that-llm-v22.ipynb") as f:
    nb = json.load(f)

print(f"Loaded v22 notebook with {len(nb['cells'])} cells")

# 2. Add sibling_style.py as a separate cell before infer.py (after gauss.py)
sibling_style_code = '''%%writefile sibling_style.py
import re
import numpy as np
from functools import lru_cache

MARKERS = [',','.', ';', ':', '!', '?','—','–','-','’','‘','“','”','"',"'",'(',')','[',']','{','}','*','`','•','…','/','\\\\',':','\\n','\\n\\n','\\n-','\\n1.','\\n2.','##','```','**','> ','\\t',' = ', ' e.g.', ' i.e.', 'However,','Overall,','In conclusion','Therefore,','Moreover,','First,','Finally,',' - ',': ', ' ; ', '  ']
WORDS = ['the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its','they','their','there','this','these','those','however','therefore','moreover','although','because','perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as','so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all','any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who']
WRE=re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)?")
SRE=re.compile(r'(?<=[.?!])\\s+')

@lru_cache(maxsize=15000)
def row(t):
    t = str(t)
    s = WRE.findall(t); lo=[z.lower() for z in s]; n=max(len(s),1); nt=max(len(t),1); ns=max(len(t.splitlines()),1)
    lens=np.array([len(z) for z in s] or [0]); sent_lens=np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [np.log1p(n),np.log1p(nt), nt/n,ns/n,len(set(lo))/n, float(np.mean(lens)), float(np.std(lens)), float(np.mean(sent_lens)), float(np.std(sent_lens)),float(np.max(sent_lens)),float(np.mean(lens>=8)),sum(c.isupper() for c in t)/nt, sum(c.isdigit() for c in t)/nt,sum(ord(c)>127 for c in t)/nt, sum(c.isspace() for c in t)/nt, sum(c=='\\n' for c in t)/nt]
    base += [min(t.count(m)/n,3) for m in MARKERS]
    cnt = {}
    for w in lo: cnt[w]=cnt.get(w,0)+1
    base += [cnt.get(w,0)/n for w in WORDS]
    return np.array(base, dtype=np.float32)

def features(texts):
    return np.stack([row(str(t)) for t in texts])

def apply_grouped_style_residual(logits, train_text, test_text, labels, y, groups, max_alpha=0.08, diagnostics_path=None):
    """OOF-estimated, conservative sibling re-ranking. Cross-family probability mass is mathematically unchanged."""
    import json
    import numpy as np
    from scipy.special import logsumexp, softmax
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score
    Xt = features(train_text)
    Xq = features(test_text)
    out = np.asarray(logits, dtype=np.float64).copy()
    lab = np.asarray(labels)
    yy = np.asarray(y)
    g = np.asarray(groups)
    diag = {}
    for family, C in [('claude', 0.003), ('gemini', 0.003), ('gpt', 0.003), ('glm', 0.03)]:
        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
        if len(cols) < 2:
            continue
        train_mask = np.isin(yy, cols)
        inds = np.flatnonzero(train_mask)
        X = Xt[inds]
        target = np.searchsorted(cols, yy[inds])
        grp = g[inds]
        oof = np.empty(len(target), dtype=np.int32)
        for a, b in GroupKFold(n_splits=5).split(X, target, groups=grp):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
            model.fit(X[a], target[a])
            oof[b] = model.predict(X[b])
        bacc = float(balanced_accuracy_score(target, oof))
        chance = 1.0 / len(cols)
        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)
        if reliability <= 0:
            diag[family] = dict(oof_balanced_accuracy=bacc, random_level=chance, alpha_mean=0.)
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
        model.fit(X, target)
        style_p = np.maximum(model.predict_proba(Xq), 1e-8)
        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)
        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]
        gate = np.clip((0.65 - margin) / 0.65, 0., 1.)
        strength = (max_alpha * reliability) * gate
        mix = (1. - strength[:, None]) * old_p + strength[:, None] * style_p
        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-8)
        diag[family] = dict(oof_balanced_accuracy=bacc, random_level=chance,
                            alpha_mean=float(strength.mean()), alpha_max=float(strength.max()),
                            changed_within_family=int(np.sum(old_p.argmax(1) != mix.argmax(1))))
    if diagnostics_path:
        with open(diagnostics_path, 'w') as f: json.dump(diag, f, indent=2)
    return out
'''

# 3. Pull Cell 3 (paths discovery) and enhance with OLMo-2 and Mistral paths
cell3_src = "".join(nb["cells"][3]["source"])
path_patch = """
# v24: Locate OLMo2-7B and Mistral-7B models and adapters
BOLMO = next((os.path.dirname(x) for x in cfgs if '/olmo-2/' in x.lower() and ('/1124-7b' in x.lower() or '/7b' in x.lower())), None)
BMIST = next((os.path.dirname(x) for x in cfgs if '/mistral/' in x.lower() and ('/7b' in x.lower() or 'v0.1' in x.lower())), None)

_olmo_heads = glob.glob('/kaggle/input/**/ntl-ts-olmo2-7b/heads.safetensors', recursive=True)
AOLMO = os.path.dirname(_olmo_heads[0]) if _olmo_heads else None

_mistral_heads = glob.glob('/kaggle/input/**/ntl-ts-mistral7b/heads.safetensors', recursive=True)
AMIST = os.path.dirname(_mistral_heads[0]) if _mistral_heads else None

print('v24 diverse models:', BOLMO, AOLMO, BMIST, AMIST)
"""
if "BOLMO" not in cell3_src:
    cell3_src += "\n" + path_patch
    nb["cells"][3]["source"] = cell3_src.splitlines(keepends=True)

# 4. Enhance infer.py (Cell 5) with:
#    a) DataParallel 8B sharding (dp: True)
#    b) Sibling-style residual in typography section
infer_src = "".join(nb["cells"][5]["source"])

# Check if l24_predict has DataParallel support
if "ThreadPoolExecutor" not in infer_src:
    print("Injecting DataParallel 8B sharding into l24_predict...")
    # Replace standard l24_predict with DataParallel sharded l24_predict
    # We find def l24_predict and replace its body up to return
    old_l24_start = "def l24_predict(spec, texts, groups=None, train_texts=None):"
    
    # We can pull the verified l24_predict from synthreaper
    with open("/tmp/synthreaper/ntl-gold-apex-champion-0-670.ipynb") as sf:
        s_nb = json.load(sf)
    s_c8 = "".join(s_nb["cells"][8]["source"])
    
    # Extract l24_predict from synthreaper
    l24_start_idx = s_c8.find("@torch.no_grad()\ndef l24_predict")
    l24_end_idx = s_c8.find("def tfidf_predict", l24_start_idx)
    s_l24 = s_c8[l24_start_idx:l24_end_idx]
    
    # Find in infer_src
    curr_l24_start = infer_src.find("@torch.no_grad()\ndef l24_predict")
    if curr_l24_start == -1:
        curr_l24_start = infer_src.find("def l24_predict")
    curr_l24_end = infer_src.find("def tfidf_predict", curr_l24_start)
    
    infer_src = infer_src[:curr_l24_start] + s_l24 + infer_src[curr_l24_end:]
    print("Successfully replaced l24_predict with DataParallel version")

# Now inject sibling_style invocation into typography section in infer_src
if "apply_grouped_style_residual" not in infer_src:
    print("Injecting sibling_style into typography section of infer.py...")
    old_typo_tail = """    _raw_sub = _v31_sub(_raw)
    _safe = _v31_sub(_safe_logits)
    _safe.to_parquet(_out, index=False)"""

    new_typo_tail = """    _raw_sub = _v31_sub(_raw)
    _reference_logits = _safe_logits.copy()
    try:
        from sibling_style import apply_grouped_style_residual
        _safe_logits = apply_grouped_style_residual(
            _safe_logits, train.text.values, test.text.values, LABELS, y, Gtr,
            max_alpha=0.08, diagnostics_path=str(_out.with_name('v40a_style_oof.json')))
        log('V40A family-only OOF residual fitted successfully')
    except Exception as _style_error:
        log('V40A residual fallback to reference:', repr(_style_error))
        _safe_logits = _reference_logits
    _safe = _v31_sub(_safe_logits)
    _safe.to_parquet(_out, index=False)"""

    if old_typo_tail in infer_src:
        infer_src = infer_src.replace(old_typo_tail, new_typo_tail)
        print("Successfully patched typography tail with sibling_style")
    else:
        print("WARNING: old_typo_tail not found in infer_src, searching alternative match...")
        # Search for _safe.to_parquet(_out, index=False)
        pos = infer_src.find("_safe.to_parquet(_out, index=False)")
        if pos != -1:
            line_start = infer_src.rfind("\n", 0, pos)
            pre = infer_src[:line_start]
            post = infer_src[pos + len("_safe.to_parquet(_out, index=False)"):]
            patch = """
    _reference_logits = _safe_logits.copy()
    try:
        from sibling_style import apply_grouped_style_residual
        _safe_logits = apply_grouped_style_residual(
            _safe_logits, train.text.values, test.text.values, LABELS, y, Gtr,
            max_alpha=0.08, diagnostics_path=str(_out.with_name('v40a_style_oof.json')))
        log('V40A family-only OOF residual fitted successfully')
    except Exception as _style_error:
        log('V40A residual fallback to reference:', repr(_style_error))
        _safe_logits = _reference_logits
    _safe = _v31_sub(_safe_logits)
    _safe.to_parquet(_out, index=False)"""
            infer_src = pre + patch + post
            print("Patched typography tail via alternative position")

nb["cells"][5]["source"] = infer_src.splitlines(keepends=True)

# 5. Insert sibling_style cell before infer.py (between Cell 4 and Cell 5)
cell_sibling = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": sibling_style_code.splitlines(keepends=True)
}

# Check if sibling_style is already a cell
has_sibling_cell = any("sibling_style.py" in "".join(c["source"]) for c in nb["cells"])
if not has_sibling_cell:
    nb["cells"].insert(5, cell_sibling)
    print("Inserted sibling_style.py cell at index 5")

# 6. Now Cell 8 becomes Cell 9: update MODELS definition
# Find the cell that contains MODELS =
for i, c in enumerate(nb["cells"]):
    c_src = "".join(c["source"])
    if "MODELS =" in c_src:
        models_cell_idx = i
        break

print(f"MODELS definition is in Cell {models_cell_idx}")
cell_models_src = "".join(nb["cells"][models_cell_idx]["source"])

# Replace MODELS with diverse backbones + dp: True
v24_models_code = """    MODELS = [
      {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.668, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.9140},
      {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 1.863, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.599, "dp": True},
      {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 3.602, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True},
    ]
    if BOLMO and AOLMO:
        MODELS.append({"name": "olmo7", "type": "l24", "path": BOLMO, "adapters": [AOLMO], "keep": 24, "layers": L, "w": 1.500, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True})
    if BMIST and AMIST:
        MODELS.append({"name": "mistral7", "type": "l24", "path": BMIST, "adapters": [AMIST], "keep": 24, "layers": L, "w": 1.500, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True})
    MODELS.append({"name": "deb", "type": "enc", "ckpt": DEB, "w": 0.344})"""

# Find where MODELS is defined in cell_models_src
m_start = cell_models_src.find("    MODELS = [")
m_end = cell_models_src.find("]", m_start)
# Also check if deb or extra append was there
if "MODELS.append" in cell_models_src[m_start:m_start+1000]:
    m_end = cell_models_src.find("MODELS.append({\"name\": \"deb\"", m_start)
    m_end = cell_models_src.find(")\n", m_end) + 2
else:
    m_end = cell_models_src.find("    ]\n", m_start) + 6

cell_models_src = cell_models_src[:m_start] + v24_models_code + cell_models_src[m_end:]
nb["cells"][models_cell_idx]["source"] = cell_models_src.splitlines(keepends=True)
print("Updated MODELS with diverse backbones and dp: True")

# 7. Write kernel-metadata.json for v24
metadata = {
  "id": "pikaaa/name-that-llm-v24-gold-master",
  "title": "name that llm v24 gold master",
  "code_file": "name-that-llm-v24.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": ["gpu"],
  "dataset_sources": [
    "dhruvtibarewal/nlm-l24-adapters",
    "seyitkaangunes/ntl-extra-api-passages",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
    "dhruvtibarewal/nlm-matched-clean",
    "dhruvtibarewal/nlm-offline-wheels",
    "josephadamski91/ntl-atlas-qwen3-8b-v1",
    "lucasboesen/ntl-free-proxy-generations",
    "nishantdahal/ntl-two-stage-lora",
    "nishantdahal/ntl-handwriting-blend",
    "dhruvtibarewal/nlm-l24-adapters-v5",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "pjmathematician/ultrabert-380m-ntl",
    "crtproduction/name-that-llm-open-pool",
    "alexchilton/ntl-ts-olmo2-7b",
    "alexchilton/ntl-ts-mistral7b"
  ],
  "kernel_sources": [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2",
    "vnton37nguyn/ntl-train-lora-qwen25-instyle"
  ],
  "competition_sources": ["name-that-llm"],
  "model_sources": [
    "alexchilton/llama-3-1-8b-base/Transformers/default/1",
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/8b/1",
    "qwen-lm/qwen-3/Transformers/14b/1",
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen2.5/Transformers/7b/1",
    "qwen-lm/qwen2.5/Transformers/14b/1",
    "allenai/olmo-2/Transformers/1124-7b/1",
    "mistral-ai/mistral/PyTorch/7b-v0.1-hf/1"
  ],
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
  "machine_shape": "NvidiaTeslaT4"
}

with open(DST_V24_DIR / "kernel-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

with open(DST_V24_DIR / "name-that-llm-v24.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully created v24 Gold Master at {DST_V24_DIR}")
