import gc, glob, json, os, re, time
import numpy as np
import pandas as pd
import torch
from torch import nn

T0 = time.time()
INPUT_DIR = "/kaggle/input"

# ============================================================ v18 = v17 with the Part Two trunks ON, source-first
# The question: v16 showed the pool_v4-trained 14B pair (lora_qwen3_14b / lora_qwen25_14b) plus a fine-tuned
# DeBERTa-v3-large cannot ship because its public run took 21,512 s. Root cause was ordering, not the models:
# the embedded public stack ran *after* both 14B trunks and was 34-43x slower (10,257 s for a block that takes
# 236 s in v15), i.e. accelerate silently offloaded it to CPU once VRAM was fragmented.
# v18 runs the embedded stack FIRST, frees its namespace, then loads the 14B pair. Everything else is v17:
#   - the SVC block is v14's curated_pool.parquet + class_weight="balanced" + flat 0.25 (the 0.610 block),
#     decoupled from STAGE2_TRUNKS so this run changes one thing only
#   - 1C family gates and 1D gated ATLAS + damped Sinkhorn as in v17
#   - Part One trunks (lora8b_v1 / lora14b_v1) are REPLACED by the Part Two pair; DeBERTa weight 0.10
# A diagnostic prints the share of each trunk's parameters that landed on CPU, so a repeat of the 43x
# slowdown is visible in the log instead of just in the wall clock.
POOL_SHARE = None            # 1A: None = v14's flat 0.25 pool weight (measured 0.610);
                             #     0.35 = class-balanced share, official train keeps 65% of each class
PATCH_UNIFIED_GROUPS = False # 1B: OFF by default. Measured on 40 real train prompts (720 rows), v10's
                             #     English-stop-word TF-IDF cut at 0.85 gives pair precision 1.00 / recall 0.74
                             #     (same-prompt distance median 0.816, different-prompt p10 0.968), i.e. the cut
                             #     already sits in the gap. The mixed content+hidden cut at 0.60 recovers only 0.34
                             #     recall on the same data, so switching would shrink the twin/Gaussian row set
                             #     from ~95% to ~87% of rows for no measured gain. Kept as a switch for a probe.
PATCH_GATES = True           # 1C: apply the four family StyleGram gates that v10 computed and discarded
PATCH_ATLAS_GATE = True      # 1D: ATLAS weight from row margin instead of a flat 10%
PATCH_DAMPED_SINKHORN = True # 1D: 5 damped steps (alpha 0.5) instead of 100 hard steps
STAGE2_TRUNKS = True         # v18: Part Two trunks (pool_v4-trained 14B pair + DeBERTa) ON, and the embedded
                             #      stack now runs BEFORE them. v16's 21,512 s run had the stack last, where
                             #      accelerate offloaded it to CPU (34-43x slower); ordering is the whole fix.
print(f"v18 switches: POOL_SHARE {POOL_SHARE} unified {PATCH_UNIFIED_GROUPS} gates {PATCH_GATES} "
      f"atlas_gate {PATCH_ATLAS_GATE} damped {PATCH_DAMPED_SINKHORN} stage2 {STAGE2_TRUNKS}", flush=True)

# Blend weights, fit on out-of-fold predictions (log probabilities are multiplied by these and summed).
W_SVC = 1.021
if STAGE2_TRUNKS:
    # Part Two checkpoints (nishantdahal/ntl-handwriting-blend). The three members split the *same* total blend mass
    # that v10 gave its two Part One trunks (plain 0.226 + 0.429 = 0.655, twin 0.079 + 0.150 = 0.229, gaussian 0.10),
    # so this stage swaps which model produces the evidence instead of re-weighting the blend by guesswork.
    LORAS = {
        "lora_qwen3_14b":  dict(w=0.35, twin_w=0.150, gauss_w=0.10, bs=8,
                                base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
        "lora_qwen25_14b": dict(w=0.20, twin_w=0.079, gauss_w=0.00, bs=8,
                                base=dict(model_type="qwen2", hidden_size=5120, num_hidden_layers=48)),
    }
    DEBERTA_MEMBER = dict(dir="deberta_v3_large", w=0.10, max_len=384)   # plain 0.35+0.20+0.10 = 0.65
else:
    LORAS = {  # Part One checkpoints, the v10 baseline
        "lora8b_v1":  dict(w=0.226, twin_w=0.079, gauss_w=0.10, bs=16,
                           base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),
        "lora14b_v1": dict(w=0.429, twin_w=0.150, gauss_w=0.0, bs=8,
                           base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
    }
    DEBERTA_MEMBER = None
TWIN = dict(thresh=0.85, min_size=2, max_size=18)   # legacy threshold kept for reference only
GROUP_THRESH = 0.60                                 # 1B: mixed-distance cut for the main-branch clusters
GAUSS_D, GAUSS_T = 128, 3.0                         # shared-prompt Gaussian reduction / temperature (prvsiyan core)

# Column order of the checkpoints' heads (checked against each checkpoint's meta.json below).
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
K = len(LABELS)
FAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ('claude', 'gpt', 'gemini', 'glm')}
# The four gated families cover 16 of the 18 labels; qwen3.8-max and kimi-k3 are single-member labs with no siblings.
assert {f: len(v) for f, v in FAMILY_IDX.items()} == {'claude': 5, 'gpt': 5, 'gemini': 4, 'glm': 2}, \
    f'unexpected family sizes {[(f, len(v)) for f, v in FAMILY_IDX.items()]}'


def find(pattern):
    hits = sorted(glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


_competition_dir = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"] if os.path.exists(p + "/train.csv"))
train = pd.read_csv(_competition_dir + "/train.csv")
test = pd.read_csv(_competition_dir + "/test.csv")
test["text"] = test["text"].fillna("").astype(str)
TR, TE = train["text"].astype(str).tolist(), test["text"].tolist()
TINY = len(TE) < 50   # the interactive 3-row test: run the extra self-checks so the hidden rerun cannot hit untested code
DEV = "cuda" if torch.cuda.is_available() else "cpu"
y = train["model"].map({l: i for i, l in enumerate(LABELS)}).values
LOGIT = np.zeros((len(test), K))   # every member adds w * log p here
print("train", train.shape, "test", test.shape, "device", DEV, flush=True)


from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def find_pool(pattern, prefer):
    """First pool file matching `pattern`, preferring a copy whose path contains `prefer` (several datasets ship one)."""
    hits = sorted(glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True))
    assert hits, f"attach the external pool {pattern}"
    hits.sort(key=lambda h: (prefer not in h, h))
    return hits[0]


# ---- pool: v14's curated pool, the best-scoring SVC block measured so far (public 0.610) ----
# v14 = curated_pool.parquet + class_weight="balanced" + a flat 0.25 on pool rows + transductive vocabulary.
# It measured +0.004 over v10's pool_v1 / balanced-only version, so it is the base v17 builds on.
SVC_POOL_WEIGHT = 0.25       # flat multiplier on pool rows, applied on top of class_weight="balanced"
SVC_POOL = "curated_pool.parquet"   # independent of STAGE2_TRUNKS: v14's 0.610 SVC block stays the base
_pool_file = SVC_POOL
pool = pd.read_parquet(find(_pool_file))
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
print(f"{_pool_file}: {len(pool)} rows after label/30-300-word/dedup filters", flush=True)

fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]
vocab_texts = fit_texts + test.text.tolist()   # transductive vocabulary, labels only on fit rows

if POOL_SHARE is None:
    # v14 weighting, verbatim: class_weight="balanced" plus a flat 0.25 on every pool row.
    fit_w = np.r_[np.ones(len(train)), np.full(len(pool), SVC_POOL_WEIGHT)]
    _cw = "balanced"
    _n = np.bincount(fit_y, minlength=K).astype(np.float64)
    _eff = fit_w * (len(fit_y) / (K * _n[fit_y]))          # what sklearn actually applies
    print(f"SVC weights: v14 flat pool weight {SVC_POOL_WEIGHT} on {len(pool)} pool rows; the pool holds "
          f"{1 - _eff[:len(train)].sum() / _eff.sum():.1%} of the total weight (v10: ~96%, 1A target: 35%)", flush=True)
else:
    # ---- 1A: class-balanced share weighting so each class's total SVC weight is identical and the
    #          official train.csv keeps (1 - POOL_SHARE) of it, whatever the pool's per-class row counts are.
    #          The overall scale matches class_weight="balanced" (total weight = number of rows), so C=0.5 keeps
    #          the same effective regularisation instead of being multiplied by a different row count.
    _pool_y = fit_y[len(train):]
    _n_tr = np.bincount(y, minlength=K).astype(np.float64)
    _n_pl = np.bincount(_pool_y, minlength=K).astype(np.float64)
    _w_tr = np.where(_n_pl[y] > 0, (1.0 - POOL_SHARE) / np.maximum(_n_tr[y], 1.0), 1.0 / np.maximum(_n_tr[y], 1.0))
    _w_pl = POOL_SHARE / np.maximum(_n_pl[_pool_y], 1.0)
    fit_w = np.r_[_w_tr, _w_pl] * (len(fit_texts) / K)
    _cw = None
    _pool_ratio = float(fit_w[len(train):].sum() / fit_w.sum())
    print(f"SVC weights: pool rows {len(pool)}, pool share of total weight {_pool_ratio:.3f} "
          f"(class totals: min {np.bincount(fit_y, fit_w, minlength=K).min():.4f} "
          f"max {np.bincount(fit_y, fit_w, minlength=K).max():.4f})", flush=True)
    assert _pool_ratio < 0.40, "pool still dominates the SVC objective"
A, B = [], []
for view, lo, hi, max_features, weight in VIEWS:
    prep = shape_text if view == "shape" else function_text if view == "function" else (lambda s: s)
    analyzer = "char" if view == "shape" else "word" if view == "function" else view
    opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                max_features=max_features, dtype=np.float32)
    if analyzer == "word":
        opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    vec = TfidfVectorizer(**opts)
    vec.fit([prep(t) for t in vocab_texts])
    A.append(vec.transform([prep(t) for t in fit_texts]) * weight)
    B.append(vec.transform([prep(t) for t in test.text]) * weight)
svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight=_cw).fit(
    hstack(A, format="csr"), fit_y, sample_weight=fit_w)
P_svc = softmax(svc.decision_function(hstack(B, format="csr")))
LOGIT += W_SVC * np.log(np.clip(P_svc, 1e-9, 1))
del A, B, svc, pool
gc.collect()
print(f"n-gram model done ({time.time() - T0:.0f}s)", flush=True)


from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


def find_base(spec):
    """Base model directory matching `spec`. Directories that carry a config but no weight file are skipped:
    swaroopmeher/ntl-public-heads added a config-only `base_config` copy, which an unguarded search would pick."""
    hits = []
    for cfg in sorted(glob.glob(f"{INPUT_DIR}/**/config.json", recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if all(c.get(k) == v for k, v in spec.items()):
            d = os.path.dirname(cfg)
            if not any(os.path.exists(os.path.join(d, f)) for f in
                       ("model.safetensors", "pytorch_model.bin",
                        "model.safetensors.index.json", "pytorch_model.bin.index.json")):
                continue
            hits.append(d)
    hits.sort(key=lambda d: (0 if "/models/" in d else 1, d))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"attach the base model {spec}")


def load_trunk(base_dir, keep, dtype):
    """First `keep` decoder layers, split over both T4s when it does not fit on one."""
    mc = AutoConfig.from_pretrained(base_dir)
    mc.num_hidden_layers, mc.use_cache = keep, False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[:keep]
    hd = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
    gib = 2 * (mc.vocab_size * mc.hidden_size + keep * (2 * mc.hidden_size * mc.num_attention_heads * hd
               + 2 * mc.hidden_size * mc.num_key_value_heads * hd + 3 * mc.hidden_size * mc.intermediate_size)) / 2 ** 30
    if gib > 13 and torch.cuda.device_count() > 1:
        mem = {i: "13GiB" for i in range(torch.cuda.device_count())}
        model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype, device_map="auto", max_memory=mem).eval()
    else:
        model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype).to("cuda:0").eval()
    model.norm = nn.Identity()
    return model


@torch.no_grad()
def merge_lora(model, adapter_dir):
    acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = acfg["lora_alpha"] / acfg["r"]
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    for k, A in sd.items():
        if ".lora_A." in k:
            name = k.split(".lora_A.")[0]
            lin = modules[name[name.index("layers."):]]
            Bm = sd[k.replace(".lora_A.", ".lora_B.")]
            lin.weight.add_(((Bm.float().to(lin.weight.device) @ A.float().to(lin.weight.device)) * scale).to(lin.weight.dtype))


@torch.no_grad()
def lora_features(ckpt_dir, base_dir, texts, bs):
    """Pooled features Z (n, 8 * hidden) and the checkpoint's heads."""
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    assert meta["labels"] == LABELS, "label order differs from the checkpoint's heads"
    cfg = meta["cfg"]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_trunk(base_dir, cfg["keep"], dtype)
    merge_lora(model, os.path.join(ckpt_dir, "adapter"))
    heads = load_file(os.path.join(ckpt_dir, "heads.safetensors"))
    layers, H = cfg["layers"], model.config.hidden_size
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in layers]).to("cuda:0").float()
    norms.load_state_dict({k[len("norms."):]: v for k, v in heads.items() if k.startswith("norms.")})
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=cfg["max_len"])["input_ids"]
    ids = [x + [tok.eos_token_id] for x in ids]
    order = np.argsort([len(x) for x in ids], kind="stable")   # similar lengths together = less padding
    Z = np.zeros((len(ids), H * len(layers)), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(model.embed_tokens.weight.device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in layers):   # fp16 overflow: redo this batch in bf16
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            model.to(dtype)
        m = enc["attention_mask"].unsqueeze(-1).float().to("cuda:0")
        m[:, 0] = 0   # the first token is an attention sink; leave it out of the mean
        den = m.sum(1).clamp(min=1)
        Z[idx] = torch.cat([nm((hs[l].to("cuda:0").float() * m).sum(1) / den) for l, nm in zip(layers, norms)], -1).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return Z, {k: v.float().numpy() for k, v in heads.items() if not k.startswith("norms.")}


from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _average_linkage(D, thresh):
    if len(D) < 2:
        return np.ones(len(D), dtype=int)
    D = np.asarray(D, dtype=np.float64).copy()
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method="average"), thresh, criterion="distance")


def _content_distance(texts):
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    return np.clip(1.0 - (X @ X.T).toarray(), 0.0, 1.0)


def mixed_groups(texts, E, thresh=GROUP_THRESH):
    """Average of the content-word TF-IDF cosine distance and the cosine distance of mean-centred,
    L2-normalised embeddings E (the same mixed metric the legacy branch uses on GLM-L24)."""
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    Z = np.asarray(E, dtype=np.float64)
    Z = Z - Z.mean(0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    D = 0.5 * (_content_distance(texts) + np.clip(1.0 - Z @ Z.T, 0.0, 2.0))
    return _average_linkage(D, thresh)


def lexical_groups(texts, thresh=TWIN["thresh"]):
    return _average_linkage(_content_distance(texts), thresh)


groups, twin_rows = None, None


def ensure_groups(E=None):
    """Main-branch prompt clusters (the row set that receives the centred twin and the shared-prompt Gaussian).
    With PATCH_UNIFIED_GROUPS the clusters use the mixed content + hidden-state distance on the first 14B trunk, so
    the clustering is deferred until that trunk's features exist; otherwise it is v10's stop-word TF-IDF cut."""
    global groups, twin_rows
    if groups is not None:
        return
    groups = mixed_groups(TE, E) if PATCH_UNIFIED_GROUPS else lexical_groups(TE)
    size = np.bincount(groups)[groups]
    twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])
    _thr = GROUP_THRESH if PATCH_UNIFIED_GROUPS else TWIN["thresh"]
    print(f"main clusters ({'mixed 14B+TF-IDF' if PATCH_UNIFIED_GROUPS else 'TF-IDF'} thresh {_thr}): "
          f"{int((size == 1).sum())} singletons, {int(twin_rows.sum())} passages in clusters of 2-18", flush=True)


if not PATCH_UNIFIED_GROUPS:   # needs no model features: settle it before the heavy passes, as v10 did
    ensure_groups()

# Gaussian shared-prompt core, credited to prvsiyan.
import itertools
K = 18
DEFAULT = {"red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw", "lam_floor": 1e-4, "n_exact": 3, "inj": True,
           "iters": 5, "repel": True, "sp_scale": 1.0}

# =================================================================== primitives ====
def _lse(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    if keepdims: return out
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())

def log_softmax(L): return L - _lse(L, axis=1, keepdims=True)

def _lw_shrinkage(X):
    """Ledoit-Wolf shrinkage coefficient for rows X that are already centred (sklearn formula, assume_centered)."""
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(X, assume_centered=True))

def _shrunk(S, a):
    d = S.shape[0]; return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)

def _two_way(Z, y, pinv, G, iters=4):
    """additive fit Z = mu[y] + P[prompt] + E by alternating means (balanced design -> converges in 1-2 rounds)."""
    P = np.zeros((G, Z.shape[1])); cnt_p = np.bincount(pinv, minlength=G).astype(float)
    cls = np.unique(y); mu = np.zeros((K, Z.shape[1]))
    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls: mu[c] = R[y == c].mean(0)
        R = Z - mu[y]; P = np.zeros_like(P); np.add.at(P, pinv, R); P /= cnt_p[:, None]
    return mu, P, cnt_p

def fit(X, y, prompts, cfg=None):
    """fit the shared-prompt Gaussian on training rows. X (N, D) raw features, y (N,) labels in 0..17, prompts (N,) group ids.
    Returns a params dict (numpy arrays + floats); temperatures default to 1 (set by the caller's cross-fit)."""
    c = dict(DEFAULT); c.update(cfg or {})
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y); N, D = X.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True); G = len(pu)
    mx = X.mean(0); sx = X.std(0); sx = np.where(sx > 1e-8, sx, 1.0); Xs = (X - mx) / sx
    if c["red"] == "none":
        V = np.eye(D)
    else:
        if c["red"] == "wpca":
            Pm = np.zeros((G, D)); np.add.at(Pm, pinv, Xs); Pm /= np.bincount(pinv)[:, None]; R = Xs - Pm[pinv]
        elif c["red"] == "pca": R = Xs
        else: raise ValueError(c["red"])
        d = int(min(c["d"], N - G - 1, D))
        # economy SVD through the (N x N) Gram matrix (N << D for the real views)
        if N < D:
            w, U = np.linalg.eigh(R @ R.T); o = np.argsort(w)[::-1][:d]; w, U = np.clip(w[o], 1e-12, None), U[:, o]
            V = (R.T @ U) / np.sqrt(w)
        else:
            _, _, Vt = np.linalg.svd(R, full_matrices=False); V = Vt[:d].T
    Z = Xs @ V; d = Z.shape[1]
    mu, P, ng = _two_way(Z, y, pinv, G)
    E = Z - mu[y] - P[pinv]; dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = E.T @ E / dof
    ae = _lw_shrinkage(E) if c["se_shrink"] == "lw" else float(c["se_shrink"]); Se = _shrunk(Se_raw, ae)
    Pc = P - P.mean(0); Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = _lw_shrinkage(Pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se)); Sp = (1 - ap) * Sp_raw + ap * (tr / d) * np.eye(d); Sp *= c["sp_scale"]
    # joint diagonalisation: A Se A' = I, A Sp A' = diag(lam)
    L = np.linalg.cholesky(Se); Li = np.linalg.inv(L); M = Li @ Sp @ Li.T; M = 0.5 * (M + M.T)
    lam, Q = np.linalg.eigh(M); lam = np.clip(lam, c["lam_floor"], None); A = Q.T @ Li
    W = V @ A.T
    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 1.0, "Tg": 1.0,
            "info": {"d": int(d), "a_e": round(ae, 4), "a_p": round(ap, 4), "lam_pct": np.round(np.percentile(lam, [5, 50, 95, 100]), 3).tolist()}, "cfg": c}

def transform(X, prm): return ((np.asarray(X, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]

_MASK = {}
def _inj_mask(n, C):
    if (n, C) not in _MASK:
        grids = np.meshgrid(*[np.arange(C)] * n, indexing="ij"); bad = np.zeros([C] * n, bool)
        for i, j in itertools.combinations(range(n), 2): bad |= grids[i] == grids[j]
        _MASK[(n, C)] = bad
    return _MASK[(n, C)]

def singleton_ll(U, m, lam):
    """(n, C) log N(u; m_c, I + diag lam) up to a constant (p integrated out)."""
    R = U[:, None, :] - m[None]; return -0.5 * (R ** 2 / (1 + lam)).sum(-1)

def joint_terms(U, m, lam):
    """a (n, C) and B (n, n, C, C) of the exact joint log-likelihood of one group (constant in c dropped)."""
    n = len(U); gam = lam / (1 + n * lam); R = U[:, None, :] - m[None]
    a = -0.5 * (R ** 2 * (1 - gam)).sum(-1); G = R * np.sqrt(gam); B = np.einsum("icd,jed->ijce", G, G)
    return a, B

def joint_table(a, B):
    """full tuple table Q[c_1, ..., c_n] = sum_i a_i + sum_{i<j} B_ij."""
    n, C = a.shape; Q = np.zeros([C] * n)
    for i in range(n):
        sh = [1] * n; sh[i] = C; Q = Q + a[i].reshape(sh)
    for i, j in itertools.combinations(range(n), 2):
        sh = [1] * n; sh[i] = C; sh[j] = C; Q = Q + B[i, j].reshape(sh)
    return Q

def exact_marginals(Q, T=1.0, inj=True):
    """per-passage log marginal posteriors (n, C) from a tuple table Q (unnormalised log joint), tempered by 1/T."""
    n = Q.ndim; C = Q.shape[0]; Qt = Q / T
    if inj and n >= 2: Qt = np.where(_inj_mask(n, C), -np.inf, Qt)
    lz = _lse(Qt); out = np.empty((n, C))
    for i in range(n):
        ax = tuple(k for k in range(n) if k != i); out[i] = (_lse(Qt, axis=ax) if ax else Qt) - lz
    return out

def em_marginals(U, m, lam, T=1.0, iters=5, repel=False, q0=None):
    """cavity EM for one group (any n >= 2): returns (n, C) log q."""
    n = len(U); v = lam / (1 + (n - 1) * lam)
    L = singleton_ll(U, m, lam) / T if q0 is None else np.log(np.clip(q0, 1e-300, None))
    q = np.exp(log_softmax(L))
    for _ in range(iters):
        s = U - q @ m; S = s.sum(0); Pm = (S[None] - s) * v                  # cavity prompt-effect means (n, d)
        R = U[:, None, :] - Pm[:, None, :] - m[None]; L = -0.5 * (R ** 2 / (1 + v)).sum(-1) / T
        if repel:   # mean-field at-most-once: discourage classes the other members already hold
            lo = np.log(np.clip(1 - q, 1e-6, None)); L = L + (lo.sum(0)[None] - lo)
        q = np.exp(log_softmax(L))
    return log_softmax(L)

LAM_DIFF = 1e6
def lam_g(prm, cfg=None):
    """lam used for groups: the fitted one, or ~infinity in the paired-difference ablation (cfg 'diff'): then the joint likelihood
    only sees within-group differences (x_i - x_j around mu_a - mu_b with covariance 2 S_e; triples: centred observations vs
    centred prototypes) and the prompt term cancels exactly."""
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {})
    return np.full_like(prm["lam"], LAM_DIFF) if c.get("diff") else prm["lam"]

def group_logpost(U, prm, T1=None, Tg=None, cfg=None):
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {}); n = len(U); lg = lam_g(prm, cfg)
    T1 = prm["T1"] if T1 is None else T1; Tg = prm["Tg"] if Tg is None else Tg
    if n == 1: return log_softmax(singleton_ll(U, prm["m"], prm["lam"]) / T1)
    if n <= c["n_exact"]:
        a, B = joint_terms(U, prm["m"], lg); return exact_marginals(joint_table(a, B), Tg, c["inj"])
    return em_marginals(U, prm["m"], lg, Tg, c["iters"], c["repel"])

def scores(X_pool, lab, prm, T1=None, Tg=None, cfg=None, U=None):
    """(n, 18) log-probabilities for every pool row; lab = detected group labels of the pool rows (singletons allowed)."""
    U = transform(X_pool, prm) if U is None else U; lab = np.asarray(lab); n = len(U); out = np.empty((n, prm["m"].shape[0]))
    sizes = np.bincount(lab); s1 = sizes[lab] == 1
    T1 = prm["T1"] if T1 is None else T1
    if s1.any(): out[s1] = log_softmax(singleton_ll(U[s1], prm["m"], prm["lam"]) / T1)
    for g in np.where(sizes >= 2)[0]:
        ii = np.where(lab == g)[0]; out[ii] = group_logpost(U[ii], prm, T1, Tg, cfg)
    return out


_GAUSSIAN_SOURCE = '# ============================================================ 0. configuration\nimport gc, glob, json, os, pathlib, re, time\nimport numpy as np, pandas as pd\ntry:\n    import torch\nexcept ImportError:   # local smoke without torch\n    torch = None\nfrom scipy.cluster.hierarchy import fcluster, linkage\nfrom scipy.sparse import csr_matrix, diags, hstack, vstack\nfrom scipy.spatial.distance import squareform\nfrom sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import LinearSVC\n\nSMOKE = os.environ.get(\'NTL_SMOKE\') == \'1\'   # local code-path check: pseudo features instead of checkpoints, CPU\nT0 = time.time()\nLABELS = sorted([\'gpt-5.5\', \'gpt-5.6-luna\', \'gpt-5.6-sol\', \'gpt-5.6-terra\', \'gpt-6-astra\',\n                 \'gemini-3.1-pro-preview\', \'gemini-3.6-flash\', \'gemini-3.7-flash\', \'gemini-3.8-flash\',\n                 \'claude-fable-5\', \'claude-opus-4-6\', \'claude-opus-4-8\', \'claude-opus-5\', \'claude-sonnet-5\',\n                 \'qwen3.8-max\', \'glm-5.3\', \'glm-5.3-flash\', \'kimi-k3\'])\nK = len(LABELS)\nLAB2I = {l: i for i, l in enumerate(LABELS)}\nFAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in (\'claude\', \'gpt\', \'gemini\', \'glm\')}\n\n# ---- test-pool grouping\nGROUP_THRESH, GROUP_LAYER, GROUP_MIN, GROUP_MAX = 0.60, 24, 2, 18   # mixed content + GLM-L24 distance, average linkage\nSTYLE_GROUP_THRESH = 0.90                                           # stylegram\'s own median-normalised content distance\n\n# ---- blend weights: log-probability multipliers of the 0.563 notebook (its L-BFGS fit on prompt-grouped OOF)\nW = {\'stylegram\': 1.516, \'stylometry_lgb\': 0.063, \'ultrabert\': 0.095,\n     \'q3_8b\': 0.0, \'q25_7b\': 0.0, \'q3_4b\': 0.0, \'glm9b\': 0.0, \'debv3l\': 0.05, \'electl\': 0.032, \'desk\': 0.132,\n     \'lora24_public\': 0.177, \'lora24_seed7\': 0.113, \'lora24_claude\': 0.177, \'lora20_q25_instyle\': 0.17,\n     \'tokattn_q35\': 0.153, \'tokattn_deb\': 0.185}\nTWIN_W = {\'q3_8b\': 1.5 * 0.219, \'q25_7b\': 1.5 * 0.141, \'q3_4b\': 1.5 * 0.083, \'glm9b\': 1.5 * 0.110, \'debv3l\': 0.25}\nLORA_TWIN_W = {\'lora24_public\': 0.15, \'lora24_claude\': 0.15}\nGATE_LAM = {\'claude_style\': 1.5, \'claude_lora\': 1.0, \'family\': 0.25}\nBALANCE_ALPHA = 0.5\nEXTRA_MIN_LABELS, EXTRA_SHARE = 12, 0.5\n\n# ---- the addition: shared-prompt Gaussian group heads on four measured views, weight = the centred twin of the same backbone,\n#      reduction / temperatures from our production fit (research/gaussian/production_weights_v8g.json, 135 training prompts)\nGAUSS_HEADS = {\'gauss_q3_8b_L22\': dict(src=\'q3_8b\', part=\'L22_mean\', d=384, T1=2.351, Tg=2.306, w=TWIN_W[\'q3_8b\']),\n               \'gauss_glm9b_L8\': dict(src=\'glm9b\', part=\'L8_mean\', d=384, T1=2.722, Tg=2.596, w=TWIN_W[\'glm9b\']),\n               \'gauss_q3_4b_L4\': dict(src=\'q3_4b\', part=\'L4_mean\', d=384, T1=2.725, Tg=2.578, w=TWIN_W[\'q3_4b\']),\n               \'gauss_debv3l_L24\': dict(src=\'debv3l\', part=\'L24_mean\', d=384, T1=3.112, Tg=3.069, w=TWIN_W[\'debv3l\'])}\nGAUSS_SETTING = dict(red=\'wpca\', se_shrink=\'lw\', sp_shrink=\'lw\', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)\n\n# ---- member specifications\nLM_HEADS = [  # key, model_type, hidden size, parts of the plain head [(layer, pool)], extra parts extracted in the same pass, C\n    (\'glm9b\', \'glm4\', 4096, [(8, \'mean\'), (3, \'mean\')], [(GROUP_LAYER, \'mean\')], 0.0025),   # first: its L24 builds the groups\n    (\'q3_8b\', \'qwen3\', 4096, [(7, \'mean\'), (22, \'mean\')], [], 0.0025),\n    (\'q25_7b\', \'qwen2\', 3584, [(6, \'mean\'), (20, \'mean\')], [], 0.0025),\n    (\'q3_4b\', \'qwen3\', 2560, [(4, \'style\'), (7, \'style\')], [(4, \'mean\')], 0.0025)]\nENC_HEADS = [  # key, input directory name, parts, C\n    (\'debv3l\', \'deberta-v3-large\', [(24, \'mean\'), (6, \'mean\')], 0.005),\n    (\'electl\', \'electra-large-discriminator\', [(21, \'mean\'), (24, \'style\')], 0.005),\n    (\'desk\', \'desklib-detector\', [(23, \'mean\'), (23, \'style\')], 0.005)]\nLORA_RUNS = [  # (trunk model_type, hidden, kept layers), pooled layers, [(key, checkpoint marker, norms file, anchor features file)]\n    ((\'qwen3\', 4096, 24), [2, 4, 6, 8, 12, 16, 20, 24], [\n        (\'lora24_public\', \'**/lora24/final/COMPLETE.json\', \'heads.safetensors\', \'anchor_features.npz\'),\n        (\'lora24_seed7\', \'**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json\', \'norms.safetensors\', \'valid_features.npz\'),\n        (\'lora24_claude\', \'**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json\', \'heads.safetensors\', \'anchor_features.npz\')]),\n    ((\'qwen2\', 3584, 20), [2, 4, 6, 8, 10, 12, 16, 20], [\n        (\'lora20_q25_instyle\', \'**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json\', \'norms.safetensors\', \'valid_features.npz\')])]\nLORA_MAX_LEN, LORA_BS = 447, 8\nTOKATTN_HEADS = [(\'tokattn_q35\', \'tokattn_final.pt\', \'qwen3_5\'), (\'tokattn_deb\', \'tokattn_debl23_final.pt\', \'deberta\')]\n\n# ---- data\nif SMOKE:\n    _all = pd.read_csv(os.environ[\'NTL_DATA\'] + \'/train.csv\')\n    _p = _all[\'prompt_id\'].drop_duplicates().sample(frac=1, random_state=0).tolist()\n    train = _all[_all[\'prompt_id\'].isin(_p[:30])].reset_index(drop=True)\n    test = _all[_all[\'prompt_id\'].isin(_p[30:42])].sample(frac=1, random_state=1).reset_index(drop=True)\n    test = pd.concat([test[test[\'prompt_id\'] == p].head(k) for p, k in zip(_p[30:42], [1, 1, 2, 2, 2, 3, 3, 4, 6, 9, 18, 18])]).reset_index(drop=True)\n    if os.environ.get(\'NTL_TINY\') == \'1\':\n        test = test.head(3).reset_index(drop=True)\n    SMOKE_Y = test[\'model\'].map(LAB2I).to_numpy(dtype=int)\nelse:\n    DATA_DIR = next(p for p in [pathlib.Path(\'/kaggle/input/competitions/name-that-llm\'), pathlib.Path(\'/kaggle/input/name-that-llm\')]\n                    if (p / \'train.csv\').exists())\n    train, test = pd.read_csv(DATA_DIR / \'train.csv\'), pd.read_csv(DATA_DIR / \'test.csv\')\ntest[\'text\'] = test[\'text\'].fillna(\'\').astype(str)\nTR, TE = train[\'text\'].tolist(), test[\'text\'].tolist()\ny = train[\'model\'].map(LAB2I).to_numpy(dtype=int)\nPROMPT = train[\'prompt_id\'].to_numpy(dtype=object)\nTINY = len(TE) < 50   # the interactive 3-row test: exercise the group paths on training rows so the hidden rerun cannot hit untested code\nDEV = \'cuda\' if torch is not None and torch.cuda.is_available() else \'cpu\'\nprint(f\'train {train.shape}  test {test.shape}  gpus {torch.cuda.device_count() if torch is not None else 0}  smoke {SMOKE}\', flush=True)\n\nLOGIT = np.zeros((len(TE), K))           # the running log-probability blend\n\n\ndef add_member(key, prob, rows=None, weight=None):\n    """Add weight * log(prob) to the blend (all rows, or the given row subset)."""\n    w = W[key] if weight is None else weight\n    lp = np.log(np.clip(prob, 1e-9, 1.0))\n    if rows is None:\n        LOGIT[:] += w * lp\n    else:\n        LOGIT[rows] += w * lp\n    print(f\'  + {key:20s} weight {w:.3f} on {len(TE) if rows is None else int(np.sum(rows))} rows  ({time.time() - T0:.0f}s)\', flush=True)\n\n\ndef softmax_rows(S):\n    S = S - S.max(1, keepdims=True)\n    P = np.exp(S)\n    return P / P.sum(1, keepdims=True)\n\n\ndef full_prob(P_part, classes):\n    """Expand a (n, len(classes)) probability block to (n, K) on the global label order."""\n    P = np.zeros((len(P_part), K))\n    P[:, classes] = P_part\n    return P\n\n\ndef group_centre_dense(X, groups):\n    X = np.asarray(X, dtype=np.float32).copy()\n    for g in np.unique(groups):\n        i = groups == g\n        X[i] -= X[i].mean(0)\n    return X\n\n\ndef self_test_rows():\n    """Training rows of the first three prompts, grouped by their true prompt (used only when the test file is tiny)."""\n    idx = np.where(np.isin(PROMPT, pd.unique(PROMPT)[:3]))[0]\n    return idx, np.unique(PROMPT[idx], return_inverse=True)[1]\n\n# ============================================================ 1. grouping the test pool\ndef average_linkage_groups(D, thresh):\n    """Cluster ids (1-based) from a square distance matrix, average linkage, cut at thresh."""\n    if len(D) < 2:\n        return np.ones(len(D), dtype=int)\n    D = np.asarray(D, dtype=np.float64).copy()\n    np.fill_diagonal(D, 0.0)\n    return fcluster(linkage(squareform(D, checks=False), method=\'average\'), thresh, criterion=\'distance\')\n\n\ndef content_cosine_distance(texts, max_features=None):\n    X = TfidfVectorizer(stop_words=\'english\', sublinear_tf=True, max_features=max_features, dtype=np.float32).fit_transform(texts)\n    return np.clip(1.0 - (X @ X.T).toarray(), 0.0, 1.0)\n\n\ndef mixed_groups(texts, E, thresh=GROUP_THRESH):\n    """Average of the content-word TF-IDF cosine distance and the cosine distance of mean-centred, L2-normalised embeddings E."""\n    if len(texts) < 2:\n        return np.ones(len(texts), dtype=int)\n    Z = np.asarray(E, dtype=np.float64)\n    Z = Z - Z.mean(0)\n    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12\n    D = 0.5 * (content_cosine_distance(texts) + np.clip(1.0 - Z @ Z.T, 0.0, 2.0))\n    return average_linkage_groups(D, thresh)\n\n\ndef normalised_content_groups(texts, thresh=STYLE_GROUP_THRESH):\n    """The stylegram\'s grouping: content distance divided by its median off-diagonal value, cut at thresh."""\n    if len(texts) < 2:\n        return np.ones(len(texts), dtype=int)\n    D = content_cosine_distance(texts, max_features=100_000)\n    med = max(float(np.median(D[np.triu_indices(len(texts), 1)])), 1e-6)\n    return average_linkage_groups(D / med, thresh)\n\n\ndef eligible(groups, lo=GROUP_MIN, hi=GROUP_MAX):\n    size = np.bincount(groups)[groups]\n    return (size >= lo) & (size <= hi)\n\n\ndef describe_groups(name, groups):\n    size = np.bincount(groups)[groups]\n    print(f\'{name}: {len(groups)} rows | singletons {int((size == 1).sum())}  2-3: {int(((size >= 2) & (size <= 3)).sum())}\'\n          f\'  4-18: {int(((size >= 4) & (size <= 18)).sum())}  >18: {int((size > 18).sum())}\', flush=True)\n\n\nG_TE, GROUP_ROWS = None, None   # set after the GLM pass (section 4)\n\n# ============================================================ 2. style n-grams + extra passages + gates\nFUNCTION_WORDS = frozenset("a an the and or but if while although because as so than that which who whom whose this these those it its "\n                           "they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with "\n                           "without for of by over under between through into about is are was were be been being have has had do does did "\n                           "can could will would shall should may might must not no nor all any some each every both either neither other "\n                           "another such only also just very too more most less least here there when where why how what whether then now "\n                           "still yet even much many few little".split())\nTOKEN_RE = re.compile(r"\\w+|[^\\w\\s]")\n\n\ndef shape_view(text):\n    return re.sub(r"[^\\W\\d_]+", lambda m: \'X\' if m[0].isupper() else \'x\', text)\n\n\ndef function_view(text):\n    return \' \'.join(t.lower() if (t.lower() in FUNCTION_WORDS or not t.isalnum()) else \'CONTENT\' for t in TOKEN_RE.findall(text))\n\n\nclass StyleGrams:\n    """Four stacked TF-IDF views: char 3-5, word 1-3, shape (char 2-6 of the x/X text), function-word (word 1-3 of the masked text)."""\n    VIEWS = [(\'char\', str, (3, 5), 250_000, 1.0), (\'word\', str, (1, 3), 180_000, 0.7),\n             (\'char\', shape_view, (2, 6), 60_000, 0.5), (\'word\', function_view, (1, 3), 50_000, 0.5)]\n\n    def fit(self, texts):\n        self.vecs = []\n        for analyzer, view, ngr, maxf, w in self.VIEWS:\n            opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float64)\n            if analyzer == \'word\':\n                opts[\'token_pattern\'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"\n            self.vecs.append((TfidfVectorizer(**opts).fit([view(t) for t in texts]), view, w))\n        return self\n\n    def transform(self, texts):\n        return hstack([v.transform([view(t) for t in texts]) * w for v, view, w in self.vecs], format=\'csr\')\n\n\ndef sparse_group_centre(X, groups, lo=GROUP_MIN, hi=GROUP_MAX):\n    """Subtract each group\'s mean row (sparse); groups outside [lo, hi] are left as they are."""\n    X = csr_matrix(X)\n    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)\n    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))\n    if not len(keep):\n        return X.copy()\n    col = np.full(len(cnt), -1); col[keep] = np.arange(len(keep))\n    rows = np.flatnonzero(col[inv] >= 0)\n    M = csr_matrix((np.ones(len(rows)), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)))\n    means = diags(1.0 / cnt[keep]) @ (M.T @ X)\n    return (X - M @ means).tocsr()\n\n\ndef svc_probs(svc, X):\n    return full_prob(softmax_rows(svc.decision_function(X)), svc.classes_)\n\n\n# ---- public in-style passages (prompt-bank protocol), one per (prompt, label); prompts answered by >= EXTRA_MIN_LABELS labels\ndef load_extra_passages():\n    sources = [(\'**/ntl_extra_passages_claude.csv\', \'model\', None), (\'**/gen_passages.csv\', \'model\', None),\n               (\'**/ntl_generations.csv\', \'label\', \'finish_reason\'), (\'**/ntl_proxy_generations.csv\', \'label\', None),\n               (\'**/ntl_generated*.csv\', \'model\', \'finish\'), (\'**/passages_clean.csv\', \'model\', None)]\n    parts = []\n    for pattern, label_col, finish_col in sources:\n        for f in sorted(glob.glob(\'/kaggle/input/\' + pattern, recursive=True)):\n            d = pd.read_csv(f)\n            if finish_col and finish_col in d:\n                d = d[d[finish_col].eq(\'stop\')]\n            d = d.rename(columns={label_col: \'model\'})[[\'prompt_id\', \'model\', \'text\']].copy()\n            d[\'prompt_id\'] = (\'pv_\' if f.endswith(\'passages_clean.csv\') else \'\') + d[\'prompt_id\'].astype(str)   # prvsiyan\'s prompts are a different bank\n            parts.append(d)\n    if not parts:\n        return pd.DataFrame(columns=[\'prompt_id\', \'model\', \'text\'])\n    x = pd.concat(parts, ignore_index=True)\n    x = x[x[\'model\'].isin(LABELS)].copy()\n    x[\'text\'] = x[\'text\'].astype(str).map(lambda t: re.sub(r\'\\s+\', \' \', re.sub(r\'^\\*\\*(.*)\\*\\*$\', r\'\\1\', t.strip())).strip())\n    x = x[x[\'text\'].str.split().str.len().between(30, 300)].drop_duplicates(subset=[\'model\', \'text\'])\n    x = x.drop_duplicates(subset=[\'prompt_id\', \'model\'])\n    x = x[x.groupby(\'prompt_id\')[\'model\'].transform(\'nunique\') >= EXTRA_MIN_LABELS].reset_index(drop=True)\n    return x\n\n\nEXTRA = load_extra_passages()\nEX_T, EX_Y, EX_G = EXTRA[\'text\'].tolist(), EXTRA[\'model\'].map(LAB2I).to_numpy(dtype=int), EXTRA[\'prompt_id\'].to_numpy(dtype=object)\nprint(f\'extra passages: {len(EXTRA)} rows, {EXTRA["prompt_id"].nunique()} prompts\', flush=True)\n\n\ndef extra_sample_weights():\n    """Per class: original rows keep (1 - EXTRA_SHARE) of the class weight, the class\'s extra rows share EXTRA_SHARE of it."""\n    n_tr, n_ex = np.bincount(y, minlength=K).astype(float), np.bincount(EX_Y, minlength=K).astype(float)\n    w_tr = np.where(n_ex[y] > 0, 1.0 - EXTRA_SHARE, 1.0)\n    w_ex = EXTRA_SHARE * n_tr[EX_Y] / np.maximum(n_ex[EX_Y], 1.0)\n    return np.concatenate([w_tr, w_ex])\n\n\ngrams = StyleGrams().fit(TR + EX_T)                       # vocabulary and idf from train + extra\nX_tr, X_ex, X_te = grams.transform(TR), grams.transform(EX_T) if len(EX_T) else None, grams.transform(TE)\n\n\ndef fit_svc(X, labels, C=0.5, **kw):\n    return LinearSVC(C=C, random_state=42, max_iter=4000).fit(X, labels, **kw)\n\n\nif X_ex is not None:\n    svc_plain = fit_svc(vstack([X_tr, X_ex]).tocsr(), np.concatenate([y, EX_Y]), sample_weight=extra_sample_weights())\nelse:\n    svc_plain = fit_svc(X_tr, y)\nP_style = svc_probs(svc_plain, X_te)\n\n# centred twin: prompt-centred training rows (extra rows centred within their own prompts) -> inferred test groups\nX_tr_c = sparse_group_centre(X_tr, PROMPT)\nif X_ex is not None:\n    svc_centred = fit_svc(vstack([X_tr_c, sparse_group_centre(X_ex, EX_G, GROUP_MIN, 10 ** 6)]).tocsr(), np.concatenate([y, EX_Y]),\n                          sample_weight=extra_sample_weights())\nelse:\n    svc_centred = fit_svc(X_tr_c, y)\nG_style = normalised_content_groups(TE)\nrows_style = eligible(G_style)\ndescribe_groups(\'stylegram groups\', G_style)\nif rows_style.any():\n    P_style[rows_style] = svc_probs(svc_centred, sparse_group_centre(X_te[rows_style], G_style[rows_style]))\nif TINY:   # exercise the centred path on training rows grouped by their true prompt (output discarded)\n    st_idx, st_g = self_test_rows()\n    acc = np.mean(svc_centred.classes_[svc_centred.decision_function(sparse_group_centre(X_tr[st_idx], st_g)).argmax(1)] == y[st_idx])\n    print(f\'  centred stylegram self-test on {len(st_idx)} training rows (seen in training): acc {acc:.2f}\', flush=True)\nadd_member(\'stylegram\', P_style)\ndel X_tr_c, svc_plain, svc_centred, P_style\n\n\n# ---- gates: style SVCs that only re-rank inside one family after the blend\ndef family_gate(texts, labels, classes, C, sample_weight=None, balanced=False):\n    g = StyleGrams().fit(texts)\n    svc = LinearSVC(C=C, class_weight=\'balanced\' if balanced else None, random_state=42, max_iter=4000).fit(\n        g.transform(texts), labels, sample_weight=sample_weight)\n    assert list(svc.classes_) == list(classes)\n    d = svc.decision_function(g.transform(TE))\n    return np.column_stack([-d, d]) if len(classes) == 2 else d\n\n\nGATES = {}\napi_claude = sorted(glob.glob(\'/kaggle/input/**/ntl_extra_passages_claude.csv\', recursive=True))\nif api_claude:\n    api = pd.read_csv(api_claude[0])\n    m = np.isin(y, FAMILY_IDX[\'claude\'])\n    GATES[\'claude\'] = family_gate([t for t, k in zip(TR, m) if k] + api[\'text\'].astype(str).tolist(),\n                                  np.concatenate([y[m], api[\'model\'].map(LAB2I).to_numpy(dtype=int)]), FAMILY_IDX[\'claude\'], 0.5)\n    print(f\'claude gate: {int(m.sum())} official + {len(api)} API rows\', flush=True)\n    del api\nelse:\n    print(\'Claude API passages not attached: Claude gate off\', flush=True)\nfor fam, C in [(\'gpt\', 1.0), (\'gemini\', 1.0), (\'glm\', 0.5)]:\n    m, me = np.isin(y, FAMILY_IDX[fam]), np.isin(EX_Y, FAMILY_IDX[fam])\n    GATES[fam] = family_gate([t for t, k in zip(TR, m) if k] + [t for t, k in zip(EX_T, me) if k], np.concatenate([y[m], EX_Y[me]]),\n                             FAMILY_IDX[fam], C, sample_weight=np.concatenate([np.ones(int(m.sum())), np.full(int(me.sum()), 0.25)]), balanced=True)\n    print(f\'{fam} gate: {int(m.sum())} official + {int(me.sum())} extra rows\', flush=True)\ndel grams, X_tr, X_ex, X_te\n\n# ============================================================ 3. stylometric features + LightGBM\nPUNCT = list(\',.;:!?()[]"\\\'-\') + [\'—\', \'–\', \'‘\', \'’\', \'“\', \'”\', \'…\', \'*\', \'/\', \'%\']\nSTOPS = sorted(ENGLISH_STOP_WORDS)\nSENT_RE, WORD_RE = re.compile(r\'(?<=[.!?])\\s+\'), re.compile(r"[A-Za-z’\']+")\n\n\ndef stylometry(t):\n    words = WORD_RE.findall(t)\n    low = [w.lower() for w in words]\n    nw, nc = max(len(words), 1), max(len(t), 1)\n    sents = [s for s in SENT_RE.split(t.strip()) if s]\n    sl = np.array([len(WORD_RE.findall(s)) for s in sents] or [0], dtype=float)\n    wl = np.array([len(w) for w in words] or [0], dtype=float)\n    f = [np.log1p(nw), len(sents), sl.mean(), sl.std(), sl.max(), sl.min(), wl.mean(), wl.std(), (wl >= 9).mean(), len(set(low)) / nw,\n         sum(c.isupper() for c in t) / nc, sum(c.isdigit() for c in t) / nc, sum(ord(c) > 127 for c in t) / nc,\n         t.count(\', \') / nw, t.count(\' and \') / nw, t.count(\' but \') / nw,\n         sum(1 for s in sents if s[:1].isupper() and s.split()[0].lower() in (\'this\', \'these\', \'it\', \'they\')) / max(len(sents), 1),\n         sum(w.endswith(\'ly\') for w in words) / nw, sum(w.endswith(\'ing\') for w in words) / nw, sum(w.endswith(\'tion\') for w in words) / nw]\n    f += [t.count(q) / nw for q in PUNCT]\n    counts = pd.Series(low).value_counts() if low else pd.Series(dtype=float)\n    f += [counts.get(s, 0) / nw for s in STOPS]\n    return f\n\n\nF_tr, F_te = (np.array([stylometry(t) for t in txt], dtype=np.float32) for txt in (TR, TE))\ntry:\n    import lightgbm as lgb\n    booster = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,\n                                 colsample_bytree=0.5, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=4).fit(F_tr, y)\n    add_member(\'stylometry_lgb\', full_prob(booster.predict_proba(F_te), booster.classes_))\n    del booster\nexcept ImportError:\n    assert SMOKE, \'lightgbm missing\'\n    print(\'smoke: lightgbm unavailable, stylometry member skipped\', flush=True)\ndel F_tr, F_te\n\n# ============================================================ 4b. shared-prompt Gaussian group head\n# Verbatim core of our research tool tools/group_gaussian.py (sha256 8a42427b77a37f56...), numpy only.\n# Model per dense view, in a reduced space z (standardise -> within-prompt-residual PCA, fitted on training rows):\n#   z_i = mu_{y_i} + p_g + e_i,  p_g ~ N(0, S_p) shared by every passage of prompt group g,  e_i ~ N(0, S_e).\n# A joint diagonalisation makes the dimensions independent; for a detected group of n passages the exact joint\n# log-likelihood of a label tuple is sum_i a_i(c_i) + sum_{i<j} b_ij(c_i, c_j). Groups of 2-3 are decoded exactly over\n# all label tuples with the at-most-once constraint; larger groups by a cavity EM with the finite-population-corrected\n# prompt-effect estimate lam / (1 + (n-1) lam); singletons integrate the prompt effect out (never used here).\nimport itertools\nK = 18\nDEFAULT = {"red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw", "lam_floor": 1e-4, "n_exact": 3, "inj": True,\n           "iters": 5, "repel": True, "sp_scale": 1.0}\n\n# =================================================================== primitives ====\ndef _lse(A, axis=None, keepdims=False):\n    m = np.max(A, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.0)\n    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m\n    if keepdims: return out\n    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())\n\ndef log_softmax(L): return L - _lse(L, axis=1, keepdims=True)\n\ndef _lw_shrinkage(X):\n    """Ledoit-Wolf shrinkage coefficient for rows X that are already centred (sklearn formula, assume_centered)."""\n    from sklearn.covariance import ledoit_wolf_shrinkage\n    return float(ledoit_wolf_shrinkage(X, assume_centered=True))\n\ndef _shrunk(S, a):\n    d = S.shape[0]; return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)\n\ndef _two_way(Z, y, pinv, G, iters=4):\n    """additive fit Z = mu[y] + P[prompt] + E by alternating means (balanced design -> converges in 1-2 rounds)."""\n    P = np.zeros((G, Z.shape[1])); cnt_p = np.bincount(pinv, minlength=G).astype(float)\n    cls = np.unique(y); mu = np.zeros((K, Z.shape[1]))\n    for _ in range(iters):\n        R = Z - P[pinv]\n        for c in cls: mu[c] = R[y == c].mean(0)\n        R = Z - mu[y]; P = np.zeros_like(P); np.add.at(P, pinv, R); P /= cnt_p[:, None]\n    return mu, P, cnt_p\n\ndef fit(X, y, prompts, cfg=None):\n    """fit the shared-prompt Gaussian on training rows. X (N, D) raw features, y (N,) labels in 0..17, prompts (N,) group ids.\n    Returns a params dict (numpy arrays + floats); temperatures default to 1 (set by the caller\'s cross-fit)."""\n    c = dict(DEFAULT); c.update(cfg or {})\n    X = np.asarray(X, dtype=np.float64); y = np.asarray(y); N, D = X.shape\n    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True); G = len(pu)\n    mx = X.mean(0); sx = X.std(0); sx = np.where(sx > 1e-8, sx, 1.0); Xs = (X - mx) / sx\n    if c["red"] == "none":\n        V = np.eye(D)\n    else:\n        if c["red"] == "wpca":\n            Pm = np.zeros((G, D)); np.add.at(Pm, pinv, Xs); Pm /= np.bincount(pinv)[:, None]; R = Xs - Pm[pinv]\n        elif c["red"] == "pca": R = Xs\n        else: raise ValueError(c["red"])\n        d = int(min(c["d"], N - G - 1, D))\n        # economy SVD through the (N x N) Gram matrix (N << D for the real views)\n        if N < D:\n            w, U = np.linalg.eigh(R @ R.T); o = np.argsort(w)[::-1][:d]; w, U = np.clip(w[o], 1e-12, None), U[:, o]\n            V = (R.T @ U) / np.sqrt(w)\n        else:\n            _, _, Vt = np.linalg.svd(R, full_matrices=False); V = Vt[:d].T\n    Z = Xs @ V; d = Z.shape[1]\n    mu, P, ng = _two_way(Z, y, pinv, G)\n    E = Z - mu[y] - P[pinv]; dof = max(N - G - len(np.unique(y)) + 1, 1)\n    Se_raw = E.T @ E / dof\n    ae = _lw_shrinkage(E) if c["se_shrink"] == "lw" else float(c["se_shrink"]); Se = _shrunk(Se_raw, ae)\n    Pc = P - P.mean(0); Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw\n    ap = _lw_shrinkage(Pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])\n    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se)); Sp = (1 - ap) * Sp_raw + ap * (tr / d) * np.eye(d); Sp *= c["sp_scale"]\n    # joint diagonalisation: A Se A\' = I, A Sp A\' = diag(lam)\n    L = np.linalg.cholesky(Se); Li = np.linalg.inv(L); M = Li @ Sp @ Li.T; M = 0.5 * (M + M.T)\n    lam, Q = np.linalg.eigh(M); lam = np.clip(lam, c["lam_floor"], None); A = Q.T @ Li\n    W = V @ A.T\n    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 1.0, "Tg": 1.0,\n            "info": {"d": int(d), "a_e": round(ae, 4), "a_p": round(ap, 4), "lam_pct": np.round(np.percentile(lam, [5, 50, 95, 100]), 3).tolist()}, "cfg": c}\n\ndef transform(X, prm): return ((np.asarray(X, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]\n\n_MASK = {}\ndef _inj_mask(n, C):\n    if (n, C) not in _MASK:\n        grids = np.meshgrid(*[np.arange(C)] * n, indexing="ij"); bad = np.zeros([C] * n, bool)\n        for i, j in itertools.combinations(range(n), 2): bad |= grids[i] == grids[j]\n        _MASK[(n, C)] = bad\n    return _MASK[(n, C)]\n\ndef singleton_ll(U, m, lam):\n    """(n, C) log N(u; m_c, I + diag lam) up to a constant (p integrated out)."""\n    R = U[:, None, :] - m[None]; return -0.5 * (R ** 2 / (1 + lam)).sum(-1)\n\ndef joint_terms(U, m, lam):\n    """a (n, C) and B (n, n, C, C) of the exact joint log-likelihood of one group (constant in c dropped)."""\n    n = len(U); gam = lam / (1 + n * lam); R = U[:, None, :] - m[None]\n    a = -0.5 * (R ** 2 * (1 - gam)).sum(-1); G = R * np.sqrt(gam); B = np.einsum("icd,jed->ijce", G, G)\n    return a, B\n\ndef joint_table(a, B):\n    """full tuple table Q[c_1, ..., c_n] = sum_i a_i + sum_{i<j} B_ij."""\n    n, C = a.shape; Q = np.zeros([C] * n)\n    for i in range(n):\n        sh = [1] * n; sh[i] = C; Q = Q + a[i].reshape(sh)\n    for i, j in itertools.combinations(range(n), 2):\n        sh = [1] * n; sh[i] = C; sh[j] = C; Q = Q + B[i, j].reshape(sh)\n    return Q\n\ndef exact_marginals(Q, T=1.0, inj=True):\n    """per-passage log marginal posteriors (n, C) from a tuple table Q (unnormalised log joint), tempered by 1/T."""\n    n = Q.ndim; C = Q.shape[0]; Qt = Q / T\n    if inj and n >= 2: Qt = np.where(_inj_mask(n, C), -np.inf, Qt)\n    lz = _lse(Qt); out = np.empty((n, C))\n    for i in range(n):\n        ax = tuple(k for k in range(n) if k != i); out[i] = (_lse(Qt, axis=ax) if ax else Qt) - lz\n    return out\n\ndef em_marginals(U, m, lam, T=1.0, iters=5, repel=False, q0=None):\n    """cavity EM for one group (any n >= 2): returns (n, C) log q."""\n    n = len(U); v = lam / (1 + (n - 1) * lam)\n    L = singleton_ll(U, m, lam) / T if q0 is None else np.log(np.clip(q0, 1e-300, None))\n    q = np.exp(log_softmax(L))\n    for _ in range(iters):\n        s = U - q @ m; S = s.sum(0); Pm = (S[None] - s) * v                  # cavity prompt-effect means (n, d)\n        R = U[:, None, :] - Pm[:, None, :] - m[None]; L = -0.5 * (R ** 2 / (1 + v)).sum(-1) / T\n        if repel:   # mean-field at-most-once: discourage classes the other members already hold\n            lo = np.log(np.clip(1 - q, 1e-6, None)); L = L + (lo.sum(0)[None] - lo)\n        q = np.exp(log_softmax(L))\n    return log_softmax(L)\n\nLAM_DIFF = 1e6\ndef lam_g(prm, cfg=None):\n    """lam used for groups: the fitted one, or ~infinity in the paired-difference ablation (cfg \'diff\'): then the joint likelihood\n    only sees within-group differences (x_i - x_j around mu_a - mu_b with covariance 2 S_e; triples: centred observations vs\n    centred prototypes) and the prompt term cancels exactly."""\n    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {})\n    return np.full_like(prm["lam"], LAM_DIFF) if c.get("diff") else prm["lam"]\n\ndef group_logpost(U, prm, T1=None, Tg=None, cfg=None):\n    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {}); n = len(U); lg = lam_g(prm, cfg)\n    T1 = prm["T1"] if T1 is None else T1; Tg = prm["Tg"] if Tg is None else Tg\n    if n == 1: return log_softmax(singleton_ll(U, prm["m"], prm["lam"]) / T1)\n    if n <= c["n_exact"]:\n        a, B = joint_terms(U, prm["m"], lg); return exact_marginals(joint_table(a, B), Tg, c["inj"])\n    return em_marginals(U, prm["m"], lg, Tg, c["iters"], c["repel"])\n\ndef scores(X_pool, lab, prm, T1=None, Tg=None, cfg=None, U=None):\n    """(n, 18) log-probabilities for every pool row; lab = detected group labels of the pool rows (singletons allowed)."""\n    U = transform(X_pool, prm) if U is None else U; lab = np.asarray(lab); n = len(U); out = np.empty((n, prm["m"].shape[0]))\n    sizes = np.bincount(lab); s1 = sizes[lab] == 1\n    T1 = prm["T1"] if T1 is None else T1\n    if s1.any(): out[s1] = log_softmax(singleton_ll(U[s1], prm["m"], prm["lam"]) / T1)\n    for g in np.where(sizes >= 2)[0]:\n        ii = np.where(lab == g)[0]; out[ii] = group_logpost(U[ii], prm, T1, Tg, cfg)\n    return out\n\n# ============================================================ 4. frozen models: plain heads, centred twins, Gaussian group heads\nif not SMOKE:\n    from transformers import AutoConfig, AutoModel, AutoTokenizer\nCACHED_PARTS = {}   # key -> {part name: (train features, test features)}\n\n\ndef find_checkpoint(model_type, hidden=None, dir_name=None):\n    """A checkpoint directory under /kaggle/input by model_type (+ hidden size) or by directory name."""\n    pattern = f\'/kaggle/input/**/{dir_name}/config.json\' if dir_name else \'/kaggle/input/**/config.json\'\n    for cfg in sorted(glob.glob(pattern, recursive=True)):\n        try:\n            c = json.load(open(cfg))\n        except Exception:\n            continue\n        if dir_name or (c.get(\'model_type\') == model_type and (hidden is None or c.get(\'hidden_size\') == hidden)):\n            return os.path.dirname(cfg)\n    return None\n\n\ndef load_truncated_decoder(path, keep, dtype):\n    cfg = AutoConfig.from_pretrained(path)\n    cfg.num_hidden_layers = keep\n    if getattr(cfg, \'layer_types\', None):\n        cfg.layer_types = cfg.layer_types[:keep]\n    kw = dict(config=cfg, dtype=dtype)\n    if DEV == \'cuda\':\n        kw[\'device_map\'] = \'auto\'\n    _m = AutoModel.from_pretrained(path, **kw).eval()\n    _cpu = sum(p.numel() for p in _m.parameters() if p.device.type == \'cpu\')\n    _all = sum(p.numel() for p in _m.parameters())\n    print(f\'  trunk {path}: {100 * _cpu / max(_all, 1):.1f}% of parameters on CPU \'f\'({time.time() - T0:.0f}s) -- >5% means accelerate offloaded and this pass will run ~40x slow\', flush=True)\n    return _m\n\n\ndef style_token_mask(tok):\n    """Vocabulary mask of \'style\' tokens: the decoded piece is an English stop word or pure punctuation."""\n    pieces = [tok.decode([i]) for i in range(len(tok))]\n    return torch.tensor([(p.strip().lower() in ENGLISH_STOP_WORDS) or bool(re.fullmatch(r\'\\s*[^\\w\\s]+\\s*\', p)) for p in pieces])\n\n\n@torch.no_grad()\ndef pooled_states(model, tok, texts, parts, decoder, bs, max_len=512):\n    """{\'L{layer}_{pool}\': (n, hidden)} masked mean pools of hidden_states[layer] (all tokens, or style tokens only)."""\n    H = model.config.hidden_size\n    out = {f\'L{L}_{pool}\': np.zeros((len(texts), H), dtype=np.float32) for L, pool in parts}\n    smask = style_token_mask(tok) if any(pool == \'style\' for _, pool in parts) else None\n    dev = next(model.parameters()).device\n    order = np.argsort([len(t) for t in texts])\n    retried = 0\n    for b0 in range(0, len(texts), bs):\n        idx = order[b0:b0 + bs]\n        if decoder:\n            batch = [(tok.bos_token or \'\') + texts[i] for i in idx]\n            enc = tok(batch, return_tensors=\'pt\', padding=True, truncation=True, max_length=max_len, add_special_tokens=False).to(dev)\n        else:\n            enc = tok([texts[i] for i in idx], return_tensors=\'pt\', padding=True, truncation=True, max_length=max_len).to(dev)\n        hs = model(**enc, output_hidden_states=True).hidden_states\n        if decoder and not all(torch.isfinite(hs[L]).all() for L, _ in parts):   # fp16 overflow: redo the batch in bf16\n            retried += 1\n            model.to(torch.bfloat16)\n            hs = model(**enc, output_hidden_states=True).hidden_states\n        m_all = enc[\'attention_mask\'].float()\n        for L, pool in parts:\n            m = m_all * smask.to(m_all.device)[enc[\'input_ids\']].float() if pool == \'style\' else m_all\n            m = m.unsqueeze(-1).to(hs[L].device)\n            out[f\'L{L}_{pool}\'][idx] = ((hs[L].float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()\n        del hs, enc\n    if retried:\n        print(f\'  {retried} batches redone in bf16\', flush=True)\n    return {k: np.nan_to_num(v) for k, v in out.items()}\n\n\ndef pseudo_parts(parts, dim=48):\n    """SMOKE only: deterministic stand-ins for the model passes (TF-IDF SVD of the texts), same downstream code."""\n    from sklearn.decomposition import TruncatedSVD\n    tf = TfidfVectorizer(analyzer=\'char\', ngram_range=(2, 4), sublinear_tf=True, min_df=2).fit(TR + TE)\n    out = {}\n    for j, (L, pool) in enumerate(parts):\n        svd = TruncatedSVD(dim, random_state=L + 7 * j).fit(tf.transform(TR))\n        out[f\'L{L}_{pool}\'] = (svd.transform(tf.transform(TR)).astype(np.float32), svd.transform(tf.transform(TE)).astype(np.float32))\n    return out\n\n\ndef extract_parts(key, parts, loader):\n    """Train and test pools for every requested part; loader() returns (model, tok, decoder flag, batch size)."""\n    if SMOKE:\n        return pseudo_parts(parts)\n    model, tok, decoder, bs = loader()\n    tr, te = pooled_states(model, tok, TR, parts, decoder, bs), pooled_states(model, tok, TE, parts, decoder, bs)\n    del model, tok\n    gc.collect()\n    if DEV == \'cuda\':\n        torch.cuda.empty_cache()\n    return {k: (tr[k], te[k]) for k in tr}\n\n\ndef decoder_loader(model_type, hidden, keep):\n    def load():\n        path = find_checkpoint(model_type, hidden)\n        assert path, f\'no {model_type} checkpoint with hidden size {hidden} under /kaggle/input\'\n        tok = AutoTokenizer.from_pretrained(path)\n        tok.padding_side = \'right\'\n        if tok.pad_token is None:\n            tok.pad_token = tok.eos_token\n        return load_truncated_decoder(path, keep, torch.float16 if DEV == \'cuda\' else torch.float32), tok, True, 8\n    return load\n\n\ndef encoder_loader(dir_name):\n    def load():\n        path = find_checkpoint(None, dir_name=dir_name)\n        assert path, f\'encoder directory {dir_name} not found under /kaggle/input\'\n        return AutoModel.from_pretrained(path, dtype=torch.float32).to(DEV).eval(), AutoTokenizer.from_pretrained(path), False, 16\n    return load\n\n\ndef concat_parts(cache, parts):\n    return (np.concatenate([cache[f\'L{L}_{p}\'][0] for L, p in parts], 1), np.concatenate([cache[f\'L{L}_{p}\'][1] for L, p in parts], 1))\n\n\ndef plain_and_twin(key, Etr, Ete, C):\n    """Plain head (test-side standardisation) for the blend; centred twin evidence on grouped rows."""\n    if W[key] > 0:\n        sc = StandardScaler().fit(Etr)\n        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Etr), y)\n        Xte = (Ete - Ete.mean(0)) / (Ete.std(0) + 1e-6) if len(Ete) >= 2 else sc.transform(Ete)\n        add_member(key, full_prob(clf.predict_proba(Xte), clf.classes_))\n    else:\n        print(f\'  {key}: plain head has weight 0 in the 0.563 blend, not fitted\', flush=True)\n    if key in TWIN_W:\n        csc = StandardScaler().fit(group_centre_dense(Etr, PROMPT))\n        twin = LogisticRegression(C=C, max_iter=5000).fit(csc.transform(group_centre_dense(Etr, PROMPT)), y)\n        if GROUP_ROWS.any():\n            pc = twin.predict_proba(csc.transform(group_centre_dense(Ete[GROUP_ROWS], G_TE[GROUP_ROWS])))\n            add_member(key + \'_twin\', full_prob(pc, twin.classes_), rows=GROUP_ROWS, weight=TWIN_W[key])\n        if TINY:\n            st_idx, st_g = self_test_rows()\n            ps = twin.predict_proba(csc.transform(group_centre_dense(Etr[st_idx], st_g)))\n            print(f\'  {key} twin self-test on {len(st_idx)} training rows (seen in training): acc {np.mean(twin.classes_[ps.argmax(1)] == y[st_idx]):.2f}\', flush=True)\n\n\ndef gaussian_group_head(name, spec, Xtr, Xte):\n    """Our addition: fit the shared-prompt Gaussian on the training prompts, score the grouped test rows jointly within their group."""\n    prm = fit(Xtr, y, PROMPT, dict(GAUSS_SETTING, d=spec[\'d\']))\n    prm[\'T1\'], prm[\'Tg\'] = spec[\'T1\'], spec[\'Tg\']\n    print(f\'  {name}: d {prm["info"]["d"]}, shrinkage e/p {prm["info"]["a_e"]}/{prm["info"]["a_p"]}, lam pct {prm["info"]["lam_pct"]}\', flush=True)\n    if GROUP_ROWS.any():\n        LP = scores(Xte[GROUP_ROWS], np.unique(G_TE[GROUP_ROWS], return_inverse=True)[1], prm)\n        add_member(name, np.exp(LP), rows=GROUP_ROWS, weight=spec[\'w\'])\n    if TINY:\n        st_idx, st_g = self_test_rows()\n        LP = scores(Xtr[st_idx], st_g, prm)\n        print(f\'  {name} self-test on {len(st_idx)} training rows (seen in training): acc {np.mean(LP.argmax(1) == y[st_idx]):.2f}\', flush=True)\n\n\nfor key, model_type, hidden, parts, extra_parts, C in LM_HEADS:\n    print(f\'[{key}] {model_type} hidden {hidden} parts {parts + extra_parts}\', flush=True)\n    cache = extract_parts(key, parts + extra_parts, decoder_loader(model_type, hidden, max(L for L, _ in parts + extra_parts) + 1))\n    if G_TE is None:   # the GLM pass comes first: its layer-24 pool defines the test groups\n        G_TE = mixed_groups(TE, cache[f\'L{GROUP_LAYER}_mean\'][1])\n        GROUP_ROWS = eligible(G_TE)\n        describe_groups(\'test groups (content + GLM L24)\', G_TE)\n        if SMOKE:\n            same_g, same_p = G_TE[:, None] == G_TE[None, :], test[\'prompt_id\'].to_numpy(dtype=object)[:, None] == test[\'prompt_id\'].to_numpy(dtype=object)[None, :]\n            off = ~np.eye(len(TE), dtype=bool)\n            print(f\'  smoke grouping: pair precision {(same_g & same_p & off).sum() / max((same_g & off).sum(), 1):.2f} \'\n                  f\'recall {(same_g & same_p & off).sum() / max((same_p & off).sum(), 1):.2f}\', flush=True)\n    CACHED_PARTS[key] = cache\n    plain_and_twin(key, *concat_parts(cache, parts), C)\n    for name, spec in GAUSS_HEADS.items():\n        if spec[\'src\'] == key:\n            gaussian_group_head(name, spec, *cache[spec[\'part\']])\n    del cache\n    CACHED_PARTS.pop(key)\n    gc.collect()\n\nfor key, dir_name, parts, C in ENC_HEADS:\n    print(f\'[{key}] {dir_name} parts {parts}\', flush=True)\n    cache = extract_parts(key, parts, encoder_loader(dir_name))\n    plain_and_twin(key, *concat_parts(cache, parts), C)\n    for name, spec in GAUSS_HEADS.items():\n        if spec[\'src\'] == key:\n            gaussian_group_head(name, spec, *cache[spec[\'part\']])\n    del cache\n    gc.collect()\n\n# ============================================================ 5. LoRA classifiers (inference only)\n@torch.no_grad()\ndef score_lora_checkpoints(trunk_spec, pool_layers, checkpoints):\n    """One truncated trunk, adapters swapped in turn. Plain readout -> blend member; centred readout -> twin on grouped rows;\n    within-Claude readout -> Claude gate."""\n    import importlib.util as iu\n    _find = iu.find_spec   # Kaggle\'s torchao build is rejected by peft; hide it (unused here)\n    iu.find_spec = lambda name, *a, **k: None if name.split(\'.\')[0] == \'torchao\' else _find(name, *a, **k)\n    from peft import PeftModel\n    import peft.import_utils as pi\n    pi.is_torchao_available = lambda *a, **k: False\n    from safetensors.torch import load_file\n    model_type, H, keep = trunk_spec\n    base = find_checkpoint(model_type, H)\n    assert base, f\'no {model_type} base checkpoint under /kaggle/input\'\n    cfg = AutoConfig.from_pretrained(base)\n    cfg.num_hidden_layers, cfg.use_cache = keep, False\n    if getattr(cfg, \'layer_types\', None) is not None:\n        cfg.layer_types = cfg.layer_types[:keep]\n    trunk = AutoModel.from_pretrained(base, config=cfg, dtype=torch.float16, device_map=\'auto\').eval()\n    for key, marker, norms_file, anchor_file in checkpoints:\n        hits = sorted(glob.glob(\'/kaggle/input/\' + marker, recursive=True))\n        assert hits, f\'checkpoint for {key} not attached ({marker})\'\n        ck = pathlib.Path(os.path.dirname(hits[0]))\n        print(f\'[{key}] {ck}\', flush=True)\n        tok = AutoTokenizer.from_pretrained(ck / \'tokenizer\')\n        tok.padding_side = \'right\'\n        if tok.pad_token is None:\n            tok.pad_token = tok.eos_token\n        lm = PeftModel.from_pretrained(trunk, ck / \'adapter\').eval()\n        state = load_file(str(ck / norms_file))\n        norms = torch.nn.ModuleList([torch.nn.LayerNorm(H) for _ in pool_layers])\n        norms.load_state_dict({k[len(\'norms.\'):]: v for k, v in state.items() if k.startswith(\'norms.\')})\n        ro = np.load(ck / \'readouts.npz\')\n        assert list(ro[\'classes\']) == LABELS and list(ro[\'pool_layers\']) == pool_layers\n        dev = lm.get_input_embeddings().weight.device\n        norms = norms.to(dev).float()\n\n        def token_ids(texts):\n            return [ids + [tok.eos_token_id] for ids in tok(list(texts), add_special_tokens=False, truncation=True, max_length=LORA_MAX_LEN)[\'input_ids\']]\n\n        def features(texts, bs):\n            ids = token_ids(texts)\n            Z = np.zeros((len(ids), H * len(pool_layers)), dtype=np.float32)\n            order = np.argsort([len(t) for t in ids], kind=\'stable\')\n            for b0 in range(0, len(order), bs):\n                idx = order[b0:b0 + bs]\n                batch = tok.pad({\'input_ids\': [ids[i] for i in idx]}, return_tensors=\'pt\').to(dev)\n                with torch.autocast(\'cuda\', dtype=torch.float16):\n                    hs = lm(**batch, output_hidden_states=True, use_cache=False).hidden_states\n                m = batch[\'attention_mask\'].unsqueeze(-1).float()\n                m[:, 0] = 0   # the first token is masked out of the mean, as in training\n                Z[idx] = torch.cat([nm((hs[L].float() * m.to(hs[L].device)).sum(1).to(dev) / m.sum(1).to(dev))\n                                    for L, nm in zip(pool_layers, norms)], -1).cpu().numpy()\n                del hs, batch\n            return Z\n\n        if (ck / anchor_file).exists():   # the checkpoint\'s own features for training rows 0..2: our forward pass must reproduce them\n            saved = np.load(ck / anchor_file)[\'features\'].astype(np.float32)\n            ours = features(TR[:3], 3)\n            rel = float(np.abs(ours - saved).mean() / (np.abs(saved).mean() + 1e-9))\n            print(f\'  parity with the checkpoint anchors: max abs {float(np.abs(ours - saved).max()):.4f}, mean rel {rel:.5f}\', flush=True)\n            assert rel < 0.02, f\'{key}: our features do not match the checkpoint\'\n        Z = features(TE, LORA_BS)\n        add_member(key, softmax_rows(Z @ ro[\'plain_coef\'].T + ro[\'plain_intercept\']))\n        if \'claude_coef\' in ro.files:\n            lc = Z @ ro[\'claude_coef\'].T + ro[\'claude_intercept\']\n            GATES[\'claude_lora\'] = lc - np.log(np.exp(lc - lc.max(1, keepdims=True)).sum(1, keepdims=True)) - lc.max(1, keepdims=True)\n        if key in LORA_TWIN_W and GROUP_ROWS.any():\n            Zc = group_centre_dense(Z[GROUP_ROWS], G_TE[GROUP_ROWS])\n            add_member(key + \'_twin\', softmax_rows(Zc @ ro[\'center_coef\'].T + ro[\'center_intercept\']), rows=GROUP_ROWS, weight=LORA_TWIN_W[key])\n        trunk = lm.unload()\n        del lm, Z\n        gc.collect()\n        torch.cuda.empty_cache()\n    del trunk\n    gc.collect()\n    torch.cuda.empty_cache()\n\n\nif not SMOKE:\n    for trunk_spec, pool_layers, checkpoints in LORA_RUNS:\n        score_lora_checkpoints(trunk_spec, pool_layers, checkpoints)\n\n# ============================================================ 6. token-attention heads + UltraBERT\nclass TokenAttention(torch.nn.Module if torch is not None else object):   # the module of the public training kernels (lucasboesen/name-that-llm-train-tokattn, -tokattn2)\n    def __init__(self, k, h=4096):\n        super().__init__()\n        self.drop, self.proj, self.score, self.out = torch.nn.Dropout(0.2), torch.nn.Linear(h, 64), torch.nn.Linear(64, 1), torch.nn.Linear(128, k)\n\n    def forward(self, x, m):\n        h = torch.tanh(self.proj(self.drop(x)))\n        a = torch.softmax(self.score(h).squeeze(-1).masked_fill(~m, -1e4), -1).unsqueeze(-1)\n        mf = m.unsqueeze(-1).float()\n        return self.out(self.drop(torch.cat([(h * mf).sum(1) / mf.sum(1).clamp(min=1), (a * h).sum(1)], -1)))\n\n\n@torch.no_grad()\ndef token_attention_probs(texts, head_file, backbone, bs=8):\n    hits = sorted(glob.glob(\'/kaggle/input/**/\' + head_file, recursive=True))\n    assert hits, f\'training-kernel output holding {head_file} not attached\'\n    d = torch.load(hits[0], map_location=\'cpu\')\n    assert list(d[\'classes\']) == LABELS\n    decoder = d.get(\'decoder\', True)\n    path = find_checkpoint(\'qwen3_5\') if backbone == \'qwen3_5\' else find_checkpoint(None, dir_name=\'deberta-v3-large\')\n    assert path, f\'backbone {backbone} not attached\'\n    cfg = AutoConfig.from_pretrained(path)\n    if decoder:\n        tc = getattr(cfg, \'text_config\', None) or cfg\n        tc.num_hidden_layers = d[\'keep\']\n        if getattr(tc, \'layer_types\', None):\n            tc.layer_types = tc.layer_types[:d[\'keep\']]\n        lm = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map=\'auto\').eval()\n    else:\n        lm = AutoModel.from_pretrained(path, dtype=torch.float32).to(DEV).eval()\n    tok = AutoTokenizer.from_pretrained(path)\n    tok.padding_side = \'right\'\n    if tok.pad_token is None:\n        tok.pad_token = tok.eos_token\n    H = d.get(\'hidden\', 4096)\n    net = TokenAttention(K, H)\n    net.load_state_dict(d[\'state_dict\'])\n    dev = lm.get_input_embeddings().weight.device\n    net, mu, sd = net.to(dev).eval(), d[\'mu\'].float().to(dev), d[\'sd\'].float().to(dev)\n    ids = tok(list(texts), add_special_tokens=not decoder, truncation=True, max_length=d[\'max_len\'])[\'input_ids\']\n    P = np.zeros((len(texts), K))\n    order = np.argsort([len(t) for t in ids])\n    for b0 in range(0, len(order), bs):\n        idx = order[b0:b0 + bs]\n        batch = tok.pad({\'input_ids\': [ids[i] for i in idx]}, return_tensors=\'pt\').to(dev)\n        out = lm(input_ids=batch[\'input_ids\'], attention_mask=batch[\'attention_mask\'], output_hidden_states=True)\n        hs = (out.hidden_states if getattr(out, \'hidden_states\', None) is not None else out.language_model_outputs.hidden_states)[d[\'layer\']]\n        n = batch[\'attention_mask\'].sum(1)\n        T = max(int(n.max()) - 1, 1)\n        x = torch.zeros(len(idx), T, H, device=dev)\n        m = torch.zeros(len(idx), T, dtype=torch.bool, device=dev)\n        for r in range(len(idx)):   # first token dropped, states rounded through fp16, as in training\n            k = int(n[r]) - 1\n            if k > 0:\n                x[r, :k] = ((torch.nan_to_num(hs[r, 1:k + 1].to(dev).float()).half().float() - mu) / sd).clamp(-8, 8)\n                m[r, :k] = True\n        P[idx] = torch.softmax(net(x, m), -1).cpu().numpy()\n        del hs, out, batch\n    del lm\n    gc.collect()\n    torch.cuda.empty_cache()\n    return P\n\n\n@torch.no_grad()\ndef ultrabert_probs(texts):\n    from transformers import AutoModelForSequenceClassification\n    folds = None\n    for root in glob.glob(\'/kaggle/input/**/ultrabert-380m-ntl\', recursive=True):\n        fs = [f for f in sorted(glob.glob(os.path.join(root, \'fold*\'))) if os.path.exists(os.path.join(f, \'config.json\'))\n              and json.load(open(os.path.join(f, \'config.json\'))).get(\'model_type\') == \'ultrabert\']\n        if len(fs) == 5:\n            folds = fs\n            break\n    assert folds, \'ultrabert-380m-ntl (5 folds) not attached\'\n    tok = AutoTokenizer.from_pretrained(folds[0], trust_remote_code=True)\n    ids = tok(texts, truncation=True, max_length=512)[\'input_ids\']\n    order = np.argsort([len(e) for e in ids])\n    P = np.zeros((len(texts), K))\n    for fp in folds:\n        em = AutoModelForSequenceClassification.from_pretrained(fp, trust_remote_code=True, dtype=torch.float16, attn_implementation=\'sdpa\').to(DEV).eval()\n        cols = [LAB2I[em.config.id2label[i]] for i in range(len(em.config.id2label))]\n        for b0 in range(0, len(order), 32):\n            idx = order[b0:b0 + 32]\n            b = tok.pad({\'input_ids\': [ids[j] for j in idx]}, return_tensors=\'pt\').to(DEV)\n            P[np.ix_(idx, cols)] += em(**b).logits.float().softmax(-1).cpu().numpy() / len(folds)\n        del em\n        gc.collect()\n        torch.cuda.empty_cache()\n    return P\n\n\nif not SMOKE:\n    for key, head_file, backbone in TOKATTN_HEADS:\n        if TINY:   # in-sample check on training rows the head was fitted on: must be high, or the checkpoint is not being reproduced\n            acc = float(np.mean(token_attention_probs(TR[:40], head_file, backbone).argmax(1) == y[:40]))\n            print(f\'  {key} in-sample check on 40 training rows: acc {acc:.2f}\', flush=True)\n            assert acc > 0.5, f\'{key} does not reproduce its training fit\'\n        add_member(key, token_attention_probs(TE, head_file, backbone))\n    add_member(\'ultrabert\', ultrabert_probs(TE))\n\n'
# ---- Stage 2 pre-flight: the embedded source has its own unguarded checkpoint search, and Kaggle now mounts
# datasets under /kaggle/input/datasets/ and models under /kaggle/input/models/. Sorted glob order therefore puts
# swaroopmeher/.../lora24/final/base_config (config.json only, no weights) ahead of the real Qwen3 base, and the
# source dies with "no file named model.safetensors". Same guard as find_base above, applied to the source text.
# The dir_name branch (encoder directories) keeps its original first-match behaviour: only the model_type search,
# which is what broke, is guarded.
_FC_OLD = '''def find_checkpoint(model_type, hidden=None, dir_name=None):
    """A checkpoint directory under /kaggle/input by model_type (+ hidden size) or by directory name."""
    pattern = f'/kaggle/input/**/{dir_name}/config.json' if dir_name else '/kaggle/input/**/config.json'
    for cfg in sorted(glob.glob(pattern, recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if dir_name or (c.get('model_type') == model_type and (hidden is None or c.get('hidden_size') == hidden)):
            return os.path.dirname(cfg)
    return None'''
_FC_NEW = '''def find_checkpoint(model_type, hidden=None, dir_name=None):
    """A checkpoint directory under /kaggle/input by model_type (+ hidden size) or by directory name."""
    pattern = f'/kaggle/input/**/{dir_name}/config.json' if dir_name else '/kaggle/input/**/config.json'
    _hits = []
    for cfg in sorted(glob.glob(pattern, recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if not (dir_name or (c.get('model_type') == model_type and (hidden is None or c.get('hidden_size') == hidden))):
            continue
        _d = os.path.dirname(cfg)
        if not any(os.path.exists(os.path.join(_d, f)) for f in
                   ('model.safetensors', 'pytorch_model.bin',
                    'model.safetensors.index.json', 'pytorch_model.bin.index.json')):
            continue   # config-only directory: Kaggle datasets now ship these beside the real weights
        _hits.append(_d)
    if not _hits:
        return None
    _hits.sort(key=lambda d: (0 if '/models/' in d else 1, d))
    return _hits[0]'''
assert _GAUSSIAN_SOURCE.count(_FC_OLD) == 1, 'embedded find_checkpoint changed shape'
_GAUSSIAN_SOURCE = _GAUSSIAN_SOURCE.replace(_FC_OLD, _FC_NEW)
assert _GAUSSIAN_SOURCE.count('config-only directory') == 1
assert _GAUSSIAN_SOURCE.count(_FC_NEW) == 1
compile(_GAUSSIAN_SOURCE, 'public_gaussian_stack', 'exec')   # must still be valid Python
print('SOURCE PATCHED: config-only checkpoint dirs skipped, /kaggle/input/models preferred', flush=True)

# Complementary feature ensemble: source Gaussian stack, no hard family gates.
_new_labels = list(LABELS)
_old_namespace = {'__name__': '__gaussian_stack__'}
exec(compile(_GAUSSIAN_SOURCE, 'public_gaussian_stack', 'exec'), _old_namespace)
assert list(_old_namespace['test']['id'].astype(str)) == list(test['id'].astype(str))
_old_cols = [_old_namespace['LABELS'].index(label) for label in _new_labels]
_old_raw = _old_namespace['LOGIT'][:, _old_cols]
_old_prob = softmax(_old_raw)

# ---- the source namespace holds the whole public stack in memory; keep only what the blend needs
_src_labels = list(_old_namespace['LABELS'])
_src_gates = {_k: np.array(_v, dtype=np.float64) for _k, _v in _old_namespace.get('GATES', {}).items()}
print(f'SOURCE KEPT: stack prob {_old_prob.shape}, gates {sorted(_src_gates)}', flush=True)
del _old_namespace
_gc = __import__('gc')
_gc.collect()
torch.cuda.empty_cache()
print(f'SOURCE DONE and namespace freed ({time.time() - T0:.0f}s)', flush=True)

for ckpt_name, mc in LORAS.items():
    ckpt = os.path.dirname(find(f"{ckpt_name}/meta.json"))
    # The shared-prompt Gaussian needs training-prompt features from the same trunk, so a trunk with
    # gauss_w > 0 pays one extra pass over the 2,699 training passages. Trunks with gauss_w == 0 skip it.
    _need_train = mc.get("gauss_w", 0.0) > 0
    _all_texts = test.text.tolist() + train.text.astype(str).tolist() if _need_train else test.text.tolist()
    _all_Z, heads = lora_features(ckpt, find_base(mc['base']), _all_texts, mc['bs'])
    Z = _all_Z[:len(test)]
    # 1B: the first trunk's last kept layer defines the main-branch prompt clusters (was: English-stop-word
    #     TF-IDF at 0.85, i.e. a purely lexical view that both missed paraphrases and joined shared topics).
    ensure_groups(Z[:, -mc['base']['hidden_size']:])
    LOGIT += mc["w"] * np.log(np.clip(softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), 1e-9, 1))
    print(f"  + {ckpt_name:20s} plain weight {mc['w']:.3f} on {len(test)} rows ({time.time() - T0:.0f}s)", flush=True)
    if twin_rows.any():
        Zc = Z.copy()
        for g in np.unique(groups[twin_rows]):
            Zc[groups == g] -= Zc[groups == g].mean(0)
        centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
        LOGIT[twin_rows] += mc["twin_w"] * np.log(np.clip(softmax(centred), 1e-9, 1))
        print(f"  + {ckpt_name + '_twin':20s} weight {mc['twin_w']:.3f} on {int(twin_rows.sum())} rows", flush=True)
    if _need_train:
        _train_Z = _all_Z[len(test):]
        _prm = fit(_train_Z, y, train['prompt_id'].values,
                   dict(red='wpca', d=GAUSS_D, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4,
                        n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0))
        _prm['T1'] = _prm['Tg'] = GAUSS_T
        print(f'  {ckpt_name}_gauss: d {_prm["info"]["d"]}, shrinkage e/p {_prm["info"]["a_e"]}/{_prm["info"]["a_p"]}, '
              f'lam pct {_prm["info"]["lam_pct"]}', flush=True)
        if twin_rows.any():
            _lp = scores(Z[twin_rows], np.unique(groups[twin_rows], return_inverse=True)[1], _prm)
            assert np.isfinite(_lp).all()
            LOGIT[twin_rows] += mc['gauss_w'] * _lp
            print(f"  + {ckpt_name + '_gauss':20s} weight {mc['gauss_w']:.3f} on {int(twin_rows.sum())} rows "
                  f"(d{GAUSS_D} T{GAUSS_T})", flush=True)
        else:
            print(f'  {ckpt_name}_gauss fitted; no eligible public-test groups', flush=True)
        # Exercise exact pair inference even on the three-row example, without touching its predictions.
        _pid = train['prompt_id'].values
        _ii = np.flatnonzero(_pid == _pid[0])[:2]
        assert len(_ii) == 2
        _qa = scores(_train_Z[_ii], np.zeros(2, dtype=int), _prm)
        assert _qa.shape == (2, K) and np.isfinite(_qa).all() and np.allclose(np.exp(_qa).sum(1), 1)
        print(f'{ckpt_name}_gauss exact-pair probability check PASS', flush=True)
        del _train_Z, _prm, _qa
    del Z, _all_Z
    gc.collect()
    print(f"{ckpt_name} done ({time.time() - T0:.0f}s)", flush=True)


# ---- Stage 2 third member: fine-tuned DeBERTa-v3-large (mean_last pool, linear 18-way head) --------------
if DEBERTA_MEMBER is not None:
    _db_root = os.path.dirname(find(f"{DEBERTA_MEMBER['dir']}/meta.json"))
    _db_meta = json.load(open(os.path.join(_db_root, "meta.json")))
    print(f"[deberta] {_db_root} meta {json.dumps({k: _db_meta[k] for k in _db_meta if k != 'labels'})}", flush=True)
    _db_tok = AutoTokenizer.from_pretrained(os.path.join(_db_root, "encoder"))
    _db_mdl = AutoModel.from_pretrained(os.path.join(_db_root, "encoder"), dtype=torch.float32).to(DEV).eval()
    _db_head = load_file(os.path.join(_db_root, "head.safetensors"))
    _db_W = _db_head["weight"].float().cpu().numpy()
    _db_b = _db_head["bias"].float().cpu().numpy()

    @torch.no_grad()
    def deberta_probs(texts, bs=16):
        ids = _db_tok(list(texts), truncation=True, max_length=DEBERTA_MEMBER["max_len"])["input_ids"]
        order = np.argsort([len(x) for x in ids])
        out = np.zeros((len(texts), K), dtype=np.float32)
        for b0 in range(0, len(order), bs):
            ix = order[b0:b0 + bs]
            b = _db_tok.pad({"input_ids": [ids[j] for j in ix]}, return_tensors="pt").to(DEV)
            h = _db_mdl(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[ix] = softmax(pooled.cpu().numpy() @ _db_W.T + _db_b)
        return out

    _db_w = DEBERTA_MEMBER["w"]
    if TINY:   # in-sample check: a broken forward (wrong pooling) collapses to ~1/18 = 0.06. The head was fitted on
        #          the external pool, so the bar is 0.25 rather than the ~0.9 it would reach on its own rows.
        _acc = float(np.mean(deberta_probs(TR[:40]).argmax(1) == y[:40]))
        print(f"  deberta_v3_large in-sample check on 40 training rows: acc {_acc:.2f}", flush=True)
        if _acc <= 0.25:   # do not kill the run over one member: drop it loudly and keep the submission valid
            print(f"  WARNING deberta_v3_large looks mis-wired (acc {_acc:.2f} <= 0.25): member dropped, "
                  f"weight {_db_w} -> 0", flush=True)
            _db_w = 0.0
    _P_db = deberta_probs(TE)
    if _db_w > 0:
        LOGIT += _db_w * np.log(np.clip(_P_db, 1e-9, 1))
    print(f"  + {'deberta_v3_large':20s} weight {_db_w:.3f} on {len(test)} rows "
          f"({time.time() - T0:.0f}s)", flush=True)
    del _db_mdl, _db_tok, _db_head   # _P_db stays: it is written to the member matrix
    gc.collect()
    torch.cuda.empty_cache()



# ---- 1C: the four family StyleGram gates the source computes and then never reads (deepthink.md Bug #5) ----
# They are fitted inside the source on official + extra + API rows; only the readout was missing. The source sorts
# its labels, so columns are mapped by name. Adding lam * log_softmax(gate) inside the family is the same as
# multiplying that family's likelihood by gate^lam: the sibling order changes, everything outside the family does not.
if PATCH_GATES:
    for _fam in ('claude', 'gpt', 'gemini', 'glm'):
        if _fam not in _src_gates:
            print(f'GATE {_fam} unavailable, skipped', flush=True)
            continue
        _g = np.asarray(_src_gates[_fam], dtype=np.float64)
        _ours = FAMILY_IDX[_fam]
        assert _g.shape == (len(TE), len(_ours)), f'gate {_fam} shape {_g.shape}'
        _pos = {l: k for k, l in enumerate([l for l in _src_labels if l.startswith(_fam)])}
        _g = _g[:, [_pos[LABELS[j]] for j in _ours]]
        _lam = 0.35 if _fam == 'claude' else 0.15
        LOGIT[:, _ours] += _lam * log_softmax(_g)
        print(f'GATE {_fam} lam {_lam} over {len(_ours)} columns (log-prob re-rank inside the family)', flush=True)
    if 'claude_lora' in _src_gates:
        print('GATE claude_lora computed but unused (patch 1C applies the four family gates only)', flush=True)
    del _src_labels, _src_gates, _g, _pos
else:
    print('GATE off (PATCH_GATES False): v10 behaviour, computed gates discarded', flush=True)

_new_prob = softmax(LOGIT.copy())
_mix_prob = 0.85 * _new_prob + 0.15 * _old_prob
assert np.isfinite(_mix_prob).all() and np.allclose(_mix_prob.sum(1), 1.0)
LOGIT = np.log(np.clip(_mix_prob, 1e-12, 1.0))
print('BLEND two-stage .85 + original four-Gaussian stack .15; labels aligned; single final Sinkhorn', flush=True)
_member_payload = {'ids': test['id'].astype(str).to_numpy(), 'labels': np.array(LABELS),
                   'base': _mix_prob.astype(np.float32), 'stack': _old_prob.astype(np.float32)}
if DEBERTA_MEMBER is not None and _db_w > 0:
    _member_payload['deberta_v3_large_prob'] = np.asarray(_P_db, dtype=np.float32)
np.savez_compressed('members_test.npz', **_member_payload)
print(f'member matrix saved: {sorted(_member_payload)}', flush=True)
del _old_raw, _old_prob, _new_prob, _mix_prob
import gc as _gc
_gc.collect()
torch.cuda.empty_cache()

# ---- ATLAS v2 at 10%, blended before the inherited final balancing (scored 0.607 recipe) ----
import pathlib
import sys as _sys
import hashlib as _hashlib
_atlas_roots = list(pathlib.Path(INPUT_DIR).rglob('atlas_bundle.json'))
assert len(_atlas_roots) == 1, 'Attach exactly one ATLAS bundle'
_atlas_root = _atlas_roots[0].parent
_atlas_meta = json.loads(_atlas_roots[0].read_text())
assert _atlas_meta['files']['adapters.safetensors'] == '29e9b32e09679ae4e3b2cdf0729c7011ecee730fdb0f20982593af0b40806106'
for _name, _expected in _atlas_meta['files'].items():
    with (_atlas_root / _name).open('rb') as _handle:
        assert _hashlib.file_digest(_handle, 'sha256').hexdigest() == _expected, _name
_sys.path.insert(0, str(_atlas_root))
from ntl_atlas.inference import predict as _atlas_predict
from ntl_rlcd.data import LABELS as _atlas_labels
assert len(set(LABELS)) == len(set(_atlas_labels)) == K
assert set(LABELS) == set(_atlas_labels)
_ap = _atlas_predict(test.text.tolist(),
    find_base(dict(model_type='qwen3', hidden_size=4096, num_hidden_layers=36)),
    _atlas_root, batch_size=1)
ATLAS_PROB = _ap[:, [_atlas_labels.index(label) for label in LABELS]]
assert ATLAS_PROB.shape == (len(test), K)
assert np.isfinite(ATLAS_PROB).all() and (ATLAS_PROB >= 0).all()
assert np.allclose(ATLAS_PROB.sum(1), 1, atol=1e-6)
print(f'ATLAS predictions done ({time.time() - T0:.0f}s)', flush=True)
_member_payload['atlas'] = ATLAS_PROB.astype(np.float32)
np.savez_compressed('members_test.npz', **_member_payload)
print(f'member matrix saved with atlas: {sorted(_member_payload)}', flush=True)

P_base = softmax(LOGIT)
if PATCH_ATLAS_GATE:
    # 1D: rows the base blend is unsure about get ATLAS, confident rows are barely touched.
    _part = np.partition(P_base, -2, axis=1)
    _margin = _part[:, -1] - _part[:, -2]
    _w_atlas = np.where(_margin < 0.25, 0.12, 0.02)[:, None]
    P = (1.0 - _w_atlas) * P_base + _w_atlas * ATLAS_PROB
    print(f'ATLAS gated: {int((_margin < 0.25).sum())} rows at 0.12, {int((_margin >= 0.25).sum())} at 0.02 '
          f'(median margin {float(np.median(_margin)):.3f})', flush=True)
else:
    P = 0.90 * P_base + 0.10 * ATLAS_PROB
    print('BLEND 0.90 * baseline + 0.10 * ATLAS (ungated, v10 behaviour)', flush=True)
P /= P.sum(1, keepdims=True)
assert np.isfinite(P).all() and np.allclose(P.sum(1), 1.0, atol=1e-6)
if len(test) >= 50:   # skip on the 3-row public test.csv
    target = np.full(K, len(P) / K)
    if PATCH_DAMPED_SINKHORN:
        # 1D: the hidden test is subsampled, so its class prior is not uniform. Five damped steps
        #     (exponent 0.50) nudge toward balance instead of forcing it the way 100 hard steps did.
        for _ in range(5):
            P *= (target / np.clip(P.sum(0), 1e-12, None)) ** 0.50
            P /= P.sum(1, keepdims=True)
        print('Sinkhorn damped: 5 steps, exponent 0.50', flush=True)
    else:
        for _ in range(100):
            P *= target / P.sum(0)
            P /= P.sum(1, keepdims=True)
        print('Sinkhorn hard: 100 steps (v10 behaviour)', flush=True)
sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]})
sub.to_parquet("submission.parquet", index=False)
print(sub["model"].value_counts().to_string())
print(f"done in {time.time() - T0:.0f}s")
