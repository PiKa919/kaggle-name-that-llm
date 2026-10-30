# G2 port specs — exact source for the five risky components

Status: extracted from public kernels, 2026-10-21. All code below is verbatim or minimally
trimmed (imports/comments kept where they carry constants). Kernel pulls live in `/tmp/g2/`
and are cited by kernel slug + cell; the local `/tmp` copies are disposable.

Evidence column: K = kirilltym 0.600 fork, P = prvsiyan "the prompt is not the author"
(v4 = 0.604, submission 57428928, scriptVersionId 360252286), Z = zbigniewkirchner 0.552,
N = nishantdahal two-stage 0.592, S = sangam21/kiza one-cell 0.604.

---

## 1. Shared-prompt Gaussian head

### 1.1 Provenance and role

- Core: P cell 18 (`# 4b. shared-prompt Gaussian group head`, "verbatim core of
  tools/group_gaussian.py, sha256 8a42427b77a37f56…"). Identical text is embedded as
  `_GAUSSIAN_SOURCE` in N (`/tmp/g2/nishantdahal…py` line 374) and K/S.
- Fit on **train** rows with true `prompt_id`; score **test** rows jointly inside inferred
  clusters. Output is a per-row log posterior (normalised over 18) that is added to the
  running log-prob blend:
  `LOGIT[rows] += w * LP`, `LP = scores(...)`.
- This is an **additive** head, not a replacement: plain member evidence stays on every row
  (`ADAPT_MODE='add'`). Groups are 2–18; singletons never enter.

### 1.2 Model / fitting math (copy-paste core)

```python
K = 18
DEFAULT = {"red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw",
           "lam_floor": 1e-4, "n_exact": 3, "inj": True, "iters": 5,
           "repel": True, "sp_scale": 1.0}

def _lw_shrinkage(X):
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(X, assume_centered=True))

def _shrunk(S, a):
    d = S.shape[0]
    return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)

def _two_way(Z, y, pinv, G, iters=4):
    P = np.zeros((G, Z.shape[1])); cnt_p = np.bincount(pinv, minlength=G).astype(float)
    cls = np.unique(y); mu = np.zeros((K, Z.shape[1]))
    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls: mu[c] = R[y == c].mean(0)
        R = Z - mu[y]; P = np.zeros_like(P); np.add.at(P, pinv, R); P /= cnt_p[:, None]
    return mu, P, cnt_p

def fit(X, y, prompts, cfg=None):
    c = dict(DEFAULT); c.update(cfg or {})
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y); N, D = X.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True); G = len(pu)
    mx = X.mean(0); sx = X.std(0); sx = np.where(sx > 1e-8, sx, 1.0); Xs = (X - mx) / sx
    if c["red"] == "none":
        V = np.eye(D)
    else:
        if c["red"] == "wpca":
            Pm = np.zeros((G, D)); np.add.at(Pm, pinv, Xs); Pm /= np.bincount(pinv)[:, None]
            R = Xs - Pm[pinv]
        elif c["red"] == "pca": R = Xs
        else: raise ValueError(c["red"])
        d = int(min(c["d"], N - G - 1, D))
        if N < D:
            w, U = np.linalg.eigh(R @ R.T); o = np.argsort(w)[::-1][:d]
            w, U = np.clip(w[o], 1e-12, None), U[:, o]
            V = (R.T @ U) / np.sqrt(w)
        else:
            _, _, Vt = np.linalg.svd(R, full_matrices=False); V = Vt[:d].T
    Z = Xs @ V; d = Z.shape[1]
    mu, P, ng = _two_way(Z, y, pinv, G)
    E = Z - mu[y] - P[pinv]; dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = E.T @ E / dof
    ae = _lw_shrinkage(E) if c["se_shrink"] == "lw" else float(c["se_shrink"])
    Se = _shrunk(Se_raw, ae)
    Pc = P - P.mean(0)
    Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = _lw_shrinkage(Pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = (1 - ap) * Sp_raw + ap * (tr / d) * np.eye(d); Sp *= c["sp_scale"]
    L = np.linalg.cholesky(Se); Li = np.linalg.inv(L); M = Li @ Sp @ Li.T
    M = 0.5 * (M + M.T)
    lam, Q = np.linalg.eigh(M); lam = np.clip(lam, c["lam_floor"], None); A = Q.T @ Li
    W = V @ A.T
    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 1.0, "Tg": 1.0,
            "info": {"d": int(d), "a_e": round(ae, 4), "a_p": round(ap, 4),
                     "lam_pct": np.round(np.percentile(lam, [5, 50, 95, 100]), 3).tolist()},
            "cfg": c}

def transform(X, prm):
    return ((np.asarray(X, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]
```

Inference core (`_MASK`, `singleton_ll`, `joint_terms`, `joint_table`, `exact_marginals`,
`em_marginals`, `lam_g`, `group_logpost`, `scores`) is P cell 18 lines 1355–1460. The
non-obvious pieces:

```python
def singleton_ll(U, m, lam):                     # (n,C), prompt integrated out
    R = U[:, None, :] - m[None]
    return -0.5 * (R ** 2 / (1 + lam)).sum(-1)

def joint_terms(U, m, lam):
    n = len(U); gam = lam / (1 + n * lam); R = U[:, None, :] - m[None]
    a = -0.5 * (R ** 2 * (1 - gam)).sum(-1)
    G = R * np.sqrt(gam); B = np.einsum("icd,jed->ijce", G, G)
    return a, B

def em_marginals(U, m, lam, T=1.0, iters=5, repel=False, q0=None):
    n = len(U); v = lam / (1 + (n - 1) * lam)
    L = singleton_ll(U, m, lam) / T if q0 is None else np.log(np.clip(q0, 1e-300, None))
    q = np.exp(log_softmax(L))
    for _ in range(iters):
        s = U - q @ m; S = s.sum(0); Pm = (S[None] - s) * v
        R = U[:, None, :] - Pm[:, None, :] - m[None]
        L = -0.5 * (R ** 2 / (1 + v)).sum(-1) / T
        if repel:
            lo = np.log(np.clip(1 - q, 1e-6, None)); L = L + (lo.sum(0)[None] - lo)
        q = np.exp(log_softmax(L))
    return log_softmax(L)

def group_logpost(U, prm, T1=None, Tg=None, cfg=None):
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {}); n = len(U)
    lg = lam_g(prm, cfg)                          # LAM_DIFF = 1e6 only for the 'diff' ablation
    T1 = prm["T1"] if T1 is None else T1; Tg = prm["Tg"] if Tg is None else Tg
    if n == 1:
        return log_softmax(singleton_ll(U, prm["m"], prm["lam"]) / T1)
    if n <= c["n_exact"]:                         # n_exact=3, exact tuple table
        a, B = joint_terms(U, prm["m"], lg)
        return exact_marginals(joint_table(a, B), Tg, c["inj"])
    return em_marginals(U, prm["m"], lg, Tg, c["iters"], c["repel"])

def scores(X_pool, lab, prm, T1=None, Tg=None, cfg=None, U=None):
    U = transform(X_pool, prm) if U is None else U
    lab = np.asarray(lab); n = len(U)
    out = np.empty((n, prm["m"].shape[0]))
    sizes = np.bincount(lab); s1 = sizes[lab] == 1
    T1 = prm["T1"] if T1 is None else T1
    if s1.any():
        out[s1] = log_softmax(singleton_ll(U[s1], prm["m"], prm["lam"]) / T1)
    for g in np.where(sizes >= 2)[0]:
        ii = np.where(lab == g)[0]; out[ii] = group_logpost(U[ii], prm, T1, Tg, cfg)
    return out
```

`exact_marginals` enforces the at-most-once mask (`_inj_mask`) for groups 2–3 (all 18^n
tuples), larger groups use cavity EM with `repel`. `inj=True` is part of the recipe.

### 1.3 Grouping (must not be shared blindly between branches)

```python
def average_linkage_groups(D, thresh):
    if len(D) < 2: return np.ones(len(D), dtype=int)
    D = np.asarray(D, dtype=np.float64).copy(); np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method='average'),
                    thresh, criterion='distance')

def content_cosine_distance(texts, max_features=None):
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True,
                        max_features=max_features, dtype=np.float32).fit_transform(texts)
    return np.clip(1.0 - (X @ X.T).toarray(), 0.0, 1.0)
```

- **Branch A** (two-stage LoRAs + their Gaussians): `G_A = average_linkage_groups(
  content_cosine_distance(TE), 0.85)`; `ROWS_A = eligible(G_A)` (size 2..18). P line 1222,
  `GROUP_A_THRESH = 0.85` line 250.
- **Branch B / built-in views**: mixed content + GLM-4-9B layer-24 distance, cut **0.60**:
  `D = 0.5*(content_cosine_distance(TE) + clip(1 - Z@Z.T, 0, 2))` where
  `Z = (E - E.mean(0)) / ||E - E.mean(0)||`; average linkage. `GROUP_LAYER=24`,
  `GROUP_MIN=2`, `GROUP_MAX=18`. P cell 11, Z lines 245–257.
- **Stylegram** uses its own median-normalised content distance cut **0.90**
  (`D / median(off-diagonal)`, `max_features=100_000`).
- Cluster ids are arbitrary; always re-index with
  `np.unique(groups[rows], return_inverse=True)[1]` before `scores`, so `np.bincount(lab)`
  has no empty leading slots. This is done in both P and N.

### 1.4 Per-view settings and weights (exact)

| head | features | d | T1 | Tg | weight |
|---|---|---|---|---|---|
| `gauss_q3_8b_L22` | Qwen3-8B L22 mean | 384 | 2.351 | 2.306 | `TWIN_W['q3_8b']=1.5*0.219=0.3285` |
| `gauss_glm9b_L8` | GLM-4-9B L8 mean | 384 | 2.722 | 2.596 | `1.5*0.110=0.165` |
| `gauss_q3_4b_L4` | Qwen3-4B L4 mean | 384 | 2.725 | 2.578 | `1.5*0.083=0.1245` |
| `gauss_debv3l_L24` | DeBERTa-v3-L L24 mean | 384 | 3.112 | 3.069 | `0.25` |
| `gauss_ts_lora8b` | public two-stage 8B pooled 8-layer feat | 128 | 3.0 | 3.0 | 0.10 |
| `gauss_ts_lora14b` | public two-stage 14B pooled feat | 128 | 3.0 | 3.0 | 0.10 |
| `gauss_twostage_s0` | own stage-2 seed 0 pooled feat | 128 | 3.0 | 3.0 | 0.05 |
| `gauss_twostage_s1` | own stage-2 seed 1 pooled feat | 128 | 3.0 | 3.0 | 0.05 |

`GAUSS_SETTING = dict(red='wpca', se_shrink='lw', sp_shrink='lw', lam_floor=1e-4,
n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)`.
The four built-in Gaussians are on the **branch-B** stack (P `GAUSS_HEADS`, lines 252–256);
the four LoRA Gaussians are on **branch A** (`GAUSS_LORA_W`, line 245; d=128 T=3 is
kirilltym's 0.600 setting for the 8B, P line 244). In P v6 only branch A is enabled, so
only the LoRA Gaussians run. Do not put the four built-in Gaussians on branch A; their
weights come from the branch-B fit and would double-count that stack.

Call site (public two-stage, P lines 2016–2035):

```python
Z = feats(TE)
add_member(key, full_prob(softmax_rows(Z @ Wp.T + bp), cols), weight=w, target=LOGIT_A)
if ROWS_A.any():
    Zc = group_centre_dense(Z[ROWS_A], G_A[ROWS_A])
    add_member(key + '_twin', full_prob(softmax_rows(Zc @ Wc.T + bc), cols),
               rows=ROWS_A, weight=twin_w, target=LOGIT_A)
if gauss_w > 0:
    Ztr = feats(TR)
    prm = fit(Ztr, y, PROMPT, dict(GAUSS_SETTING, d=GAUSS_LORA['d']))
    prm['T1'] = prm['Tg'] = GAUSS_LORA['T']
    if ROWS_A.any():
        LP = scores(Z[ROWS_A], np.unique(G_A[ROWS_A], return_inverse=True)[1], prm)
        add_member('gauss_' + key, np.exp(LP), rows=ROWS_A, weight=gauss_w, target=LOGIT_A)
```

`add_member` logs its argument, so pass `np.exp(LP)` (probabilities), not `LP`.

### 1.5 Runtime / correctness notes

- Fit is `O(N·D²)` in the worst branch but N=2699; `d` is clipped to
  `min(d, N-G-1, D)`. For the 8B pooled features D=8*4096=32768, so use the `N < D`
  Gram path (already in `fit`).
- `feats(TR)` is recomputed per adapter in P; for v10, compute once per trunk and reuse
  (same adapter, same features). P recomputes because it streams adapters.
- `np.unique(lab, return_inverse=True)` in the call site is mandatory.
- The three-row public test exercises `ROWS_A.any()==False`; keep the `TINY` self-test on
  train prompts (P `self_test_rows`) so the hidden 4000-row run does not hit untested code.

---

## 2. Within-family gates

### 2.1 What actually shipped

- **0.604 (P v4) and 0.600 (K)**: gates **OFF**. P line 237:
  `GATES_ON = False  # the within-family gates of version 1 are off: the two-stage stack
  reports no gain from them and the 0.600 fork dropped them`. P line 353 asserts
  `not GATES_ON`. K/S contain the gate training code inside the embedded source but never
  apply `GATES` (only the Claude gate of the older lineage is referenced in Z).
- **0.552 (Z)**: Claude-only gate, two re-rankers, applied after Sinkhorn. This is the
  only gate with measured LB gain (+0.003 with both gates; CV Claude-recall 0.393→0.438).
- Family gates for gpt/gemini/glm exist in the 0.563 lineage (P/Z) with
  `GATE_LAM['family']=0.25`, but were dropped. Do not enable them by default.

### 2.2 Gate training (Z lines 298–362)

Style views (`SPEC`) used by every gate:

```python
SPEC = [('char', 3, 5, False, 250000, 1.0), ('word', 1, 3, False, 180000, 0.7),
        ('shape', 2, 6, False, 60000, 0.5), ('function', 1, 3, False, 50000, 0.5)]

def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)

def function_text(text):   # same FUNCTION_WORDS set as P; stop word -> lower, content -> CONTENT
    ...
```

Vectorizer options per view: `analyzer=actual, ngram_range=(lo,hi), lowercase=False,
sublinear_tf=True, min_df=2, max_features=maxf, dtype=float64`; for word views
`token_pattern=r"(?u)\b\w+\b|[^\w\s]"`; multiply each block by its `w`, `hstack`.

Claude style gate (structure from Z 334–362; identifier names are reconstructed, the
constants/SPEC/C/weights are the contract):

```python
CLAUDE_GATE_LAM = 1.5
CLAUDE_LORA_LAM = 1.0
CLAUDE_IDX = np.array([i for i, l in enumerate(LABELS) if l.startswith('claude')])
_api_hits = glob.glob('/kaggle/input/**/ntl_extra_passages_claude.csv', recursive=True)
_api = pd.read_csv(_api_hits[0])
_m = np.isin(y, CLAUDE_IDX)
_ft = [t for t, k in zip(TR, _m) if k] + _api['text'].astype(str).tolist()
_fy = np.concatenate([y[_m], _api['model'].map({l: i for i, l in enumerate(LABELS)}).values])
A, B = [], []
for analyzer, lo, hi, lower, maxf, w in SPEC:
    tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
    actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
    opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True,
                min_df=2, max_features=maxf, dtype=np.float64)
    if actual == 'word':
        opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
    v = TfidfVectorizer(**opts)
    A.append(v.fit_transform([tf(t) for t in _ft]) * w)
    B.append(v.transform([tf(t) for t in TE]) * w)
_svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A, format='csr'), _fy)
assert list(_svc.classes_) == list(CLAUDE_IDX)
DC_TE = _svc.decision_function(hstack(B, format='csr'))
```

External data: Kaggle dataset `seyitkaangunes/ntl-extra-api-passages`, file
`ntl_extra_passages_claude.csv` (Claude rows only); if absent, the gate must disable with a
warning (Z line 365), not silently train on 5-class official rows only. `shape`/`function`
views are the same four StyleGrams views used by the embedded source (there they are named
`shape_view`/`function_view`); reuse one implementation, do not duplicate the word regex.

Claude LoRA head: checkpoint `ntl-qwen3-8b-lora-claude-head`, `readouts.npz` has
`claude_coef` / `claude_intercept` (5 columns). It is a log-softmax, Z line 660–667:

```python
lc5 = Z @ ro['claude_coef'].T + ro['claude_intercept']
lc5 -= lc5.max(1, keepdims=True)
LC_TE = lc5 - np.log(np.exp(lc5).sum(1, keepdims=True))
```

### 2.3 Application (Z 828–836) — after Sinkhorn, before submission

```python
pred = P.argmax(1)
if DC_TE is not None:
    _isc = np.isin(pred, CLAUDE_IDX)
    _s = np.log(np.clip(P[:, CLAUDE_IDX], 1e-12, 1)) + CLAUDE_GATE_LAM * DC_TE
    if LC_TE is not None:
        _s = _s + CLAUDE_LORA_LAM * LC_TE
    _new = CLAUDE_IDX[_s.argmax(1)]
    pred[_isc] = _new[_isc]          # non-Claude rows untouched
```

Family variant (P 2329–2338, off by default):

```python
logP = np.log(np.clip(P, 1e-12, 1.0))
for fam in ('claude', 'gpt', 'gemini', 'glm'):
    idx = FAMILY_IDX[fam]
    score = logP[:, idx] + (1.5 if fam == 'claude' else 0.25) * GATES[fam]
    if fam == 'claude' and 'claude_lora' in GATES:
        score = score + 1.0 * GATES['claude_lora']
    inside = np.isin(pred, idx); new = idx[score.argmax(1)]
    pred[inside] = new[inside]
```

Gate SVCs (P 1380–1403): `family_gate` fits the same 4-view StyleGrams + `LinearSVC`
(`C=0.5` Claude; `C=1.0` gpt/gemini; `C=0.5` glm), gpt/gemini/glm use
`class_weight='balanced'` and extra-row sample weight **0.25**, official rows weight 1.0. For the Claude gate, S's `family_gate` call passes neither
`class_weight` nor sample weights; treat that as the reference, not a general rule.

**Port decision for v10**: ship the code path but keep `GATES_ON=False` until a grouped CV
shows a gain. The 0.600/0.604 lineage deliberately dropped gates.

---

## 3. Spaced em-dash guard (post-Sinkhorn)

### 3.1 Invariant and placement

S lines 403–415. (Not present in K's 0.600 fork and **not** in P v4's 0.604 submission;
it ships only in the sangam21/kiza 0.604 one-cell variant.) In official `train.csv`,
**86 rows** contain `' \u2014 '`; 100% belong to exactly four classes and **0** rows belong
to the other 14:

`claude-opus-4-6`, `claude-opus-5`, `glm-5.3`, `glm-5.3-flash`.

The guard runs **after** Sinkhorn (so the balanced posterior is what gets corrected) and
**before** Hungarian, in S only. It only zeroes forbidden classes on rows that already contain the
spaced em dash and whose argmax is outside the allowed set; it does not force a specific
allowed class.

```python
_SPACED_ALLOWED = {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}
_SPACED_FORBIDDEN = np.array([l not in _SPACED_ALLOWED for l in LABELS])
_typo_fixed = 0
for _i, _t in enumerate(test.text):
    if ' \u2014 ' in _t or ' — ' in _t:
        if LABELS[P[_i].argmax()] not in _SPACED_ALLOWED:
            P[_i, _SPACED_FORBIDDEN] = 0.0
            P[_i] /= P[_i].sum()
            _typo_fixed += 1
if _typo_fixed > 0:
    print(f'Spaced em-dash invariant corrected {_typo_fixed} impossible predictions',
          flush=True)
```

Notes:
- `' \u2014 '` and `' — '` are the same literal; the second is there for readability. Keep
  both checks (they are intentional in the source, not a bug).
- Test-side texts are `fillna('').astype(str)` in P/N; do that before the guard.
- The guard is **not** a general punctuation rule. Do not extend it to unspaced `—`,
  curly quotes, or other classes; 0.600 lineage only ships this one invariant.
- Guard changes are counted; log the count for the hidden run.

---

## 4. Balancing variants (and the mix that precedes them)

### 4.1 v4 / 0.600 exact order

P/K: branch log-prob sums → per-branch softmax → **85/15 probability mix** → **one full
Sinkhorn** (100 iters, uniform target) → argmax. S (sangam21/kiza 0.604) inserts the
em-dash guard, then Hungarian, before argmax; P v4 and K 0.600 do not.

K exact (`/tmp/g2/kirilltym/0-600.py` lines 379–397):

```python
_old_raw = _old_namespace['LOGIT'][:, _old_cols]
_old_prob = softmax(_old_raw)
_mix_prob = 0.85 * _new_prob + 0.15 * _old_prob
assert np.isfinite(_mix_prob).all() and np.allclose(_mix_prob.sum(1), 1.0)
LOGIT = np.log(np.clip(_mix_prob, 1e-12, 1.0))
P = softmax(LOGIT)
if len(test) >= 50:
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
```

(`_new_prob` = two-stage branch; `_old_prob` = four-Gaussian 0.563 stack, label-aligned
first. P v4 expresses this as
`P = MIX_A*softmax(LOGIT_A) + (1-MIX_A)*softmax(LOGIT_B)`, `MIX_A=0.85`; current P sets
`MIX_A=1.0` because branch B is disabled.)

### 4.2 Soft class-mass variant (Z, not the 0.600 path)

```python
BALANCE_ALPHA = 0.5
target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
for _ in range(50):
    P *= (target / (P.sum(0) + 1e-12))[None, :]
    P /= P.sum(1, keepdims=True) + 1e-12
```

Z reports: never negative in-domain (+0.002..+0.007), and under genre shift a collapsed
posterior went 0.12→0.19. Caution is explicit in both sources: balanced accuracy weights
recall equally but the hidden **class mixture is unequal and changes** between public and
private; over-balancing can hurt. K/P v4 use the hard-uniform 100-iter variant and scored
0.604; do not swap in `BALANCE_ALPHA=0.5` while claiming v4 parity. (P defines
`BALANCE_ALPHA=0.5` but never uses it.)

Skip Sinkhorn entirely when `len(test) < 50` (interactive 3-row file), in both variants.

### 4.3 ATLAS placement (v5 experiment, not in 0.604)

P 2266–2296: `P = softmax(LOGIT_A)`; then
`P = 0.90*P + 0.10*P_ATLAS` (ATLAS labels remapped first); then Sinkhorn; then (disabled)
gates. Blend is before balancing, verify rows sum to 1 after the mix.

### 4.4 Hungarian (0.604 kernel; high risk, off by default for v10)

S lines 418–434 (present only in sangam21/kiza; neither P v4 nor K 0.600 contains this
block):

```python
try:
    from scipy.optimize import linear_sum_assignment
    _hungarian_changes = 0
    for _gid in np.unique(groups):
        _members = np.where(groups == _gid)[0]
        if 12 <= len(_members) <= 18:          # dense prompt clusters only
            _cost = -np.log(np.clip(P[_members, :], 1e-12, 1.0))
            _r_idx, _c_idx = linear_sum_assignment(_cost)
            for _r, _c in zip(_members, _c_idx):
                if P[_r].argmax() != _c:
                    _hungarian_changes += 1
            for _r, _c in zip(_members, _c_idx):
                P[_r, :] = 0.0
                P[_r, _c] = 1.0
    print(f'Hungarian matching resolved {_hungarian_changes} intra-prompt duplicate collisions',
          flush=True)
except Exception as _e:
    print(f'Hungarian matching skipped: {_e}', flush=True)
```

Here `groups` is the 0.85 content-cluster assignment defined at S lines 173–180. The
assignment is hard one-hot over 18 classes and is applied **after** the em-dash guard.
Our own earlier LB test of Hungarian regressed 0.353→0.344; K's 0.604 is not an ablation
of Hungarian (it is the whole one-cell recipe). Keep it behind a flag, default off, until
grouped-CV evidence on the v10 stack exists.

---

## 5. LoRA member loading compatibility

Three incompatible on-disk formats circulate. Never point a loader at the wrong one.

### 5.1 Format A — nishantdahal public two-stage (`ntl-two-stage-lora`)

Used by N and P `public_two_stage`.

```
<ckpt>/meta.json
<ckpt>/adapter/adapter_config.json
<ckpt>/adapter/adapter_model.safetensors     # peft names: ...lora_A.weight / ...lora_B.weight
<ckpt>/heads.safetensors                     # plain.weight, plain.bias,
                                             # centered.weight, centered.bias, norms.*
```

Contract from P `check_public_meta` (lines 840–850):
- `meta['labels']` must be 18 unique labels, set-equal to the 18 authors; **order is the
  checkpoint's head order** (N's order is opus-4-6, opus-4-8, opus-5, sonnet-5, fable-5 —
  not sorted).
- `meta['cfg']['keep']` int in `1..base.num_hidden_layers`; `cfg['layers']` exactly 8
  unique ints in `1..keep`; `cfg['max_len'] == 447`.
- Public: `lora8b_v1` base qwen3/H4096/36 layers, keep 24; `lora14b_v1` qwen3/H5120/40,
  keep 27 (N markdown; P passes base spec but reads `cfg['keep']`, so trust meta).
- Trunk: final `model.norm = nn.Identity()`; fp16; split over T4s if the fp16 estimate
  exceeds 13 GiB; merge adapter into weights with `W += (B@A) * lora_alpha/r`.
- Features: tokenize with `add_special_tokens=False`, truncate to `cfg['max_len']`, append
  `eos_token_id`, sort by length, right pad, masked mean over tokens **with position 0
  excluded**, each pooled layer through its own `LayerNorm`, concat 8×hidden.
- Readout: `softmax(Z @ plain.weight.T + plain.bias)`; column order is `meta['labels']`,
  remap with `cols = [LAB2I[l] for l in meta['labels']]`.
- Blend weights (N, prompt-grouped OOF): 8B `w=0.226 twin_w=0.079 bs=16`; 14B
  `w=0.429 twin_w=0.150 bs=8`. N's separate pool SVC is `W_SVC=1.021`; it is not part of
  P's branch-A blend. P v4 adds its own two seeds at `w=0.18 twin_w=0.06` each.

### 5.2 Format B — lightningv08 / lucasboesen LoRA24 (“COMPLETE.json”)

Used by P `score_lora_checkpoints` (P 1840–1925); derived from
`lightningv08/ntl-lora24-fivefold-training` and `name-that-llm-train-lora24-s7`.

```
<ckpt>/COMPLETE.json                          # marker only
<ckpt>/tokenizer/                             # tokenizer saved with the checkpoint
<ckpt>/adapter/                               # peft adapter, loaded via PeftModel
<ckpt>/heads.safetensors  or  norms.safetensors
<ckpt>/readouts.npz
<ckpt>/anchor_features.npz  or  valid_features.npz
```

- `readouts.npz`: `classes` (must equal the code's `LABELS` exactly), `pool_layers` (must
  equal the trunk's list), `plain_coef`/`plain_intercept`, `center_coef`/`center_intercept`,
  optional `claude_coef`/`claude_intercept`.
- Norms: `LayerNorm(H)` per pooled layer; weights under `norms.*` in the safetensors file.
- Pooled layers differ by trunk: Qwen3-8B `[2,4,6,8,12,16,20,24]` keep 24; Qwen2.5-7B
  `[2,4,6,8,10,12,16,20]` keep 20. `max_len=447`, bs=8, first token masked, fp16 with
  bf16 retry on non-finite batches.
- Anchor check: the checkpoint ships its own features for train rows 0..2; require
  `mean(|ours-saved|)/(mean(|saved|)+1e-9) < 0.02`, else abort. This catches wrong
  tokenizer/pool/truncation.
- P uses `peft.PeftModel`, plus a monkey-patch to hide `torchao` from peft on Kaggle.
  If v10 pre-merges adapters, reproduce the same forward (do not use the template's
  `merge_lora` on this format without checking names; A uses a different key layout for
  the spine).
- Weights (P): `lora24_public 0.177`, `lora24_seed7 0.113`, `lora24_claude 0.177`,
  `lora20_q25_instyle 0.17`, `lora24_instyle 0.17`; twins
  `LORA_TWIN_W={'lora24_public':0.15,'lora24_claude':0.15,'lora24_instyle':0.15}`.

### 5.3 Format C — prvsiyan own two-stage adapters (`full_s0`, `full_s1`)

Used by P `two_stage_lora` (P 2060–2105). Own LoRA implementation, **not peft**, vendored
trainer module (`verified_own_trainer`):

```python
lora_cls = verified_own_trainer(PUBLIC_A_INPUTS['own_root'])
model, tok, meta = lora_cls.load_for_inference(str(...), base_dir=..., device=DEV)
assert meta['labels'] == LABELS and meta.get('init_adapter'), 'not a stage-2 checkpoint'
hp = meta['hp']
seqs = lora_cls.encode(tok, texts, hp['max_len'], hp['bos'], hp.get('bos_id'))
out  = lora_cls.predict_all(model, seqs, list(range(len(seqs))), tok.pad_token_id, DEV, bs=bs)
# out['z'] pooled features, out['lp'] logits, out['bad'] non-finite rows
Zc = group_centre_dense(Z[ROWS_A], G_A[ROWS_A])
lc = model.logits_ctr(torch.tensor(Zc, device=DEV)).float().cpu().numpy()
```

`meta` fields: `labels` (must equal the code's sorted 18), `init_adapter`, `steps`, `hp`
with `max_len`, `bos`, optional `bos_id`. Dataset:
`prvsiyan/ntl-two-stage-lora-adapters-v1`, dirs `full_s0` / `full_s1`; base Qwen3-8B
H4096. No `readouts.npz`; heads live in the model state (`logits` / `logits_ctr`).

### 5.4 Label-order trap (all formats)

Two orders exist in the wild:

- **sorted** (P/K/S, and `readouts.npz['classes']`): `claude-fable-5, claude-opus-4-6,
  claude-opus-4-8, claude-opus-5, claude-sonnet-5`.
- **nishant** (N `LABELS`, and `meta.json['labels']` of Format A): `claude-opus-4-6,
  claude-opus-4-8, claude-opus-5, claude-sonnet-5, claude-fable-5`.

Always remap probabilities/columns through the checkpoint's own label list into the
competition output order before blending. K/S do this with
`_old_cols=[LABELS.index(label) for label in _new_labels]`; P Format A does it with
`cols=[LAB2I[l] for l in meta['labels']]`. Never assume `sorted()`.

### 5.5 Trunk sharing / unload

P reuses one truncated trunk per `(model_type, hidden, keep)` and swaps adapters; after the
last adapter it calls `trunk = lm.unload()`. If v10 merges adapters instead, keep one trunk
per base and rebuild weights per checkpoint (or reload) — do not merge two adapters into
the same trunk cumulatively. P also hides `torchao` from peft (`iu.find_spec` patch and
`pi.is_torchao_available = lambda *a, **k: False`); keep that on Kaggle.

---

## 6. v10 assembly checklist (what to port, in order)

1. Build `G_A` from test content TF-IDF, average linkage, cut 0.85; `ROWS_A = size 2..18`.
2. Per branch-A member: plain log-prob; centred-twin log-prob on `ROWS_A` (weight
   per member); Gaussian fit on train features of the same view, `d=128`, `T1=Tg=3.0`,
   score `ROWS_A` jointly and add at 0.10 (public 8B/14B) or 0.05 (own seeds).
3. Branch-B stack only if `ENABLE_BRANCH_B`: own `G_TE` mixed content+GLM cut 0.60, own
   Gaussians at the table weights, own stylegram groups cut 0.90.
4. Mix: `P = 0.85*softmax(LOGIT_A) + 0.15*softmax(LOGIT_B)` (v4) or branch-A only (v6).
5. Optional ATLAS: `P = 0.90*P + 0.10*P_ATLAS` (labels remapped).
6. Sinkhorn if `len(test)>=50`: 100 iters, uniform target (not `BALANCE_ALPHA`).
7. Spaced em-dash guard (S fork only; not part of P v4 / K 0.600 parity).
8. Hungarian (S fork only) if explicitly enabled; dense groups 12..18; log changed count.
9. Gates only if `GATES_ON`; Claude gate last, non-Claude rows untouched.
10. Argmax in `LABELS` order; assert ids unique, 18 valid labels, no nulls, exact row count.
