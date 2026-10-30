I'll start by examining the actual state of the repository before judging anything.
Thinking: Let me read the key files. The user says "3 newly authored modules" — sibling_features.py (80KB!), test_time_calibration.py (42KB), short_passage_engine.py (56KB). Also mentions HodgeRank in the question — let me check whether HodgeRank exists anywhere. And "8 pairwise classifiers", "Balanced Accuracy from 0.54".

Let me check the master_sota/submission_sota.py and existing postproc/fusion.

Key adversarial angles:
- Does HodgeRank exist? grep.
- Is "0.54" real? Check logs / actual CV.
- Does Sinkhorn-Knopp already exist in test_time_calibration.py?
- Does sibling_features contain pair routing for 8 pairs? Check.

Let me do parallel greps.
Thinking: There's an existing docs file. Let me read the key artifacts: the master submission, docs/INVALIDATION_v2.md, exp 20/21 invalidation evidence, and the actual measured CV/LB numbers. The user claims 0.54 balanced accuracy — need to verify against logs.

Let me read in parallel.
Thinking: There's already an INVALIDATION_v2.md which is very thorough. Note also there's a file `experiments/deepseek_v4_integration_review.md` that appears to contain... my own thinking? That's odd — it looks like a leak of previous agent output. Let me check it, it may be a prior run of this exact task. I should be careful not to just copy it. Actually it says "Thinking: Let me read the key files..." — that's a transcript artifact from a previous session saved into the repo. Interesting but I'll note it.

Now I need to verify the three new modules actually exist and do what's claimed. Let me read them. They're large (2168, 1028, 1378 lines). Let me read the docstrings/APIs and check whether they've been tested.

Key checks:
1. sibling_features.py — 101 features with pair routing? Does it run? Any tests?
2. test_time_calibration.py — Sinkhorn. Does alpha/reg default exist, and has it been validated? Is there a `__main__` self-check?
3. short_passage_engine.py — truncation invariant.

And critically: are they integrated/imported anywhere? Is there any measured evidence they help? The user asks "how to push 0.54 → 0.70+". The honest answer per prior invalidation: hygiene fixes worth +0.02–0.05, and 0.70 requires representation change. I should push back on 0.70 claim but still give a maximal-value integration.

Let me read the module headers and public APIs.
Thinking: Now let me actually run the self-tests in these modules and check whether they work. Also check whether they're imported anywhere (integration status).

Also important: the user asks "8 pairwise classifiers" and sibling_features with pair routing. And asks for optimal alpha/reg/threshold. I need measured values, not guesses. Since I can run CPU TF-IDF locally (allowed per AGENTS.md: local is for code authoring/unit testing, no heavy model training), I can run a quick grouped-CV experiment to find optimal alpha/reg/threshold empirically. That would be the highest-value part of the answer — actual measured numbers.

Let me check: data/train.csv exists? data dir listing.

Let's plan:
1. Check integration status (grep imports).
2. Run module self-tests to verify they pass.
3. Run an experiment: build OOF n-gram probabilities with StratifiedGroupKFold on prompt_id, then measure:
   - Sinkhorn alpha sweep × reg sweep on bacc
   - placement vs HodgeRank (does order matter numerically?)
   - short gate threshold sweep with the actual short model
   - sibling_features added to pairwise classifiers: does it help vs TF-IDF-only?

That's a real experiment. Time cost: TF-IDF + LR 5-fold locally ~105s per earlier doc. Feasible.

Let me first check data availability and imports.
Thinking: Confirmed: **none of the 3 modules are imported by any code path**. They are dead code — no test file in tests/ either. The only references are themselves and the subagent orchestration script.

Also note experiments/21 run? Let me check whether subagents produced any validation. And check the docs dir for a review already done.

Let me now run the module self-tests to verify they at least work, and time the extractor (for OOM/latency claim on 2699+4000 rows).

Then run the real experiment.
Thinking: Both modules pass their self-tests. Good. Note sibling_features is 2.35ms/doc → 2699+4000 = 6699 docs → ~16s single-threaded. Fine, not a latency problem. Memory 3.2MB for 4k. No OOM.

Now the key: short_passage_engine self-test? Let me check if it has a self-test function. And then run the real OOF experiment.

Plan for experiment (CPU only, allowed):
1. Load train (2699). 5-fold StratifiedGroupKFold on prompt_id.
2. Views:
   a. n-gram 4-view (as in notebook) → OOF log-probs.
   b. sibling_features (101) → per-fold StandardScaler+LR → OOF log-probs (base 18-way), and also as pairwise feature augmentation.
   c. short passage engine features → LR → OOF log-probs.
3. Measure bacc of each view alone, fusion, then optimize alpha/reg for Sinkhorn on OOF, and short gate threshold.

Important subtlety for the Sinkhorn question: the hidden test set groups by prompt, and passages within a prompt share a topic. Sinkhorn on OOF must be applied per prompt cluster (only within a prompt group). In OOF, a fold contains complete prompt groups (since grouped folds), so we can apply Sinkhorn per prompt cluster within the OOF matrix. That's the right simulation. Good — prompt-grouped folds mean each prompt appears entirely in one fold, so per-cluster OT on the fold's OOF rows is a faithful emulation of test-time per-prompt OT.

Also: "before or after HodgeRank?" — HodgeRank here is a per-row potential projection of pairwise corrections. Since it's a row-wise affine operation on logits and Sinkhorn is a column coupling across rows, they don't commute in general. I can measure both orders empirically. That's the definitive answer.

Let me write the experiment script. Keep it efficient: TF-IDF 4-view with 200k+ features on 2699 docs and 5-fold LR fits = ~105s per the doc. Plus sibling features 16s. Total maybe 3-5 min. Acceptable.

Let me be careful with time. Let me first build and cache the OOF matrices to a npz so I can iterate on calibration hyperparameters without refitting.

Let me write experiments/23_integration_gate.py.

Design details:
- views: reuse the exact 4-view TF-IDF config from submission_sota.py.
- OOF: for each fold fit LR C=5.0 class_weight balanced on train part, predict_proba on val part.
- sibling features: extract once for all train (unsupervised normalization — it's robust z-score using batch stats, slight transductive leak but feature-wise only, no labels; acceptable, and it's what would happen at test time anyway since we extract on train+test together or train-only. To be strict, extract on train only for fitting, then transform test with train stats. Let me check the API for fit/transform... `extract_sibling_features(texts, method="robust")` normalizes using the batch itself. For rigor I'll fit normalization on train fold only. Simpler: extract on train once, and note the caveat. Actually to be honest and quick, I'll extract on all train rows at once (no labels involved = transductive but label-free). CV numbers remain essentially unbiased w.r.t. class labels. I'll note it.

Actually better: for the pairwise classifier test, what matters is fold-wise fitting of the classifier. Feature normalization on full train unlabeled is fine (standard practice, and test-time we'd normalize using train+test). I'll note it.

- short engine: `ShortPassageFallbackModel.fit(texts, labels, word_counts)` then predict_proba. Need to check fit signature and whether it internally truncates. Let me check.

Let me grep the API signatures.
Thinking: Good. Now let me write the experiment. I'll build and cache OOF matrices, then sweep.

Important design decision for the honest measurement: the real pipeline's `p_semantic` is the LoRA view (unavailable locally). I'll use the n-gram view as the semantic surrogate, as INVALIDATION_v2 did, and state that clearly. The calibration hyperparameters (alpha, reg) are properties of the posterior *shape*; they should ideally be refit on the real fused OOF, but I can give a defensible range and the rule for choosing on the notebook.

Let's write the script.

Steps:
1. Load data, encode labels.
2. 5-fold StratifiedGroupKFold by prompt_id.
3. Build 4-view TF-IDF (as in notebook) → 5-fold OOF probs (p_ng_oof), and full-train-fit predictions for test later (not needed).
4. sibling features → 5-fold OOF probs base 18-way, plus per-pair augmented classifiers.
5. short engine → 5-fold OOF? That's expensive: ShortPassageFallbackModel fits word+char TFIDF on augmented rows (~2699*? rows). Let's test timing per fold. Might be ~60s/fold. 5 folds = 5 min. Acceptable maybe. But let me first check the gate threshold using simpler surrogate: fit the short model once on 80% and evaluate on held-out. Hmm.

Actually, let's stage: exp A = ngram OOF + sibling features + pairwise augmentation + Sinkhorn/centering sweep. exp B = short engine gate threshold. Run A first, cache.

Let me write exp A now.

For the pairwise classifiers, the notebook uses 8 pairs. sibling_features has 10 pairs. The relevant question: does adding the pair-routed dense features to each pairwise classifier improve the pairwise decision over TF-IDF alone? I can measure directly: for each pair, OOF pairwise accuracy with (a) TF-IDF only, (b) TF-IDF + pair-routed sibling features, (c) sibling features only. And then the end-to-end effect on 18-way bacc when injected via coherent residual.

Memory: hstack of sparse TF-IDF (2699 x ~390k) with 101 dense → convert dense to csr, fine. 390k * 2699 sparse. Memory OK. But careful: pairing with LogisticRegression on hstack of 390k sparse + 101 dense scaled — need to scale dense features so they're comparable. Use sibling features (robust z, clipped) which are roughly in [-5,5]; multiply by small factor. LR with C=2 will handle.

Actually there's a latency concern for the notebook: 8 pairwise fits on the full 4-view matrix take 4.6s (measured). Adding dense won't change much. Fine.

Let me write it.
Thinking: I have a leftover weird line in apply_ot: 
```
q = sinkhorn_knopp_transport(p, reg=reg, alpha=1.0 if alpha >= 1 else None) \
    if False else sinkhorn_knopp_transport(p, reg=reg, alpha=1.0, max_iter=max_iter)
```
That's ugly dead code. Let me clean it. Also the semantics: alpha=1.0 in sinkhorn then blend (1-alpha) — that's the module's documented "convex blend"? Let me check the module's `alpha` semantics: in sinkhorn_knopp_transport, alpha is the blend: "alpha=0 is the exact identity, alpha=1 imposes balanced column mass". So the function itself blends. So I can just call sinkhorn_knopp_transport(p, reg=reg, alpha=alpha) directly. Let me check the function signature/behavior.
Thinking: Note: `adaptive_neighborhood_centering` has no `groups` param — it derives neighbors from posteriors or embeddings. That's a design weakness for this competition: we *know* prompt clusters at test time? Actually no — test.csv doesn't give prompt_id. So neighborhoods must be inferred. But we could cluster by embedding. The module falls back to sqrt-P cosine on posteriors — that's weak but usable.

Important: my experiment should compare centering using TRUE prompt groups (upper bound) vs posterior-derived (what's actually available). I'll note the gap. For the grid, I'll just use the default (posterior-derived) and also an oracle variant using group means computed from groups, to show the ceiling.

Let me fix the script: remove the weird dead code, use the module alpha directly, and add oracle-group centering variant.

Also `sinkhorn_knopp_transport(p, reg=reg, alpha=alpha)` handles blending already. Good.

Let me also reconsider: in my `apply_ot` I call sinkhorn per cluster with alpha applied inside, which blends (1-alpha)*p + alpha*balanced per cluster. Correct.

Let me rewrite those parts.
Thinking: Two findings already:

1. **`sibling_features` returns 93 features, not 101** — `N_FEATURES` earlier printed 101 but the matrix is (2699, 93). Let me check: `extract_sibling_features` with method="robust" maybe drops constant columns? Yes likely it drops zero-variance columns. That means the schema is *data-dependent* → column order can shift between train and test! That's a genuine integration bug: `get_pair_feature_indices` indexes into the full 101-schema, but the returned matrix is 93 columns, so slicing `S[:, idx]` is **misaligned**. Let me verify.

That's a serious defect worth surfacing. Let me check lines around 1912-1998.

2. `get_pair_feature_indices` raises KeyError for ("glm-5.3","claude-opus-4-6") — which is one of the 8 shipped pairs. So 1 of 8 shipped pairs has no routing entry. Need fallback.

Let me inspect both.
Thinking: So `extract_sibling_features` claims (N, N_FEATURES) = (N, 101), but returned (2699, 93). Let me check: earlier `extract_sibling_features(t)` on 500 docs gave shape (500, 101). On 2699 docs it gave 93. So the width depends on the batch! That means `_robust_zscore` must be dropping something, or... wait nbytes earlier: (4000, 101) for `t*8` (4000 docs of 500 unique). Now 93 for 2699. Hmm, that's not the raw count — it's that `extract_sibling_features(df["text"].tolist())` returned 93. 

Wait, actually the print said "sibling matrix (2699, 93)". So width varies by batch. Let me check `_robust_zscore` — maybe it drops degenerate columns. Let me look.

This is a genuine, serious bug: **the column schema silently changes with the batch**, so `get_pair_feature_indices` (indices into the 101-wide schema) will point at the wrong columns. Let me confirm by reading `_robust_zscore`.
Thinking: robust_zscore preserves width. So width 93 vs 101 means `extract_raw_sibling_features` returned 93 columns? No — earlier call on 500 docs returned 101. Hmm wait, in this run I did `S = extract_sibling_features(df["text"].tolist())`. df has 2699 rows... unless some rows have empty text and the registry... no, width is column count fixed by registry.

OH WAIT. Maybe pandas read_csv mis-parsed and `df["text"]` is a *Series*, not a list... no, `.tolist()`.

Hmm, but the printed shape was (2699, 93). Let me directly test. Maybe N_FEATURES is 101 but FEATURE_NAMES has duplicates? Or maybe extract on 500 docs returns 101 because... no.

Let me just test directly.
Thinking: N_FEATURES is 93, not 101. Earlier I printed "N_FEATURES 101" — wait, my first command printed `N_FEATURES 101 pairs 10`. Now it prints 93. That's bizarre... unless the file changed between the two calls. The file was modified Oct 21 02:30. Hmm, did something modify it? Let me re-check. Actually the first run printed "N_FEATURES 101". Now 93.

Wait — maybe there are two versions and `experiments/archive/sibling_features_prev.py` is the 101 version, and `src/` is 93? No, my first import was `from src.sibling_features`.

Could the file have been regenerated by a background agent between my calls? Let me check mtime and grep for N_FEATURES definition.
Thinking: The files are being **actively modified right now** — `sibling_features.py` mtime 02:34:38, current time 02:35:04, and `test_time_calibration.py` 02:34:48. Something (another agent/process) is still writing these files. My earlier read at 02:31 saw 101 features; now it's 93. The user said "3 newly authored modules" — apparently a background subagent (experiments/20_run_3_deepseek_subagents.py) is still generating/rewriting them.

This is important context for the review, and it also means I must pin a specific snapshot. Let me check if a process is running.

Also the line numbers changed (now 1405 for N_FEATURES vs earlier 2168-line file with different structure). The file shrank? Let me check wc -l now.

I need to be careful: the file is a moving target. Let me check running processes, then re-verify current state and note the instability in my report.

Given the task is a review + integration design, I should base my answer on the current snapshot and explicitly warn that the modules are still being rewritten (mtime within seconds of my reads), which is itself a blocker: you cannot submit an artifact whose schema changes at 02:34.

Let me check for running processes.
Thinking: Critical finding: **Subagent 1 is still running and rewriting `src/sibling_features.py` while I review it.** The file went from 2168 → 1955 lines and 101 → 93 features during my review. This is the single most important operational finding: I am reviewing a moving target. Any integration I write against feature indices can silently break when the subagent's next write lands.

Also note the subagent was told "10 most confused sibling pairs" with specific rationale for each. But it emits only 93 features vs "101" claimed in the user's prompt and in the module's own docstring.

Now, key: the shipped 8 pairs differ from the module's 10 pairs:
- Shipped: ("glm-5.3","claude-opus-4-6") — module has no such pair (it has glm-5.3/glm-5.3-flash).
- Module pairs not shipped: ("claude-fable-5","claude-opus-4-8"), ("gemini-3.1-pro-preview","gemini-3.7-flash"), ("gpt-5.5","gpt-6-astra").

So 1 of 8 needs a fallback. I'll handle with union of both pairs' routed indices.

Let me continue the experiment. Wait for subagent to finish? It may keep rewriting. I'll proceed and pin by copying a snapshot to experiments/snapshot/ so my measurements are reproducible. Actually better: measure now, and note the hash. Let me compute md5 and copy the snapshot.

Let me update the experiment to be robust: use `get_pair_feature_indices` with fallback (union of the two families' pairs or top-k by |mean diff|), and handle missing pair gracefully.

Actually a cleaner, stronger design for integration (and one I can defend): don't rely on the *static routing table* at all. Instead compute, inside each training fold, the discriminative feature subset for that pair via a cheap F-statistic / mutual information on the fold's training rows. Reasons: (1) the routing table is hand-authored and its correctness is unverified/unmeasured, (2) the routing table is missing one shipped pair, (3) it's about to change under us, (4) selecting on fold-train is leak-free and adapts to the actual data. I can measure both: static routing vs fold-fit selection.

Let me implement: `select_pair_features(S_train, yp_train, k=12)` using absolute standardized mean difference (Cohen's d) between the two classes, top-k. Then pair classifier uses TF-IDF + S[:, sel].

That's the "fastest and safest" answer: no OOM (101 dense cols is 3.2 MB for 4000 rows), no latency (4.6s for 8 pairwise fits measured, dense adds nothing), and fold-internal selection avoids routing-table drift.

Let me modify the experiment accordingly and rerun. Also fix the KeyError path.

Also should I test whether the dense features add value to the *base 18-way* model too? Yes — that's cheap and tells whether sibling_features is worth anything at all. Add view: S-only 18-way LR, and concat with TF-IDF 18-way LR. This is the decisive question for "should this go into the pipeline at all".

Let me restructure the script a bit. I'll add:
- [2a] 18-way LR on S alone, and on hstack(X, S*10).
- [2b] pairwise with fold-fit selection.

Careful with scale when hstacking dense with sparse TF-IDF: TF-IDF values are sublinear_tf ~ small; dense z-scores in [-8,8]. Multiply dense by 0.1 to keep comparable? With C=5 regularization it matters. I'll standardize dense to unit variance and scale by 0.3, a reasonable choice; and also try raw. To limit combinatorial explosion, I'll use StandardScaler→ *0.25.

Hmm, but the honest thing: I should tune only coarsely and report. Let me add two variants: S*1.0 and S*0.1, and pick better as the integration default (documenting it).

Let me write it.
The modules are being rewritten *while* I review them — subagent 1 is still running. Let me pin a snapshot so measurements are reproducible, then finish the sweep.
Thinking: Also fix the print loop to include "routed". And the injection: now I should test both `pair_logodds` (dense-augmented) and `pair_logodds_tfidf`. Let me update the injection section to sweep both sources and weights.

Let me update the print and injection blocks.
