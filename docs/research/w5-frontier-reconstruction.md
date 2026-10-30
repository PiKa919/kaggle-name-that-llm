# W5 — Frontier Reconstruction: what a 0.65–0.70 solution to *Name That LLM* looks like

**Competition:** Kaggle *Name That LLM* — 18 frontier models, 40–250-word passages, balanced accuracy.
**Current state:** our public LB 0.600; best public notebook 0.604 (0.607 reported historical); LB #1 0.688.
**Scope:** reconstruct the missing +0.06–0.08 from leaderboard/reputation signals, artifact forensics, published ablations, and model/data-scale extrapolation. This report deliberately does **not** re-dissect the public notebooks (W1 does that) or the literature (W4 does that); it asks what the top teams have that the public stack does not.

**Tag legend**
- `[PROVEN]` — measured on this competition's LB/CV, or directly observed in a public artifact.
- `[LIKELY]` — repeated independent observations, or strong transfer evidence from an adjacent setting.
- `[SPECULATIVE]` — inference from score gaps; no direct NTL evidence.

---

## 0. TL;DR

1. **The 0.688 score is not explained by any public notebook or public checkpoint.** The entire public lineage tops out at 0.604–0.607 on LB, and every public increment after 0.593 is +0.003 to +0.006. The remaining +0.08 is a different regime, not a better decoder.
2. **Three regimes can produce +0.08, in order of likelihood:** (a) a much larger, format-matched, provenance-cleaned corpus generated through the actual 18 APIs, used to train several 8B/14B/27B LoRA bags; (b) a training-objective change the public stack lacks — same-prompt contrastive/hard-negative loss plus hierarchy/pairwise sibling specialists; (c) a pure artifact-reuse ensemble with nested weight selection and per-class calibration. (a) and (b) can coexist; (c) is worth +0.01–0.03 at most.
3. **Hard prompt grouping is a proven trap on this LB.** Three independent teams lost LB score with recovered-group Hungarian assignment despite +0.08–0.12 OOF. Soft Gaussian group heads are worth only ~+0.005 on LB. Do not build the frontier on grouping.
4. **The fastest defensible gains for our team are hierarchical and provenance-based, not architectural:** within-family gates (+3.7 pts on the 5-way Claude subset), hierarchy with external data only below lab level (+2.4 pts on a CPU stylegram), and removing misrouted Gemini-3.7/GLM external rows. These are cheap and already have public evidence.
5. **Balanced accuracy amplifies rare-class fixes.** The public stack's weakest classes (`claude-fable-5` recall 0.11 in Alexey's 0.42 blend; `gpt-5.6-sol` 0.21; GLM siblings ~0.21) are worth more than another 0.002-weight ensemble member. A frontier solution must be judged by per-class recall, not overall accuracy.

---

## 1. Evidence base and method

### 1.1 Leaderboard snapshot (2026-10-21)

| Rank | Team | Public LB |
|---:|---|---:|
| 1 | Hod Carriers | 0.688 |
| 2 | Vishak K Bhat | 0.667 |
| 3 | Adisak Sukul | 0.658 |
| 4 | Harveen Singh Chadha | 0.644 |
| 5 | Le Viet | 0.637 |
| 6 | Maniceet Sahay | 0.634 |
| 7 | Søren Ravn Andersen | 0.634 |
| 8 | Aarush112006 | 0.633 |
| 9 | TF | 0.631 |
| 10 | kulshekhar | 0.630 |

Source: `kaggle competitions leaderboard name-that-llm -s`, 2026-10-21. The top two are 2.1 and 3.0 points apart; #3–#10 are separated by <1 point. That shape is consistent with one team having a materially better data/compute regime rather than LB noise (binomial SE at 1,600 public rows is ~1.2 points, but a 2.1-point gap plus a 3.0-point gap to #3 is unlikely to be pure noise).

### 1.2 Public score ladder (what is actually proven)

| Version / notebook | Change | Public LB |
|---|---|---:|
| Travis Nguyen in-style stack | 5 LoRA + frozen heads + stylegrams | 0.563 |
| prvsiyan v1 | same stack + 4 shared-prompt Gaussian heads | 0.569 |
| prvsiyan v2 | + public two-stage 8B/14B branch, 85/15 mix | 0.593 |
| prvsiyan v3 | + own two-stage seeds (38,642-passage pool) | 0.599 |
| prvsiyan v4 | + Gaussian head on every two-stage LoRA | **0.604** |
| prvsiyan v5 | + Qwen3-4B early-layer member (10%) | 0.603 (completed whole-notebook; member effect not isolated) |
| Joseph Adamski | 0.90 × V4 + 0.10 × ATLAS | **0.607** (reported historical) |

Sources: `prvsiyan/the-prompt-is-not-the-author` markdown (V1–V5 table); `josephadamski91/teach-the-model-10-atlas` metadata. The 0.607 is the author's reported historical score, not independently re-run. The key observation is the **shape**: going from a 0.563-style stack to the full public frontier bought +0.041, and the two-stage LoRA was +0.024 of it. Grouping/Gaussian refinements bought +0.011 total.

### 1.3 What we inspected

- **Notebooks (code + markdown):** `teach-the-model-10-atlas`, `the-prompt-is-not-the-author`, `0-595-two-stage-lora-gaussian-olmo2`, `two-stage-lora-gaussian-full-hungarian-0-604`, `0-600-teach-the-model-remove-the-prompt`, `0-597-two-stage-lora-olmo2-smollm3`, `name-that-llm-qwen-lora-ensemble` (nested OOF), `the-prompt-recovery-trap`, `test-conditional-author-decoding`, `math-view-of-name-that-llm`, `ntl-0612-apex-invariants-hungarian`, `ntl-apex-0612-atlas`.
- **Datasets (metadata + small files):** Oleg corpus, prvsiyan extended, Seyit Claude passages, Hailey Cheng free-quota, Maniceet paid OpenRouter/OpenAI, Illia prompt bank, pjmathematician reconstructed prompts and UltraBERT, Zbigniew Claude-head/in-style LoRAs, lightningv08 artifact bundle (27B/OLMo2/SmolLM3/Qwen2.5/Gemma/OOF heads), Rudra fusion heads, Joseph ATLAS, Nishant two-stage adapters.
- **Discussions:** 749447 (data scale), 750213 (grouping simulation), 750706 (train structure), 750462 (early layers), 750224 (Claude extra data/hierarchy), 750947 (external data), 751370 (Gemini-3.7 routing), 750365 (authorship verification), 750675 (fingerprints/nonlinear heads), 751268 (9-hour/Qwen3.5 load), 750390 (grouped CV), 750924 (slow `/kaggle/input`).
- **Paper:** arXiv 2610.19583 (ATLAS), fetched directly.
- **Reputation signal:** CLIST's standings snapshot maps team names to Kaggle handles: #1/#2 `Vishak K Bhat` → `vishakkbhat`, `Hod Carriers` → `tononnh`, `Adisak Sukul` → `adisak`, `Maniceet Sahay` → `maniceet`, `Nishant Dahal` → `nishantdahal`, `Seyit kaan Gunes` → `seyitkaangunes`. `tononnh` and `vishakkbhat` have **no public NTL kernel or dataset**; `adisak` and `maniceet` have discussion/dataset activity. The public-notebook authors sit at rank ~40+ (0.61). The frontier is private.

Limitations are listed in §7. The most important: **no public writeup from #1–#3 exists**, so the top path is a reconstruction, not a verified recipe.

---

## 2. Decomposing the 0.081 gap

### 2.1 Base-model vs ensemble arithmetic

The public stack's own ablations give a rough production function:

| Layer | Best public evidence | Balanced accuracy |
|---|---|---:|
| TF-IDF + LinearSVC, grouped CV | Adisak / Yang Kuang Ou | 0.326–0.372 |
| Frozen Qwen3-4B early layers + LR | Alexey Osipenko | 0.391 (layer 3), 0.414 (6-layer concat) |
| Frozen Qwen3-4B blend with TF-IDF/stylometry | Alexey | 0.439 CV / 0.420 LB |
| Qwen3-8B two-stage LoRA, plain head | prvsiyan own pool | 0.5675 OOF |
| Same, oracle prompt centring | prvsiyan | 0.6447 OOF |
| Full public ensemble (8B+14B+SVC+stylometry+frozen+Gaussian) | prvsiyan v4 / Sangam | **0.604 LB** |

Two conclusions:

1. **The best single public model is ~0.567 OOF, and the ensemble adds ~+0.037.** That is a normal ensemble lift from diverse members.
2. **0.688 requires either a base model around 0.63–0.64 (then +0.04–0.05 ensemble), or an ensemble that adds ~+0.12 over the best single model.** The latter is implausible with the current member set: adding a 15th correlated member buys <0.005. Therefore the frontier almost certainly has a **materially stronger base classifier**, not just more post-processing.

### 2.2 Why the gap cannot be closed by more Gaussian/Hungarian tuning

The measured post-processing budget on this LB is small:

- Two-stage + Gaussian + full Hungarian (Sangam): 0.604.
- Joseph's ATLAS 10% (a genuinely different evidence source): +0.003.
- Own two-stage seeds: +0.006 (0.593→0.599).
- Gaussian head on every LoRA: +0.005 (0.599→0.604).
- V5 with early-layer member: **0.603** (completed whole-notebook result; the 0.604→0.603 difference is not a causal estimate of the member).
- Exact conditional author decoding (mtoshidesu: 0.520 modified vs 0.522 base): **no LB gain**.
- Recovered-group assignment (Yang Kuang Ou): OOF 0.327→0.419, **LB 0.336→0.308**.

This is a saturated post-processing surface. A team at 0.688 did not get there by tuning Sinkhorn iterations.

### 2.3 Balanced-accuracy arithmetic: the weak-class lever

Balanced accuracy is the unweighted mean of 18 recalls. If three sibling-confused classes sit at recall 0.2 while the other 15 sit at 0.65, the score is `(15×0.65 + 3×0.2)/18 = 0.575`. Raising those three to 0.55 gives `(15×0.65 + 3×0.55)/18 = 0.633` — **+0.058 from three classes alone**. Alexey's 0.420 blend reported `claude-fable-5` recall 0.11, `gpt-5.6-sol` 0.21, GLM ~0.21; the 0.604 stack is certainly better, but the same asymmetry persists. A frontier team that solves the three or four worst sibling pairs can gain more than a team that adds another ensemble member.

This is the strongest argument that the missing technique is **targeted sibling work**, not another global model.

---

## 3. Artifact table: the actual frontier ingredients

Costs assume Kaggle 2×T4, 9-hour inference ceiling, and offline rerun. "Reuse" means no training, only inference.

| Artifact | Type | What it gives | Evidence | Cost / risk |
|---|---|---|---|---|
| `prvsiyan/ntl-public-in-style-pool-v1` | 38,642-passage pool built from 15 public datasets | The pool behind the 0.599→0.604 two-stage branch | prvsiyan markdown: own LoRA OOF 0.5675 plain, 0.6447 oracle-centred | Reuse for training; licensing/filter provenance must be checked |
| `nishantdahal/ntl-two-stage-lora` | Public 8B/14B LoRA checkpoints | The 0.592 two-stage branch | prvsiyan V2 0.593 | Reuse; ~1–2 h inference |
| `prvsiyan/ntl-two-stage-lora-adapters-v1` | Own two-stage adapters, 2 seeds | +0.006 (0.593→0.599) | prvsiyan V3 | Reuse; 2 seeds |
| `lightningv08/ntl-olmo2-native24-final-inference-assets` | Bundle: Qwen3.8-27B NF4 LoRA, OLMo2 native24, SmolLM3, Qwen2.5 native20, Gemma, TabICL/tabular, OOF Qwen3.5 heads | The only public **27B** classifier: final 8 decoder layers (56–63), rank 16/alpha 32, 28,390 external responses + 2 official epochs; plus orthogonal small-backbone members | Dataset metadata; `0.597` notebook uses OLMo2/SmolLM3, **not** 27B or ATLAS | Heavy: 27B NF4 inference on 2×T4; not proven to beat 8B/14B stack |
| `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head` | 8B LoRA with a 5-way Claude head | 5-way Claude: 0.484 vs 0.456 plain; 0.493 with gate | Zbigniew comment in 750224 | Reuse; cheap |
| `zbigniewkirchner/ntl-qwen3-8b-lora-instyle` | In-style 8B LoRA | 0.574 standalone | Dataset/notebook | Reuse |
| `pjmathematician/ultrabert-380m-ntl` | UltraBERT-380M, 5 folds | Fixed 0.15 share in the nested OOF notebook | lvnew notebook; dataset files | Reuse; ~0.15 weight cap |
| `rudrabhaskar/name-that-llm-qwen-fusion-heads` | Pre-fitted heads + selection JSONs | Component OOF: SVM 0.372, Qwen8 plain 0.446, centre 0.466, joint 0.467, DeBERTa 0.384; fusion 0.507–0.547; sparse 2/3/4 replies 0.511/0.539/0.542 | `qwen8_fusion_selection.json`, `qwen14_fusion_selection.json` | Reuse; no GPU training |
| `josephadamski91/ntl-atlas-qwen3-8b-v1` | 18 source-specific adapters on frozen Qwen3-8B-Base | ATLAS likelihood evidence; +0.003 at 10% on LB | Dataset metadata; arXiv 2610.19583 | Reuse; 18 forwards/sample is expensive in full, fine at 10% mix |
| `maniceet/name-that-llm-openrouter-openai-generations` | 7,753 paid passages, 10 labels, competition-style single paragraph | Real API corpus from the current #6; heavy GLM coverage (4,667 rows: 1,661 full + 2,238 flash + 178/590 Z.AI) and ~3.0k GPT | Dataset metadata, discussion 750947 | Reuse only within-family (GLM/GPT); labels are publisher claims; Gemini 3.7 rows may be routed 3.8 |
| `seyitkaangunes/ntl-extra-api-passages` | ~5,000 Claude-only passages, 1,000 new prompts | Best public within-family data | 750224; Zbigniew's +2.4 hierarchy gain | Reuse; within-family only |
| `olegzholobov/name-that-llm-response-corpus` | 135,171 unique responses total, 24,015 in the 40–250-word range; 1,902 exact prompt groups with all 18 models in the full corpus; provenance | The largest public 18-model corpus | Dataset README | Reuse; heavy format mismatch (line breaks/bold/length); in-range subset does not preserve all 1,902 complete groups |
| `prvsiyan/prvsiyans-extended-name-that-llm-dataset` | 9,290 passages, 510 prompts, 18 models; 4,926 clean; API-logged; misrouted responses quarantined | Cleanest public 18-model corpus; provenance discipline | Dataset metadata | Reuse; best default external source |
| `haileycheng/name-that-llm-same-prompt-passages` | 1,002 passages, 12 labels, free Kaggle quota | Free same-prompt blocks; reproducible collector | Dataset metadata | Reuse; missing 6 labels |
| `dophuchung/ntl-proxy-generations`, Zbigniew `ntl-extra-gpt-passages`, Lucas Boesen proxy data | Proxy/free API passages | Extra GPT/Claude coverage | Cited in prvsiyan lineage | Reuse; route shift |
| `illiadolenko/ntl-prompt-bank-v1` | 12,295 English prompts + config | Prompt-distribution-matched generation bank; train prompts excluded and n-gram-checked | Dataset metadata | Reuse for data generation |
| `pjmathematician/ntl-train-prompts-reconstructed` | 150 reconstructed train prompts + protocol meta + `infer_prompts.py` | Diagnostic prompt reconstruction; useful for group analysis, **not** test prompts | Dataset + script | CPU/LLM calls; not a test-time solution |
| `kirilltym/ntl-frozen-heads-and-ce-projection` | Public fitted heads | Reuse without re-extracting features | Referenced by mtoshidesu notebook | Reuse |

**What is missing from the public artifact set:** a 100k+ passage corpus generated with the exact competition template and exact model snapshots; a 27B model *proven* to beat the 8B/14B stack; any public writeup of a >0.61 LB result. Those three absences define the frontier.

---

## 4. Ranked paths to 0.65–0.70

Expected ranges are incremental over our 0.600, not additive across paths. Costs are rough.

### Path A — Private-scale, format-matched, provenance-clean corpus + multi-backbone LoRA bags

**Thesis:** the top team bought or generated far more in-domain data than the public pool, cleaned it to the competition's exact format, and trained several large LoRA classifiers whose errors are decorrelated. This is the most likely explanation of a 2-point gap to #2 and an 8-point gap to the public stack.

**Steps**

1. **Corpus (100k–300k passages).** Generate prompt blocks through the actual 18 APIs. Template: `Write a single paragraph of about N words, with no title or headings.` Draw `N` from the competition train word-count distribution (median ~101, range 36–274). Keep exact prompt text, requested model, served model, route, provider, timestamp, token counts, finish reason.
2. **Format matching (non-negotiable).** Strip titles, headings, bullets, bold, line breaks; collapse to one paragraph; drop <40 or >250 words; preserve case, straight/curly quotes, em/en dashes and spacing. Public evidence: raw external text is 94% line-break and 60% bold, median 174 words vs competition 101; adding it to TF-IDF dropped 0.326→0.279. `[PROVEN]`
3. **Label-noise audit.** Quarantine rows whose `served_model` differs from the request. Gemini-3.7 was deprecated on 2026-10-08 and requests are routed to 3.8 (discussion 751370); GLM OpenRouter provider routing can mix 5.3/5.3-flash. Training on misrouted rows teaches the wrong sibling boundary. `[PROVEN issue]`
4. **Two-stage LoRA, several seeds/backbones.** Stage 1: 1 epoch on external, class-weighted CE. Stage 2: 2 epochs on the 2,699 official rows at half LR. Backbones: Qwen3-8B, Qwen3-14B, Qwen3.5-9B, Qwen3.8-27B NF4. Readout: 8 pooled layers through per-layer LayerNorm, linear head; LoRA on all linear layers, r16/α32. Truncation augmentation (20% of rows to 40–80 words). `[LIKELY]` based on public two-stage recipe and 27B asset.
5. **Hierarchy, not flat 18-way.** Train a 6-way lab head, then within-lab line/version heads using external data **only below the lab level**. Zbigniew: +2.4 points on a CPU stylegram with this design; Claude gate 0.456→0.493. `[PROVEN on subsets]`
6. **Ensemble with nested prompt-fold weights.** Log-prob average across seeds/backbones + SVC + stylometry + frozen early-layer heads + UltraBERT + ATLAS at small weight. Select weights on inner prompt folds, evaluate on outer, and **simulate 1/3/6 replies per prompt** because the LB behaves like a sparse pool.
7. **Balanced decision.** Equal-mass Sinkhorn or prior-corrected argmax; optionally tune per-class biases on OOF with shrinkage. Do **not** hard-assign one author per recovered group.

**Expected:** 0.64–0.68 `[LIKELY]`; 0.69+ only if the corpus is both large and truly format-matched.
**Cost:** $1k–$10k API (depends on model mix), 30–100 GPU-hours of training across sessions, 1–3 h inference.
**Risks:** route/snapshot mismatch; licensing; 9-hour inference; 27B memory; overfitting official rows in stage 2; **rules risk** (Section 6 requires external data to be public and free to all, or to meet the host's Reasonableness Standard, and winners must publish it — so any paid corpus should be released as a public Kaggle Dataset before the deadline, and budget should stay defensible under "minimal cost").
**Why it explains #1:** #6 (0.634) publicly shared a 7.7k paid corpus and said it gave "meaningful gains"; the 107-vote thread is entirely about whether money buys the win; the only public 27B classifier exists but is not in any 0.60+ public notebook, consistent with teams using it privately.

### Path B — Training-objective change on public data: same-prompt contrastive + pairwise siblings

**Thesis:** the public stack minimizes plain cross-entropy on a topic-confounded label. A team that explicitly optimizes **author identity given the same prompt** can get a stronger base model without buying data.

**Steps**

1. Build prompt blocks from the public corpora (Oleg, prvsiyan, Seyit, Maniceet, Hailey). Within each block, all passages share the prompt and differ by author.
2. Add a supervised-contrastive / InfoNCE term to the LoRA CE loss:
   - positives: same author, different prompt;
   - hard negatives: different authors, **same prompt** (the exact confound);
   - temperature ~0.05–0.1, loss weight 0.1–0.3.
   Literature evidence: +5.4 pp authorship accuracy on blogs, +11 pp top-k diverse authors (W4 §5.2). `[LIKELY]` transfer.
3. Add a **pairwise sibling tournament** for the confusable sets: GPT-5.6 luna/sol/terra, the 5 Claudes, GLM-5.3 vs flash, Gemini-3.7 vs 3.8. Train small linear/MLP heads on frozen LoRA features or stylegrams; combine with Bradley–Terry or a second-stage stack. Public family gates already show +3.7 pts on the Claude subset; full pairwise is the natural extension. `[LIKELY] +0.005–0.02`
4. Multi-seed bagging (3–5 seeds) and nested weight selection. `[LIKELY] +0.01`
5. Keep the Gaussian group head at low weight (0.10) and only on tight clusters; never hard assignment.

**Expected:** 0.62–0.66 `[LIKELY]`; 0.67+ only if contrastive + pairwise both transfer strongly.
**Cost:** 1–2 retrains (6–10 GPU-hours each), no API spend.
**Risks:** contrastive loss can collapse or overfit 2.7k rows; pairwise heads need enough per-pair data; public evidence is adjacent-domain, not NTL.

### Path C — Artifact-reuse ensemble with nested calibration

**Thesis:** most of the public frontier is already packaged as checkpoints and pre-fitted heads. Recombining them with proper nested weight selection and per-class calibration is the cheapest way to move from 0.600 to ~0.62.

**Steps**

1. Mount: Nishant/prvsiyan two-stage adapters, Zbigniew Claude-head/in-style, Rudra fusion heads, UltraBERT, OLMo2/SmolLM3/Qwen2.5, ATLAS. Skip 27B unless runtime allows.
2. Refit only lightweight readouts within prompt folds; reuse frozen features.
3. Nest weight selection: 5 outer prompt folds, 3 inner folds; include sparse-pool scenarios (1/3/6 replies) in the inner objective.
4. Calibrate: temperature scaling per member, then per-class logit bias tuned on OOF with ±1.5-logit cap and shrinkage; compare equal-mass Sinkhorn vs prior-corrected argmax.
5. Report per-class recalls and the sibling confusion matrix; choose the blend that maximizes the **worst-class** recall, not overall accuracy.

**Expected:** 0.61–0.63 `[PROVEN direction]`; +0.01–0.02 over naive reuse.
**Cost:** 0 training; 1–2 h inference; mostly engineering.
**Risks:** public heads may encode train-fold leakage; nested selection is essential; runtime of many members.

### Path ranking

| Rank | Path | Expected | Cost | Evidence strength | Recommendation |
|---:|---|---:|---|---|---|
| 1 | A: paid/format-matched corpus + multi-backbone bags | 0.64–0.68 | High | Strong indirect (LB shape, #6 data, 27B asset) | Only path with a credible route to #1 |
| 2 | B: contrastive + pairwise siblings | 0.62–0.66 | Medium | Literature + family-gate evidence | Best ROI if no API budget |
| 3 | C: artifact-reuse + nested calibration | 0.61–0.63 | Low | Direct public ablations | Do this first as the safe floor |

**Recommended sequence:** C (1–2 days) → B (1–2 weeks) → A only if budget and time allow. C gives a reliable 0.61–0.62; B is the only cheap route to 0.65; A is the only route that plausibly explains 0.688.

---

## 5. Missing grandmaster techniques, ranked by expected value

| # | Technique | Why it is missing from the public stack | Evidence | Expected Δ | Tag |
|---:|---|---|---|---:|---|
| 1 | **Targeted sibling corpora, not uniform volume** | Public corpora are class-unbalanced and mostly lab-level; #6's corpus is deliberately GLM/GPT-heavy | Claude gate 0.456→0.493; hierarchy +2.4 pts; Maniceet 7.7k corpus | +0.01–0.03 | `[PROVEN on subsets]` |
| 2 | **Provenance/label-noise filtering** | Public datasets publish labels, not always served-model; Gemini-3.7 routing and GLM provider routing poison sibling boundaries | 751370; prvsiyan quarantines misroutes; Maniceet keeps `served_model` | +0.005–0.02 | `[LIKELY]` |
| 3 | **Competition-format restoration** | External corpora have line breaks/bold/longer median; raw addition hurts | HALX: 0.326→0.308→0.279; competition 0% line breaks/bold | +0.01–0.03 | `[PROVEN negative if skipped]` |
| 4 | **Same-prompt contrastive / hard-negative loss** | No public NTL notebook uses it; all use plain CE | W4 §5.2: +5.4 pp blogs, +11 pp diverse authors | +0.01–0.03 | `[LIKELY]` |
| 5 | **Full pairwise sibling tournament** | Public stack has only family gates (Claude/GPT/Gemini/GLM), often off | Claude gate +3.7 pts on 5-way; sibling errors 55–58% of total | +0.005–0.02 | `[LIKELY]` |
| 6 | **Multi-seed LoRA bags + nested weight selection** | Public weights are hand-set from the 0.563 lineage; seeds are 1–2 | Own seeds +0.006; lvnew nested selection | +0.01–0.02 | `[LIKELY]` |
| 7 | **Per-class calibration for balanced accuracy** | Public notebooks use equal-mass Sinkhorn, not OOF-tuned per-class biases | GoPain simulation: equal mass ≤0.003 from optimal; balancing to true frequencies −0.040 | +0.005–0.015 | `[PROVEN direction]` |
| 8 | **Early/mid-layer + style-token pooling** | Public stack uses layers 4/8/22/24, but many heads use last-layer/mean | Alexey: layer 3 0.391 vs last 0.311; concat 0.414 | +0.005–0.015 | `[PROVEN]` |
| 9 | **Length-stratified training / truncation augmentation** | Public two-stage recipes do not stratify by length | <60 words ~0.27 vs >160 ~0.46 accuracy (AGENTS benchmark) | +0.005–0.01 | `[LIKELY]` |
| 10 | **27B NF4 LoRA** | Only one public asset; not used in any 0.60+ public notebook | 28,390 external + 2 official epochs, final 8 layers | +0.00–0.02 | `[SPECULATIVE]` |
| 11 | **Prompt reconstruction at test time** | Tried publicly; overfits train layout | 0.501 OOF → 0.351 LB; 0.419 OOF → 0.308 LB | Negative | `[PROVEN negative]` |
| 12 | **Hard Hungarian / injective assignment** | Tried by at least four teams | 0.353→0.344; 0.345→0.343; 0.336→0.308 | Negative | `[PROVEN negative]` |
| 13 | **Transductive pseudo-labeling of test** | No public evidence; runtime risk | None | Unknown, possibly +0.01 | `[SPECULATIVE]` |
| 14 | **Authorship-verification metric learning** | One forum thread, no implementation | Verification literature; no NTL result | Unknown | `[SPECULATIVE]` |

### 5.1 The single most underrated technique: hierarchy with external data below the lab level

The public failure mode is dumping extra data into the 18-way classifier. Seyit and Zbigniew both report the opposite: use external data to choose **which Claude/GPT/GLM**, not to teach the lab. Zbigniew's hierarchy gave **+2.4 points** on a plain CPU stylegram model, with a bootstrap CI excluding zero. That is larger than most public ensemble increments and costs almost nothing. Any frontier reconstruction that ignores hierarchy is leaving a proven gain on the table.

### 5.2 The single most overrated technique: prompt-group assignment

Every public attempt to recover prompt groups on the test distribution and enforce one-author-per-group lost LB score, despite large OOF gains. The mechanism is understood (GoPain simulation): with 1–2 answers per prompt, centring subtracts the author signal; wrong merges make the injective constraint forbid correct labels; and the hidden test is subsampled/incomplete. Keep only soft Gaussian evidence on tight clusters at low weight. `[PROVEN negative]`

GoPain's controlled simulation (20 pools of 4,000 passages, plain decoding ≈0.35) makes the boundary explicit:

| Prompt group source | Answers/prompt | Centring | Hungarian | Joint (≤1/author) |
|---|---:|---:|---:|---:|
| True groups | 1 | 0.000 | 0.000 | 0.000 |
| True groups | 2 | −0.067 | +0.006 | +0.024 |
| True groups | 3 | +0.002 | +0.013 | +0.042 |
| True groups | 18 | +0.101 | +0.117 | +0.115 |
| Recovered, train-like CV | 18 | +0.098 | +0.118 | +0.111 |
| Recovered, test | 1 | −0.102 | −0.004 | −0.019 |
| Recovered, test | 2 | −0.057 | +0.004 | +0.015 |

The public LB is the "recovered, test, 1–2 answers" row, not the "train-like CV" row. This single table explains why every OOF-positive grouping notebook lost LB, and why a dense-18 OOF gain of +0.10 cannot be used to justify a test-time assignment. Caveat: the simulation's base accuracy (~0.35) is below the real 0.60 stack, so read the signs and the sparsity pattern, not the exact magnitudes.

### 5.3 What the top team probably did *not* do

- A novel decoder. The public decoder space (Hungarian, cavity, exact marginals, Sinkhorn, gates) is saturated and measured.
- A bigger public model alone. The 27B asset exists but is absent from the 0.597 notebook; 8B→14B buys little in the public stack.
- Test-time prompt reconstruction. It is publicly tried and fails on LB.
- A single magic feature. Punctuation/char n-grams are already in the 0.37-SVC and the 0.60 stack.

The remaining explanations are data scale/quality and training objective. That is the frontier.

---

## 6. Validation protocol for any frontier claim

The public notebooks' biggest methodological weakness is that their OOF does not simulate the hidden test layout. Adopt this protocol before believing any +0.01:

1. **Grouped outer CV:** `StratifiedGroupKFold(5, shuffle=True, random_state=42)` on `prompt_id`.
2. **Sparse-pool inner simulation:** for each validation prompt, subsample 1, 3 and 6 replies; measure the post-processing (centring, Gaussian, Sinkhorn) on each. The LB behaves like a sparse pool; dense-18 OOF overstates grouping gains by 0.05–0.10.
3. **Nested weight selection:** outer prompt folds for evaluation, inner prompt folds for blend weights. Never reuse the same OOF to both select and report.
4. **Per-class recall + sibling confusion:** report all 18 recalls, not overall accuracy. Optimize the worst three.
5. **LB delta only from complete submissions:** the 3-row commit run is a runtime/format check, not a score. Public LB at ~1,600 rows has ~1.2-point binomial noise; do not chase <1-point deltas.
6. **Runtime gate:** stage weights to `/tmp`, log `/kaggle/input` MB/s, and skip the lowest-weight member if loading falls behind. A 0.600 fork timed out because Qwen3.5-9B loaded in 1,206 s instead of 22 s; the cause was slow input reads, not the head (751268, 750924).
7. **Submission integrity:** 4,000 rows, exact 18 labels, no nulls, correct parquet schema.

---

## 7. Gaps and uncertainties

1. **No verified writeup from #1–#3.** The top teams' captains (`tononnh`, `vishakkbhat`) have no public NTL artifacts, and no public post describes a >0.61 method; the 0.688 recipe is inferred. Treat Path A as a hypothesis, not a fact.
2. **Hidden test layout is unknown.** Hosts have not answered the direct questions in 750213/750768. Public LB is reported as 40% by one participant, while the competition brief says 20–30%; the private split may be by passage or by prompt. This changes the value of grouping by ±0.05.
3. **Self-reported scores.** The 0.607 ATLAS result is historical metadata, not an independent rerun; several 0.60+ notebooks are remixes without new scores.
4. **27B accuracy is unknown.** The asset documents training, not a score. It may be slower without being better.
5. **Paid-corpus scaling is unproven.** #6 shared 7.7k passages and reports gains, but no dose-response curve exists. Diminishing returns are plausible; prompt diversity may matter more than raw count (749447).
6. **Label noise may impose a ceiling.** If Gemini-3.7 and GLM-5.3/flash snapshots are intrinsically overlapping in the official data, some sibling recall is unrecoverable. The Gemini-3.7 routing issue is real; its prevalence in the official train is unknown.
7. **`swaroopmeher/glm-4-9b-trunc25-ntl` returned 403** and could not be inspected; the GLM branch of the public stack is therefore less documented than the others.
8. **No public NTL result for contrastive loss, pairwise tournaments, or authorship verification.** These are literature-backed but not competition-proven; Path B is a bet.
9. **The "0612" Apex family is unverified.** `kiza123123/ntl-0612-apex-invariants-hungarian` and its remixes carry no score claim in the notebook, and their companion datasets return 403. The name suggests a 0.612 result, which would raise the public ceiling above 0.604–0.607; treat it as unconfirmed.

---

## 8. Actionable summary for our team

**If we have 2 days:** do Path C. Reuse the public checkpoints/heads, nest the blend weights, add per-class calibration, and verify with sparse-pool simulation. Target 0.61–0.62.

**If we have 2 weeks and no API budget:** do Path B. Build prompt blocks from public corpora, add same-prompt contrastive loss and a full sibling tournament, keep the Gaussian head at 0.10, and bag 3 seeds. Target 0.63–0.65.

**If we have budget:** do Path A. Generate 100k+ format-matched passages with provenance, filter misroutes, train 8B/14B/27B two-stage LoRAs with 3–5 seeds, and ensemble hierarchically. Target 0.65–0.68.

**Do not:** spend more time on Hungarian assignment, test-time prompt reconstruction, or another correlated 0.002-weight member.

---

## 9. Citations and local evidence

**Leaderboard / competition**
- Leaderboard: https://www.kaggle.com/competitions/name-that-llm/leaderboard (snapshot 2026-10-21).
- Team-name → handle mapping: https://clist.by/standings/name-that-llm-classification-balanced-accuracy-score-71276398/ (retrieved 2026-10-21).
- Test-layout questions: https://www.kaggle.com/competitions/name-that-llm/discussion/750213, /750768.
- Data-scale debate: https://www.kaggle.com/competitions/name-that-llm/discussion/749447.
- Train structure: https://www.kaggle.com/competitions/name-that-llm/discussion/750706.
- Grouping simulation: https://www.kaggle.com/competitions/name-that-llm/discussion/750213 (GoPain comment; notebook `locdunhat/when-does-test-time-prompt-grouping-help`).
- Early layers: https://www.kaggle.com/competitions/name-that-llm/discussion/750462.
- Claude extra data / hierarchy: https://www.kaggle.com/competitions/name-that-llm/discussion/750224.
- Paid external data: https://www.kaggle.com/competitions/name-that-llm/discussion/750947.
- Gemini-3.7 routing: https://www.kaggle.com/competitions/name-that-llm/discussion/751370.
- Runtime / Qwen3.5 load: https://www.kaggle.com/competitions/name-that-llm/discussion/751268, /750924.
- Authorship verification: https://www.kaggle.com/competitions/name-that-llm/discussion/750365.
- Fingerprints/nonlinear heads: https://www.kaggle.com/competitions/name-that-llm/discussion/750675.

**Notebooks**
- `prvsiyan/the-prompt-is-not-the-author` (V1–V5 ladder, 0.604, oracle centring 0.6447, 38,642 pool).
- `sangam21/two-stage-lora-gaussian-full-hungarian-0-604`.
- `josephadamski91/teach-the-model-10-atlas` (0.607 historical, ATLAS 10%).
- `lightningv08/0-595-two-stage-lora-gaussian-olmo2` and `0-597-two-stage-lora-olmo2-smollm3`.
- `lightningv08/name-that-llm-qwen-lora-ensemble` (nested OOF; 0.4752/0.5412/0.5716 at 1/3/6 replies).
- `kirilltym/0-600-teach-the-model-remove-the-prompt`.
- `yangkuangou/name-that-llm-the-prompt-recovery-trap` (0.501 OOF → 0.351 LB; 0.419 → 0.308).
- `mtoshidesu/test-conditional-author-decoding` (0.520; exact marginals, no gain).
- `chovyxu/math-view-of-name-that-llm`.
- `zbigniewkirchner/extra-data-hierarchical-classification-2-4-pts`.

**Datasets**
- `prvsiyan/ntl-public-in-style-pool-v1`, `prvsiyan/ntl-two-stage-lora-adapters-v1`, `prvsiyan/prvsiyans-extended-name-that-llm-dataset`.
- `nishantdahal/ntl-two-stage-lora`.
- `lightningv08/ntl-olmo2-native24-final-inference-assets` (27B/OLMo2/SmolLM3/Qwen2.5/Gemma/OOF heads).
- `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head`, `...-instyle`.
- `pjmathematician/ultrabert-380m-ntl`, `pjmathematician/ntl-train-prompts-reconstructed`.
- `rudrabhaskar/name-that-llm-qwen-fusion-heads`.
- `josephadamski91/ntl-atlas-qwen3-8b-v1`.
- `maniceet/name-that-llm-openrouter-openai-generations`.
- `seyitkaangunes/ntl-extra-api-passages`.
- `olegzholobov/name-that-llm-response-corpus`.
- `haileycheng/name-that-llm-same-prompt-passages`.
- `illiadolenko/ntl-prompt-bank-v1`.
- `kirilltym/ntl-frozen-heads-and-ce-projection`.

**Paper**
- Ko et al., *Who's Behind the Text? Robust Model Authorship Attribution* (ATLAS), arXiv:2610.19583. https://arxiv.org/abs/2610.19583

**Local artifacts**
- Pulled notebooks and dumps: `scratch/w5_notebooks/`, `scratch/w5_more/`, `scratch/w5_trap/`.
- Fusion-head ablations: `scratch/fusion_heads/qwen8_fusion_selection.json`, `qwen14_fusion_selection.json`, `qwen8_sparse_robustness.json`.
- Reconstructed prompts: `scratch/recon/infer_prompts.py`, `train_prompts_reconstructed.csv`.
- 27B/OLMo2/SmolLM3 bundle metadata: `scratch/w5_meta_lv08/dataset-metadata.json`.
- Artifact metadata: `scratch/meta_*`.

---

### Appendix A — Score-gap scenarios (probability-weighted)

| Scenario | Mechanism | Implied #1 score | Probability | Tag |
|---|---|---:|---:|---|
| A1 | 100k+ paid, format-matched corpus + 8B/14B/27B seed bags | 0.66–0.69 | 0.40 | `[LIKELY]` |
| A2 | Same corpus, 8B/14B only | 0.64–0.67 | 0.20 | `[LIKELY]` |
| B1 | Public data + contrastive/pairwise/hierarchy breakthrough | 0.64–0.67 | 0.20 | `[SPECULATIVE]` |
| B2 | 27B/multi-backbone scale on public data | 0.62–0.65 | 0.10 | `[SPECULATIVE]` |
| C | Unknown structural exploit (leak, test-specific trick) | 0.68+ | 0.10 | `[SPECULATIVE]` |

The 0.10 residual for "unknown structural exploit" is deliberate: no public evidence supports a leak, but the 2.1-point gap to #2 is large enough that it cannot be dismissed. Any claim of a leak would need evidence from the test construction, which is not public.

### Appendix B — Minimum experiment matrix for the next 10 days

| Experiment | Data | Model | Folds | Success criterion |
|---|---|---|---|---|
| E1 hierarchy | official + Seyit/Maniceet | stylegram + 8B LoRA | 5 prompt folds | +0.01 over flat 18-way on OOF and LB |
| E2 contrastive | public prompt blocks | Qwen3-8B LoRA | 5 prompt folds | +0.01 OOF at 3/6 replies |
| E3 calibration | existing OOF | none | nested | +0.005 LB or +0.01 worst-class recall |
| E4 seed bag | official + pool | 3× 8B LoRA | 5 prompt folds | +0.005 OOF, <9 h inference |
| E5 format audit | external corpora | TF-IDF probe | grouped CV | quantify route/format shift before training |
| E6 pairwise GPT/Claude/GLM | official + within-family corpora | linear on frozen features | 5 prompt folds | +0.01 on the family subset |

Run E1/E3 first: cheapest, publicly evidenced. E2 is the highest-upside training change. E4/E5 are hygiene. E6 is the targeted sibling fix that balanced accuracy rewards most.
