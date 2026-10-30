# W4 — Literature & Methods for 18-Way LLM Authorship Attribution

**Competition:** Kaggle *Name That LLM* — 18 frontier models, 40–250-word passages, balanced accuracy.
**Current state:** LoRA classifiers (Qwen3-8B/14B) + TF-IDF SVC + stylometry, public LB ≈ 0.600; public SOTA notebooks ≈ 0.604; top LB ≈ 0.688.
**Scope of this report:** what the literature says that is *actionable* on 2×T4, ≤9 h, offline. Each technique carries a measured effect, a cost estimate, a risk note, and an evidence tag.

**Tag legend**
- `[VALIDATED: n]` — effect measured on a real benchmark, with n = number of independent settings/datasets.
- `[PLAUSIBLE]` — mechanism is validated but the exact transfer to 18-way short-passage attribution is inferred.
- `[SPECULATIVE]` — no direct evidence; included only because it is cheap to test.

---

## 0. TL;DR — ranked shortlist

Deltas are **incremental balanced accuracy vs. the current 0.600 ensemble**, not additive (they overlap). Ranges are deliberately conservative.

| # | Technique | Mechanism | Strongest evidence | Expected Δ | Cost on 2×T4 / 9 h | Risk | Tag |
|---|-----------|-----------|--------------------|-----------|--------------------|------|-----|
| 1 | **OOF per-class logit bias / threshold calibration** | Tune one additive bias per class on out-of-fold logits to maximize macro-recall; constrain shifts to ±1.5 logits, shrink to 0. | CAP: CIFAR-100-LT balanced error 61.94 → 56.38 (−5.6 pp); GATC: +1.5–4 pp balanced acc across 7 model families; GLA/GCA: CIFAR-100-LT 61.53 → 53.68 (−7.9 pp). | **+0.5 – 1.5 pp** | Minutes; no GPU | Overfits OOF if unconstrained; use nested folds and shrinkage. | `[VALIDATED: 3]` mechanism, `[PLAUSIBLE]` magnitude |
| 2 | **Multi-layer + late-interaction readout** | Instead of last-layer mean pooling: concatenate mean/max of layers ~8/16/24 **and** a ColBERT-style MaxSim over token states; train a small head. | Where Does Authorship Signal Emerge: 4× performance gap from scoring mechanism alone; late interaction defers consolidation to later layers. LIGHT: all-layer fusion = new SOTA OOD, +50% R@8 Fanfiction, MRR 18.03 vs 14.65. | **+0.5 – 2.0 pp** | Medium: feature extraction 30–45 min; head retrain 1–2 h | More features → overfit on 2.7k rows; keep head linear / low-rank. | `[VALIDATED: 2]` components, `[PLAUSIBLE]` combined |
| 3 | **Prompt-group soft prior correction (no hard assignment)** | Within each test prompt group, estimate class composition with a kNN pre-pass over embeddings; shift each sample's logits toward the estimated group prior; shrink toward global prior and cap the shift. Never solve a hard assignment. | OT marginal-capacity: macro-F1 0.322 → 0.623 with composition-matched marginal; kNN pre-pass closes 69% of oracle gap. Hard Hungarian on our data: LB 0.353 → 0.344. | **+0.5 – 2.0 pp** | Low: minutes after embeddings exist | Test groups are *incomplete* rounds; a wrong composition estimate can hurt. Paper shows benefit can go negative when coverage is already sufficient → apply only to low-coverage classes/groups. | `[VALIDATED: 1]` OT, `[PLAUSIBLE]` here |
| 4 | **Contrastive hard-negative auxiliary loss (same-prompt negatives)** | Add InfoNCE/ supervised-contrastive term to the LoRA CE loss; positives = same model on other prompts, hard negatives = *other models on the same prompt* (topic-controlled). | Isolating authorship from content: blogs 0.3928 → 0.4469 (+5.4 pp); top-k diverse authors 0.7062 → 0.8158 (+11 pp); preserves zero-shot. | **+0.5 – 1.5 pp** (largest on sibling pairs) | Medium: one retrain (~2–3 h) | Loss weight needs tuning; hard negatives from the same prompt are exactly our confound, so this is the highest-leverage training change. | `[VALIDATED: 2]` AA, `[PLAUSIBLE]` here |
| 5 | **More data + length-truncation augmentation** | Add the external 18-model corpus; during training randomly crop 20% of samples to 40–80 words. | Idiosyncrasies: 10 samples → 40.3% (5-way); accuracy saturates ~256 tokens; short-text accuracy is our known weak slice (~0.27 <60 words vs ~0.46 >160). | **+1.0 – 3.0 pp** | High: data curation + 2–3 retrains | External corpus must be label-compatible and prompt-disjoint; truncation can destroy punctuation cues if overdone. | `[VALIDATED: 1]` data scale, `[PLAUSIBLE]` truncation |
| 6 | **Layer-selective LoRA / asymmetric rank** | Apply LoRA only to the layers with highest gradient/Fisher signal; allocate higher rank to early-mid layers. | Aletheia: 81 experiment rows, 14 models, consistent gains over uniform LoRA; FoRA: beats LoRA/DoRA with fewer adapted layers; LIGHT: early-mid layers carry style. | **+0.3 – 1.0 pp** at lower cost | Low–medium: one probe + retrain | Most evidence is generative/NLU, not authorship; measure on grouped CV. | `[PLAUSIBLE]` |
| 7 | **Hierarchical auxiliary head (lab → sibling)** | Multi-task: predict 6 labs and 18 models jointly; use lab logits as an auxiliary loss, not a hard cascade. | FAID: family-level contrastive + auxiliary task improves unseen-domain/generator accuracy (62.78% / 66.55%); our lab-level signal is strong (~60% linear). | **+0.3 – 1.0 pp** | Low: same backbone, extra head | Hard cascade propagates lab errors; only use as auxiliary. | `[PLAUSIBLE]` |
| 8 | **ATLAS-style per-source likelihood rerank (ambiguous subset only)** | For the lowest-confidence 5–10% of samples, score each candidate model's likelihood under a source-conditioned adapter and rerank. | ATLAS: 74.27% vs 65.27% (+9.0 pp) under joint domain+version shift. | **+0.2 – 0.8 pp** | Very high: 18 forward passes/sample; 4k × 18 is infeasible in 9 h on 8B → restrict to ≤300 samples or a 1.5B scorer | Budget blow-up; adapter quality on 2.7k rows. | `[VALIDATED: 1]` mechanism, `[SPECULATIVE]` budget |
| 9 | **Backbone/seed/feature-view ensemble** | Average calibrated probabilities from Qwen3-8B + Qwen3-14B + TF-IDF SVC + stylometry GBM, with rank averaging. | Public notebooks at 0.604 vs top 0.688 imply ~8 pp is available from engineering, not one trick; Idiosyncrasies shows embedding choice matters (LLM2vec 97.1 vs BERT 91.1). | **+0.5 – 1.5 pp** | High: 2–3 models, but inference still <30 min for 4k | Correlated errors; calibrate before averaging. | `[PLAUSIBLE]` |
| 10 | **Hard Hungarian / balanced assignment per prompt** | Solve linear sum assignment on each prompt cluster. | Directly tested here: CV +0.10, **LB −0.009** (0.353 → 0.344). OT paper explains why: incomplete groups and composition mismatch. | **Negative** | Low | Do not ship. | `[VALIDATED: negative]` |

**Recommended order of work:** #1 → #3 → #2 → #4 → #5. Items #1 and #3 are cheap enough to test in one day and are independent of retraining.

---

## 1. Thread 1 — LLM authorship / source attribution architectures

### 1.1 The base fact: LLM outputs are strongly separable in-domain

**Idiosyncrasies in Large Language Models** (Sun et al., 2025; arXiv:2502.12150) is the cleanest evidence that this task is tractable with a discriminative encoder:

- 5-way classification (ChatGPT, Claude, Grok, Gemini, DeepSeek): **97.1%** held-out accuracy with a fine-tuned LLM2vec embedding model; chance = 20%.
- Instruct models 96.3%, base models 87.3%.
- **Within one family, across 4 Qwen-2.5 sizes: 59.8%** — sibling discrimination is much harder, matching our error analysis (55–58% of errors are sibling confusions).
- 10-way (5 chat + 5 instruct, incl. Phi-4): **92.2%**.
- OOD prompts: "strong out-of-distribution generalization" to prompts outside the training distribution.
- Data scale: as few as **10 training samples → 40.3%** 5-way; saturates around **10k samples**.
- Length: accuracy improves with tokens and **saturates beyond ~256 tokens**; for chat/instruct models, ~50% accuracy from a **single token** (mostly style/markup priors).
- Transformations: word shuffling causes only a slight drop; rephrasing/translation still >90%; summarization stays above chance → signal is both word-distributional and semantic.

**Implication for us:** a fine-tuned 7–8B encoder/decoder readout is the right backbone; 384 tokens is sufficient; the 18-way problem is dominated by sibling separation, not by lab separation. This is consistent with our lab-vs-sibling error split.

### 1.2 Source-conditioned likelihood (ATLAS)

**Who's Behind the Text? Robust Model Authorship Attribution** (arXiv:2610.19583) treats attribution as a likelihood ratio rather than a 18-way softmax:

- One LoRA + learned prefix per source model; score = clipped aggregate of token-level log-likelihood under each source-conditioned model.
- **74.27% vs 65.27%** (+9.0 pp) under joint domain + version shift on MAABench.
- Cost: 18 adapters and **18 forward passes per sample**.

**Verdict:** the mechanism is the best published answer to *version shift* (our gpt-5.6-luna vs sol vs terra problem), but the cost is prohibitive for 4,000 samples. Use only as a reranker on a small ambiguous subset, or distill the 18 likelihoods into a student.

### 1.3 Multi-level contrastive + auxiliary task (FAID)

**FAID** (arXiv:2505.14271) fine-tunes XLM-R with:

- multi-level contrastive loss enforcing that same-family pairs are closer than cross-family pairs, and human–LLM collaborative text sits between;
- an auxiliary human-vs-LLM binary head.

Results: in-domain English 96.41% accuracy / 95.77 macro-F1; **unseen domains 62.78%**, **unseen domain + generator 66.55%**. The drop is the honest number for OOD; the paper's contribution is that the auxiliary + contrastive terms improve OOD relative to CE-only baselines.

**Implication:** add a **lab-level auxiliary head** and a **same-prompt hard-negative contrastive term**; do not expect in-domain numbers to transfer to unseen prompts.

### 1.4 Triplet margin on embeddings (T5-CSBoost)

**T5-CSBoost** (arXiv:2607.14113) adds a margin triplet loss (λ = 0.5, margin = 1.0) on decoder embeddings to a T5 source-attribution model:

- MAGE average: T5-Sentinel 86.29 → **88.47** (+2.2 pp).
- **Paraphrasing task: 63.87 → 68.19** (+4.3 pp).
- Unseen domains: 89.55 → **93.67** (+4.1 pp).
- Under 30% word synonym replacement: **>0.998** accuracy, vs RoBERTa 0.938 → 0.508.

**Implication:** a cheap geometric regularizer buys robustness to the kind of surface variation that separates sibling models. Combine with #4 rather than using both separately.

### 1.5 What does *not* transfer

- **TRACE / GhostWriteBench** (arXiv:2603.28054): token-rank transition fingerprints for 50k-word books. Long-form; not applicable to 250-word passages.
- **SiBot** (PMC13552777): 22-class hybrid reports 92.38% under prompt-aware 10-fold vs 68.2% for TF-IDF+RF. Treat with caution: the class set and label semantics differ from ours; the useful signal is that prompt-aware splitting still permits high accuracy when the corpus is large and classes are coarse.
- **Watermark / perplexity attribution:** no evidence it survives the competition's "authentic completion" preprocessing; skip.

---

## 2. Thread 2 — Representation readout: layers, pooling, late interaction

### 2.1 The scoring mechanism matters as much as the backbone

**Where Does Authorship Signal Emerge in Encoder-Based Language Models?** (arXiv:2605.19908):

- Holding the backbone fixed, changing only the **scoring/readout** produces a **four-fold performance gap**.
- **Mean pooling forces style consolidation in early-mid layers**; **late interaction (ColBERT-style MaxSim) defers the decision to later layers**.
- Stylistic features are **linearly available at every layer**, but the readout determines which layer the model actually uses.

**Implication:** our current "last hidden state → mean pool → linear" readout is likely leaving 1–2 pp on the table. Two cheap alternatives:

1. **Layer concatenation:** mean-pool layers 8/16/24 (and maybe 32), concatenate, L2-normalize, linear head.
2. **Late interaction:** for each class, store a small set of token embeddings (or a learned class token bank) and score with MaxSim; average with the mean-pool logits.

### 2.2 All-layer fusion is SOTA OOD

**Layered Insights / LIGHT** (arXiv:2503.00958):

- Fuses representations from multiple transformer layers instead of one.
- New SOTA on out-of-domain authorship attribution; **+50% R@8 on Fanfiction**; **MRR 18.03 vs 14.65** (LUAR) on Reddit transfer.
- Analysis: different layers specialize in different stylistic abstractions; OOD robustness comes from the fusion, not from a single "best" layer.

**Implication:** supports #2 above, and explains why our early-mid-layer probe (0.39) beat the final layer (0.31): the final layer has abstracted away from style. Fuse rather than choose.

### 2.3 Patch/token granularity

**HALvest-Contrastive** (arXiv:2407.20595):

- Retrieval-style attribution with patch-level late interaction.
- Optimal patch size **n\* ≈ 0.18·√S** (S = token count) — for S ≈ 200–350, n\* ≈ 2.5–3.4 tokens.
- Token-level late interaction is competitive; whole-word patches are mid; fixed large patches lose signal.

**Implication:** if we implement MaxSim, use **word-level or 2–3-token patches**, not sentence-level pooling. With 384 tokens and batch 8–16, the similarity matrix is small; cost is dominated by the backbone, not the readout.

### 2.4 First-token / head-only readouts

**CodeT5-Authorship** (arXiv:2506.17323) uses an encoder-only model with the **first token** plus a 2-layer classification head; competitive with much larger decoder models on code authorship.

Combined with the Idiosyncrasies finding that ~50% of 5-way accuracy is available from the first token, this suggests a **cheap auxiliary feature**: the first 16–32 tokens' hidden states. It is nearly free and may capture greeting/formatting priors that sibling models share or differ on.

### 2.5 Practical recipe (ranked by cost/benefit)

1. Extract hidden states at layers 8, 16, 24, 32 for all train/test passages at 384 tokens (one forward pass, no grad).
2. Build three views: mean-pool concat, max-pool concat, MaxSim over 3-token patches.
3. Train a logistic regression / small MLP per view on grouped folds; average calibrated probabilities.
4. Compare against the current single-layer mean-pool baseline on **prompt-grouped** CV, not random CV.

---

## 3. Thread 3 — Transductive / semi-supervised methods for grouped test data

### 3.1 Why hard assignment fails here

Our own result is the strongest evidence: Hungarian assignment on clustered test prompts **raised CV by +0.10 but dropped LB from 0.353 to 0.344**. The test set is a **subsample of prompts**, so a group does not contain exactly one example per class. Hard one-per-class constraints then force wrong reassignments.

The OT literature explains the failure precisely. **Disentangling Marginal Capacity from Geometry in Optimal-Transport Label Transfer** (arXiv:2610.14853):

- Under uniform source weighting, each class's **transport capacity is proportional to its source frequency**. When the target composition differs, a class can be starved even when the embedding separates it well.
- Starved mass is absorbed by large classes **with high confidence**, so output-side statistics and transport entropy do not reveal the error.
- Correcting the marginal to the target composition: **macro-F1 0.322 → 0.623**, micro recall 0.393 → 0.718 on single-cell data.
- A **kNN pre-pass** composition estimator closes **69%** of the oracle gap on single-cell and **59%** on Office-Home.
- BBSE / RLLS / MLLS close only **4.7–17.8%** on single-cell and 29–43% on Office-Home; they assume p(x|y) is invariant, which a prompt/domain shift violates.
- **Important negative result:** the correction's benefit is concentrated in **capacity-limited classes** and can become **negative when coverage is sufficient**. Apply selectively.
- Design rule: floor estimated shares **relative to uniform (0.5/K)**, never with an absolute floor — otherwise a class that is never predicted stays at zero recall forever.

**Implication:** the right transductive operation is **soft, partial, and composition-aware**, not a hard matching. Per prompt group:

1. Get embeddings and base probabilities.
2. Estimate the group's class composition with a kNN vote over the training set (or a small EM/MLLS step), floored at 0.5/18.
3. Compute a log-prior shift `log q_group(k) - log p_global(k)`, multiply by a temperature α, and **cap** the shift at ±1.0–1.5 logits.
4. Shrink toward the global prior by group size: α_eff = α · n_group/(n_group + τ).
5. Skip the correction for groups with < 4–6 passages, or where the base classifier's entropy is already low.

### 3.2 Transductive MAP / PT-MAP

**PT-MAP** (arXiv:2006.03806) alternates pseudo-label refinement with a power-transform + Sinkhorn-style normalization:

- 1-shot miniImageNet: PT alone 75.60 → **PT+MAP 82.92** (+7.3 pp).
- Assumes a **balanced query set**; performance degrades when the query composition is skewed.

**Implication:** PT-MAP is the strongest published transductive method for *balanced* groups. Our groups are unbalanced and incomplete. Use the *soft* version (a few iterations of confidence-weighted prototype re-estimation + Sinkhorn with a floored marginal), not the hard balanced version.

### 3.3 Sinkhorn label allocation and unbalanced OT

- **SLA** (arXiv:2102.08622): reinterprets self-training as OT between examples and classes with **flexible upper-bound constraints** on class proportions — exactly the "partial assignment" primitive we need.
- **OTAMatch** (PMID:39008384): OT assignment with ε-greedy posterior regularization and curriculum bias correction for noisy pseudo-labels.
- **scOT-LT** (PMC13313528): unbalanced OT tolerates composition mismatch between source and target; relevant because our test mixture differs from train.
- **Label propagation through OT** (arXiv:2110.01446): graph-based propagation with OT-derived transition costs; a cheap alternative if we build a kNN graph over test+train embeddings.
- **FlexDA** (arXiv:2306.04621): flexible distribution alignment; useful if we want a single hyperparameter controlling how far the test prior may deviate.

### 3.4 Label-shift estimation

- **BBSE** (arXiv:1802.03916): confusion-matrix inversion to estimate target priors; cheap, but assumes label shift only (p(x|y) fixed).
- **FMAPLS** (arXiv:2511.18615): Bayesian online label-shift tracking with Dirichlet priors; overkill here.
- **Open-set label shift** (arXiv:2505.05868): relevant if a test passage comes from a model variant not in the 18; but the competition guarantees the label set, so skip.

**Recommended implementation:** kNN composition estimate + capped log-prior shift. It is ~50 lines of NumPy, needs no training, and is evaluated inside the grouped CV loop.

---

## 4. Thread 4 — Balanced-accuracy optimization

Balanced accuracy is a **macro-recall** metric. The literature is unambiguous that (a) training loss, (b) decision thresholds, and (c) class priors each need separate treatment.

### 4.1 Theoretically grounded balanced losses (GLA/GCA)

**Improved Balanced Classification with Theoretically Grounded Loss Functions** (Cortes, Mohri, Zhong; arXiv:2512.23947):

- **GLA** generalizes logit adjustment to the general cross-entropy family.
- **GCA** adds **class-dependent confidence margins** to class-weighted losses; its consistency bound scales as 1/√p_min instead of 1/p_min.
- CIFAR-100-LT (ρ = 1000): CE 38.45 → GCA **33.17** balanced error (−5.3 pp); GLA 33.99.
- CIFAR-100-LT (ρ = 100): CE 61.53 → GLA **53.68** (−7.9 pp); GCA 54.02.
- Tiny ImageNet (ρ = 1000): CE 70.23 → GCA **64.88** (−5.4 pp).
- Class-weighted CE and focal loss are *worse* than CE in several settings.

**Implication:** if we retrain, replace plain CE with **GLA (logit adjustment)** first; it is a one-line change (`logits + τ·log prior`) and is the best average performer. Add **GCA margins** only if we can tune per-class margins on OOF.

### 4.2 Class-attribute priors (CAP)

**Class-attribute Priors** (arXiv:2401.14343):

- Learns a **per-class margin/prior** from class attributes (e.g., difficulty), not just frequency.
- CIFAR-100-LT: CE 61.94 → CAP **56.38** Err_bal (−5.6 pp); beats frequency-only and AutoBalance.
- Key insight: **balanced accuracy is optimized by class-specific bias, and frequency is not the right attribute** — class difficulty/heterogeneity is.

**Implication:** our 18 classes differ enormously in difficulty (sibling pairs vs singleton labs). A single global prior correction is suboptimal. Fit **one bias per class** on OOF, initialized at `-log p(y)` and then free-tuned with shrinkage. This is technique #1 in the shortlist.

### 4.3 Group-aware threshold calibration

**Group-Aware Threshold Calibration** (arXiv:2509.02592):

- Group-specific thresholds (groups defined by class or metadata) improve **balanced accuracy by 1.5–4%** over SMOTE/CT-GAN across 7 model families.
- The gain comes from aligning thresholds to **per-group recall**, not from resampling.

**Implication:** if we can cluster the 18 models into sibling groups (e.g., GPT-5.6 variants, Claude Opus variants), calibrate thresholds **within group** rather than globally. This directly targets the 55–58% sibling-error share.

### 4.4 Logit adjustment baseline

**Menon et al. logit adjustment** (arXiv:2007.07314, referenced by both CAP and GLA/GCA) is the canonical `logit + τ log prior` method. τ = 1 is Bayes-optimal for balanced error under label shift; τ is often tuned down (0.5–1.0) in practice.

### 4.5 Practical calibration protocol

1. Collect **OOF logits** with the exact grouped folds used for model selection.
2. Fit per-class biases by coordinate ascent on **macro-recall** (not log-loss), with:
   - initialization `b_k = -log p_k`;
   - L2 shrinkage toward the initialization, strength chosen by a nested fold;
   - hard cap `|b_k - b_k^init| ≤ 1.5`.
3. Fit a single temperature on OOF log-loss (does not change argmax but fixes ensemble averaging).
4. Re-check per-class recall and the sibling confusion matrix; a gain in macro-recall that comes from one class cannibalizing another is a red flag.
5. Apply the same biases to the test logits **before** any transductive correction.

---

## 5. Thread 5 — Fine-tuning recipe for attribution

### 5.1 LoRA configuration that fits 2×T4

| Knob | Recommendation | Rationale |
|------|----------------|-----------|
| Base | Qwen3-8B (primary), Qwen3-14B (secondary) | 8B fits one T4 in fp16 with room for activations; 14B needs `device_map="auto"` + `max_memory={0:"14GiB",1:"14GiB"}`. No bf16 on Turing → `torch.float16`. |
| Rank | r = 16, α = 32, dropout 0.05 | Standard; r = 32 only if underfitting. |
| Target modules | all linear (`q,k,v,o,gate,up,down`) | Style signal is distributed; restricting to q/v loses recall. |
| Max length | 384 | 250 words ≈ 330 tokens; Idiosyncrasies saturates at 256. |
| Batch | 8 with grad accumulation 2 (effective 16) | Fits 16 GB; sort by length to minimize padding. |
| LR | 1e-4 (LoRA), cosine, 3–5% warmup | Standard for 7–8B LoRA. |
| Epochs | 3–4 | 2,699 rows; early stop on grouped val macro-recall. |
| Precision | fp16 + gradient checkpointing | Required on T4. |

**Cost estimate:** ~1.5–2.5 h per 3-epoch run on 2×T4 for 8B; merge adapters before inference; 4k test passages at 384 tokens ≈ 15–20 min.

### 5.2 Contrastive hard negatives (highest-leverage training change)

**Isolating authorship from content** (arXiv:2411.18472) is the direct template:

- InfoNCE with **hard negatives synthesized from a semantic-similarity model**, plus style/content separation terms.
- Blogs: simple fine-tune 0.3928 → detangle **0.4469** (+5.4 pp); category-balanced evaluation 0.4192 → 0.4423.
- Top-k most diverse authors: 0.7062 → **0.8158** at k=10 (+11 pp); 0.6930 → 0.7742 at k=25.
- Fanfiction: 0.7741 → 0.7924 (+1.8 pp).
- Zero-shot capability preserved on PAN challenges.

**Our adaptation:** for each anchor passage, positives = same model on *other* prompts (or another passage from the same prompt if available), hard negatives = **other models on the same prompt**. This is exactly the topic-controlled negative that the paper argues for, and it is free — we already have prompt IDs.

Loss: `L = CE + λ·InfoNCE`, λ ∈ {0.1, 0.3, 0.5}; temperature 0.07; batch composition must guarantee same-prompt negatives (use a prompt-balanced sampler).

### 5.3 Auxiliary heads

- **Human-vs-LLM head:** not useful here (all passages are LLM), but a **lab head** (6 classes) is. FAID shows auxiliary tasks improve OOD.
- **Family/sibling hierarchy:** predict lab and model jointly; weight the lab loss at 0.2–0.3. Do **not** hard-cascade (a lab error is unrecoverable).
- **Length head:** predict the word-count bucket; a small auxiliary task that forces the encoder to represent length, which is a known confound. Optional.

### 5.4 Augmentation: what to do and what not to do

**Do:**
- **Random truncation** to 40–80 words on ~20% of samples. Justification: our accuracy is much lower on <60-word passages (~0.27 vs ~0.46), and Idiosyncrasies shows 256 tokens is sufficient but shorter inputs lose accuracy gradually. Truncation forces the model to use early-token style cues.
- **Length-stratified batching** so short passages are not drowned out.
- **Prompt-level mixup of negatives** (same-prompt hard negatives) as in §5.2.

**Do not:**
- **Paraphrase augmentation.** T5-CSBoost shows paraphrasing is a severe distribution shift for attribution (T5-Sentinel 63.87% on the paraphrase task vs 92.71% in-domain). Paraphrasing training text destroys the fingerprint we are trying to learn.
- **Back-translation** for the same reason.
- **Synonym replacement** as augmentation; it is a *test-time robustness* perturbation, not a training signal.

**Evidence status:** truncation is `[PLAUSIBLE]` (mechanism well supported, direct A/B on this task absent); paraphrase harm is `[VALIDATED: 2]` (T5-CSBoost; Isolating authorship treats content perturbations as the enemy).

### 5.5 Data scale and external corpora

Idiosyncrasies: 10 samples → 40.3%; performance rises with data and saturates ~10k. We have 2,699 rows (≈150 per class). The external 18-model corpus (~1,902 prompt groups × 18) is therefore likely worth **+1–3 pp**, but only if:

- labels use the same 18 identifiers;
- prompts are disjoint from the competition's 150 train prompts and do not overlap test domains;
- passages are filtered to 40–250 words and single-paragraph;
- the external corpus's generation settings (temperature, system prompt) are close to the competition's.

**Risk:** a corpus generated with different prompts/styles can teach *topic* shortcuts. Validate with a **prompt-grouped** holdout and a **cross-corpus** holdout (train on external, test on competition) before mixing.

### 5.6 Distillation from API models / LLM judges

No paper found that distilling a large judge into a small attribution classifier improves 18-way accuracy. The nearest evidence is negative or orthogonal:

- LLM judges are competitive on coarse style attribution (arXiv:2510.13898) but expensive and poorly calibrated.
- ATLAS's per-source likelihood is a better use of compute than a judge.
- **Verdict:** skip distillation; spend the budget on #4 and #5.

### 5.7 Sibling-pair auxiliary loss

No direct paper on pairwise sibling losses for LLM attribution. The mechanism is supported by:

- FAID's multi-level contrastive hierarchy (family → model);
- Isolating authorship's hard negatives;
- our own confusion structure (sibling errors dominate).

A cheap version: add a **pairwise logistic loss** for the top-5 confused sibling pairs (e.g., `glm-5.3` vs `glm-5.3-flash`), using their logit difference as the score. Weight 0.1–0.2. Tag `[SPECULATIVE]`; test only if #4 is already in.

---

## 6. Thread 6 — Verification, leakage, and protocol

### 6.1 Topic/prompt leakage inflates scores

**Addressing Topic Leakage in Cross-Topic Evaluation for Authorship Verification** (arXiv:2407.19164):

- Random topic sampling leaves **mean topic cosine similarity 0.836** between train/test; heterogeneity-informed sampling (HITS) lowers it to **0.766** (max 0.920 → 0.837).
- Models that rely on topic shortcuts can lose **0.25 F1** (0.80 → 0.56) under HITS.
- Rankings are unstable across random topic draws: the same model set can produce different top performers.

**The Topic Confusion Task** (arXiv:2104.08530):

- When the author–topic pairing is flipped between train and test, **pretrained language models have much larger cross-group (topic-driven) error** than stylometric + POS features.
- Stylometric + POS + word n-grams reaches **83.3%** cross-topic accuracy vs 80.9% for character-masking (P = 0.04).
- Masking infrequent words reduces cross-group error by ~10–15%.

**Our empirical numbers are the same phenomenon:** grouped CV 0.217 vs random CV 0.369 for the linear baseline; LoRA ensemble CV 0.77 → LB 0.54. The 0.23 gap is prompt leakage, not model quality.

### 6.2 Mandatory protocol

1. **StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)** grouped on `prompt_id`, stratified on `model`.
2. **All** preprocessing, feature selection, calibration, and post-processing fitted **inside** the fold. In particular, the per-class biases and the transductive prior must be fit on the training folds and applied to the validation fold.
3. Report:
   - overall balanced accuracy;
   - **per-class recall** (18 numbers);
   - **sibling confusion matrix** (5 GPT, 4 Gemini, 5 Claude, 2 GLM, singletons);
   - **length-stratified** accuracy (<60, 60–100, 100–160, >160 words);
   - **lab-level** accuracy and sibling-conditional accuracy.
4. **Simulate the hidden test's incompleteness:** for each validation prompt group, randomly drop 20–50% of its passages, then evaluate the transductive correction. A method that only works on complete 18-model rounds is not deployable.
5. **Nested selection:** use an inner grouped fold to choose λ, α, bias caps; never tune on the same OOF logits used to report.
6. **Adversarial check:** train on competition data, validate on external corpus (and vice versa). A large drop signals topic/style shortcut learning.

### 6.3 What to log for every experiment

- CV macro-recall and per-class recall.
- Public LB (when submitted).
- The **CV→LB gap**. A method that improves CV but widens the gap (e.g., Hungarian) is a leakage artifact, not a gain.
- Runtime and GPU memory; 9 h is a hard ceiling.

---

## 7. What the literature does **not** answer (gaps)

1. **No paper on "incomplete rounds" assignment.** The OT literature assumes balanced or known-composition target sets; the competition's test prompts are subsampled. The closest is SLA's flexible upper bounds and scOT-LT's unbalanced OT, but neither is evaluated on "≤1 example per class per group with unknown missingness."
2. **No direct study of 18-way sibling attribution among frontier models.** Idiosyncrasies stops at 5-way and 4 Qwen sizes; ATLAS is 2-way source verification under shift; FAID is family-level. Our 18-way, 5-sibling-per-lab setting is unpublished.
3. **No measured effect of truncation augmentation on authorship attribution.** The mechanism is supported by length-sensitivity results, but the A/B is missing. Treat as `[PLAUSIBLE]`.
4. **No paper on prompt-grouped transductive calibration for this task.** HITS/topic-confusion establish the leakage; they do not propose a deployable correction.
5. **No public write-up of the 0.688 LB solution.** The gap between 0.604 (public notebooks) and 0.688 (top) is undocumented; likely ensemble diversity + external data + calibration, but unverified.
6. **No evidence that LLM-as-judge or distillation beats a fine-tuned 8B encoder** for 18-way short-passage attribution. The cost/benefit is unfavorable.

---

## 8. References

**Attribution / LLM fingerprints**
- Sun et al., *Idiosyncrasies in Large Language Models*, arXiv:2502.12150.
- Huertas-Tato et al., *Isolating authorship from content with semantic embeddings and contrastive learning*, arXiv:2411.18472.
- *Who's Behind the Text? Robust Model Authorship Attribution* (ATLAS), arXiv:2610.19583.
- *FAID: Fine-Grained AI-Generated Text Detection Using Multi-Task Auxiliary and Multi-Level Contrastive Learning*, arXiv:2505.14271.
- *T5-CSBoost: Adversarial Perturbation Resistant LLM Fingerprinting*, arXiv:2607.14113.
- *Where Does Authorship Signal Emerge in Encoder-Based Language Models?*, arXiv:2605.19908.
- *Layered Insights: Generalizable Analysis of Authorial Style by Leveraging All Transformer Layers* (LIGHT), arXiv:2503.00958.
- *HALvest-Contrastive*, arXiv:2407.20595.
- *CodeT5-Authorship*, arXiv:2506.17323.
- *Authorship Attribution in the Era of LLMs: Problems, Methodologies, Challenges*, arXiv:2408.08946.
- *Attribution Quality in AI-Generated Content: Benchmarking Style Embeddings and LLM Judges*, arXiv:2510.13898.
- *Who Wrote the Book? Detecting and Attributing LLM Ghostwriters* (GhostWriteBench / TRACE), arXiv:2603.28054.
- *From Text to Source: Results in Detecting Large Language Model-Generated Content*, arXiv:2309.13322.

**Transductive / OT / label shift**
- *Disentangling Marginal Capacity from Geometry in Optimal-Transport Label Transfer*, arXiv:2610.14853.
- *PT-MAP: Leveraging Feature Distribution in Transfer-based Few-Shot Learning*, arXiv:2006.03806.
- *Sinkhorn Label Allocation*, arXiv:2102.08622.
- *OTAMatch*, PMID:39008384.
- *scOT-LT: Optimal transport for label transfer in single-cell multi-omics integration*, PMC13313528.
- *Label Propagation Through Optimal Transport*, arXiv:2110.01446.
- *FlexDA*, arXiv:2306.04621.
- *BBSE: Detecting and Correcting for Label Shift with Black Box Predictors*, arXiv:1802.03916.
- *FMAPLS*, arXiv:2511.18615.
- *Open-set Label Shift*, arXiv:2505.05868.

**Balanced classification / calibration**
- Cortes, Mohri, Zhong, *Improved Balanced Classification with Theoretically Grounded Loss Functions*, arXiv:2512.23947.
- *Class-attribute Priors* (CAP), arXiv:2401.14343.
- *Group-Aware Threshold Calibration*, arXiv:2509.02592.
- Menon et al., *Long-tail learning via logit adjustment*, arXiv:2007.07314.

**Protocol / leakage**
- *Addressing Topic Leakage in Cross-Topic Evaluation for Authorship Verification* (HITS / RAVEN), arXiv:2407.19164.
- *The Topic Confusion Task: A Novel Evaluation Scenario for Authorship Attribution*, arXiv:2104.08530.

**Fine-tuning / PEFT**
- *Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures*, arXiv:2604.15351.
- *FoRA: Fisher-orthogonal Rank Adaptation for Parameter-Efficient Fine-Tuning*, arXiv:2605.29317.
- *RDP LoRA: Geometry-Driven Identification for Parameter-Efficient Adaptation*, arXiv:2604.19321.

---

## 9. One-page implementation plan

| Day | Work | Success gate |
|-----|------|--------------|
| 1 | OOF per-class bias calibration; measure on grouped CV + length strata | Macro-recall gain ≥ 0.3 pp without any class recall dropping > 1 pp |
| 1 | Soft prompt-group prior correction with kNN composition; simulate 20–50% missingness | Gain ≥ 0.3 pp on incomplete-group simulation, no loss on complete groups |
| 2 | Multi-layer + MaxSim readout head; compare to single-layer mean pool | Gain ≥ 0.5 pp on grouped CV |
| 2–3 | LoRA retrain with same-prompt hard-negative InfoNCE + lab auxiliary head | Gain ≥ 0.5 pp over CE-only, same seed/data |
| 3–4 | Add external corpus + truncation augmentation; ablations on 10% / 20% / 40% crop | Gain ≥ 0.5 pp; CV→LB gap does not widen |
| 4 | Layer-selective LoRA / rank allocation | Gain ≥ 0.2 pp at equal or lower runtime |
| 5 | Ensemble + calibration; final submission validation | `submission.parquet` schema + 18-label check; runtime < 30 min |

**Hard rules**
- No hard Hungarian assignment.
- No paraphrase augmentation.
- No post-processing fitted on test.
- Every claimed gain reported with per-class recall and the CV→LB gap.
