# W1 — Forensic dissection of the top public "Name That LLM" notebooks

**Date:** 2026-10-21
**Author:** W1 (research worker)
**Scope:** every public notebook with a credible score ≥0.59, plus the direct upstream lineage (0.541–0.574) needed to attribute gains.
**Our reference:** local pipeline at **0.600 public LB** (two-stage Qwen3-8B + 14B LoRA, 4-view TF-IDF LinearSVC, centred twins, sibling stylometry, length-gated typographic priors, global Sinkhorn).
**Method:** pulled each notebook with `kaggle kernels pull -m` into `scratch/w1_pull*/`, converted cells to `_code.py` / `_md.md`, diffed each against the 0.592 base, read author claims, and re-checked train-only invariants against `data/train.csv`.

> **Reading the numbers.** The public LB is roughly 20–30% of ~4,000 hidden rows, i.e. ~800–1,200 rows. A one-split delta below ~0.01 is inside sampling noise. Rankings below therefore weight *paired/OOF evidence and replication across notebooks*, not the headline LB digit alone. Every score in the tables is **claimed by the author**; none is independently reproducible from here because the hidden labels are unavailable.

---

## 1. TL;DR — what actually moved the needle

1. **The public frontier is one lineage, not many independent ideas.** 0.592 (Nishant) → +Gaussian readout 0.595–0.604 (prvsiyan / kirilltym) → +10% ATLAS 0.607 (joseph) → post-Sinkhorn invariants + Hungarian 0.604–0.612 (sangam / kiza). Most "new" notebooks are forks of these three moves.
2. **The single most credible technique is the shared-prompt Gaussian head.** It replaces sample-mean centring inside inferred prompt groups with a model that integrates the prompt effect out. Measured **+0.0100 [0.0026, 0.0172]** pooled on 2/3-passage fragments, **+0.0408** complete-prompt accuracy, and it beat the matching centred head in **68/68** view-by-scenario cells (prvsiyan). It moved 0.599 → 0.604 and 0.592 → 0.595.
3. **The largest replicated LB gain is the ATLAS 10% blend.** A Qwen3-8B-Base adapter trained on the official 2,699 rows, scoring source-conditioned token likelihood, blended once at 10% before Sinkhorn. **+0.007 on the exact 0.600 base**, and **7/7 positive** across unrelated notebooks (+0.002 to +0.007).
4. **In-style extra data is the best *member* upgrade.** A Qwen3-8B LoRA trained on the official rows + **9,818 extra passages from 709 prompts** raised held-out member accuracy **0.456 → 0.516** and LB **0.569 → 0.574** at weight 0.17 + twin.
5. **Treat the post-Sinkhorn "shields" and Hungarian forcing as high-risk.** The train invariants are real (verified below: 86/86 spaced em-dashes in 4 classes; 0 curly quotes in 1,049 Claude+GLM rows), but the only *measured* punctuation veto was **+0.0004 OOF**, two public forks lost **0.003–0.005**, and our own earlier Hungarian test **degraded LB 0.353 → 0.344** despite inflating CV.
6. **Nothing here changes the lab-vs-sibling structure.** Every top notebook is still a lab classifier plus within-family re-decision; the gains come from prompt-effect removal, extra in-style data, and one orthogonal member (ATLAS), not from a better 18-way semantic classifier.

---

## 2. Score ladder (claimed) and lineage

| Claimed LB | Notebook | Mechanism relative to parent | Claim source | Verification status |
|---:|---|---|---|---|
| **0.612** | `kiza123123/ntl-0612-apex-invariants-hungarian` | copy of 0.604 fork + curly-quote hard invariant + Hungarian widened to clusters 9–18 | title token "0612" only (no markdown, no explicit score statement) | **Unverified**; score digit inferred from title convention, no version link |
| **0.607** | `josephadamski91/teach-the-model-10-atlas` | kirilltym v20 (0.600) + ATLAS 10% pre-Sinkhorn | md + scored v1 link (360203672) | Claim internally consistent; old-version endpoint 404 prevents source binding |
| **0.604+** | `sangam21/two-stage-lora-gaussian-full-hungarian-0-604` | kirilltym 0.85/0.15 Gaussian mix + spaced-em-dash guard + hard Hungarian 12–18 | title | Code read; no md; gains not isolated |
| **0.604** | `prvsiyan/the-prompt-is-not-the-author` (V4) | 0.563 stack + 4 Gaussian heads + 85/15 mix + gates | md + official sub `57428928` | Strongest documented lineage; V5 = 0.603 |
| **0.604** | `kiza123123/ntl-0604-dual-lora-gaussian-hungarian` | **byte-identical to sangam21** (only cell marker differs) | title | Duplicate of sangam21, not independent evidence |
| **0.600** | `kirilltym/0-600-teach-the-model-remove-the-prompt` (v20) | nishant 0.592 + Gaussian on two-stage 8B + 85/15 old-stack mix | title; joseph attributes 0.600 to v20 | md itself says V19 0.594, V4 0.595, combined "no official score yet" |
| **0.60** | `shalingonge/day-4-0-60-master-multi-gauss-protected-apex` | 0.604 fork + exact injective marginals for clusters 2–6 + Hungarian 12–18 + family gates | title only | Unverified; code read |
| **0.597** | `lightningv08/0-597-two-stage-lora-olmo2-smollm3` | 0.595 + SmolLM3 routed log-prob at coefficient 0.03 | title | No OOF; single author |
| **0.595** | `lightningv08/0-595-two-stage-lora-gaussian-olmo2` | kirilltym Gaussian + OLMo2 at coefficient 0.03 | title | md explicitly says "No leaderboard score is claimed for this candidate" |
| **0.595** | `kirilltym/two-stage-lora-gaussian-readout` (v4) | nishant + Gaussian on two-stage 8B | lightningv08 md attribution | md says "No score is known for this modified version" |
| **0.592** | `nishantdahal/two-stage-lora-on-the-public-pool` | base: 8B+14B two-stage + n-gram + twins + Sinkhorn | md | CV 0.593; our diff anchor |
| **0.574** | `zbigniewkirchner/0-574-in-style-qwen3-8b-lora-fully-public` | prvsiyan 0.569 + in-style 8B LoRA (weight 0.17 + twin) | md | Held-out member table |
| **0.569** | `prvsiyan/...` V1 | Travis 0.563 stack + Gaussian heads | md | Version table |
| **0.563** | `vnton37nguyn/ntl-lora-instyle-0-563` | 0.556 stack + in-style Qwen2.5-7B LoRA + gates | md | Component CV given |
| **0.552** | `zbigniewkirchner/0-552-adaptive-centering-lora-claude-gates` | lucas 0.549 + own LoRA + Claude gates | md | Version table |
| **0.552** | `liurouhan/...sinkhorn-0-25` | experimental harness on zbigniew: alpha grid 0/0.1/0.25/0.5; selects 0.25 | md | Same score as parent; no new member |
| **0.550** | `lucasboesen/lora-x-adaptive-centering` (v4) | gates + punctuation veto on 0.549 | md | v5 = stronger gates, no LB digit |

Pulled but not central: `alexchilton/ntl-0592-plus-0569` (adds the whole 0.569 probability vector at weight 0.5; **no score claimed**), `saddamaditya/apex-attribution-sovereign-shield-multi-lora` (Apex v4 = 0.556 per joseph's table; shield + multi-LoRA), `lightningv08/0-541-...` (old adaptive-centering branch).

---

## 3. Technique catalog

### T1. Shared-prompt Gaussian group head — **the highest-quality measured gain**

**Where:** prvsiyan (0.604), kirilltym v4/v20, embedded verbatim in sangam21, joseph, kiza, shalingonge, lightningv08, saddamaditya.
**Idea.** A passage feature is modelled as `x = class_mean + prompt_effect + noise`, with the prompt effect a Gaussian shared by every passage answering the same (unknown) prompt. Instead of subtracting the group mean — which for a pair removes half the author signal — the head integrates the prompt effect out and decodes the group jointly.

Key implementation facts (from prvsiyan / kirilltym):
- Apply only to test groups of **2–18** inferred by 0.60 average-linkage on the mean of content-word TF-IDF cosine and mean-centred GLM-4-9B L24 cosine.
- **Residual PCA 128**, **temperature 3.0**, **log-prob weight 0.10**, four measured views: **Qwen3-8B L22, GLM-4-9B L8, Qwen3-4B L4, DeBERTa-v3-large L24**.
- Pairs/triples: **exact injective posterior** (18·17·16 = **4,896** tuples for a triple). Larger groups: **cavity EM** with finite-population correction `lam/(1+(n-1)lam)`.
- Head weight equals the centred-twin weight of the same backbone — i.e. the Gaussian head *replaces/augments* the twin, it is not a new tuned hyperparameter.
- Fitting uses train prompt IDs; inference uses no test labels or test prompt IDs.

**Measured evidence.**
- Nested prompt-grouped harness, 8 seeds, paired prompt bootstrap: **+0.0100 [+0.0026, +0.0172]** pooled 2/3-passage fragments, **+0.0408** complete prompts, **68/68** view-by-scenario wins vs the matching centred head.
- Early Fable notebook: **0.518 → 0.533** LB.
- prvsiyan: **0.599 → 0.604** (+0.005). kirilltym: **0.592 → 0.595** (+0.003).
- Author's diagnostic: on pairs, centred logistic heads score **0.05–0.10 below** plain heads; this is why the Gaussian head exists.

**Why it matters for us:** our pipeline already has *centred twins*. This is the principled upgrade, and it reuses our existing two-stage 8B features. **Best expected value per unit of new modelling.**

**Risk:** medium. Cluster inference threshold is the only tuned object; the measured gain is on fragments, and the hidden test is believed to be fragment-heavy, but a prompt-heavy hidden pool would shrink it.

---

### T2. ATLAS 10% blend — **largest replicated LB delta**

**Where:** josephadamski91 (0.607), seven ATLAS-variant notebooks.
**Idea.** A separate Qwen3-8B-Base adapter (dataset `josephadamski91/ntl-atlas-qwen3-8b-v1`, version 2, adapters SHA `29e9b32e…`) trained for **5 epochs on the official 2,699 rows**; it scores each passage by **source-conditioned token likelihood** and outputs 18 class probabilities. Blend once, after the whole ensemble, before the final Sinkhorn:

```python
P = 0.90 * P + 0.10 * ATLAS_PROB
P /= P.sum(1, keepdims=True)
# then the original 100 Sinkhorn iterations, unchanged
```

Column order is remapped by label name; the adapter is called at `batch_size=1`; all files SHA-validated.

**Measured evidence (joseph's "Other results" table, all allegedly official LB):**

| Parent notebook | Parent | +ATLAS | Δ |
|---|---:|---:|---:|
| kirilltym v20 | 0.600 | **0.607** | **+0.007** |
| nishant 0.592 (+early layers) | 0.592 | 0.596 | +0.004 |
| InStyle v1 | 0.563 | 0.567 | +0.004 |
| prvsiyan v1 | 0.569 | 0.572 | +0.003 |
| prvsiyan v1 (ATLAS inside Gaussian groups) | 0.569 | 0.572 | +0.003 |
| Apex v4 | 0.556 | 0.558 | +0.002 |
| Lucas v3 | 0.549 | 0.551 | +0.002 |

**7/7 positive, mean +0.0036, max +0.007.** This is the only technique with independent replication across *different* parent stacks.

**Risk:** medium. Single author; adapter trained only on the 2,699 official rows (prompt-overfit risk on unseen prompts); ATLAS deltas of +0.002 are noise-level. The 10% cap limits downside. Note the adapter is *not* a normal LoRA classifier readout — it is a likelihood model, hence its errors are less correlated with the rest of the stack.

---

### T3. In-style extra data (LoRA + stylegram) — **best member upgrade**

**Where:** zbigniew 0.574 (Qwen3-8B), vnton 0.563 (Qwen2.5-7B), prvsiyan 0.563 stack.
**Idea.** Add public prompt-bank passages where every prompt is answered by **≥12 of 18 models**, so topic cannot act as a label shortcut. zbigniew raised usable prompts from ~240 to **709** by generating missing GPT/Gemini answers with the free Kaggle Benchmarks proxy — **9,818 extra passages**. Train the same 24-layer LoRA recipe on official + extra, then add at **weight 0.17 + centred twin**, exactly like the existing in-style Qwen2.5 member.

**Measured evidence.**
- zbigniew, 539 held-out rows: public LoRA24 **0.456**; own Claude-head LoRA **0.484**; in-style LoRA **0.516** (+0.060 over public). LB **0.569 → 0.574** (+0.005).
- vnton stylegram component: **0.3845 → 0.400** (+0.0155) with in-style extra rows (per-class weight share 0.5, prompts ≥12 labels); held-out blend **+0.007**.
- vnton adaptive-centred stylegram component: 5-fold **0.3849 → 0.4245**; 3-fold held-out **0.3656 → 0.4115**.

**Risk:** low-medium. The "≥12 labels per prompt" filter is the key leak guard. Cost is a LoRA training run on Kaggle (fits the 9 h budget), plus pool curation.

---

### T4. Within-family gates — cheap, data-dependent

**Where:** zbigniew 0.552, lucas v4/v5, vnton 0.559+, saddamaditya.
**Idea.** If the blend predicts a family, re-decide only inside it: Claude 5-way stylegram LinearSVC (+ optionally a 5-way DeBERTa head and a LoRA Claude head); GPT / Gemini / GLM analogues. Non-family predictions are never touched.

**Measured evidence.**
- zbigniew: fork 0.541 → **0.544** with Claude gate (+0.003); stylegram grouped CV **0.3845 → 0.3967** (+0.0122).
- lucas: Claude gate + punctuation veto together **+0.0018 OOF** (veto alone +0.0007); v4 LB **0.550** (from 0.549).
- lucas GPT gate, the clean ablation: **without** extra GPT rows **−0.0004 OOF**; **with** 1,342 pooled free-proxy GPT passages **+0.0052 OOF**; v20 vs v19 **+0.0063 [+0.0019, +0.0119]**.
- zbigniew's within-Claude LoRA head: member 0.4838 → **0.4932** on 539 rows when gated.

**Risk:** low. The lesson is explicit: **the gate is worthless without balanced extra data**; with it, +0.003–0.006. Our pipeline already has sibling stylometry, so incremental value may be smaller than the public forks saw.

---

### T5. Post-Sinkhorn orthographic invariants ("shield")

**Where:** sangam21 (spaced em-dash), kiza 0612 (curly quotes), saddamaditya (both), lucas v4 (rejected), vnton (rejected).
**Train facts, independently re-verified on `data/train.csv`:**
- **Spaced em-dash `" — "`: 86 rows (3.19%).** Exactly 4 classes: `claude-opus-4-6` 40, `claude-opus-5` 38, `glm-5.3` 5, `glm-5.3-flash` 3. **86/86 = 100%**, zero in the other 14 classes.
- **Any em-dash `"—"`: 1,052 rows (39.0%).** Gemini: **1/600** (one in `gemini-3.7-flash`). Claude 94–124/150; GLM 119–123/150. This confirms the lab-level signal but shows why "any em-dash ⇒ not Gemini" is too blunt.
- **Curly quotes (`’ “ ” ‘`): 0 rows in 1,049 Claude+GLM rows; 607 rows in the other 1,650.** Verified exactly.
- **`gpt-5.6-sol` straight apostrophes: 0/150.** (saddamaditya claim; verified.)
- The shield zeroes (not merely penalizes) forbidden classes post-Sinkhorn, then renormalizes.

**Conflicting evidence.**
- lucas's version (penalize Claude/GLM on curly, Gemini on 2+ em-dashes): **+0.0004 OOF**; two public forks with the same rule **lost 0.003–0.005**; proxy Gemini text uses em-dashes, so the invariant does not hold off-distribution.
- vnton: "orthographic shield and alpha 0.25 were neutral/negative in our prompt-grouped CV and are not used."
- sangam's rule is much narrower (spaced em-dash only, 3.2% of rows, 4 allowed classes); it plausibly has near-zero ripple. kiza 0612's curly hard-zero is broader and riskier: 607/1,650 non-Claude/GLM rows carry curly quotes, and free-proxy Claude/GLM text violates the invariant.

**Verdict:** the spaced em-dash guard is the only defensible variant; keep it **post-Sinkhorn**, restrict to exactly 4 allowed classes, and gate it behind a check that the hidden prompt distribution still looks like train. Do **not** resurrect the broad −2 log-penalty veto.

---

### T6. Hard Hungarian / injective marginals — **highest risk, contested**

**Where:** sangam21 (dense clusters 12–18, hard one-hot), kiza 0612 (9–18), shalingonge (soft exact marginals 2–6; hard 12–18), our own earlier experiment.
**Idea.** In a detected prompt cluster, no model may appear twice. Solve `linear_sum_assignment` on `−log P` and one-hot the result.

**Evidence for:** sangam21 0.604+ and kiza 0612 0.612 titles; the clusters are dense (12–18), where a near-complete 18-model round is plausible; kiza claims "99.6% cluster purity down to size 9 at threshold 0.85".
**Evidence against:** our own documented result — Hungarian inflated CV by +0.10 but **degraded public LB 0.353 → 0.344**, because the hidden test is subsampled and does not contain balanced 18-model rounds. Hard one-hot also destroys calibrated probabilities for the subsequent gates.
**More principled variant:** shalingonge's exact injective marginals for clusters 2–6 — a DP over permutations that reweights probabilities instead of forcing a hard assignment. This is the version worth testing if we test any.
**Verdict:** do not adopt hard Hungarian on the strength of titles alone; it contradicts our only direct LB experiment. If explored, restrict to clusters ≥12, use soft marginals, and A/B on the public LB.

---

### T7. Orthogonal members at fixed small coefficients (OLMo2 / SmolLM3)

**Where:** lightningv08 0.595 / 0.597.
**Idea.** Independently trained native-24-layer OLMo2 and SmolLM3 routed readouts, each entering the log-prob sum at **coefficient 0.03** before the original softmax and 100-step uniform Sinkhorn; SmolLM3 uses a 5,400-response warmup + official fine-tune, frozen routed readouts, plus a frozen Gaussian term at weight 0.10 within eligible groups.

**Evidence:** 0.592 → 0.595 title (OLMo), then 0.597 (SmolLM3). The md itself says "that result does not establish its contribution" and "No leaderboard score is claimed for this candidate." No prompt-disjoint OOF exists for the full stack.
**Verdict:** cheap diversity if the checkpoints are already public, but the evidence is title-only and the coefficient is unablated. Lower priority than T1–T3.

---

### T8. Test-time machinery already worth copying (even though our base has variants)

- **Adaptive centring dose–response (lucas, pooled 150-prompt sim):**
  - replace-mode (centred only): **+0.000** at 1–2 passages/prompt, **+0.019** at 1–5, **+0.034** at 3–8, **+0.056** at all 18.
  - add-mode from cluster size 2 (plain + centred evidence): **+0.024 / +0.042 / +0.043 / +0.052**.
  - This is why every strong notebook uses add-mode from size 2, not replace-mode from size 4.
- **Cluster threshold:** mix of content TF-IDF and GLM-4-9B L24 cosine, average linkage at **0.60**; at 0.65 same-prompt pair recall 0.89 vs 0.78 for TF-IDF alone, precision 0.97. At 0.60 the false-positive pair rate is ~0.012 with recall ~0.72. Stylegram uses its own median-normalized distance at **0.90**.
- **Test-set standardization** of frozen-LM features (label-free, transductive) is used by lucas and prvsiyan; cheap and orthogonal.
- **Partial Sinkhorn:** lucas uses `alpha=0.5` with target `(1−alpha)·observed + alpha·uniform`, 50 iterations, CV "never negative (+0.002..+0.007)". liurouhan tested 0/0.1/0.25/0.5 and submitted 0.25 — same 0.552 as the alpha=0.5 parent, so alpha is not a sensitive knob.
- **Two-stage training itself:** stage 1 on the public pool (8B member CV **0.483**), stage 2 on official rows (8B **0.569**, 14B **0.569**); the two LoRAs disagree on **22%** of rows, which is the blend gain. Truncate at 24/27 layers, pool 8 depths, merge adapters, fp16, no final RMSNorm.
- **Weight/evidence accounting:** log-prob sum, not probability average; twins and Gaussian evidence carry explicit separate weights; gates run after Sinkhorn and only inside a family.

---

## 4. Diff vs our 0.600 base — what we already have vs what is missing

| Component | Our base | Public frontier | Gap |
|---|---|---|---|
| Two-stage Qwen3 8B + 14B LoRA | yes | yes | none |
| 4-view TF-IDF LinearSVC (char/word/shape/function-word) | yes | yes | none |
| Centred twins in inferred groups | yes | yes, add-mode from size 2, weight 1.5× member | **check gating/threshold; measured dose–response favors size 2** |
| Sibling stylometry | yes | within-family gates (Claude/GPT/Gemini/GLM) | **gates add +0.003–0.006 when fed balanced extra data** |
| Typographic priors | yes (length-gated) | broad veto rejected; narrow spaced-em-dash shield | **keep narrow, do not broaden** |
| Global Sinkhorn | yes | yes (partial alpha 0.25–0.5) | check whether ours is full-strength |
| Shared-prompt Gaussian head | **no** | yes, 4 views, weight 0.10 | **T1 — add** |
| ATLAS likelihood member | **no** | yes, 10% pre-Sinkhorn | **T2 — add** |
| In-style extra-data LoRA | **no** | yes, 9,818 rows / 709 prompts, weight 0.17 + twin | **T3 — train** |
| Post-Sinkhorn invariants | partial (priors, not hard) | narrow em-dash guard, contested curly guard | low priority |
| Hungarian / injective marginals | **no** (and previously harmful) | 0.604–0.612 forks | **avoid hard version** |
| OLMo2 / SmolLM3 members | **no** | 0.03 coefficients | optional |

---

## 5. Ranked techniques

Ranking weighs **claimed gain**, **evidence quality** (paired OOF/bootstrap > replicated LB deltas > single LB delta > title only), **risk**, and **effort**.

| # | Technique | Claimed gain | Best evidence | Replications | Risk | Effort |
|---:|---|---|---|---|---|---|
| **1** | Shared-prompt Gaussian head (4 views, w=0.10, PCA128, T=3) | +0.005 LB; +0.010 OOF | +0.0100 [+0.0026,+0.0172], 68/68 cells, 8 seeds | 2 independent (prvsiyan, kirilltym) + 6 forks | Medium | Medium (reuses 8B features) |
| **2** | ATLAS 10% pre-Sinkhorn blend | +0.007 on 0.600; +0.002–0.007 elsewhere | 7/7 positive variants | 7 notebooks, 1 author | Medium | **Low** (~10 lines + 1 adapter) |
| **3** | In-style extra-data LoRA (9,818 rows / 709 prompts, w=0.17 + twin) | +0.005 LB; +0.060 member acc | 539-row held-out table; component CV +0.0155 | 2 (8B, Qwen2.5-7B) | Low-Medium | High (training) |
| **4** | Within-family gates + balanced extra rows | +0.003 Claude; +0.005–0.006 GPT | Paired OOF/sim with CIs; clean no-data ablation (−0.0004) | 3 | Low | Low-Medium |
| **5** | Narrow spaced-em-dash guard (post-Sinkhorn, 4 classes) | ~+0.001–0.004 (inferred, not isolated) | Train 86/86 verified; broad veto measured +0.0004 and forks −0.003..−0.005 | 1 narrow, 2 negative | **High** (distribution shift) | Very low |
| **6** | Hard Hungarian on dense clusters | +0.004–0.012 (confounded with guards) | Titles only; our own LB −0.009 | Forks copy each other | **Very high** | Low |
| **7** | OLMo2 / SmolLM3 at 0.03 | +0.003 (title) | No OOF; author disclaims causality | 1 | Medium | Medium |
| **8** | Whole-notebook probability stacking (0.592 + 0.569 at w=0.5) | no score claimed | None | 1 | Medium | Low |

---

## 6. Most actionable 3 changes vs our 0.600 base

### Change 1 — Add a shared-prompt Gaussian head on our two-stage 8B features
- Reuse the existing 8B LoRA forward; extract one to four dense views (start with the 24-layer pooled/readout feature we already compute, then Qwen3-8B L22, GLM-4-9B L8, Qwen3-4B L4, DeBERTa-v3-large L24 if cheap).
- Fit the class-mean + shared-prompt-covariance model on official train with prompt IDs; PCA-128 residuals; temperature 3.
- At test, infer groups with the 0.60 TF-IDF+GLM-L24 average-linkage cut; apply the Gaussian log-prob at **weight 0.10** only on groups of 2–18; exact decode for 2–3, cavity EM above.
- Blend it *in addition to* the centred twins (add-mode), mirroring kirilltym's 85/15 composition if we also keep the older stack.
- Expected: **+0.003–0.010**; the paired OOF evidence is the strongest in the public corpus. Main implementation risk is group inference and numerical stability, both already solved in the public code.

### Change 2 — Blend ATLAS at 10% before Sinkhorn
- Attach `josephadamski91/ntl-atlas-qwen3-8b-v1` (version 2) + the `ntl_atlas` inference package; validate adapter SHA `29e9b32e…`.
- `P = 0.90·P + 0.10·ATLAS_PROB`, renormalize, then run our existing Sinkhorn unchanged.
- Expected: **+0.004–0.007** (largest measured delta at the exact 0.600 parent). Because the adapter is an official-data-only likelihood model, cap at 10% and verify on our grouped CV that it does not degrade the Claude/GLM sibling recalls.

### Change 3 — Train one in-style 8B LoRA on the filtered public pool
- Curate prompts answered by **≥12 of 18** models; add zbigniew's free-proxy GPT/Gemini rows to lift prompt count toward 709; target ~9–10k passages.
- Fine-tune the same 24-layer Qwen3-8B recipe (stage 1 on pool, brief stage 2 on official), add at **weight 0.17 + centred twin**, exactly like the existing LoRA members.
- Expected: **+0.005** LB, with the largest member-level accuracy jump measured publicly (**0.456 → 0.516**). This is the highest-compute action and the one that most changes the ensemble's information set.

**If only one day of compute is available:** do Change 2 first (hours, +0.007, 10 lines), then Change 1. Change 3 is the right next training run but needs pool curation + a Kaggle GPU slot.

**Do not do:** broad punctuation veto, hard Hungarian, or ATLAS weight >0.15 without an A/B. The first two have negative direct evidence; the third is outside the measured range.

---

## 7. Verification log

**Verified locally against `data/train.csv` (2,699 rows, 150 prompts):**
- Spaced em-dash `" — "`: **86 rows**; distribution `claude-opus-4-6` 40, `claude-opus-5` 38, `glm-5.3` 5, `glm-5.3-flash` 3; **zero elsewhere**. Matches sangam21's comment exactly.
- Any em-dash: 1,052 rows (39.0%); Gemini 1/600; Claude 94–124/150; GLM 119–123/150. Confirms lab signal and why "any em-dash" is too broad.
- Curly quotes: **0/1,049** Claude+GLM; **607/1,650** other rows. Matches kiza 0612's comment exactly.
- `gpt-5.6-sol` straight apostrophes: **0/150**.
- `kiza123123/ntl-0604` is a **byte-identical copy** of `sangam21` apart from the code-cell marker — it is not independent evidence for 0.604.

**Not verified / cannot verify from here:**
- joseph's 0.607 binding to a specific scored script version (legacy endpoint 404; author says code identical to scored v1).
- kiza 0612's **0.612** (title only, no markdown, no version link, no explanation).
- sangam21's Hungarian "resolved N collisions" count — runtime-dependent and not printed in the static source.
- Whether the Gaussian head's OOF gain transfers to our exact stack; the public gains are on the 0.563/0.592 parents.
- ATLAS transfer to a 0.600 stack with our typographic priors (only the 0.600 kirilltym parent is directly analogous).
- Hidden-test prompt mixture (the Gaussian head and Hungarian both depend on fragment density; the public LB is a single subsample).

**Comment/discussion audit:** notebook markdown caveats were captured in full. External search was attempted but the configured search provider hit its plan limit; no independent Kaggle-discussion signal is included. The authors' own caveats are quoted in the tables above and are unusually explicit (e.g. lightningv08: "No leaderboard score is claimed"; kirilltym v4: "No score is known"; prvsiyan: "old-version endpoint 404").

---

## 8. Practical notes for the team

- **Public LB deltas below ~0.01 are one-split noise.** Use them to rank hypotheses, not to declare wins. The Gaussian head is the only technique here with a paired, prompt-grouped CI; ATLAS is the only one with 7/7 cross-notebook replication.
- **Everything is a fork.** Before crediting a "0.604" or "0.612", check the code diff. 0.604 (kiza) = 0.604+ (sangam) exactly; 0.612 adds one curly-quote rule and widens Hungarian; 0.607 adds one 10% blend.
- **The lab/sibling split is unchanged.** Gains come from (a) removing the prompt effect (Gaussian), (b) extra in-style data, (c) one orthogonal likelihood member (ATLAS), (d) within-family gates. No notebook demonstrates a better 18-way semantic backbone.
- **Our 0.600 is already inside the top cluster.** The realistic public ceiling is ~0.607–0.612; the remaining gap is ~1 point, and all candidate moves are small, correlated, and noisy.

---

## Appendix — artifacts

- Pulled notebooks and extracted code:
  - `scratch/w1_pull/{nishantdahal,kirilltym,prvsiyan,sangam21,josephadamski91,lucasboesen,zbigniewkirchner,liurouhan,saddamaditya,lightningv08}_*/`
  - `scratch/w1_pull2/{alexchilton,kiza123123(0604,0612),lightningv08(0597),kirilltym(readout),vnton37nguyn,zbigniewkirchner(0574),shalingonge}_*/`
- Each directory contains `_code.py`, `_md.md`, `kernel-metadata.json`, and the raw `.ipynb`.
- Local train data used for verification: `data/train.csv`.
- Key constants quoted above are from `nishantdahal.../_code.py` (W_SVC 1.021; LoRA weights 0.226/0.429; twin 0.079/0.150; TWIN 0.85/2/18), `prvsiyan.../_code.py` (MIX_A 0.85; GROUP 0.60/24/2/18; weight 0.10; PCA128; T=3; 100 Sinkhorn iters), and `josephadamski91.../_code.py` (ATLAS 0.90/0.10 + SHA).
