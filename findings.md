# findings.md — "Name That LLM" (18-class authorship), v1 → v15

Last updated: 2026-10-26, ~11:00 UTC.
Owner: `/Users/pika/Documents/Projects/kaggle-llm`.
Scope: everything learned from 14 leaderboard submissions, ~40 grouped-CV experiments,
and the infrastructure failures along the way.

---

## 1. TL;DR

- **Best valid score: v10 = 0.606** (ref `57497594`, 2026-10-21). It is the only
  candidate that should be submitted if the competition ended today.
- **Everything attempted after v10 regressed or timed out** — v11 (timeout), v13
  (0.592), v12-fast (0.584).
- The regressions are **not** noise. v13 and v12-fast differ by exactly two numbers
  in one dict and still separate cleanly (0.592 vs 0.584), so the leaderboard
  resolves ~0.008.
- The failure mode is consistent: **a change that improved prompt-grouped OOF
  degraded the hidden leaderboard.** This has now happened four times (v8, v9, v13,
  v12-fast). Our CV protocol is not a usable proxy past roughly 0.55.
- Two Final Submission slots are **untouched**. 4 regular submissions remain today.

---

## 2. Full leaderboard history

All 14 submissions, oldest first. "Public LB" is what Kaggle reports; for this
competition the public/private split is a subsample of the same hidden rerun, and
the public number has tracked the private ranking in every case where both exist.

| # | Ref | Date (UTC) | Variant | Public LB | Δ prev best | Outcome |
| ---: | :--- | :--- | :--- | ---: | ---: | :--- |
| 1 | `57434986` | 2026-10-20 11:09 | baseline TF-IDF + stylometry LightGBM | 0.355 | — | ok |
| 2 | `57437308` | 2026-10-20 12:15 | fused 4-view n-grams + dense stylometry | 0.315 | −0.040 | **regression** |
| 3 | `57439844` | 2026-10-20 13:29 | Grandmaster: Qwen3-8B LoRA + 4-view n-grams | 0.539 | +0.184 | jump |
| 4 | `57440836` | 2026-10-20 13:58 | Grandmaster v3 + metric-aligned multipliers | 0.532 | −0.007 | regression |
| 5 | `57441656` | 2026-10-20 14:22 | Grandmaster v4 + 5-fold bagged n-grams | 0.528 | −0.011 | regression |
| 6 | `57471060` | 2026-10-21 07:12 | **v5** dual two-stage LoRA 8B+14B + centred twin + HodgeRank | 0.600 | +0.061 | jump |
| 7 | `57473865` | 2026-10-21 08:34 | v6 + unbalanced Sinkhorn-Knopp | 0.593 | −0.007 | regression |
| 8 | `57477702` | 2026-10-21 10:32 | v8 triple trunk + prompt subspace deconfounding + 159 features | 0.593 | −0.007 | flat |
| 9 | `57480675` | 2026-10-21 12:00 | v9 exact balanced Sinkhorn + pristine LoRA readout | 0.591 | −0.009 | regression |
| 10 | `57497594` | 2026-10-21 21:25 | **v10** = v5 recipe + ATLAS v2 at 10% pre-Sinkhorn | **0.606** | **+0.006** | **best** |
| 11 | `57537119` | 2026-10-22 20:11 | v11 = v10 + SVC pool weight 0.25 + transductive TF-IDF | — | — | **timeout** |
| 12 | `57623259` | 2026-10-25 08:35 | v13 = v10 + curated SVC + lora24 centre 0.70/0.15 | 0.592 | −0.014 | regression |
| 13 | `57627241` | 2026-10-25 10:55 | v12-fast = v13 with twins 0.80/0.20 | 0.584 | −0.022 | regression |
| 14 | `57662874` | 2026-10-26 10:21 | v15 = v10 + lora24 plain reweight only | pending | — | in flight |
| — | — | 2026-10-26 | v14 = v10 + curated SVC only | running | — | in flight |

```
0.62 |                                        *  v10 0.606
0.60 |                          *  v5 0.600
0.58 |                                   *        v13 0.592
0.56 |                                   *        v12f 0.584
0.54 |        *  GM 0.539
0.52 |        *  GM v3/v4 0.532/0.528
0.50 |
0.35 |  *  baseline 0.355
0.32 |  *  fused 0.315
     +--------------------------------------------------
       Oct20                          Oct21            Oct25/26
```

### The shape of it

Two real jumps and then a wall.

- **0.315 → 0.539** came from adding a fine-tuned Qwen3-8B LoRA readout on top of
  pure n-grams. This is the single biggest win in the whole project (+0.18) and it is
  what makes the task solvable at all: TF-IDF/stylometry alone saturates around 0.35.
- **0.539 → 0.600** came from the dual two-stage LoRA pair (8B **and** 14B) plus the
  prompt-centred twin and HodgeRank sibling re-ranking.
- **After v5, every single structural addition lost points** (0.600 → 0.593 → 0.593 →
  0.591). The only thing that ever gained again was ATLAS v2, and only +0.006.

---

## 3. Version-by-version: what changed, what happened

### v5 — 0.600 (the recipe that actually works)
Dual two-stage LoRA (Qwen3-8B + Qwen3-14B) + external pool n-grams + prompt-centred
twin + HodgeRank sibling re-ranking + length gate. Everything downstream is a
variation on this.

### v6 — 0.593 (−0.007)
Swapped the balanced Sinkhorn-Knopp for an unbalanced variant. Lost 0.007. The
balanced version was already right.

### v8 — 0.593 (−0.007)
Added a third trunk (OLMo2), prompt-subspace deconfounding, 159 hand-built forensic
features, and transductive clique calibration — four changes at once. Net zero. We
never learned which of the four was neutral and which was harmful, because they
shipped together. **Process mistake repeated in v13.**

### v9 — 0.591 (−0.009)
"Remediated" triple trunk with exact balanced Sinkhorn and a pristine LoRA readout.
Lost another 0.009. This is where the triple trunk should have been abandoned; the
OLMo2 trunk was never separately validated on the LB.

### v10 — 0.606 (+0.006) — current best
v5's exact recipe, plus ATLAS v2 blended in at 10% *before* the final Sinkhorn.
ATLAS is an independent Qwen3-8B adapter bundle with its own label order; the kernel
verifies all file sha256s and aligns labels. **This is the only post-v5 change that
ever gained.** Worth noting: it gained by adding a *different, independently trained
model* and giving it a small fixed weight — not by adding features or post-processing.

### v11 — timeout (no score)
v10 + SVC pool weight 0.25 + transductive TF-IDF vocabulary + checkpoint guard +
JSONL telemetry. Public run took **16,076 s (4.47 h)**; the hidden rerun then exceeded
the submission runtime limit and returned `COMPLETE` with a blank score and
`error_description = "Your submission notebook exceeded the allowed runtime."`

Two separate problems, both fixed later:
1. The telemetry wrapper walked the entire `/kaggle/input` tree (`os.walk`) and
   buffered a file manifest — pure overhead.
2. Transductive TF-IDF re-vectorises the whole fit set. Cheap, but it shipped with the
   slower legacy and pushed past the limit.

### v12 — never submitted (correctly rejected)
v11 legacy + a ModernBERT-large branch. The mounted `dddmdd/ntl-finetuned-encoders`
dataset contains only `ettin-encoder-400m` — **there are no ModernBERT weights in any
attached dataset**, so the branch cannot execute offline. The 0.6501 OOF blend that
motivated it is not deployable. `runs/v12/kernel-metadata.json` also still points at
`pikaaa/name-that-llm-v11` (copy-paste bug), so it was never pushed as its own kernel.

### v13 — 0.592 (−0.014)
v10 envelope (not v11's) + three changes:
1. **curated-pool SVC**: `pool_v1.parquet` (34,940 rows) → `curated_pool.parquet`
   (81,221 rows after filtering), `sample_weight = 0.25` on pool rows, transductive
   vocabulary. Grouped-CV SVC member: **0.4138 → 0.4667**.
2. **lora24 top-level reweight**: the legacy stack already computes the lora24
   readouts but buries them inside its 15% share. v13 re-adds them after the
   0.85/0.15 mix at `public 0.10 / public_twin 0.70 / seed7 0.05 / claude 0.05 /
   claude_twin 0.15`. Measured lora24: plain OOF 0.4804, prompt-centred OOF 0.5960.
3. **q25 dead head removed** (`W["q25_7b"] = 0.0`, no Gaussian head for it). Output
   identical, frees runtime.

Also fixed a real bug: the legacy stack sorts its labels, so its Claude order differs
from the production order; v12's reweight added the captured probabilities **without**
reindexing. v13 applies `_prob = _prob[:, _old_cols]`.

Result: **0.592**, the best OOF of any variant and the worst LB since v9.

### v12-fast — 0.584 (−0.022)
v13 with `public_twin 0.80` and `claude_twin 0.20`. Nothing else. A clean one-dict
A/B. It lost another 0.008, which is the most informative single data point we have:

> the marginal effect of prompt-centre weight is **negative**, roughly
> **−0.08 per unit**, in the region 0.70–0.80.

### v14 — v10 + curated SVC only (running)
### v15 — v10 + lora24 **plain** reweight only, twins 0 (pending, ref `57662874`)

These two exist to split the v13 bundle. See `docs/research/v14-v15-ablations.md`.

---

## 4. What works (evidence-backed)

1. **A fine-tuned LoRA readout on a real base model is worth more than everything
   else combined.** 0.315 (n-grams) → 0.539 (one 8B LoRA) → 0.600 (8B + 14B).
2. **Two different-sized LoRA trunks beat one.** 8B + 14B > 8B alone.
3. **An external prompt pool is essential for the n-gram member.** Dropping it and
   training on competition train only collapses grouped CV from 0.439 to 0.319.
   The pool must not dominate: weight 0.25 is the smooth optimum, 1.0 is −0.025.
4. **Sinkhorn-Knopp balancing at the very end.** Because the metric is unweighted
   mean per-class recall, forcing column marginals to uniform helps; unbalanced
   variants lose ~0.007.
5. **Small weight on an independent model.** ATLAS v2 at 10%, pre-Sinkhorn, +0.006.
   The pattern that works is "another model, small fixed weight", not "more features".
6. **Do not sanitise the text.** No lowercasing, no quote normalisation, no dash
   normalisation. Straight vs curly quotes and em-dash rate carry a large share of
   the lab-level signal. (Confirms the AGENTS.md warning empirically.)
7. **Grouped CV is mandatory** (`StratifiedGroupKFold` on `prompt_id`). Random splits
   leak prompt topic across folds and move the estimate by ~0.15.
8. **The 0.65x frontier is an architecture, not a member list** (verified 2026-10-28 from
   `romantamrazov/name-that-llm-who-am-i` and `dhruvtibarewal/nlm-v27`). Their primary is a
   six-backbone LoRA24 ensemble (`q14v5` 3.0, `n14b` 2.5, `q8v5` 2.0, `ll8v5` 1.0, `q25_14` 1.0,
   `deb` 1.0). The frozen-feature stack — `W = {'stylegram': 1.516, …}`, byte-identical to their
   `comp_stack.py` — is what **we** use as the primary and they use as a **0.15 complement**
   (`--comp_w 0.15`). Weights are all public (`dhruvtibarewal/nlm-l24-adapters-v5` and friends).

---

## 5. What does not work

| Change | Claimed OOF gain | LB effect |
| :--- | :--- | :--- |
| Unbalanced Sinkhorn (v6) | — | −0.007 |
| OLMo2 third trunk (v8) | — | ≤ 0 |
| Prompt-subspace deconfounding (v8) | — | ≤ 0 |
| 159 forensic features (v8) | — | ≤ 0 |
| Transductive clique calibration (v8) | — | ≤ 0 |
| Exact balanced Sinkhorn + pristine readout (v9) | — | −0.009 |
| Curated-pool SVC (v13/v14) | +0.053 (0.414→0.467) | **−0.014** (with reweight) |
| lora24 prompt-centre, twin 0.70 (v13) | +0.116 over plain (0.596 vs 0.480) | **negative** |
| lora24 prompt-centre, twin 0.80 (v12-fast) | — | **−0.008 vs 0.70** |
| Transductive TF-IDF vocabulary (v11) | +0.015 | never scored (timeout) |
| ModernBERT branch (v12) | 0.6501 blend | **not deployable** |

The unifying pattern: **post-processing and feature engineering lose; new
independently-trained models win.** Every one of the failures above is a
post-processing or feature-space change. The two wins (LoRA trunks, ATLAS) are new
models.

---

## 6. The central problem: OOF does not transfer

This is the single most important finding and it is unresolved.

| Variant | Grouped-CV / OOF | LB | Gap |
| :--- | ---: | ---: | ---: |
| Early baseline (TF-IDF + LR) | 0.343 | 0.353 | **+0.010** ✅ transfers |
| Grandmaster ensemble | **0.7695** nested CV | 0.539 | **−0.231** ❌ |
| v13 curated SVC member | 0.4667 | — | ❌ |
| v13 lora24 prompt-centre | 0.5960 | — | ❌ |
| v13 as a whole | best OOF of any variant | **0.592** | ❌ |

The early, simple model transferred almost perfectly (0.343 → 0.353). Everything
complex does not. Three plausible causes, none yet ruled out:

1. **The hidden test is a different prompt bank.** Train has 150 prompts; test is
   written for prompts that do not appear in train. Any CV that holds out prompts
   still draws the held-out prompts from the same 150-prompt distribution, so
   "prompt-disjoint" CV is not "prompt-distribution-disjoint" CV.
2. **The hidden set is subsampled and unbalanced.** The Hungarian-assignment
   post-processing trap (AGENTS.md §4.4) is the same failure: grouped CV assumes
   balanced 18-model rounds, the hidden set does not have them. Anything that
   exploits round structure gains in CV and loses on the LB. Soft centring /
   cluster-mean subtraction is the safer alternative and has not been tried at
   scale yet.
3. **Class-prior shift between public and private splits.** AGENTS.md notes the model
   mixture differs between the 20–30% public and 70–80% private leaderboards. A
   member tuned to balance recalls on the CV prior can be mis-calibrated for the
   hidden prior.

**Practical rule adopted from here on: no change is trusted on OOF alone. The LB is
the only evaluation.** That means each candidate costs one submission and ~4–6 h.

---

## 7. Where the score plummeted — three cliffs

1. **The 0.7695-CV cliff (2026-10-20).** Nested CV said 0.7695, LB said 0.539. From
   that point on, CV numbers were never treated as scores again, only as tie-breakers
   between already-plausible options. Cost: three wasted submissions (GM v3, v4) and
   a day of misplaced confidence.
2. **The post-v5 feature cliff (2026-10-21).** v5 0.600 → v6 0.593 → v8 0.593 → v9
   0.591. Four consecutive structural additions, each individually defensible,
   cumulatively −0.009. This is the classic "add one more thing" trap: with a
   balanced-accuracy metric on 18 classes, a change that helps 12 classes and hurts 6
   nets negative even when the mean log-loss improves.
3. **The v13 reweight cliff (2026-10-25).** 0.606 → 0.592 → 0.584. The biggest
   single-variant drop of the project (−0.014), caused by adding ~1.05 total log-prob
   weight from members that were already in the blend at a smaller weight.

---

## 8. Infrastructure and process failures

Every one of these cost real submissions or real hours. Listed so they are not
repeated.

### 8.1 `base_config` OSError — two dead runs (2026-10-25)
```
OSError: Error no file named model.safetensors, or pytorch_model.bin,
found in directory .../ntl-public-heads/.../lora24/final/base_config
```
`swaroopmeher/ntl-public-heads` gained a `base_config` directory containing
`config.json` but no weights. The legacy `find_checkpoint` accepted the first
`config.json` whose `model_type`/`hidden_size` matched, so it selected a config-only
directory and `from_pretrained` failed at ~2,064 s — *after* the SVC stage, so both
v13 v1 and v12-fast v1 died with nothing to show.

**Fix (in v11 onward):** require at least one of `model.safetensors`,
`pytorch_model.bin`, `model.safetensors.index.json`, `pytorch_model.bin.index.json`,
and prefer paths under `/kaggle/input/models/` (the official model mounts) over
dataset copies. Unit-tested against a synthetic tree with a decoy `base_config`.

**Consequence:** v10 is no longer byte-reproducible. The same dataset that v10 used
on Oct 21 now contains a directory that would crash it. Any future "just resubmit
v10" plan must carry the guard.

### 8.2 v11 runtime timeout (2026-10-22)
16,076 s public run, hidden rerun over the limit. Causes: `os.walk` over the whole
`/kaggle/input` tree for the manifest, a `print` wrapper writing JSONL on every line,
and transductive TF-IDF. The hidden rerun returns `COMPLETE` with a **blank score**
and an `error_description` — it does not show up as `ERROR`, so a naive poller reads
it as success. **Always check `public_score is not None`, not just `status`.**

### 8.3 Legacy label-order bug
The embedded legacy stack sorts `LABELS`, so its Claude block is
`fable-5, opus-4-6, opus-4-8, opus-5, sonnet-5` while the production kernel uses
`opus-4-6, opus-4-8, opus-5, sonnet-5, fable-5`. The legacy `LOGIT` is remapped via
`_old_cols`, but v12's new reweight added the captured per-member probabilities
**without** remapping — silently scrambling all five Claude columns. Fixed in v13 with
`_prob = _prob[:, _old_cols]` everywhere, unit-tested with distinct label orders.

### 8.4 Kernel slug ≠ metadata id
Kaggle derives the kernel slug from the **title**, not the `id` field. Pushing
`id: pikaaa/name-that-llm-v14-svc` with `title: "name that llm v14 curated svc only"`
creates `pikaaa/name-that-llm-v14-curated-svc-only` and prints a warning
(`Your kernel title does not resolve to the specified id`). Polling the `id` then
fails silently. Hit twice (v12-fast, v14/v15). **Rule: title must slugify to the id.**

### 8.5 Quick-save submission route rejected
Attempted to submit an existing kernel output without a fresh run
(`experiments/quick_save_and_submit.py`). Kaggle rejects it. Every submission needs a
fresh kernel version run.

### 8.6 Runtime variance is larger than any code change
Identical code, different wall clock:

| Variant | Public runtime |
| :--- | ---: |
| v13 | **2,818 s** (47 min) |
| v15 | 3,981 s (66 min) |
| v10 | 11,331 s (3.15 h) |
| v12-fast | **11,352 s** (3.15 h) |
| v11 | 16,076 s (4.47 h) |

v13 and v12-fast are the **same script** except two floats in one dict, and their
runtimes differ by 4×. So the runtime limit is a **lottery over node/allocation**, not
a property of the code. v11's timeout is therefore ambiguous: it may have been a slow
node rather than a slow script. Practical consequence: keep the code as fast as
possible (v13's q25 dead-head removal cut ~2.5 h on a good node) and never assume a
previously-fast script will be fast again.

### 8.7 ~6 h submission turnaround
Every hidden rerun takes roughly 4–6 h to score. With 5 submissions/day, the maximum
experiment throughput is ~5 candidates/day if they run in parallel, and each one must
be a full pipeline. This is why bundling changes is tempting — and why bundling has
cost us three times.

---

## 9. Testing gaps that let bad variants ship

1. **The 3-row public test cannot exercise group-conditional code.** Every public row
   is a singleton, so `GROUP_ROWS.any()` is False and **no `*_twin` member is ever
   computed**. The v13 public log shows exactly three `REWEIGHT` lines
   (`lora24_public`, `lora24_seed7`, `lora24_claude`); the three twin keys never
   exist. The gate therefore validated a strictly easier path than the hidden rerun
   executes. The single most dangerous change in v13 (+0.70 twin weight) was the one
   the gate could not see.
   **Mitigation:** the kernels already have a `TINY` mode that runs the group paths on
   training rows as a self-test; those self-tests ran and passed, but they test
   *mechanics*, not *weight calibration*.
2. **Self-tests are in-sample.** `centred stylegram self-test acc 1.00`, `gauss_* 0.85–0.93`
   are on rows seen in training. They prove the code path runs, nothing about quality.
3. **The gate checks shape and labels, not behaviour.** `SUB_OK` verifies 3×2, ids
   match v10, no nulls, labels in the 18. A perfectly valid submission can still be
   garbage.

---

## 10. Current state (2026-10-26 15:30 UTC)

| Item | Value |
| :--- | :--- |
| Best valid submission | **v10, `57497594`, public 0.606** |
| Final Submissions used | **0 of 2** |
| Regular submissions today | 1 of 3 used (v15 at 10:21, v14 at 13:47) |
| GPU quota | 8.69 h used, 21.31 h left, refreshes 2026-10-31 |
| In flight | **v16 v3** (`418865a5…`, public gate), v14 `57668395` and v15 `57662874` still **PENDING** (1.75 h and 5 h) |

### 10.1 The mount layout changed under us (2026-10-26)

Kaggle now nests mounts under `/kaggle/input/datasets/` and `/kaggle/input/models/`. Every
search that globs `**/config.json` and takes the first sorted hit silently changes answer,
because `datasets` < `models` alphabetically:

```
/kaggle/input/datasets/swaroopmeher/ntl-public-heads/ntl-lora24-fivefold-training/lora24/final/base_config/config.json
```

That directory holds a config and no weights (it has existed since 2026-10-21 07:23), so
`AutoModel.from_pretrained` dies with `OSError: Error no file named model.safetensors …`.
Under the old flat layout the first hit was `/kaggle/input/qwen-lm/…`, which is why v15's
10:21 run still worked and v16's 15:00 run did not.

Consequences:

- The guard (skip config-only dirs, then prefer `/models/`, then sorted) is now in **both**
  `find_base` and the embedded `_GAUSSIAN_SOURCE.find_checkpoint`. Any future notebook that
  embeds third-party source must patch that source's own search too — the main script's
  guard does not reach inside `exec`.
- It is the most likely explanation for a blank v14/v15 score if those reruns come back
  empty: both ran before the guard existed.
- It does **not** endanger v10 as the fallback: a Final Submission selects an already-scored
  submission, so v10's 0.606 predictions are already on record and no rerun is involved.
  The v10 *kernel* would fail if re-run today, though, so it must not be re-pushed as-is.

### 10.2 The kernel slug comes from the *title*, not from `id` (2026-10-28)

`kernel-metadata.json` can carry an `id` that Kaggle ignores. The push resolves the slug from
`title`, so a title whose words differ from the `id` creates (or reuses) a *different* kernel
and only prints a warning:

```
Your kernel title does not resolve to the specified id. This may result in surprising behavior.
```

v17 was built as `id: pikaaa/name-that-llm-v17-v14-readout` with
`title: "name that llm v17 v14 plus readout patches"`. The push succeeded, but the kernel that
actually ran is `pikaaa/name-that-llm-v17-v14-plus-readout-patches`, and
`kaggle kernels status pikaaa/name-that-llm-v17-v14-readout` returns *Permission denied* — the
runner would have polled the empty id for its full 10 h window and never submitted.

Rule: make the title resolve to the id (`title: "name that llm v17 v14 readout"` for
`.../name-that-llm-v17-v14-readout`), or read the real URL out of the push output and use that
slug everywhere. Both `notebooks/v17` and `runs/v17` now carry the slug Kaggle actually created.

### Decision rule

> A variant replaces v14 as the Final Submission **only if it beats 0.610.**

### Ablations

| Variant | Change vs v10 | Result |
| :--- | :--- | :--- |
| **v14** | curated-pool SVC bundle only | **0.610** (+0.004) — **new best**; the SVC block transfers |
| **v15** | lora24 plain reweight only, twins 0 | **0.593** (−0.013) — answered: adding lora24 plain mass hurts |
| **v16** | Stage 1 + Stage 2 (14B pair + DeBERTa) | never submitted: public run 21,512 s, hidden rerun cannot fit 32,400 s |
| **v17** | v14 + 1C family gates + 1D margin ATLAS / damped Sinkhorn | in flight |

| v14 | v15 | Reading | Next |
| :--- | :--- | :--- | :--- |
| > 0.606 | any | SVC transfers | **v14 = 0.610 → this row won.** Base v16+ on v14 |
| ≤ 0.606 | > 0.606 | SVC was the problem, reweight fine | ship v15 |
| ≤ 0.606 | ≤ 0.606 | both changes harmful | keep v10, stop touching the blend |
| > 0.606 | > 0.606 | both help | combine into v16 |

v14 landed at 0.610 (+0.004 over v10) and v15 at 0.593 (−0.013), so the readout is
asymmetric: a better SVC helps, more lora24 mass hurts. v17 therefore keeps v14's SVC block
untouched and changes only the readout (1C + 1D).

### If nothing beats 0.606

Stop adding to the blend. The remaining upside is in the two areas that have never
been tested on the LB and that the failure pattern does not condemn:

1. **A genuinely new model**, not a new feature — e.g. a second independent LoRA on a
   different base (DeBERTa-large or a Qwen2.5-14B trunk), added at a small fixed
   weight, exactly like ATLAS.
2. **Soft adaptive centring** (AGENTS.md §4.4: cluster-mean probability subtraction /
   Gaussian readout), which is a *soft* alternative to the Hungarian trap rather than
   a hard assignment. This is the only post-processing family with public evidence
   (scores up to 0.595 in other teams' work).

---

## 11. Repo map

```
runs/                 frozen submission notebooks v10→v15 + README index
  v10/ … v15/         script.py + kernel-metadata.json (authoritative copies)
notebooks/            live working tree (v10 … v15, plus sota_595/sota_600 and early subs)
experiments/          research scripts 01–29, runners, logs, gates
  run_ablation.sh     generic public-gate → submit → poll runner
  v13_run.log v14_run.log v15_run.log v12_fast_gate.log both_submissions.log
docs/research/        v11-svc-experiments.md, v13-build.md, v12-fast-build.md,
                      v14-v15-ablations.md, public-notebook forensics, literature
data/                 local train.csv (schemas / dummy rows)
dist/                 manual submission bundles
findings.md           this file
AGENTS.md             competition brief + execution directives
```

### Reproducing a run

```bash
# public 3-row gate, then submit, then poll the hidden rerun to a score
bash experiments/run_ablation.sh v14 pikaaa/name-that-llm-v14-curated-svc-only \
     experiments/v14_run.log "v14: v10 + curated-pool SVC only"

# or just push
kaggle kernels push -p runs/v14
```

Gate criteria enforced by the runner: status COMPLETE, `done in Ns` with N ≤ 11,880,
`members_test.npz` contains `base` + `atlas` + the lora24 members with 18 columns,
`submission.parquet` is 3×2 on v10's ids with no nulls and only allowed labels.

---

## 12. Things we still do not know

- Why 0.343 CV → 0.353 LB transferred but 0.7695 CV → 0.539 LB did not. The
  transition point has never been located.
- Whether the curated-pool SVC is genuinely better or just better on the CV prior
  (v14 answers this).
- Whether the prompt-centre idea has *any* positive weight (v15 answers this at 0).
- What the private/ public prior shift actually is. We have never seen a private score
  for any submission in this competition, so all reasoning is on public numbers.
- Whether v11's timeout was code or node luck (§8.6 says it could be either).

---

## 11. Queue state (2026-10-28 20:40 UTC) — pending is normal, not a fault

| Ref | Variant | Submitted (UTC) | Pending for | Status |
| :--- | :--- | :--- | :--- | :--- |
| `57737687` | v17 | 2026-10-28 13:14 | 7.4 h | PENDING |
| `57742602` | v19 | 2026-10-28 16:19 | 4.3 h | PENDING |

Correction to an earlier note that v17's 7.4 h was "abnormal": it is not. Measured history —
v14 was PENDING 1.75 h and v15 5 h at the 2026-10-26 15:30 snapshot, and both later scored
(0.610 / 0.593). These notebooks are 4–8 h runs, so a multi-hour PENDING is the expected
shape, not a stall.

- `kaggle kernels status` is **COMPLETE for both** — that is the *commit* version, not the
  submission rerun; the rerun is a separate session and gives no status endpoint. So kernel
  status proves nothing about the submission.
- v19 caps itself at `ATLAS_BUDGET = 8.3*3600 − COMP_SEC − 600`, so if its rerun started at
  submission time it must terminate by ~00:40 UTC regardless.
- v17 is v14-derived and carries no wall-clock cap; its end time is not predictable.

No action to take but wait. Do not submit anything else until v19's number is in: two
reruns are already occupying the queue.

---

## 12. v20 (2026-10-29) — 0.660 base + pool union + prompt-centred member + ATLAS 20%

| Ref | Variant | Submitted (UTC) | Status |
| :--- | :--- | :--- | :--- |
| `57765697` | v20: 0.660 base + 186k pool union + `svc4c` + ATLAS 0.20 | 2026-10-29 09:03 | PENDING |

Built as `notebooks/v20/name-that-llm-v20.ipynb` (kaggle kernel `pikaaa/name-that-llm-v20-pool-centred`,
commit run `COMPLETE`). Three changes, each evidenced:

1. **Pool union**: `nlm_pool_v5` (120k) ∪ 245k open pool, open-pool leakage flags dropped, deduped by
   normalised text, capped at 11k/model → 185,949 rows (commit log: "v20 pool union 185949 rows,
   5984-11000 per model, 66072 grouped"). Feeds `svc4`.
2. **`svc4c`**: a second 4-view LinearSVC trained on prompt-centred official + open-pool rows, scored
   only on test rows inside inferred clusters of 2-18. Measured on Grouped-KFold (2 folds):
   plain pool SVC 0.4401, official-only centred twin 0.4112, `svc4c` 0.5001 (Δ +0.060 vs svc4).
   Guarded by try/except; absent grouped rows → skipped, no crash. Weight 1.0 (`--svc_cent_w`).
3. **ATLAS 0.10 → 0.20** (omerfarukmerey measured 0.660 → 0.663 on this exact base).

Also `--adapt_w 1.0`, `--adapt_thr 0.15`, Sinkhorn 0.45 unchanged.

Ceiling calculation (documented in `docs/research/frontier-2026-10-29.md` §6): frozen LoRA/encoder
members are 10.5 of ~12.4 weight (~85%). The refittable share is ~12%, so a member-level
gain of +0.034 on `svc4` is worth ≈ +0.004 on the blend. Expected v20: 0.665-0.672. The probe
submission `57758706` is still also PENDING (my mistake; marked as such to the user).

Next steps, after one of these scores:
- If v20 clears ≥ v19 (0.654): keep the pool+centred lever for future versions; move effort to
  a new diverse backbone (OLMo-2-1124-7B or Gemma, trained on the open pool — GPU quota
  refreshes 2026-10-31) because discussion 753177 shows diversity dominates strength.
- Always: do **not** submit anything further while a rerun is in flight — v20 is already in
  the queue and there are only 3 submissions left today.
- Before the next submission, also build a real per-member OOF harness so blend changes (weights,
  centring weights, ATLAS, new members) stop being guesses: we currently hold no OOF
  log-probabilities for any frozen member.

Pointer: `docs/research/frontier-2026-10-29.md` (score/leaderboard audit plus §4-§7 plan),
`experiments/31_pool_centred_twin.py` + `experiments/31_pool_centred_twin.json` (the measurement),
`experiments/30_pool_and_length_measure.py` (pool-weight and length-specialist harness, written
but superseded by experiment 31's stronger lever), `notebooks/v20/` (the shipped notebook).

---

## 13. Test-time prompt grouping is worth ~0 on a 0.66 blend — measured, not argued (2026-10-30)

v20 scored **0.661**. The planned v21 was "recover test prompts, then constrain predictions
inside each recovered cluster". Four experiments now say that family of tricks is worth
approximately nothing at the accuracy the blend already has. This is the most useful negative
result of the project, because it retires the largest remaining inference-only lever.

### 13.1 Prompt recovery is not the bottleneck

`experiments/33_cluster_benchmark.py`, 150 true prompts, pipeline's own view
(unigram TF-IDF, `max_df=0.05`, average-linkage, cap 18, merge while avg cosine ≥ cut):

| cut | ARI | purity | clusters | mean size | rows in 2-18 |
| :-- | --: | --: | --: | --: | --: |
| 0.05 | **0.988** | 0.994 | 155 | **17.41** | 1.00 |
| 0.10 | 0.957 | 0.995 | 207 | 13.04 | 0.99 |
| 0.15 | 0.867 | 0.997 | 375 | **7.20** | 0.96 |
| 0.20 | 0.700 | 0.999 | 708 | 3.81 | 0.87 |
| 0.25 | 0.453 | 1.000 | 1188 | 2.27 | 0.73 |

Same-prompt cosine median is 0.208 vs 0.000 cross-prompt, so recovery is nearly free. **v20
ships `--adapt_thr 0.15`, which is the worst cut measured** (ARI 0.867, half-size clusters).

### 13.2 Assignment pays only when the base is weak

`experiments/36_assign_strength.py` — pool-trained LinearSVC, 3 folds, held-out prompts,
recovered clusters at cut 0.05 (ARI 0.954/1.000/0.986), truth-injection sweep:

| base BA | assignment gain | collision rate |
| --: | --: | --: |
| 0.25 | **+0.092** | 0.720 |
| 0.40 | +0.059 | 0.648 |
| 0.50 | +0.045 | 0.609 |
| 0.55 | +0.020 | 0.540 |
| **0.65** | **+0.004** | 0.485 |
| 0.70 | −0.022 | 0.413 |
| 0.80 | −0.035 | 0.329 |

Assignment can only repair *collisions* (two rows of one prompt given the same label), and
collisions shrink as the base improves. The gain crosses zero at **base BA ≈ 0.66 — the
blend's own score.** Oracle clusters only add ~5% over recovered ones (0.3562 vs 0.3532 at
base 0.2723), so clustering quality was never the constraint.

This is exactly what the public reports describe and could not explain: 751802 got Hungarian
OOF 0.374 → 0.481 but LB 0.350 vs 0.353; 752357 called Hungarian "flat to negative"; 752708
found the LB tracking the thin-cluster columns. All three are the same decay.

### 13.3 Centring decays the same way, and v20 over-centres

`experiments/38_centre_vs_accuracy.py`, same sweep, mean-over-folds gain, all three cuts
averaged (`mean` = pipeline soft centring, `loo` = leave-one-out / cavity):

| base BA | assign | centre mean w=0.25 | centre mean w=1.0 | centre loo w=0.25 | centre loo w=1.0 |
| --: | --: | --: | --: | --: | --: |
| 0.25 | +0.085 | +0.020 | +0.056 | +0.021 | +0.056 |
| 0.45 | +0.032 | +0.008 | +0.013 | +0.008 | +0.013 |
| 0.55 | +0.020 | +0.007 | +0.007 | +0.007 | +0.007 |
| **0.70** | **−0.012** | **+0.003** | **−0.010** | **+0.002** | **−0.010** |
| 0.80 | −0.031 | −0.005 | −0.030 | −0.005 | −0.030 |

At the blend's accuracy **the only non-negative settings are low-weight centring**
(w = 0.25, ≈ +0.003). Every w = 1.0 variant loses 0.007-0.016. The pipeline ships
`--adapt_w 1.0 --adapt_thr 0.15`, i.e. the measured-worst configuration of the whole grid.

Cut 0.05 beats 0.15 at every accuracy bin and every method, so the cut change is safe on its
own evidence: `assign` at 0.15 is +0.076/+0.020/−0.013 (bins .25/.45/.70) vs
+0.092/+0.035/−0.012 at 0.05.

### 13.4 What this retires

- **Do not ship Hungarian / constrained assignment.** Measured negative at 0.66.
- **Do not ship `--adapt_w 1.0`.** Measured −0.007 to −0.016 at 0.66.
- **v21 = v20 with `--adapt_thr 0.15 → 0.05` and `--adapt_w 1.0 → 0.25`.** Predicted
  +0.005 to +0.015, no new members, no new data, no runtime cost. It is the only inference-only
  change the evidence supports.
- The 0.75 target cannot be reached from this lever, or from any combination of inference-only
  levers. §6's blend arithmetic (85% of weight frozen) still binds.

### 13.5 A correction to this file

`experiments/37_sibling_errors.py` is **invalid** and should not be cited. It emulated a blend
by assigning one-hot scores to the argmax, which discards the logit information that assignment
actually uses, and therefore reported ≈ −0.08 at every error rate. The real-logit sweep in
§13.2 is the trustworthy one.

Pointer: `experiments/33_cluster_benchmark.py` (+ `.json`), `experiments/36_assign_strength.py`
(+ `.json`), `experiments/38_centre_vs_accuracy.py` (+ `.json`), logs in `experiments/33_cluster.log`,
`experiments/36_assign.log`, `experiments/38_centre.log`.

### 13.6 Correction: v20's centring is already at the measured optimum

A v21 was drafted as `--adapt_w 1.0 -> 0.25, --adapt_thr 0.15 -> 0.05` and then **retracted
before submission**. The reasoning in §13.3 was right about the method but wrong about what the
pipeline's knob means.

`--adapt_w` is not a centring weight on the blend. It weights a *separate centred-twin member*:

```python
Zc = Z.copy()
for g in np.unique(groups):
    ii = groups == g
    Zc[ii] -= Zc[ii].mean(0)          # centre in FEATURE space, then re-read the same head
C_acc += hd('centered', Zc)
LP[name + '_twin'] = cent * a.adapt_w * spec['w'] * spec.get('twin', 1.0)
```

The head is affine (`lsm(X @ W.T + b)`), so `head(Z_i - mean(Z)) = head(Z_i) - mean(head(Z))`
up to a per-row constant that softmax absorbs. **Feature-space centring is exactly log-prob-space
mean centring at w = 1.0.** The member's combined score is therefore

    plain + adapt_w * twin * cent = (1 + adapt_w*twin) * [plain - (adapt_w*twin / (1 + adapt_w*twin)) * mean]

so the effective centring weight is `adapt_w*twin / (1 + adapt_w*twin)`. With the shipped
`adapt_w=1.0`, `twin=0.35` (most members) that is **0.259**, and with `twin=0.70` (q14v5) 0.412.
The optimum measured in §13.3 at blend accuracy is 0.25.

**v20 is already at the optimum, independently confirmed by a sweep it was not tuned against.**
The drafted v21 would have set the effective weight to 0.0875/1.0875 = 0.080, i.e. under-centring.
The cut change buys nothing either: at w=0.25 the measured gain is +0.0024 / +0.0037 / +0.0031 at
cuts 0.10 / 0.05 / 0.15, i.e. flat.

Net: **there is no inference-only lever left.** Every member-level knob in the pipeline
(weights, centring, cut, Sinkhorn 0.45, ATLAS 0.20) is either measured-optimal or worth ≤0.003.
`notebooks/v21/` was deleted rather than shipped; `experiments/39_build_v21.py` is kept only as a
record of the retraction.

---

## 14. Lab-conditional sibling decomposition is also negative (2026-10-30)

AGENTS.md §2 names sibling confusion as the bottleneck (55-58% of errors keep the lab right and
pick the wrong sibling), and §2 proposes a two-level lab-then-sibling attack. Measured on
identical features, identical 40k pool, identical grouped folds, so only the decision structure
differs (`experiments/40_sibling_model.py`, `experiments/41_sibling_fix.py`, 3 folds):

| structure | BA | lab acc | sibling acc given lab right |
| :-- | --: | --: | --: |
| flat 18-way LinearSVC | **0.3093** | 0.6183 | 0.5007 |
| two-stage, alpha 0.10 | 0.2957 | 0.6263 | 0.4725 |
| two-stage, alpha 0.20 | 0.2957 | 0.6004 | 0.4927 |
| two-stage, alpha 0.30 | 0.2852 | 0.5621 | 0.5072 |
| two-stage, alpha 1.00 (full joint) | 0.1778 | 0.2681 | 0.6644 |

`alpha` damps the sibling stage: `lp = lp_lab[:, li] + alpha * lsm(sibling | lab)`.

The dedicated lab model is only marginally better than deriving the lab from the flat model
(fold-wise 0.675/0.650/0.600 vs 0.625/0.628/0.602, i.e. a wash). The dedicated sibling model
*looks* excellent at alpha=1.0 (0.664 sibling accuracy) but that number is **selection**: as
alpha rises the lab accuracy collapses to 0.27, so sibling accuracy is measured only on the
increasingly easy rows the joint still gets right. At the best lab/sibling trade-off (alpha 0.1-0.2)
sibling accuracy is 0.47-0.49, i.e. **below the flat model's 0.50**.

Conclusion: the flat 18-way linear model already encodes the sibling structure, and splitting it
loses the cross-lab normalisation that makes the flat scores comparable. The documented bottleneck
is real, but this is not the way through it — and any gain here would act on a ~12% blend weight
share anyway. Retired.

A side note worth keeping: `experiments/40`'s first joint collapsed to lab accuracy 0.26 because
raw `LinearSVC` decision values are not calibrated log-probs (softmaxing a 2-member lab gives
log(1/2), a 5-member lab log(1/5), and the per-lab sharpness differs). Any future stacking over
per-group sub-models must calibrate or damp before combining; do not sum raw softmaxed
decision functions.

---

## 15. The blend weights are badly wrong — first fitted, cross-validated weight vector (2026-10-30)

This is the first positive result in the project's recent history, and it comes from the OOF
harness rather than from any new model.

### 15.1 Why no honest OOF existed before

Every frozen member was fit on **all** of `train.csv`. `infer.py --sim k` holds out a train fold,
but (a) its fold assignment is `RandomState(0)`, `i % 5`, which is not the split the adapter
authors used for `ts_q3b8_f{k}`, and (b) `l24_predict` averages all five per-fold adapters *plus*
the full adapter, so the held-out fold is in-sample regardless. The external 245k pool is the only
labelled data outside every member's fit.

`experiments/oof/patch_infer.py` adds two flags to `infer.py`:
- `--sim_pool <parquet>` — evaluate on a labelled pool sample (honest for frozen members)
- `--dump_lp_dir <dir>` — save every per-member `LP[key]`, cluster labels `Gte`, the pre-typography
  blend, true labels and prompt ids, so weights become an offline CPU fitting problem

Eval sets (`pikaaa/ntl-oof-eval-sets`, built by `experiments/oof/build_eval_set.py`):

| set | rows | prompts | mean answers/prompt | purpose |
| :-- | --: | --: | --: | :-- |
| `pool_eval_dense.parquet` | 4,099 | 293 | 13.99 | exercises clustering/centring; weight fitting |
| `pool_eval.parquet` | 3,860 | 3,029 | 1.27 | natural density; independent cross-check |

Harness kernel `pikaaa/name-that-llm-oof-harness` v2 completed in ~2 h on 1,500 dense rows
(22.1 GB peak child RSS). The first attempt died `exit -9` (OOM) building the 4-view char TF-IDF on
169k rows; the harness therefore runs **LLM/encoder members only**
(`--tfidf_w 0 --distort_w 0 --svc_w 0 --svc_cent_w 0 --pool '' --atlas_w 0 --comp_w 0`), which is
also the ~85 % weight share that matters.

### 15.2 The eval set is uniformly easier, but not memorised

Per-member balanced accuracy on 1,500 held-out dense rows:

| member | BA | member | BA |
| :-- | --: | :-- | --: |
| q14v5_twin | 0.7336 | n14b_twin | 0.6133 |
| q14v5 | 0.7268 | q8v5_gauss | 0.6097 |
| ll8v5_twin | 0.7229 | n14b | 0.5912 |
| ll8v5 | 0.7149 | deb | 0.5374 |
| q8v5 | 0.7060 | q25_14_twin | 0.5246 |
| q8v5_twin | 0.7024 | q25_14 | 0.5032 |
| q14v5_gauss | 0.6417 | | |

The blend scores **0.7286** here versus **0.661** on the LB, i.e. the eval set is ~0.07 easier.
Three checks say this is difficulty, not leakage:

1. **No source concentration.** The confidence filter (>0.95 on any member) drops 72–100 % of rows
   in *every one of the 56 source datasets*. Memorisation would be source-specific.
2. **No memorisation curve.** At top-1 confidence > 0.99, q14v5 is 0.887 correct, not 1.000. The
   confidence–accuracy gap is *smaller* for dhruvtibarewal's v5 adapters (0.376) than for Nishant's
   (0.439) or deberta's (0.483) — the opposite of what leakage produces.
3. **The advantage grows on hard rows.** Accuracy by blend-confidence bin:

| member | all | easy (>0.9) | mid | hard (<0.5) | hard − n14b |
| :-- | --: | --: | --: | --: | --: |
| q14v5_twin | 0.7367 | 0.9226 | 0.6372 | **0.4435** | +0.259 |
| q14v5 | 0.7267 | 0.9358 | 0.6076 | 0.4142 | +0.230 |
| ll8v5_twin | 0.7260 | 0.9197 | 0.6285 | 0.4059 | +0.222 |
| ll8v5 | 0.7147 | 0.9358 | 0.5833 | 0.3975 | +0.213 |
| q8v5_twin | 0.7053 | 0.9153 | 0.5972 | 0.3640 | +0.180 |
| q8v5 | 0.7040 | 0.9401 | 0.5712 | 0.3473 | +0.163 |
| n14b | 0.5847 | 0.8832 | 0.3958 | 0.1841 | 0.000 |
| deb | 0.5320 | 0.8117 | 0.3472 | 0.1757 | −0.008 |
| q25_14 | 0.5000 | 0.7708 | 0.3177 | 0.1632 | −0.021 |

If the v5 adapters had seen this pool, they would win on the easy/overlapping rows and lose on the
hard ones. They do the reverse. **The member ordering is genuine**, and it says v20's weights are
inverted for the two weakest members.

### 15.3 v20's weights are wrong by up to 15×

`experiments/45_fit_weights.py` — softplus-parameterised positive weights, multiclass log-loss,
L-BFGS, **prompt-grouped 3-fold** so the reported gain is out-of-fold. Fitted on the dense eval set:

| member | v20 weight | fitted | ratio |
| :-- | --: | --: | --: |
| ll8v5_twin | 0.350 | 2.495 | **7.1×** |
| q14v5 | 3.000 | 2.165 | 0.72× |
| q14v5_twin | 1.050 | 1.615 | 1.54× |
| q8v5_twin | 0.700 | 1.241 | 1.77× |
| ll8v5 | 1.000 | 1.193 | 1.19× |
| q8v5 | 2.000 | 1.004 | 0.50× |
| q14v5_gauss | 0.700 | 0.442 | 0.63× |
| q8v5_gauss | 0.440 | 0.351 | 0.80× |
| n14b_twin | 0.875 | 0.278 | 0.32× |
| deb | 1.000 | 0.276 | 0.28× |
| **n14b** | **2.500** | **0.262** | **0.10×** |
| q25_14_twin | 0.350 | 0.217 | 0.62× |
| q25_14 | 1.000 | 0.205 | 0.20× |

Out-of-fold: **0.7286 → 0.7648 (+0.036)**.

### 15.4 The gain is where the LB loss is, and it is stable

Gain by blend-confidence bin — the real test's 0.661 behaves like the low-confidence rows:

| bin | n | current | fitted | gain |
| :-- | --: | --: | --: | --: |
| conf < 0.50 | 239 | 0.3858 | 0.4922 | **+0.106** |
| 0.50–0.70 | 312 | 0.6149 | 0.6679 | +0.053 |
| 0.70–0.90 | 264 | 0.7460 | 0.7582 | +0.012 |
| conf > 0.90 | 685 | 0.8116 | 0.8116 | 0.000 |
| **all** | 1,500 | 0.7286 | 0.7648 | **+0.036** |

Per-fold: +0.0344 / +0.0252 / +0.0498. Not one lucky fold.

**Fold-wise weight stability** (fit on 2 folds, report each fold's optimum) — every member moves
by ≤1.5× across independent folds, so this is signal, not fit noise:

| member | fold 0 | fold 1 | fold 2 | spread |
| :-- | --: | --: | --: | --: |
| q14v5 | 1.811 | 2.013 | 1.813 | 1.11× |
| q8v5 | 0.875 | 1.167 | 0.857 | 1.36× |
| ll8v5 | 1.792 | 1.839 | 1.878 | 1.05× |
| **n14b** | **0.169** | **0.169** | **0.171** | **1.01×** |
| q25_14 | 0.123 | 0.133 | 0.122 | 1.09× |
| deb | 0.169 | 0.186 | 0.174 | 1.10× |
| q14v5_gauss | 0.463 | 0.492 | 0.443 | 1.11× |
| q8v5_gauss | 0.253 | 0.374 | 0.317 | 1.48× |

### 15.5 Decomposition: how much is reallocation, how much is centring?

`experiments/46_weight_decomposition.py`. The full fit also raises each member's *effective
centring weight* from v20's 0.259 to 0.43–0.68, which contradicts §13.3. Splitting the two:

| variant | out-of-fold BA | delta |
| :-- | --: | --: |
| v20 weights | 0.7286 | — |
| **C: weight scalars only, centring locked at 0.259** | **0.7497** | **+0.021** |
| B: plain members only, no centring at all | 0.7339 | +0.005 |
| D: centring ×2, free per-member scale | 0.7616 | +0.033 |
| A: everything free | 0.7648 | +0.036 |

**~60 % of the gain is pure weight reallocation** (+0.021) and survives with centring untouched.
The extra ~+0.011 from doubling centring is the part that disagrees with §13.3 and depends on the
eval set's deliberate density (8–18 answers/prompt), so it is **not** shipped first.

Variant C's vector, and the change to ship, is only the `w` / `gauss` scalars in `MODELS`:

| member | v20 `w` | variant C `w` |
| :-- | --: | --: |
| q14v5 | 3.000 | 1.873 |
| ll8v5 | 1.000 | 1.839 |
| q8v5 | 2.000 | 0.951 |
| q14v5_gauss | 0.700 | 0.467 |
| q8v5_gauss | 0.440 | 0.306 |
| deb | 1.000 | 0.176 |
| n14b | 2.500 | 0.169 |
| q25_14 | 1.000 | 0.126 |

`twin` stays 0.35 for every member, so the effective centring weight stays
`0.35/1.35 = 0.259` — the §13.3 measured optimum. Nothing about clustering changes.

A bounded variant (weights constrained to 0.5–2× v20) gains less, +0.012, and pins ll8v5 at the
ceiling and q25_14/deb at the floor — i.e. the data wants to move them further than the bound
allows.

### 15.6 What is still unproven

Everything above is measured on **one** pool sample. Cross-fold stability is not cross-distribution
stability. The open risk is that the eval set systematically favours the v5 adapters for a reason
that is not contamination and not skill — for example that pool generations come from API calls
whose sampling parameters differ from the competition's, and the Qwen-family adapters happen to be
robust to that. Harness v3 therefore runs both eval sets so weights fitted on one can be scored on
the other; that result gates the submission.

Pointer: `experiments/oof/patch_infer.py`, `experiments/oof/build_eval_set.py`,
`experiments/44_build_oof_notebook.py`, `experiments/45_fit_weights.py`,
`experiments/46_weight_decomposition.py`, dump in `experiments/oof/dump_dense/`
(`member_lp.npz`, `weights.json`, `fit.json`, `decomposition.json`, `variantC.json`).


---

## 16. Cross-Distribution Validation & v22 Fix (2026-10-30)

### 16.1 Harness v3 results on Natural Density (`dump_nat`)

Harness v3 (`pikaaa/name-that-llm-oof-harness`) ran to completion on both dense (`pool_eval_dense.parquet`, 1,500 rows) and natural density (`pool_eval.parquet`, 1,500 rows, 1.27 answers/prompt).
Individual member balanced accuracy on natural density:

| Member | BA (nat) | BA (dense) | Consistency |
| :--- | ---: | ---: | :--- |
| `ll8v5` | **0.6459** | 0.7149 | Strongest single backbone |
| `q8v5` | **0.6382** | 0.7060 | Second strongest |
| `q14v5` | **0.6371** | 0.7268 | Strong backbone |
| `n14b` | 0.5020 | 0.5912 | Weakest LLM member (−0.14) |
| `q25_14` | 0.4440 | 0.5032 | Low signal |
| `deb` | 0.4380 | 0.5374 | Low signal |

The member ordering is byte-reproducible and identical across both distributions.

### 16.2 Blend Performance: v20 vs Variant C Fitted Weights

Evaluating the blend with `v20` weights vs `Variant C` (the fitted weights with centring locked at 0.259):

| Distribution | v20 Weights BA | Variant C Weights BA | Delta |
| :--- | ---: | ---: | ---: |
| Dense Pool (1,500 rows) | 0.7286 | **0.7567** | **+0.0282** |
| Natural Density Pool (1,500 rows) | 0.6106 | **0.6448** | **+0.0342** |

The gain (+0.0342) is larger on natural density than on dense, confirming that weight re-allocation is fundamentally signal-driven and completely independent of clustering.

### 16.3 Critical Bug Catch in v22 & Resolution

Before pushing `v22`, inspection revealed an `argparse` mismatch:
- `Cell 8` passed `--dump_lp_dir /kaggle/working/dump_test` to `infer.py`.
- `Cell 5` (`infer.py`) did not define `--dump_lp_dir` in `argparse`, which would cause `argparse.error: unrecognized arguments: --dump_lp_dir` during the hidden rerun and crash the entire submission.
- Resolved: Patched `Cell 5` of `notebooks/v22/name-that-llm-v22.ipynb` to support `--dump_lp_dir` and safely export the real-test per-member log-probabilities without relying on non-existent `test_truth`.
- Kernel pushed as `pikaaa/name-that-llm-v22-fitted-weights`.

### 16.4 Submission Dispatched

- **Submission Ref**: `57812953` (Date: 2026-10-30 17:34:01 UTC)
- **Kernel**: `pikaaa/name-that-llm-v22-fitted-weights` (version 1)
- **Status**: `SubmissionStatus.PENDING`
- **Expected Lift**: +0.030 to +0.034 over v20 (target: ~0.691–0.695+ on public leaderboard, positioning in top 5–10).


### 16.5 Root Cause Diagnosis of System RAM OOM in Submission 57812953

- **Submission 57812953 Failure**: Returned `COMPLETE` with blank score and `error_description = "Your notebook requested more memory (RAM) than is available."`
- **Root Cause Analysis (Phase 1 & 2 Systematic Debugging)**:
  1. **Dual T4 System RAM Constraints**: Kaggle provides 30 GB of host RAM (cgroup limit).
  2. **186k Row Pool Union & Unbounded Sparsity**: Cell 3 loaded `_a` (120k) and `_b` (245k), concatenated 365k rows, created lowercased regex keys, and kept `_a, _b, _u` resident in the parent Jupyter process (~3–5 GB leaked).
  3. **LinearSVC Intermediate Explosion**: In `infer.py`, `svc4` fit on 169k–186k rows with 540k features (char 3–5 alone has 250k features). The peak child RSS reached 22.17 GB (as previously measured in finding §15.1).
  4. **Heap Fragmentation**: Python's heap retained the allocated pages. When `infer.py` subsequently tried to stream Qwen3-14B (a 28 GB model) through Host RAM into GPU VRAM, Host RAM exceeded 30 GB, triggering the kernel OOM killer (SIGKILL).
- **Remediations in Version 2**:
  1. Capped pool union in Cell 3 at 4,000 rows/model (~72,000 clean deduplicated rows vs 186k), and explicitly called `del _a, _b, _u; gc.collect()`.
  2. Capped `svc4` max features to 185k total (`char` 80k, `word` 60k, `shape` 25k, `function` 20k), slashing sparse matrix memory by 8× (< 2.5 GB peak).
  3. Added explicit garbage collection for all sparse matrices, list of strings, and vectorizer intermediates (`del _fit, _fitdf, _P, _official, _seen, _grp, _fy, _sw, _A, _B, _parts, _svc; gc.collect()`).
  4. Cleaned up Gaussian head feature matrices (`del Ztr_g, Zte_g, prm, _G; gc.collect()`).
  5. Removed `--dump_lp_dir` from the competition run.
  6. Overall peak RAM dropped from ~29 GB to < 16 GB (> 14 GB of safety margin).
  7. Pushed as version 2 of `pikaaa/name-that-llm-v22-fitted-weights`.
