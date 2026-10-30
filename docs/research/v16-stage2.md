# v16 — deepthink Stage 1 + Stage 2 on the v10 base

**Base:** `notebooks/v10/script.py` (public LB **0.606**, ref `57497594`).
**Kernel:** `pikaaa/name-that-llm-v16-stage2`, script sha256 `418865a5…` (v3), archived in `runs/v16/`.
**Rule:** v16 replaces v10 only if it beats 0.606 on the LB. Otherwise v10 stays the fallback.

## What the doc claimed, and what the code actually does

`deepthink.md` lists six bugs. All six were re-checked against the real v10 source before
touching anything, and the three that shaped v16 are:

| Claim | Verdict on v10 |
| :--- | :--- |
| Pool dominates the SVC objective | **True, differently than stated.** v10 has no `SVC_POOL_WEIGHT`; it uses `class_weight="balanced"` on `train ∪ pool_v1`, so per class the pool holds ~96% of the weight. The doc's 88.27% figure comes from v13/v14's flat `0.25` multiplier, which never existed in v10. |
| `NEW_LORA_GAUSSIAN` only ever ran for the 8B trunk | **True.** `if ckpt_name == 'lora8b_v1':` at v10 line 333 gated the Gaussian fit, so `lora14b_v1` never got one. |
| `GATES` computed and never read | **True, and stronger than stated.** In v10 the gates are not even in the main script — they live inside the embedded `_GAUSSIAN_SOURCE` and are dropped with its namespace. |

The other three (naive group-mean centring at n<18, disjoint clustering metrics, hard
uniform Sinkhorn) are real but were handled with a switch or a measured decision, not a
blind patch — see below.

## Changes, each behind a named switch

| Switch | Default | Change |
| :--- | :--- | :--- |
| `POOL_SHARE` | `0.35` | Class-balanced share weighting: every class's total SVC weight is identical and official `train.csv` keeps 65% of it, whatever the pool's per-class counts are. Total weight still equals the row count, so `C=0.5` keeps v10's effective regularisation. |
| `PATCH_UNIFIED_GROUPS` | **`False`** | The doc's mixed content+hidden clustering. **Measured off.** |
| `PATCH_GATES` | `True` | Applies the four family StyleGram gates that v10 computed and discarded, reading them out of the source namespace and remapping columns by label name (the source sorts its labels; our Claude order differs). |
| `PATCH_ATLAS_GATE` | `True` | ATLAS weight 0.12 on rows whose top-1/top-2 margin is < 0.25, 0.02 otherwise, instead of a flat 0.10. |
| `PATCH_DAMPED_SINKHORN` | `True` | 5 steps at exponent 0.50 instead of 100 hard steps. |
| `STAGE2_TRUNKS` | `True` | Swaps the two Part One trunks for `lora_qwen3_14b` + `lora_qwen25_14b` + `deberta_v3_large` and `pool_v1` for `pool_v4`. |

### Why `PATCH_UNIFIED_GROUPS` is off

Measured on 40 real competition prompts (720 rows):

| Grouping | clusters | rows in 2–18 | mean size | pair precision | pair recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| v10 stop-word TF-IDF @ 0.85 | 104 | 681/720 | 14.3 | **1.000** | **0.742** |
| mixed content+hidden @ 0.60 | 215 | 624/720 | 7.7 | 1.000 | 0.341 |
| mixed @ 0.50 | 437 | 417/720 | 4.5 | 1.000 | 0.120 |

Same-prompt content distance median 0.816, different-prompt p10 0.968 — v10's 0.85 cut
already sits inside the gap, and no cluster exceeded 18 rows. The mixed metric only shrinks
the twin/Gaussian row set. Kept as a switch for a future probe.

### Stage 2 blend mass

The three new members split the **same total mass** v10 gave its two trunks (plain 0.655,
twin 0.229, gaussian 0.10) instead of inventing new weights:

| Member | plain | twin | gaussian |
| :--- | ---: | ---: | ---: |
| `lora_qwen3_14b` | 0.350 | 0.150 | 0.10 (d128, T3.0) |
| `lora_qwen25_14b` | 0.200 | 0.079 | — |
| `deberta_v3_large` | 0.100 | — | — |
| **total** | **0.650** | **0.229** | **0.10** |

`gauss_w > 0` is what decides whether a trunk also pays a forward pass over the 2,699
training passages (the Gaussian needs same-trunk training features), so qwen25 skips it.

## Verified before pushing

- Checkpoint specs read from the real `meta.json` files: `lora_qwen3_14b` base `Qwen/Qwen3-14B-Base`, keep 27, layers `[2,4,6,9,13,17,22,27]`, max_len 447; `lora_qwen25_14b` base `Qwen/Qwen2.5-14B`, keep 32, layers `[3,5,8,11,16,21,27,32]`; `deberta_v3_large` `pool: mean_last`, max_len 384, head `weight (18,1024)`. All three label orders equal v10's `LABELS`.
- `pool_v4.parquet`: 75,783 rows, **zero** exact-text and **zero** `prompt_id` overlap with official `train.csv`, all 18 labels present (2,338–6,736 each), all rows already 30–300 words.
- `find_base` ported from v13: skips config-only directories (`swaroopmeher/ntl-public-heads` gained a weightless `base_config` after Oct 21).
- Local unit harnesses on real data: pool weighting (share 0.350, per-class totals equal at 4360.11), clustering, gate column mapping for all four families (including the Claude order mismatch), ATLAS gate + damped Sinkhorn (rows sum to 1, all 18 classes reachable), and the full trunk loop with synthetic features (train pass only when `gauss_w > 0`, twin/Gaussian touching only twin rows, exact-pair check passing).
- The DeBERTa in-sample check is a **warning, not an assert**: if the head does not reproduce its fit (acc ≤ 0.25 on 40 training rows) the member is dropped with a loud log line rather than killing a run that has already consumed a submission slot.

## Public-run failure history (all three attempts, nothing submitted)

The runner only submits after the public gate passes, so none of these spent a submission.

| Attempt | Died at | Cause | Fix |
| :--- | ---: | :--- | :--- |
| v1 (`0a7ed9f6…`) | 7 s | my own assert: the four gated families cover 16 labels, not 18 (`qwen3.8-max`, `kimi-k3` are single-member labs) | assert now checks the real sizes `claude 5 / gpt 5 / gemini 4 / glm 2` and names the two ungated labels |
| v2 (`fa4cd413…`) | 39 min | `OSError: no file named model.safetensors … lora24/final/base_config` inside the embedded source | the source has its **own** `find_checkpoint`; patched at runtime (below) |
| v3 (`418865a5…`) | — | — | pushed |

### The `base_config` bug, and why it only appeared now

Kaggle moved mounts under `/kaggle/input/datasets/` and `/kaggle/input/models/`. Alphabetically
`datasets` < `models`, so `**/config.json` now yields
`datasets/swaroopmeher/…/lora24/final/base_config/config.json` **before**
`models/qwen-lm/qwen-3/transformers/8b-base/1/config.json`. That `base_config` directory holds a
config and nothing else, so `AutoModel.from_pretrained` fails on it. Under the old flat layout the
first hit was `/kaggle/input/qwen-lm/…`, which is why v15's run at 10:21 today still worked.
The file itself has existed since 2026-10-21 07:23; only the sort order changed.

`find_base` (the main script's lookup, ported from v13) already had the guard. The embedded
`_GAUSSIAN_SOURCE` had its own copy without one. v3 replaces that function at runtime via
`str.replace` (asserted to match exactly once, then `compile()`d) with:

- skip any candidate directory that has no `model.safetensors`, `pytorch_model.bin`, or a sharded
  index — i.e. exactly the same guard as `find_base`;
- prefer paths containing `/models/`, then sorted order — which reproduces v15's choice of
  `/kaggle/input/models/…` for every model_type under the new layout;
- the `dir_name` branch (encoder directories) gets the same guard: a config-only encoder directory
  would fail identically, so skipping it can only help.

Unit-tested on a fake mount tree: the unpatched function picks the weightless `base_config` (reproducing
the v2 failure), the patched one picks the real `/models/` copy, the encoder `dir_name` lookup skips a
weightless sibling, sharded weights are accepted, and a genuine miss still returns `None`.

## Risks

1. **`POOL_SHARE=0.35` is the biggest single change.** v10's SVC was ~96% pool-weighted. Cutting the pool to 35% of each class's weight is the doc's Patch 1A, but it is untested here and could easily move the score either way.
2. `deberta_v3_large`'s head may have been fitted with a different pooling than masked mean; the public run's warning line will say so.
3. Runtime: v14 with an 81k-row pool SVC took 6,741 s and v10 took ~4,000 s, so v16 should land in the 6,500–9,000 s range against a 32,400 s limit.
4. Public 3-row test has no multi-row clusters, so `twin_rows` and the Gaussian paths are invisible to the public gate; only the hidden rerun exercises them.

## What v16 is not

Stage 3 items — the spectral readout bank, the full-text Qwen3.8-27B member, within-lab
hierarchical Gaussian heads, short-crop augmentation, a non-Qwen trunk — are **not** in
v16. They need either a >27B model (infeasible on 2×T4) or new trained backbones
(multi-day). Per the doc, those are what the 0.673–0.710 tier requires; v16 is the
0.640–0.650 tier.
