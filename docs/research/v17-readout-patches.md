# v17 — v14 base + the two cheap readout patches (1C + 1D)

Pushed 2026-10-28 09:17 UTC.

```
kernel   pikaaa/name-that-llm-v17-v14-plus-readout-patches   (version 1)
script   notebooks/v17/script.py   92,702 bytes
sha256   27db3bb71c24e2d07fe29443b66f18a4216a840e29fa0faa3c8ef54d0c2f3b34
archive  runs/v17/
```

The `id` in `kernel-metadata.json` said `…-v17-v14-readout`, but Kaggle derives the slug from
the **title**, so the live kernel is the `…-plus-readout-patches` one (findings §10.2).

## Why this variant

v14 (0.610) is the best valid submission and the new base. v16's Stage 2 is dead on runtime
(21,512 s public → the 4,000-row hidden rerun cannot fit 32,400 s), and every attempt to add
lora24 mass lands at 0.584–0.593. What is left that is cheap, measured in deepthink, and has
never been on the LB is the **readout**: v10 computed a family gate tensor and then threw it
away, and ATLAS was applied at a flat 10% to every row.

v17 therefore changes **nothing** in the evidence layer and exactly two things after it.

## Exact change set vs v14

| # | Change | Code |
| :--- | :--- | :--- |
| 1C | Apply the four family StyleGram gates the embedded source computes and discards, remapped by label name into our column order (the source sorts labels; our Claude order differs) | `LOGIT[:, family] += lam * log_softmax(gate)` with `lam = 0.35` for claude, `0.15` for gpt / gemini / glm |
| 1D-a | ATLAS weight from the row's top-1/top-2 margin instead of a flat 0.10 | `w = 0.12 if margin < 0.25 else 0.02` |
| 1D-b | Damped Sinkhorn instead of 100 hard steps | 5 steps, `P *= (target / colsum) ** 0.50` |

Unchanged: the whole SVC block (`curated_pool.parquet`, `class_weight="balanced"`, flat
`SVC_POOL_WEIGHT=0.25`, transductive vocabulary), every blend weight in `W` / `TWIN_W` /
`LORA_TWIN_W`, the twin and Gaussian paths, the clustering cut (1B stays `False`), and
`EXTRA_LORA24 = {}`.

`POOL_SHARE = None` reproduces v14's weighting exactly — verified against `runs/v14/scriptv14.py`:
the two blocks are byte-identical except `class_weight="balanced"` → `class_weight=_cw`, where
`_cw == "balanced"` in this branch.

Stage 2 stays in the file but off (`STAGE2_TRUNKS = False`), so the pool is `curated_pool.parquet`
and the trunks are the Part One 8B/14B LoRA pair.

## What the public gate can and cannot see

- **Can see:** that the script runs, the runtime envelope, the SVC block, and the 3-row
  submission's ids/labels.
- **Cannot see:** 1D-b entirely — the damped Sinkhorn is behind `if len(test) >= 50`, and the
  public test has 3 rows. It also cannot see the twin/Gaussian paths (all 3 rows are
  singletons) or the effect of any gating (3 rows carry no signal).

So a green public run is a *plumbing* gate. The score is the only evidence.

## Runtime budget

v14's public run was 6,741 s and v15's 3,981 s; v17 adds no model and no pass over the corpus,
only a gate tensor already in memory. Expect ≈6,750–7,000 s, i.e. well inside the runner's
11,880 s gate and inside the hidden rerun's 32,400 s.

## Decision rule

| v17 vs 0.610 | Reading | Next |
| :--- | :--- | :--- |
| > 0.610 | the readout was the remaining upside | v18 = v17 + `POOL_SHARE 0.35`; then re-open 1B |
| ≈ 0.610 (±0.005) | gates and gated ATLAS are neutral | try each of 1C and 1D alone, then 1A |
| < 0.610 | the flat 10% ATLAS and the discarded gates were already tuned | revert to v14 and look at data, not readout |

Any variant replaces v14 only if it beats **0.610**.
