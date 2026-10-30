# v11 SVC experiments — pool weighting, transductive TF-IDF, views

All numbers are **3-fold `StratifiedGroupKFold` grouped on `prompt_id`**, 18-class
balanced accuracy, SVC member alone (not the full ensemble). Random seed 42.
Fit = competition train + external pool; folds only split the competition train
rows, every pool row is present in every fit.

Reference recipe: LinearSVC(C=0.5), `class_weight="balanced"`, views
`char(3,5)/250k`, `word(1,3)/180k`, `shape(2,6)/60k`, `function(1,3)/50k`,
`min_df=2`, sublinear TF-IDF, punctuation-preserving token pattern.

## Pool composition and weight

`pool_v1.parquet` = 34,940 usable rows (30–300 words, deduped). With
`class_weight="balanced"` the raw pool dominates the 2,699 train rows ~13:1, so
the pool's effective sample weight is the main regularisation knob.

| config | pool | pool weight | BA | min recall | median recall |
|---|---|---:|---:|---:|---:|
| A_v1 (v10 recipe) | pool_v1 | 1.00 | 0.4138 | 0.073 | 0.403 |
| D_v1 | pool_v1 | 0.50 | 0.4216 | 0.067 | 0.407 |
| E_v1 | pool_v1 | 0.25 | 0.4390 | 0.073 | 0.420 |
| A_v1 | pool_v1 | 0.35 | 0.4297 | 0.073 | 0.420 |
| A_v1 | pool_v1 | 0.10 | 0.4275 | 0.100 | 0.427 |
| A_v1 | pool_v1 | 0.05 | 0.4045 | 0.133 | 0.393 |
| A_v1 | pool_v1 | ~0 | 0.3189 | 0.107 | 0.277 |

Smooth peak at **0.25**; train-only collapses, so the external pool is essential
but must not dominate. **v11 v3 uses 0.25.**

## SVC regularisation / vocabulary

| config | C | min_df | BA |
|---|---:|---:|---:|
| pw0.25 | 0.25 | 2 | 0.4323 |
| pw0.25 | 0.50 | 2 | **0.4390** |
| pw0.25 | 1.00 | 2 | 0.4256 |
| pw0.25 | 2.00 | 2 | 0.4138 |
| pw0.25 | 0.50 | 3 | 0.4386 |
| pw0.35 | 0.50 | 2 | 0.4297 |

C=0.5 and min_df=2 are already optimal; min_df=3 is a tie.

## Views

| views | pool | pw | BA |
|---|---|---:|---:|
| A: char(3,5)/250k, word(1,3)/180k, shape(2,6)/60k, function(1,3)/50k | v1 | 0.25 | **0.4390** |
| B (Vishak): char(1,5)/400k, word(1,3)/300k, shape(2,6)/80k, function(1,4)/100k | v1 | 1.00 | 0.4186 |
| B | v1 | 0.25 | 0.4356 |
| A | mixed | 1.00 | 0.3897 |
| B | mixed | 1.00 | 0.3916 |
| A | mixed | 0.25 | 0.4064 |
| A | mixed | 0.10 | 0.4090 |
| A | v1 0.25 + mixed 0.10 | — | 0.4434 (min rec 0.127) |

Mixed pool = `ntl_pool_v4` + `nlm_matched_clean` + `ntl_mega_corpus`, deduped and
sampled to 34,940 rows. It never wins alone; a small 0.10 share on top of v1 adds
only +0.0045 BA, inside fold noise. Not used.

## Transductive TF-IDF (the v4 change)

The private 0.667 recipe fits each vectorizer on `pool + train + test` texts
(labels unused) and transforms test. Grouped CV simulates this by fitting on
`train_fold + pool + val_fold` and predicting the val fold.

| config | BA | min recall | median recall |
|---|---:|---:|---:|
| non-transductive, A views, pw0.25 | 0.4390 | 0.073 | 0.420 |
| **transductive, A views, pw0.25** | **0.4538** | **0.113** | 0.407 |
| transductive, A views, pw0.40 | 0.4530 | 0.113 | 0.403 |
| transductive, A views, pw0.10 | 0.4430 | 0.127 | 0.410 |
| transductive, B views, pw0.25 | 0.4497 | 0.107 | 0.390 |

**+0.0148 BA and +0.040 worst-class recall** over the non-transductive SVC, with
A views still the best set. This is the v4 candidate; it costs only the
vectorizer fit over `+N_test` documents.

## Kernel ledger

| kernel | purpose | GPU | result |
|---|---|---|---|
| `pikaaa/ntl-svc-bench` | pool/views screen | no | A_v1 0.4138, A_mix 0.3897, B_mix 0.3916 |
| `pikaaa/ntl-svc-bench2` | weight sweep | no | B_v1 0.4186, pw0.5 0.4216, pw0.25 0.4390 |
| `pikaaa/ntl-svc-bench3` | weight tails | no | pw0.10 0.4275, pw0.05 0.4045, pw0 0.3189 |
| `pikaaa/ntl-svc-bench4` | C / min_df / pw0.35 | no | C0.5+pw0.25 best (0.4390) |
| `pikaaa/ntl-svc-bench5` | mixed pool at optimal weight | no | v1+0.10mix 0.4434 (noise-level gain) |
| `pikaaa/ntl-svc-bench6` | transductive A views | no | **0.4538** at pw0.25 |
| `pikaaa/ntl-svc-bench7` | transductive B views | no | 0.4497 (A views still better) |
| `pikaaa/ntl-svc-bench8` | transductive + v1/mixed share | no | **0.4605** at mix0.10 (min rec 0.147); mix0.05 0.4553 |
| `pikaaa/ntl-svc-bench9` | class-mass equalisation | no | pending |
| `pikaaa/ntl-svc-bench10` | family gates on SVC member | no | **+0.0033** only (Gemini +11, Claude -2) -> skip |
| `pikaaa/ntl-svc-bench11` | curated 80k pool, transductive | no | pending |
| `pikaaa/ntl-svc-bench12` | Sinkhorn vs block-Hungarian decode | no | running |

## Curated pool (`ankiteshtiwari/ntl-curated-pool`)

80,115 rows, 18 classes (min 2,701 / max 6,235), 13 upstream sources, `register_ok`
quality flag, and **zero exact-text overlap with competition train**. It is a superset
of `pool_v1` (33,734 shared rows) plus newer sources.

Two traps found while auditing a fresh public pipeline that uses it:

1. **`split == "train_regen"` rows (2,168) are regenerations on official train
   prompts.** They do not overlap train text but leak train prompt topics into a
   grouped CV fold, so they must be dropped for honest validation. The final fit may
   keep them, but benchmarks here drop them.
2. The pool is large enough that fold-time TF-IDF fit dominates CPU time; bench11
   fits each view on `train_fold + pool + val_fold` (transductive), ~80k docs.

## Public OOF members found 2026-10-22 (bigger lever than SVC tuning)

`dddmdd/ntl-finetuned-encoders` publishes 5 prompt-grouped folds of
ModernBERT-large (two variants, A/B) plus a 5-fold UltraBERT-380M, together with
`oof_*.csv` probabilities. `pierrevanvolsem/name-that-llm-final-models` publishes
OOF probabilities for two pool-trained Qwen3.5 backbones and a linear model.

Audit on the 2,699 train rows (all folds prompt-grouped; every prompt sits wholly
inside one fold, so no sibling leakage):

| member | OOF balanced acc | min class recall |
|---|---:|---:|
| ModernBERT-large A | 0.5442 | 0.289 |
| ModernBERT-large B | 0.5494 | 0.268 |
| UltraBERT-380M | 0.5349 | 0.293 |
| Pierre 0.8B / 4B / linear blend | 0.4694 | 0.173 |
| SVC v1 transductive (bench6) | 0.4538 | 0.113 |
| SVC v1+mix transductive (bench8) | 0.4605 | 0.147 |

Blending Pierre + ModernBERT A/B + UltraBERT reaches **0.5883 OOF** on train,
before any of the v11 LoRA/stylegram members. The ModernBERT models are the only
members here that are stronger than the whole SVC line, so v12 should add them.
A/B are two training variants of the same base; B is marginally stronger alone but
A+B is better in a blend (+0.0085 over B alone against Pierre).

`pikaaa/ntl-public-oof` is a private 2 MB dataset with the six OOF CSVs (source and
licence in `ATTRIBUTION.md`), used by `pikaaa/ntl-svc-oof-weight-fit` to fit the
ModernBERT weight against the final SVC recipe in log space.

## v11 runtime (3-row public test)

| stage | v10 | v11 v2 (+sdpa) | v11 v3 (sdpa removed, reference ATLAS) |
|---|---:|---:|---:|
| n-gram SVC | 206 s | ~270 s | 198 s |
| lora8b | 700 s | 1,053 s | 886 s |
| lora14b | 1,112 s | 1,764 s | 1,691 s |
| legacy stack (tokattn_q35 dominates) | ~8,500 s | ~11,000 s | ~13,700 s |
| ATLAS | 525 s | 845 s | 652 s |
| **total** | **11,339 s (3.15 h)** | **17,896 s (4.97 h)** | **16,076 s (4.47 h)** |

`attn_implementation="sdpa"` was slower than the default on T4 and was reverted.
Batched ATLAS (`source_chunk=3`, batch 4) was also slower on the 3-row probe and
was reverted to the reference `predict(batch_size=1)` path. The dominant runtime
is the legacy stack's Qwen3.5-9B token-attention load, which is I/O-bound on the
mounted safetensors and varies by node.
