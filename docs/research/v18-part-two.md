# v18 — Part Two trunks ON, with the embedded stack moved first

Built 2026-10-28, not yet pushed (waiting on v17's score, which decides whether 1C/1D stay on).

```
kernel   pikaaa/name-that-llm-v18-part-two-source-first   (title resolves to the id)
script   notebooks/v18/script.py
sha256   0051c2abd7b6fe3e…
archive  runs/v18/
```

## Why not train a member from scratch

The obvious reading of "get a trained member" is to train a Qwen3-14B LoRA ourselves. The published
recipe for the checkpoints everybody uses says that is the wrong lever:

| Fact (from `nishantdahal/two-stage-lora-on-the-public-pool`) | Number |
| :--- | :--- |
| Qwen3-8B LoRA, public pool only (stage 1) | 0.483 grouped CV |
| Qwen3-8B LoRA, pool + host data (stage 1+2) | **0.569** |
| Qwen3-14B LoRA, same recipe | **0.569** (14B = 8B) |
| 8B + 14B + n-gram blend | 0.593 |

Two consequences:

1. A freshly trained 14B scores **0.569 standalone** — below the 0.610 blend it would be added to.
   It is also the *same* recipe and the *same* pool as `lora8b_v1` / `lora14b_v1`, which v10 already
   uses. Re-training it is a re-roll, not new evidence.
2. Cost: 90 min on an RTX 5090 in bf16. A T4 has no bf16 and roughly a third of the memory bandwidth,
   so the same run is 5–8 h. Our remaining weekly quota is ~21 h, and the inference run needs its own
   2–3 h. One from-scratch member would consume the whole week for an expected ±0.004.

The same author's ablation list agrees: *"a larger backbone on its own (14B = 8B)"*, *"extra family
gates on top of the blend"* (what v17's 1C does — they measured 0.559 → 0.559), *"member additions
moved the score by at most 0.004 in either direction"*.

## What actually separates 0.610 from 0.654

The 0.654 public notebook is *our stack* — same `lucasboesen` kernels, same `nishantdahal` two-stage
LoRA, same ATLAS — plus four things we do not have:

| Missing ingredient | Role there | Cost to us |
| :--- | :--- | :--- |
| Part Two trunks (`nishantdahal/ntl-handwriting-blend`: `lora_qwen3_14b`, `lora_qwen25_14b`, `deberta_v3_large`) | the reference itself (fleongg's *Part Two Complement*) | **already attached**, blocked only by v16's runtime bug |
| Kirill's spectral style-head bank (`kirilltym/ntl-spectral-style-heads-paired-4404`) | 9–10% after ATLAS, reuses LoRA24 features, no extra forward | cheap: one dataset, one extra readout |
| Full-text Qwen3.8-27B NF4 classifier | 10% on every row | expensive: 27B forward on 4,000 rows |
| Bayesian typography likelihood (train-only, short stylistic markers) | 0.651 → 0.654 | cheap: CPU |

So the 0.610 → 0.65 gap is **not** training, it is running evidence we already have plus two prefit
assets. v18 attacks the first row of that table.

## What v18 changes

Exactly one modelling change against v17: **the Part One trunks are replaced by the Part Two pair**
(`lora_qwen3_14b` w 0.35 + twin 0.150 + gauss 0.10, `lora_qwen25_14b` w 0.20 + twin 0.079, DeBERTa
0.10), which is v16's Stage 2 setting — the one that never got to the leaderboard.

Plus the structural fix that makes it runnable:

- the embedded public stack now executes **before** any 14B trunk, and its namespace is deleted
  immediately (only `_old_prob` and the four gate tensors survive);
- the source's `load_truncated_decoder` now prints the share of parameters that landed on CPU, so a
  repeat of the offload regime is visible in the log rather than only in the wall clock;
- `SVC_POOL = "curated_pool.parquet"` is decoupled from `STAGE2_TRUNKS`, so this run keeps v14's
  0.610 SVC block instead of silently switching to `pool_v4` at 35% share (v16 did both at once).

## Runtime

| Phase | v16 (stack last) | v18 expectation |
| :--- | :--- | :--- |
| n-gram SVC | 515 s | ~515 s |
| embedded stack | 17,967 s | ~2,900 s (v15 ran the whole job in 3,981 s) |
| 14B pair + DeBERTa | 2,674 s | ~2,700 s |
| ATLAS | 769 s | ~770 s |
| **public total** | **21,512 s** | **~7,000 s** |

Gate is 11,880 s; hidden rerun must fit 32,400 s.

## Correction: which block actually blew up in v16

An earlier draft of this note, and the header comment of `notebooks/v18/script.py`, blamed the
`lora24_public` member for a 10,257 s block. That is wrong. Re-read of
`/tmp/v16public/name-that-llm-v16-stage2.log` against
`/tmp/v15public/name-that-llm-v15-lora24-plain-reweight-only.log` gives:

| Member | v15 (no 14B resident) | v16 (14B pair resident) | slowdown |
| :--- | ---: | ---: | ---: |
| `lora24_public` | 24 s | 44 s | 1.8× |
| `tokattn_q35` | **115 s** | **13,885 s** | **121×** |
| `tokattn_deb` | 105 s | 38 s | — |
| `ultrabert` | 189 s | 1,739 s | 9.2× |
| whole embedded stack (member time) | 2,848 s | 17,640 s | 6.2× |

The 40-row in-sample check inside `tokattn_q35` alone took 227 s in v15 and 10,245 s in v16. So the
offload hit the attention-pooled member hardest — `tokattn_q35` is 121× slower — while the 5-fold
LoRA24 readouts barely moved (1.8×). The fix is unchanged — run the stack while VRAM is still clean —
but the attribution above is the measured one.
The header comment in the pushed kernel (v1, sha `0051c2ab…`) still carries the old wording; it is a
comment only and does not affect execution. It will be corrected when the next version is cut.

## Local verification

`/tmp/v18_smoke.py` runs the whole file in `NTL_SMOKE=1` with the trunks/DeBERTa/ATLAS stubbed, on 30
training prompts and a 69-row pseudo test. It reaches the end (`done in 54s`) and prints the ordered
evidence that the reordering is sound:

```
SOURCE PATCHED: config-only checkpoint dirs skipped, /kaggle/input/models preferred
SOURCE KEPT: stack prob (69, 18), gates ['gemini', 'glm', 'gpt']
SOURCE DONE and namespace freed (53s)
GATE gpt lam 0.15 over 5 columns / gemini 4 / glm 2   (claude needs the API dataset)
BLEND two-stage .85 + original four-Gaussian stack .15
ATLAS gated: 69 rows at 0.12, 0 at 0.02
Sinkhorn damped: 5 steps, exponent 0.50
```

## Measured result (public run, 2026-10-28 15:49 → 18:52 UTC)

Total **10,874 s**. The ordering fix worked — the source ran first, its namespace was freed, then the
trunks — but the run is still unshippable, and the reason is not the one this note predicted.

| phase | predicted | measured |
| :--- | ---: | ---: |
| n-gram SVC | 515 s | 717 s |
| embedded stack | ~2,900 s | **8,336 s** |
| Part Two trunks + DeBERTa | ~2,700 s | 2,260 s |
| ATLAS | ~770 s | 384 s |
| **total** | ~7,000 s | **10,874 s** |

`members_test.npz` contains `base`, `atlas`, `deberta_v3_large_prob`, `ids`, `labels`, `stack`, and the
3-row `submission.parquet` is valid, so the pipeline itself is correct. 10,874 s for **3** test rows
cannot fit 32,400 s for 4,000, so this variant stays off the leaderboard.

**The offload hypothesis is disproved.** The new diagnostic reported `0.0% of parameters on CPU` for
every trunk (glm4_9b, qwen3-8b, qwen2.5-7b, qwen3-4b), and the log has no OOM, no fallback and no
`device_map` warning. Yet one member is still 45× slower than the same member in v15, which ran the
identical block:

| member | v15 (whole job 3,981 s) | v18 (source first) | ratio |
| :--- | ---: | ---: | ---: |
| `debv3l` | 1,678 s | 1,329 s | 0.8× |
| `lora24_public` / `seed7` / `claude` | 24 / 10 / 13 s | 32 / 17 / 17 s | ~1.4× |
| `tokattn_q35` | **115 s** | **5,188 s** | **45×** |
| `tokattn_deb` | 105 s | 35 s | 0.3× |
| `ultrabert` | 189 s | 857 s | 4.5× |

So the cost is concentrated in `tokattn_q35` (and to a lesser degree `ultrabert`), with the GPU
holding every parameter. The plausible cause is host-level variance — memory-bandwidth contention on
a shared T4 host — not our own VRAM management. We cannot prove that from inside the run, and the
honest statement is: **unexplained, measured twice, and specific to the token-attention member.**

Two facts that make it moot for the frontier path: the public 0.654 pipeline sets
`NTL_DROP_Q35=1` and **drops `tokattn_q35` entirely**, and the author's ablation shows that turning it
back on *loses* 0.011 (0.651 → 0.640). A member that is both slow and harmful is not worth chasing.

## Decision rule

| v18 | Reading | Next |
| :--- | :--- | :--- |
| ≥ 0.63 | Part Two is the real gap | v19 = v18 + spectral heads, then + 27B full-text |
| 0.61–0.63 | pair helps a little | keep, then add spectral heads |
| < 0.61 | the pair does not transfer (as lora24 re-weighting did not) | revert to v14; go to the fork route |
