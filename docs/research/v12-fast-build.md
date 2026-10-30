# v12-fast — runtime-corrected v12 (0.80 / 0.20 lora24 twins)

## Why this exists

The literal `notebooks/v11/script.py` and `notebooks/v12/script.py` cannot score:
v11's public 3-row run was **16,076 s (4.47 h)**, already beyond the inferred
~4 h submission limit.  v12 inherits that legacy, adds a dead ModernBERT branch
(the mounted encoder dataset only contains Ettin), and has the pre-fix legacy
label order.  Submitting either as-is is a guaranteed timeout, not a score.

v12-fast keeps the only meaningful v12 change — the lora24 prompt-centre
weighting — on the validated v13 code path:

- v10 runtime envelope (not the v11 legacy)
- curated-pool SVC (`ankiteshtiwari/ntl-curated-pool`), q25 dead-head removed
- corrected legacy label remap (`_old_cols`)
- no ModernBERT / no `dddmdd`
- `EXTRA_LORA24 = {public 0.10, public_twin 0.80, seed7 0.05, seed7_twin 0.0,
  claude 0.05, claude_twin 0.20}`

v13 is the same code with `public_twin 0.70, claude_twin 0.15`, so the pair is a
one-dict A/B on the prompt-centre weight.

## Artifact

- `notebooks/v12_fast/script.py`
  - sha256 `e1e1ca488020a557d1d690d550f09213c4ef78c6baf3802911402b43a2ac89a0`
  - 76,372 bytes, `py_compile` and `ast.parse` clean
  - no `modernbert`, `dddmdd`, or `pool_v1` references
- `notebooks/v12_fast/kernel-metadata.json`
  - id `pikaaa/name-that-llm-v12`, GPU on, internet off, 12 datasets
- `experiments/build_v12_fast.py` — deterministic builder from the v13 source

## Run plan

`experiments/run_v12_fast_public_then_submit.sh`:

1. wait for the Saturday 00:00 UTC GPU quota reset (wake 00:01)
2. push `pikaaa/name-that-llm-v12`, capture the version
3. poll the public 3-row run
4. gate:
   - status `COMPLETE`
   - `done in Ns` with `N <= 11,880` (v10 envelope 11,331 s)
   - `members_test.npz` contains `base`, `atlas`, `lora24_public_prob`
   - `REWEIGHT lora24_public +0.10` and `REWEIGHT lora24_claude +0.05`
   - if twins are exercised, their weights are exactly 0.80 / 0.20
   - `submission.parquet` is 3x2 on the v10 ids with valid labels
5. submit the gated version (regular submission, not a Final)
6. poll the hidden rerun and print `V12FAST_FINAL`

Expected public gate: ~3.1 h.  Expected hidden rerun: ~3.7 h.  If v13 and v12
run concurrently, both scores should land by roughly 07:00 UTC; if Kaggle
serialises the GPU runs, by roughly 14:00 UTC.  The 30 h weekly quota covers
both (v13 ~6.8 h + v12-fast ~6.8 h).

## Public 3-row caveat

The 3-row public pool usually has no eligible prompt groups, so the `*_twin`
members are absent there and their weights are only exercised on the ~4k hidden
pool.  The public gate therefore checks the base members and the reweight lines;
if the twins appear it also checks their exact weights.  The hidden test is the
first place the 0.80/0.20 centre actually changes predictions.
