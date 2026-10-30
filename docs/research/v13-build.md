# v13 build: v10 envelope + curated-pool SVC + lora24 prompt-centre

Status: **built and scheduled**, not yet scored. v10 (`57497594`, public **0.606**) remains the
only valid fallback. v11 (`57537119`) is dead: `SubmissionStatus.COMPLETE` with blank score and
`error_description = "Your submission notebook exceeded the allowed runtime."`

## Why v13 exists

`v10` public runtime was **11,331 s (3.15 h)** and it scored on the hidden rerun. `v11` public
runtime was **16,076 s (4.47 h)** and its hidden rerun timed out. The binding constraint is
therefore the submission runtime, not model quality. v13 must stay inside v10's compute envelope
while adding only changes that are free at test time.

`v12` is rejected. It inherits v11's slower legacy and adds the SVC/ModernBERT paths. The
`dddmdd/ntl-finetuned-encoders` dataset contains only **`ettin-encoder-400m`** (5 folds,
11.75 GB). The ModernBERT-large OOF files in `/tmp/ddd/` are real, but there are **no
ModernBERT weights** in any mounted dataset, so v12's `MODERNBERT_PROB` branch cannot run at
test. The 0.6501 OOF blend that used ModernBERT is therefore not deployable and is not the v13
target.

## v13 changes (zero extra model passes)

1. **Curated-pool SVC instead of `pool_v1`.**
   - Dataset: `ankiteshtiwari/ntl-curated-pool` (43,323,083-byte parquet).
   - Filters: drop `split == "train_regen"`, keep `register_ok`, keep the 18 labels,
     30-300 words, deduplicate text. Result: **87,734 rows, all 18 classes**.
   - `LinearSVC(C=0.5, class_weight="balanced", max_iter=5000)` with
     `sample_weight = 1.0` for train and **0.25** for pool.
   - TF-IDF vocabulary is fit on `train + pool + test` (transductive, unlabeled test text only);
     the classifier is still fit only on `train + pool`.
   - Bench11 grouped-CV SVC: **0.4667** versus v10's full-pool **0.414**.
2. **Top-level lora24 prompt-centred reweight.**
   - The legacy block computes the readouts, but they sit inside its 15% share with
     `LORA_TWIN_W = 0.15`, i.e. 0.0225 effective.
   - v13 captures `lora24_*` probabilities in the legacy `add_member` and adds them again at the
     top level, after the 0.85/0.15 mix and before ATLAS:
     `lora24_public 0.10`, `lora24_public_twin 0.70`, `lora24_seed7 0.05`,
     `lora24_seed7_twin 0.0`, `lora24_claude 0.05`, `lora24_claude_twin 0.15`.
   - Measured lora24 signal: plain **0.4804** prompt-grouped OOF, prompt-centred **0.5960**.
     The D/F fit on the public diverse base put the centre at 1.1 with ModernBERT; 0.70 is
     deliberately below that because the test-time groups are detected, not true prompts.
3. **Dead `q25_7b` LM head removed.**
   - `W["q25_7b"] = 0.0` and there is no `gauss_q25_7b` group head (the four measured views are
     q3_8b, glm9b, q3_4b, debv3l). Its train+test forward pass is pure dead weight.
   - Removal is output-identical and frees roughly 10 minutes of the 4 h budget.

No ModernBERT, no Ettin, no extra LoRA passes, no extra ATLAS pass.

## Label-order bug fixed

The legacy stack sorts its labels, so its Claude order is
`claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5`.
The production kernel uses
`claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5, claude-fable-5`.
The legacy `LOGIT` is already remapped by `_old_cols`, but v12's top-level reweight added the
captured `lora24` probabilities **without** reindexing them. v13 applies
`_prob = np.asarray(_prob)[:, _old_cols]` before every reweight and in the saved member payload.
A unit test with distinct label orders gives a zero max-diff against the expected remapped
result.

## Runtime gate

The background job `experiments/run_v13_public_then_submit.sh` waits until **Saturday 00:01 UTC**
(the weekly GPU reset), pushes `pikaaa/name-that-llm-v13`, and runs the public 3-row kernel.
It submits **only if**:

- the public run is `COMPLETE`;
- `done in` is **<= 11,880 s (3.30 h)**;
- `members_test.npz` contains `base`, `atlas`, and `lora24_public_prob`, all 18 columns;
- the 3-row `submission.parquet` has the same ids as the v10 public submission, no nulls, and
  only the 18 valid labels.

Otherwise it stops without spending a submission. The submission is a regular first-of-day
submission, not one of the two Final Submissions. v10 stays the Final fallback until v13 beats
0.606.

## Evidence and expectation

- v10 public: 11,331 s, LB **0.606**.
- v11 public: 16,076 s, submission timed out.
- SVC curated pool: **0.4667** grouped-CV (bench11).
- lora24 prompt-centred OOF: **0.5960**.
- lora24 plain OOF: **0.4804**.
- v13 expected range: **0.63-0.66** LB, with the main uncertainty being test-time prompt-group
  detection. The top-10 cutoff observed earlier was about **0.644-0.649**; this is an attempt to
  reach it, not a verified result.

Artifacts: `notebooks/v13/script.py`, `notebooks/v13/kernel-metadata.json`,
`experiments/build_v13.py`, `experiments/v13_run.log`.
