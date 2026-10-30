# Notebooks & Submission Kernels

This directory contains the working Jupyter notebooks and self-contained Kaggle submission scripts developed across the competition.

## Directory Structure

| Directory | Primary File | Description | Best Public Score |
| :--- | :--- | :--- | :--- |
| `baseline_sub/` | `baseline.ipynb` | TF-IDF (word + char n-grams) + Stylometry LightGBM | 0.355 |
| `fused_sub/` | `fused.ipynb` | 4-view n-grams + dense stylometrics (early fusion) | 0.315 |
| `grandmaster_sub/` | `grandmaster.ipynb` | First LLM LoRA integration (Qwen3-8B readout + n-grams) | 0.539 |
| `sota_595/` | `script.py` | 8B+14B two-stage LoRA with early Gaussian readouts | 0.595 |
| `sota_600/` | `script.py` | The v5 baseline: dual LoRA (8B+14B) + centred twin + HodgeRank + Sinkhorn | 0.600 |
| `v10/` | `script.py` | Clean v5 recipe + 10% ATLAS v2 pre-Sinkhorn | **0.606** |
| `v11/` | `scriptv11.py` | v10 + curated pool SVC weight 0.25 (runtime timeout in rerun) | — |
| `v12/` | `scriptv12.py` | ModernBERT-large test branch (rejected: offline weight missing) | — |
| `v12_fast/` | `scriptv12f.py` | v13 with twin weights 0.80/0.20 (isolated A/B test) | 0.584 |
| `v13/` | `scriptv13.py` | Curated pool SVC + lora24 top-level twin reweight | 0.592 |
| `v14/` | `scriptv14.py` | Clean ablation: v10 + curated pool SVC **only** | 0.610 |
| `v15/` | `scriptv15.py` | Clean ablation: v10 + lora24 plain reweight **only** | 0.593 |
| `v16/` | `scriptv16.py` | Deepthink Stage 1+2 architecture (gated: 21.5k s public run) | — |
| `v17/` | `scriptv17.py` | v14 + family StyleGram gates + margin-gated ATLAS & damped Sinkhorn | 0.611 |
| `v18/` | `scriptv18.py` | Part Two trunks ablation with source-first staging | — |
| `v19/` | `name-that-llm-v19.ipynb` | The 0.654 frontier: 6-backbone LoRA24 primary + frozen complement | 0.654 |
| `v20/` | `name-that-llm-v20.ipynb` | 0.654 base + 186k pool union + prompt-centred `svc4c` + ATLAS 0.20 | **0.661** |
| `v22/` | `name-that-llm-v22.ipynb` | Fitted LLM blend weights (Variant C) via honest OOF pool harness | In flight (v2) |
| `master_sota/` | `ensemble.py` | Master stacking scripts and ensemble blending prototypes | — |
| `oof/` | `eval_harness.ipynb` | Honest out-of-fold pool evaluation harness for weight fitting | — |

## Kaggle Deployment

For authoritative frozen push scripts with attached `kernel-metadata.json`, see the [`runs/`](../runs) directory.
Push a run directly using:

```bash
kaggle kernels push -p runs/v20
```
