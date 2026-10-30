# Experiments & Empirical Research

This directory contains standalone, reproducible Python scripts, evaluation harnesses, and benchmarks developed to test specific mathematical and architectural hypotheses offline.

## Experiment Directory Map

| Number / Prefix | Script | Purpose & Key Finding |
| :--- | :--- | :--- |
| **01–04** | `01_baseline_tfidf.py`<br>`02_stylometric_lgbm.py`<br>`03_stacked_meta_ensemble.py`<br>`04_deberta_baseline.py` | Classical baselines. Proved that character n-grams + quotation/dash typography achieve ~0.35 Balanced Accuracy, but saturate. |
| **05–12** | `05_qwen_lora.py`<br>`06_frozen_qwen_probe.py`<br>`07_anova_quotient_lda.py`<br>`08_honest_cv.py`<br>`10_contrastive_lora_train.py` | Foundation model feature probes. Proved early-to-mid transformer layers retain significantly higher stylometric fidelity (~0.39) than final task-semantic layers (~0.31). |
| **13–18** | `13_master_grand_ensemble.py`<br>`14_subsampled_cv_postproc.py`<br>`15_group_recovery_audit.py`<br>`17_sibling_and_length_error_audit.py` | Grouping audits and length error analysis. Passages <60 words score 0.27 vs 0.46 for >160 words. |
| **19–24** | `19_deepseek_adversarial_invalidation.py`<br>`20_invalidation_evidence.py`<br>`21_invalidation_honest.py`<br>`23_integration_gate.py` | Multi-agent adversarial audit of early ensemble claims. Exposed in-sample leakage and invalidation of Hungarian assignment. |
| **25–29** | `25_cluster_recovery.py`<br>`26_prompt_recovery_content.py`<br>`27_end_to_end.py`<br>`29_global_vs_cluster_ot.py` | Prompt cluster recovery benchmarks. Unigram TF-IDF with average linkage achieves ARI 0.988 at cut 0.05. |
| **30–32** | `30_pool_and_length_measure.py`<br>`31_pool_centred_twin.py`<br>`32_build_v20.py` | External pool union (186k rows) and prompt-centred LinearSVC member (`svc4c`), measuring +0.060 member BA jump. Built v20 (0.661 LB). |
| **33–38** | `33_cluster_benchmark.py`<br>`36_assign_strength.py`<br>`38_centre_vs_accuracy.py` | **The Grouping Saturation Law.** Proved that Hungarian assignment and mean-centering gains decay to zero at 0.66 base BA and turn negative at 0.70+. |
| **39–41** | `39_build_v21.py`<br>`40_sibling_model.py`<br>`41_sibling_fix.py` | **v21 Retraction & Hierarchical Decomposition.** Showed flat 18-way models outperform two-stage lab→sibling models (0.3093 vs 0.2957). |
| **44–48** | `44_build_oof_notebook.py`<br>`45_fit_weights.py`<br>`46_weight_decomposition.py`<br>`47_build_v22.py`<br>`48_patch_v22_memory.py` | **Honest OOF Pool Harness & Weight Fitting.** Discovered severe weight misallocation in existing public blends (Variant C: +0.0342 BA on natural distribution). Patched Kaggle Dual T4 Host RAM OOM. |
