# runs/ — archived submission notebooks, v10 → v19

Every variant that produced (or was meant to produce) a leaderboard entry, frozen as
the exact `script.py` that was pushed plus its `kernel-metadata.json`. These are the
authoritative copies — `notebooks/` is the live working tree and keeps changing.

Push one with:

```bash
kaggle kernels push -p runs/v14
```

| Dir | Kernel id | Public LB | Ref | Script sha256 | One-line description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `v10` | `pikaaa/name-that-llm-v10-atlas` | **0.606** | `57497594` | `ebae105f…` | **Best.** Two-stage LoRA 8B+14B + pool n-grams + centred twin + 4 Gaussian group heads + ATLAS v2 at 10% pre-Sinkhorn |
| `v11` | `pikaaa/name-that-llm-v11` | — (timeout) | `57537119` | `de7c5426…` | v10 + SVC pool weight 0.25 + transductive TF-IDF + checkpoint guard + JSONL telemetry. Public run 16,076 s, hidden rerun exceeded the runtime limit |
| `v12` | `pikaaa/name-that-llm-v11` (metadata bug) | never submitted | — | `3bc2b1fc…` | v11 legacy + ModernBERT-large branch. **Dead**: the mounted encoder dataset has no ModernBERT weights, so the branch cannot run offline |
| `v12_fast` | `pikaaa/name-that-llm-v12-fast` | 0.584 | `57627241` | `757a5627…` | v13 with `public_twin 0.80 / claude_twin 0.20` — the one-dict A/B of v13 |
| `v13` | `pikaaa/name-that-llm-v13` | 0.592 | `57623259` | `b6fdd413…` | curated-pool SVC + lora24 prompt-centre reweight (`0.70 / 0.15`) + q25 dead head removed |
| `v14` | `pikaaa/name-that-llm-v14-curated-svc-only` | **0.610** | `57668395` | `8eccd0cf…` | v10 + curated-pool SVC **only**; lora24 reweight disabled |
| `v15` | `pikaaa/name-that-llm-v15-lora24-plain-reweight-only` | **0.593** | `57662874` | `fb785104…` | v10 + lora24 **plain** reweight only (`0.10/0.05/0.05`), twins off |
| `v16` | `pikaaa/name-that-llm-v16-stage2` | never submitted | — | `418865a5…` | deepthink Stage 1 + Stage 2: pool share 0.35, family gates applied, margin-gated ATLAS, damped Sinkhorn, qwen3/qwen25-14B LoRA pair + DeBERTa-v3-large on `pool_v4`. Public run **21,512 s**, so the 4,000-row hidden rerun cannot fit 32,400 s — runtime gate failed, no slot spent |
| `v17` | `pikaaa/name-that-llm-v17-v14-plus-readout-patches` | running | — | `27db3bb7…` | **v14 base + exactly two readout changes**: 1C apply the four family StyleGram gates, 1D margin-gated ATLAS weight + damped 5-step Sinkhorn. SVC block, lora24 weights and Stage 2 untouched (Stage 2 off) |
| `v18` | `pikaaa/name-that-llm-v18-part-two-source-first` | public run in flight | — | `0051c2ab…` | **v17 + Part Two trunks** (`lora_qwen3_14b` / `lora_qwen25_14b` / DeBERTa from `ntl-handwriting-blend`), with the embedded public stack moved **before** them so accelerate cannot offload it. An **ablation only**: it keeps our 0.563-grade stack as the primary, which the frontier evidence says cannot reach 0.65 |
| `v19` | `pikaaa/name-that-llm-v19-0-654-reproduction` | reproducing | — | `56329d42…` | **The public 0.654 pipeline**, reproduced with full credits and one defensive path fix (Llama-3.1-8B resolved from the asset-cache kernel). Six-backbone LoRA24 primary + our frozen stack at 0.15 + ATLAS + typography. Notebook, not script |

`v10/public_3row_submission.parquet` is the 3-row public output kept as the reference
the submission gates compare ids against.

## Reading order

1. `findings.md` (repo root) — full score history, what worked, what regressed, the
   infrastructure failures.
2. `docs/research/v14-v15-ablations.md` — the exact v10→v13 change set and the answered
   v14/v15 matrix (v14 0.610, v15 0.593).
3. `docs/research/v16-stage2.md` — the Stage 2 build and why it cannot ship (21,512 s).
4. `docs/research/v17-readout-patches.md` — v17 = v14 + 1C family gates + 1D gated ATLAS
   and damped Sinkhorn, plus what the 3-row public gate cannot see.
5. `docs/research/v18-part-two.md` — why a from-scratch trained member cannot reach 0.70 (0.569
   standalone, 5-8 h on T4), plus an errata correcting which member blew up in v16.
6. `docs/research/v19-frontier-architecture.md` — **read this first.** The 0.65x frontier uses a
   six-backbone LoRA24 primary and our whole stack as a 0.15 complement; verified from their code,
   their ablations and the live leaderboard.
7. `docs/research/v11-svc-experiments.md` — the grouped-CV sweep that motivated the
   SVC pool weight (and that failed to transfer).
8. `docs/research/v13-build.md`, `docs/research/v12-fast-build.md` — build rationale.

## Caveats

- `v10` is **not byte-reproducible** against today's datasets: `swaroopmeher/ntl-public-heads`
  gained a `base_config` directory after Oct 21, which has `config.json` but no weights.
  v10's unguarded `find_checkpoint` would now pick it and crash. v11 onward carry the
  guard; v10 is kept verbatim as the historical best.
- `v11` and `v12` are kept for the record only. Do not push them: v11 exceeds the
  runtime limit, v12 needs weights that are not attached.
- `v12/kernel-metadata.json` still points at `pikaaa/name-that-llm-v11` (copy-paste
  bug). It was never pushed as its own kernel.
