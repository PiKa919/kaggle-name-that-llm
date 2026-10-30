# v19 — the 0.65x frontier is a different primary, not more members

Written 2026-10-28 after a verification pass that overturned three earlier claims. Everything below is
either read out of someone else's published code or pulled live from the Kaggle API; nothing is
inferred from our own offline CV.

## The claim that started it

We were told the public 0.654 notebook is "our stack plus four members we lack" (Part Two trunks,
spectral style heads, a full-text Qwen3.8-27B, the typography likelihood). That was wrong on two of
the four rows and wrong about the architecture. The correction changes the plan.

## What the 0.654 actually is

`romantamrazov/name-that-llm-who-am-i` is `dhruvtibarewal/nlm-v27` (0.651) plus a typography
likelihood. Its pipeline is two files:

| file | role | ours? |
| :--- | :--- | :--- |
| `comp_stack.py` | the public frozen-feature stack — `W = {'stylegram': 1.516, …}`, `GATE_LAM`, `BALANCE_ALPHA`, `GAUSS_HEADS` | **byte-identical to our v14/v18 embedded source** (sha `91f107b4fd34`) |
| `infer.py` | the primary: a six-backbone LoRA24 ensemble | **we have no equivalent code at all** |

The weights are decisive:

| | ours (v10–v18) | nlm-v27 / 0.654 |
| :--- | :--- | :--- |
| primary | the frozen-feature stack ("the 0.563 blend") | `q14v5` w 3.0, `n14b` w 2.5, `q8v5` w 2.0, `ll8v5` w 1.0, `q25_14` w 1.0, `deb` w 1.0 |
| that same frozen stack | **is** the primary | a **complement at weight 0.15** (`--comp_w 0.15`) |

So the 0.65x frontier uses our entire pipeline as 15% of its input. No amount of member tuning on our
side of that ratio reaches 0.65: our primary is a 0.563-grade stack, theirs is a 0.63-grade
six-backbone ensemble.

Not present in the 0.654, despite what we said earlier: a 27B member (`27B` appears zero times in its
code; `Qwen3.8` zero times; `nf4` only in a bitsandbytes config) and the spectral style heads
(`kirilltym/ntl-spectral-style-heads` is not in its dataset list). Those belong to the 0.643/0.652
lineage.

## The evidence that member tuning cannot close the gap

The original author's own ablation table, each row a single change on a fixed base:

| base | change | result |
| :--- | :--- | :--- |
| nlm-v27 (0.651) | blend in their own five-backbone model | 0.648 |
| nlm-v27 (0.651) | swap `q8v5` for a 27B NF4 member (0.642 standalone) | 0.638 |
| nlm-v27 (0.651) | re-enable the 9B token-attention member | 0.640 |

Every member-level change on a strong base loses 0.003–0.013. Their summary: *"adding more LoRA
backbones or members to a strong ensemble moved the score by at most 0.004 in either direction."*

## Live leaderboard, 2026-10-28

`page_size=200`: top **0.728** (Hod Carriers), six teams ≥ 0.702, the 200th row is **0.651**. Our
0.610 is outside the top 200. Every row above us is at or beyond the public notebook frontier, so the
frontier is the entry ticket, not the win.

## Input verification before spending a submission

All 14 datasets, 9 kernel sources and 7 model sources were checked through the API:

- `dhruvtibarewal/nlm-l24-adapters-v5` — 54 files: `q14v5`, `q8v5`, `ll8v5`, each with `_f0…_f4`
  and a final, each with `adapter_config.json` + `adapter_model.safetensors` + `heads.pt`.
  *Trap:* `dataset_list_files` defaults to `page_size=20` and silently truncates. The first page looks
  like `q8v5` and all five `q14v5` folds are missing. They are not. Paginate before concluding.
- `dhruvtibarewal/nlm-l24-adapters` — 88 files, includes `heads/q3_8bx.npz` and `ts_q3b8*`,
  `ts_q3b14*`, `ts_q25_7b`, `ts_llama8*`, `ts8_s7*`.
- `dhruvtibarewal/nlm-matched-clean` — `nlm_pool_v5.parquet`, `nlm_matched_clean.parquet`.
- `nishantdahal/ntl-two-stage-lora` — `lora8b_v1`, `lora14b_v1` + `pool_v1.parquet`.
- `nishantdahal/ntl-handwriting-blend` — `lora_qwen3_14b`, `lora_qwen25_14b`, `deberta_v3_large`,
  `pool_v4.parquet`.
- `josephadamski91/ntl-atlas-qwen3-8b-v1` — `atlas_bundle.json` present.
- `pjmathematician/ultrabert-380m-ntl`, `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head`,
  `seyitkaangunes/ntl-extra-api-passages`, `abdullah1afify/ntl-generations`,
  `dophuchung/ntl-proxy-generations`, `lucasboesen/ntl-free-proxy-generations`,
  `prvsiyan/prvsiyans-extended-name-that-llm-dataset` — all present.
- All 9 kernel sources are `COMPLETE` with their output files (`tokattn_final.pt`,
  `enc_*_train.npz`, the lora24-s7 adapters, the GLM/ELECTRA/Desklib weights, the Llama asset cache).

## What v19 changes, and what it does not

Base: `alexchilton/name-that-llm-0-654-full-ensemble-what-failed`, which is the 0.654 pipeline plus a
commit-only cheap path (`RERUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None`). Its
`comp_stack.py` and `gauss.py` are byte-identical to Roman's; its `infer.py` is Roman's *pre-v39*
version, i.e. the code that scored 0.654. (Roman's current `infer.py` adds a "v39 per-row active
member normalization" experiment that blends 0.5/0.5 with the reference and writes
`candidate_reference_0654.parquet`; we deliberately use the reference version.)

Exactly one change:

```python
def _llama_dir():
    try:                       return mdir('/llama-3.1/', '/8b/')
    except AssertionError:     return os.path.dirname(glob.glob('/kaggle/input/**/llama_asset_verification.json', recursive=True)[0])
BL = _llama_dir()
```

Reason: the Llama-3.1-8B weights are attached as the `avazaku/name-that-llm-verified-llama-asset-cache`
kernel, not as a `model_sources` entry, so the author's `/llama-3.1/`+`/8b/` path glob has no obvious
match. The fallback resolves the cache by its own manifest file and asserts `config.json` sits next to
it. The cheap commit run prints which branch fired, so the log answers this rather than a guess.

Everything else — the six `MODELS` entries and their weights, `--svc_w 1.5`, `--adapt_w 1.0`,
`--sinkhorn 1.0`, `--post 1`, `--atlas_w 0.10 --atlas_margin 0.3`, `--comp_w 0.15` — is untouched.

## Files

```
kernel   pikaaa/name-that-llm-v19-0-654-reproduction
notebook notebooks/v19/name-that-llm-v19-0-654-reproduction.ipynb   (sha256 56329d420c7088fc…)
archive  runs/v19/
log      experiments/v19_commit.log  (cheap commit run)
```

## Decision rule

| v19 score | reading | next |
| :--- | :--- | :--- |
| 0.643–0.655 | reproduction holds; we are on the frontier | one measured change at a time from here |
| 0.60–0.64 | an input is missing or a stage degraded | read the log for which member failed its in-sample check |
| ≤ 0.61 | the pipeline did not reproduce on our account | revert; the 0.610 lineage stays our best |

Runtime is self-limiting: `ATLAS_BUDGET = 8.3*3600 - COMP_SEC - 600`, so a slow rerun covers fewer
ATLAS passages rather than overrunning. Published reruns of identical code span 0.643–0.651 for that
reason.
