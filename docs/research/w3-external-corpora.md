# W3 — External corpora & artifact inventory for "Name That LLM"

**Date:** 2026-10-21 (evening, updated through 22:30 local)
**Author:** W3 (research worker)
**Scope:** every public Kaggle dataset/artifact relevant to *Name That LLM* (18-way authorship attribution, balanced accuracy): text corpora, merged pools, LoRA/head weights, prompt banks, tools, mirrors. Judge each on: (a) does it contain competition-label-consistent data our stage-1 pool lacks, (b) is it a drop-in member for our 0.600 stack, (c) license/safety for a public notebook.
**Our reference pipeline:** local 0.600 public LB (two-stage Qwen3-8B + 14B LoRA trained on `pool_v1` of nishantdahal/ntl-two-stage-lora + 4-view TF-IDF LinearSVC, centred twins, sibling stylometry, typographic priors, global Sinkhorn).
**Companion docs:** W1 (notebook forensics) already ranked *techniques*; this doc ranks *data and artifacts to attach*, with fresh measurements.

> **Reading the numbers.** All scores below are **claimed by the dataset/notebook authors** (titles, markdown, dataset pages) unless printed under "measured". My measured numbers come from two probes I ran locally today (section 3). The transfer probe is a *linear-SVM proxy for stage 1*, not our LoRA; treat deltas as directional evidence, not guaranteed LB deltas. Public LB is ~800–1,200 rows; a delta under ~0.01 on LB is inside sampling noise.

---

## 1. TL;DR

1. **The best attachable data upgrade I measured is `ahmedberatozer/ntl-pool-v4` + three new small sources (chosunghwan, pardheev, sharipov): transfer BA 0.3623 (pool_v1, our current) → 0.3986 (pool_v4) → 0.4068 (+new3).** +0.045 vs today's stage-1 pool in an identical linear probe, train texts disjoint from all pools (verified 0 exact overlaps). The gain concentrates exactly where our weak classes are (kimi 0.11→0.35, gemini-3.1 0.49→0.60, claude-opus-4-8 0.24→0.32, claude-fable 0.16→0.23 recall on train).
2. **`prvsiyan/ntl-public-in-style-pool-v1` is the curated sibling of pool_v1 (38,642 rows, per-model typography veto).** It alone transfers 0.3682 (> pool_v1 0.3623) and it is what the 0.563→0.604 lineage trains stage-1 on. CC-BY-SA-4.0, 16–18 MB. Cheap and safe.
3. **Fidelity to competition labels varies 4× across sources:** chosunghwan 0.366–0.374 exact / 0.88 family (best public source I have measured), vs tadeuhayashida 0.09, kenandemirci 0.12, maniceet 0.14, bingli283 0.17, msemenoff 0.17. pool_v4 already contains ~23k rows from sources scoring ≤0.19 exact — **including `arxitekton`/`arxitekton_v2` (15,433 rows, 21% of pool_v4) whose provenance is not traceable to any public Kaggle dataset I could find.** Filter or downweight before retraining.
4. **Model artifacts worth attaching (drop-in members, not pools):** zbigniew `lora-instyle` and `lora-claude-head` (Apache-2.0; W1: held-out member 0.456→0.516, LB 0.569→0.574 at w=0.17), josephadamski `atlas-qwen3-8b-v1` (W1: +0.007, 7/7 replications, ATLAS blended at 10%), `pjmathematician/ultrabert-380m-ntl` (OOF 0.408, Apache-2.0, already used by the 0.563 notebook), `lightningv08` OLMo2 assets and `swaroopmeher` frozen mirrors (5-fold lora24-s7, in-style, tokattn).
5. **New (this week) sources not yet in any pool:** chosunghwan/ntl-gemini-new-prompts (2,331 Gemini rows, fidelity 0.879 fam — best measured), pardheev/ntl-proxy-extra-passages (2,977, GPT-heavy), sharipov7ar (750 clean same-prompt rows), kenandemirci (1,625 GLM/Gemini — API ids match labels but probe fidelity 0.24 family: **do not trust without validation**), trnhip/ntl-api-gen, dedruck/ntl-proxy-gen, ahmedberatozer/ntl-gen2s3-q8, dhruvtibarewal/nlm-l24-adapters (page claims a 0.611 LB notebook; contents unverified).
6. **Big-but-unverified:** `olegzholobov/name-that-llm-response-corpus` (394 MB, license "other", 3.4k downloads, wide 18-model format — not downloaded, rows unverified) and `illiadolenko/ntl-merged-corpus-v1` (124.7 MB CSV, license **unknown**, 53,860-row claim). Neither blocks us; both are optional cross-checks.

---

## 2. Method

- **Discovery:** 6 `kaggle datasets list -s …` searches (ntl / name that llm / name-that-llm / attribution / stylometry / updated-sorted), 2 `kaggle kernels list` searches, then a **usage matrix** built by pulling 10 notebooks (`kaggle kernels pull -m`) and extracting every `kaggle.com/datasets/...` and `/kaggle/input/...` reference from their cells. API rate limit (HTTP 429) forced ~6–10 s sleeps between calls; a few listings were retried.
- **Rows/sizes:** every "rows" figure marked *measured* was computed locally from the downloaded file; "page claim" = author's stated number; sizes from the Kaggle list API (bytes) or `kaggle datasets files`.
- **Fidelity probe** (`fid`, section 3.1): TF-IDF word(1–2, 120k, `lowercase=False`, `\S+` tokens) + char_wb(2–5, 250k) → LinearSVC(C=1, balanced), **fit on the full official train.csv** (2,699 rows; OOF BA of the same recipe = 0.3115). For each external source: predict its rows and compare to the source's own labels — `exact` = model id agreement, `fam` = lab agreement (gpt/gemini/claude/qwen/glm/kimi). A high score means "this source's labels and style are competition-consistent". Absolute value is optimistic (fit-on-train), so only cross-source comparison is meaningful.
- **Transfer probe** (`xfer`, section 3.2): train the same recipe on a pool variant, predict **official train.csv** (prompt-disjoint: 150 train prompt ids vs pool prompt spaces), balanced accuracy over 18 classes. Noise floor between variants ≈ ±0.003. **Leakage check: 0 exact-text overlaps between any pool/new source and train.csv** (all checked).
- **What I did *not* do:** train any LoRA, run the Gaussian/Sinkhorn post-processing, or verify most "LB" claims (impossible locally — hidden labels).

---

## 3. Measurements

### 3.1 Source fidelity to competition labels (fit-on-train probe)

Sorted by family agreement. `v4:<src>` = the subset of pool_v4 attributed to that source.

| Source (or pool_v4 subset) | rows | exact | fam | note |
|---|---:|---:|---:|---|
| v4:chosunghwan | 1,802 | **0.374** | **0.882** | best; gemini-only |
| chosunghwan (raw file) | 2,331 | 0.366 | 0.879 | current version |
| v4:zakir | 729 | 0.314 | 0.698 | |
| v4:hailey | 902 | 0.302 | 0.718 | |
| v4:lucas_v4 | 853 | 0.299 | 0.726 | |
| v4:pv1 | 697 | 0.298 | 0.700 | prvsiyan own-prompts |
| sharipov7ar | 750 | 0.281 | 0.721 | |
| v4:lucas | 426 | 0.277 | 0.718 | |
| lucasboesen (raw) | 2,880 | 0.276 | 0.722 | |
| v4:alp (alperen) | 3,463 | 0.272 | 0.746 | |
| v4:keshav | 496 | 0.258 | 0.825 | small, clean |
| seyitkaangunes (raw) | 4,998 | 0.257 | 0.639 | Claude-heavy |
| v4:lxd | 4,309 | 0.257 | 0.550 | |
| v4:seyit | 4,837 | 0.256 | 0.639 | |
| v4:jadkutbi | 596 | 0.253 | 0.676 | |
| prvsiyan_ext (pvx) | 4,926 | 0.253 | 0.584 | |
| v4:aswathvanan | 1,875 | 0.247 | 0.640 | |
| v4:illiadolenko | 448 | 0.243 | 0.746 | |
| v4:zz (ziyuzhou) | 984 | 0.240 | 0.749 | |
| v4:pvx | 3,821 | 0.237 | 0.556 | |
| v4:kaplinskiy | 875 | 0.231 | 0.669 | |
| v4:abhinav | 762 | 0.230 | 0.699 | |
| **pool_v1 (current stage-1)** | 34,940 | **0.226** | **0.593** | nishant |
| v4:abdullah | 516 | 0.225 | 0.684 | |
| v4:dedruck | 1,390 | 0.224 | 0.688 | |
| v4:dophuchung | 942 | 0.220 | 0.708 | |
| **prvsiyan in-style pool** | 38,642 | **0.230** | **0.604** | typography-vetoed |
| v4:zbig | 1,882 | 0.209 | 0.686 | |
| zbigniew_gpt (raw) | 2,003 | 0.207 | 0.686 | |
| v4:arxitekton | 12,160 | 0.199 | 0.535 | **provenance unknown** |
| v4:willrice | 10,248 | 0.192 | 0.536 | biggest single src |
| v4:arxitekton_v2 | 3,273 | 0.180 | 0.358 | **worse on v2** |
| v4:bingli283 | 1,214 | 0.166 | 0.733 | |
| v4:msemenoff | 2,031 | 0.171 | 0.400 | |
| v4:maniceet | 7,452 | 0.143 | 0.404 | |
| pardheev (raw) | 2,977 | 0.232 | 0.695 | GPT-heavy |
| kenandemirci (raw) | 1,625 | 0.116 | 0.241 | API ids match labels; style does not |
| v4:tadeuhayashida | 1,437 | 0.093 | 0.543 | 1 model only |

Reading: the pool builders' intuition (exclude willrice/maniceet/tadeuhayashida/arxitekton families) matches the probe — those are the low-fidelity blocks. Reference point: the probe's own OOF BA on train is 0.3115, so a source at exact ≈0.31 is roughly as self-consistent as the competition data itself under this feature space.

### 3.2 Pool-variant transfer (train on pool → predict official train.csv)

| Training pool | rows | BA | acc | Δ vs pool_v1 |
|---|---:|---:|---:|---:|
| pool_v1 (nishant, our current stage-1) | 34,940 | 0.3623 | 0.362 | — |
| with 14 low-fidelity v4 subsets removed ("no_lowfid") | 27,194 | 0.3456 | 0.346 | −0.017 (but loses GLM/kimi classes entirely → misleading) |
| prvsiyan in-style pool v1 | 38,642 | 0.3682 | 0.368 | +0.006 |
| pool_v1 + prvsiyan in-style | 40,982 | 0.3716 | 0.372 | +0.009 |
| pool_v1 + {14 downloaded low-not-needed …} "external union" | 18,621 | 0.3509 | 0.351 | −0.011 |
| **pool_v4 (all, text-dedup)** | **71,925** | **0.3986** | 0.399 | **+0.036** |
| pool_v4, dedup (group,model) pairs | 54,186 | 0.3845* | 0.385 | +0.022 |
| pool_v4, groups ≥12 models | 33,487 | 0.3845 | 0.385 | +0.022 |
| pool_v4, groups ≥16 models | 18,574 | 0.3460 | 0.346 | −0.016 |
| pool_v4, groups =18 only | 6,858 | 0.2982 | 0.298 | −0.064 |
| prvsiyan + pool_v4 | 73,266 | 0.3990 | 0.399 | +0.037 |
| pool_v4 + {ch,pa,sh} | 76,181 | **0.4068** | 0.407 | **+0.045** |
| prvsiyan + pool_v4 + new4 | 79,147 | 0.3990 | 0.399 | +0.037 |
| pool_v4 + new4 (incl. kenandemirci) | 77,806 | 0.4038 | 0.404 | +0.041 |
| pool_v4 + kenandemirci only | 73,550 | 0.4016 | 0.402 | +0.039 |
| pool_v4 + chosunghwan only | 72,454 | 0.3990 | 0.399 | +0.037 |

\* coincidence that ge12 and dedup-pair both print 0.3845; different subsets.

new3 = chosunghwan (2,331) + pardheev (2,977) + sharipov (750) = 6,058 rows. new4 adds kenandemirci (1,625).  

**Interpretation.** (a) Doubling the pool (v4) is worth ~+0.03 in the probe and mostly fixes minority-lab recall. (b) Deduping never helps here — more in-style data beats "cleaner" data. (c) The tiny new sources are net-positive when they are high-fidelity (ch,pa,sh: +0.008 over v4) and net-negative when not (kenandemirci drags the new4 batch from 0.4068 → 0.4038). (d) Requiring full 18-model rounds (ge16/ge18) throws away most of the signal — abandon that instinct.

---

## 4. Master inventory

Tags: **K**=kirilltym 0.600 · **N**=nishantdahal 0.592 base · **P**=prvsiyan 0.604 · **L**=lightningv08 0.595 · **Z**=zbigniew 0.574/0.552 · **V**=vnton37nguyn 0.563 · **J**=josephadamski 0.607 · **A**=alexchilton · **KZ**=kiza/sangam forks · **S**=swaroopmeher mirrors. "Claimed effect" as stated by authors.

### A. Merged pools & corpora (data)

| ref | size | license | updated | type | models | rows (measured) | used by | claimed effect |
|---|---|---|---|---|---|---|---|---|
| ahmedberatozer/ntl-pool-v4 | 27.5 MB | CC-BY-SA-4.0 | 10-21 | merged pool parquet | 18 | 71,925 | ≥3 forks of K lineage (page title says superset) | newer superset pool; page: "consolidated"; 61 dl |
| nishantdahal/ntl-two-stage-lora | 293 MB | CC-BY-SA-4.0 | 10-19 | pool_v1 + 8B/14B two-stage LoRA + heads | 18 | 34,940 (pool) | **ours**, N, L, P, K lineage | LB 0.592 (base notebook) |
| prvsiyan/ntl-public-in-style-pool-v1 | 15.7 MB | CC-BY-SA-4.0 | 10-19 | curated pool, typography veto per model | 18 | 38,642 | **P** | stage-1 pool of the 0.604 notebook |
| prvsiyan/ntl-two-stage-lora-adapters-v1 | n/a (page null) | CC-BY-SA-4.0 | 10-19 | adapters | 18 | unverified | **P** | v1 adapter bundle |
| zibowang819/ntl-v1-artifacts | 198 MB | CC0-1.0 | 10-21 | 3× Qwen3-4B LoRA + public_pool_v2.parquet (18 MB) | ? | unverified | — | none claimed |
| olegzholobov/name-that-llm-response-corpus | 394 MB | **other** | 10-15 | wide 18-model response corpus (8 parquet files) | 18 | page: "18 models / 1,902 prompt groups" (unverified) | — | none stated |
| illiadolenko/ntl-merged-corpus-v1 | 124.7 MB | **unknown** | 10-21 | merged external CSV | ? | page claim 53,860; v4 includes 448 of its rows | — | documents pool_v1 dirtiness (6,506 dups/1,103 groups) |
| illiadolenko/ntl-generated-corpus-v1 | 139 KB | CC0-1.0 | 10-21 | clean generated CSV | ? | unverified | — | — |
| kenandemirci/ntl-glm-gemini-generations | 490 KB | CC BY 4.0 | 10-21 | GLM/Gemini same-prompt-bank | glm+g3.6/3.8 | 1,625 | — | fills GLM; **probe says mis-styled** |

### B. Generation sources (small; most already merged into pool_v4 or the 15-source lineage)

| ref | size | license | updated | rows (measured) | used by | note |
|---|---|---|---|---|---|---|
| seyitkaangunes/ntl-extra-api-passages | 1.6 MB | CC BY 4.0 | 10-16 | 4,998 | K,N,L,P,V,Z-tier | Claude-heavy, 1000 prompts |
| willrice/name-that-llm-openrouter-harvest | 9.7 MB | CC BY 4.0 | 10-18 | ~10,248 in v4 | K,N,L,P | biggest, mid fidelity |
| lucasboesen/ntl-free-proxy-generations | 794 KB | CC BY 4.0 | 10-20 | 2,880 | K,N,L,P | seed-7 run used in S mirrors |
| prvsiyan/prvsiyans-extended-… | 17.2 MB | CC BY-SA-4.0 | 10-19 | 4,926 | K,N,L,P | also source of cosmorin audit |
| prvsiyan/ntl-own-prompt-generations-v1 | 861 KB | CC BY-SA-4.0 | 10-16 | ~700 | K,P | |
| maniceet/…openrouter-openai… | 7.6 MB | CC BY 4.0 | 10-20 | 7,753 | v4 only | low fidelity (0.14) |
| chosunghwan/ntl-gemini-new-prompts | 736 KB | CC BY 4.0 | 10-21 | 2,331 | — (new) | **best fidelity measured** |
| pardheev/ntl-proxy-extra-passages | 1.3 MB | CC BY 4.0 | 10-21 | 2,977 | — (new) | GPT-heavy, proxy |
| sharipov7ar/…benchmarks-proxy | 929 KB | CC BY 4.0 | 10-21 | 750 | — (new) | 12 labels, clean |
| zbigniewkirchner/ntl-extra-gpt-passages | 845 KB | CC BY 4.0 | 10-18 | 2,003 | K,L,P,Z | GPT+Gemini |
| haileycheng/…same-prompt-passages | 829 KB | CC BY 4.0 | 10-18 | 902 | K,L,P | free-quota round |
| abdullah1afify/ntl-generations | 347 KB | CC BY 4.0 | 10-20 | 516 | K,L,P | |
| dophuchung/ntl-proxy-generations | 323 KB | CC BY 4.0 | 10-17 | 942 | K,L,P | |
| kaplinskiy/ntl-gpt-proxy-passages | — | CC BY 4.0 | ≤10-18 | 875 | K,P | |
| keshavatearth/ntl-gpt-new-prompts-v1 | — | CC BY 4.0 | ≤10-18 | 496 | K,P | |
| ziyuzhou1207/ntl-zz-proxy-generations | — | CC BY 4.0 | ≤10-18 | 984 | K,P | |
| michaelsemenoff/ntl-own-extra-v1 | — | CC BY 4.0 | ≤10-18 | 2,031 | K,L,P | low fidelity |
| alperen5252525/…vertex-gemini-proxy-gpt… | — | CC BY 4.0 | ≤10-18 | 3,463 | K,L,P | |
| lxdlxdlxd/ntl-synth-generations-v1 | — | CC BY ? | ≤10-18 | 4,309 | K,L,P | |
| abhinavsaipodugu/…gpt-gemini-passages | 632 KB | CC BY 4.0 | 10-21 | 762 | v4 | |
| aswathvanan/ntl-proxy-passages | 935 KB | CC BY-SA-4.0 | 10-21 | 1,875 | v4 | |
| bingli283/ntl-gen-pilot-v1 | 2.4 MB | CC BY 4.0 | 10-21 | 1,214 | v4 | low fidelity |
| dedruck/ntl-proxy-gen | 643 KB | CC BY 4.0 | 10-21 | 1,390 | v4 | |
| trnhip/ntl-proxy-gen + ntl-api-gen | 555 KB + 2.2 MB | CC BY 4.0 | 10-21 | 1,346 in v4 | v4 (proxy only) | api-gen new |
| ahmedberatozer/ntl-h-extras / ntl-gen-q8 / ntl-gen2s3-q8 … | 0–? | CC BY 4.0 / Apache-2.0 | 10-21 | unverified | — | quantized model runs |
| cosmorin/…audited-external-response-sample | 77 KB | CC BY 4.0 | 10-21 | 180 | — | audit excerpt of prvsiyan ext. |

### C. Model artifacts (weights, heads, adapters)

| ref | size | license | updated | type | rows/OOF | used by | claimed effect |
|---|---|---|---|---|---|---|---|
| zbigniewkirchner/ntl-qwen3-8b-lora-claude-head | 294.7 MB | Apache-2.0 | 10-16 | Qwen3-8B LoRA + head | held-out 0.4838 plain → 0.4932 w/ Claude gate | Z, V, L-tier | +0.005 LB in blend (0.552 nb) |
| zbigniewkirchner/ntl-qwen3-8b-lora-instyle | 186.5 MB | Apache-2.0 | 10-19 | in-style Qwen3-8B LoRA | member 0.456→0.516 held-out | Z(0.574), P | LB 0.569→0.574 @ w=0.17 (per W1) |
| josephadamski91/ntl-atlas-qwen3-8b-v1 | ~463 MB adapter + code | other (code Apache-2.0, weights CC-BY-SA-4.0 per files) | 10-20 | Qwen3-8B-Base adapter + `ntl_atlas`/`ntl_rlcd` code | official 2,699 rows | J (0.607) | ATLAS 10% pre-Sinkhorn: +0.007 on 0.600 base, 7/7 notebooks (per W1) |
| pjmathematician/ultrabert-380m-ntl | ~380 MB + wheels | Apache-2.0 | 10-14 | 5-fold fine-tunes of UltraBERT-380M | **OOF BA 0.408** | V | standalone CV 0.408 |
| lightningv08/ntl-olmo2-native24-final-inference-assets | page 0 (npz bundle) | other | 10-21 | OLMo2 + qwen25-native20 + smollm3 + tabular tokens/anchors, manifests | — | L (0.595/0.597) | OLMo2 @ coeff 0.03 |
| swaroopmeher/ntl-public-heads | page 0 (multi-npz) | CC-BY-SA-4.0 | 10-21 | frozen copies: nishant pool, lora24-s7 5-fold readouts, in-style, tokattn | — | S consumers | mirrors for stability |
| swaroopmeher/ntl-public-data-run20 / sub1020 | 15.4 / 20.3 MB | CC BY 4.0 / CC-BY-SA-4.0 | 10-21 | verbatim copies of 5 datasets each (fixed versions) | — | — | reproducibility mirrors |
| swaroopmeher/ntl-public-encoders | page 0 | other | 10-21 | DeBERTa-v3-large, ELECTRA-large, desklib detector weights | — | — | encoder copies |
| kingofeverything/ntl-lora-f0 | 116 MB + head.pt | CC0-1.0 | 10-19 | LoRA adapter + head + meta.json | — | — | none; 5 dl |
| rudrabhaskar/…qwen-fusion-heads | 92 MB | Apache-2.0 | 10-15 | qwen8/qwen14 heads joblib + text SVM + fusion manifests | — | — | older fusion |
| rudrabhaskar/…original-offline-models | 644 MB | MIT | 10-15 | offline base models bundle | — | — | utility |
| huangkx666/ntl-electra-weights | 1.34 GB | Apache-2.0 | 10-19 | ELECTRA-large discriminator finetuned | — | — | member candidate |
| bbucxi/qwen35-9b-safetensors | ? | **unknown** | — | base weights for token-attn head | — | V (0.563) | required by kirilltym's token-head recipe |
| dhruvtibarewal/nlm-l24-adapters | page 0 | MIT | 10-21 | "Identify the model behind text" artifacts | — | — | page claims notebook **LB 0.611** (unverified) |
| nishantdahal/ntl-two-stage-lora (checkpoints) | see A | CC-BY-SA-4.0 | 10-19 | two-stage 8B/14B LoRA + heads + anchors | LB 0.592 base | ours | stage-2 base of most 0.59+ stacks |

### D. Prompts, features, tools

| ref | size | license | updated | what | use |
|---|---|---|---|---|---|
| pjmathematician/ntl-train-prompts-reconstructed | 47 KB | CC-BY-SA-4.0 | 10-15 | LLM-reconstructed hidden prompts for the 150 train prompt_ids + topics | prompt-cluster inference, diagnostics; **train prompts only** |
| lwolwo/diverse-prompts-for-llm-style-attribution | small | CC0-1.0 | — | 500 synthetic prompts, 17 domains, `{n}` word-count placeholder | generate fresh in-style data |
| illiadolenko/ntl-prompt-bank-v1 | 3.2 MB | CC0-1.0 | 10-21 | 7.6 MB prompts.csv (g0000-style bank) | same-prompt generation protocol |
| verracodeguacas/name-that-llm-prism-feature-atlas | ~1.6 GB (est) | CC-BY-SA-4.0 | 10-15 | frozen Qwen3-1.7B/4B/8B layer caches + 468 style features, **train rows only** | exploratory: 8B+style CPU head 0.4297 (reused-prompt diagnostics, not LB) |
| ahmedberatozer/ntl-code | — | CC0-1.0 | ≤10-20 | code | reference |
| zkskhurram/… | 18 KB | CC0-1.0 | 10-17 | stub | none |
| torpidoff/djalilname-that-llm-offline-package | multi-GB (pythia/gpt2/distilroberta + wheels) | Apache-2.0 | 10-16 | offline install kit with generic models | not competition-specific |
| axelst/fingerprinting-{ox,omen,union}-alpha-corpus | — | CC0-1.0 | — | unrelated LLM fingerprinting corpora (different task/models) | **not applicable** |

### E. pool_v4 composition (measured, for filter decisions)

Raw: 71,925 rows, all unique texts, 8,831 groups, only 381 groups with 18 models (2,202 groups ≥12 = 49,289 rows), 9,433 duplicate (group,model) pairs (max 10 passages for one pair).
Per-source: arxitekton 12,160 · willrice 10,248 · maniceet 7,452 · seyit 4,837 · lxd 4,309 · pvx 3,821 · alp 3,463 · arxitekton_v2 3,273 · msemenoff 2,031 · zbig 1,882 · aswathvanan 1,875 · chosunghwan 1,802 · tadeuhayashida 1,437 · dedruck 1,390 · lucas_v3 1,322 · bingli283 1,214 · zz 984 · dophuchung 942 · hailey 902 · kaplinskiy 875 · lucas_v4 853 · abhinav 762 · zakir 729 · pv1 697 · jadkutbi 596 · abdullah 516 · keshav 496 · illiadolenko 448 · lucas 426 · abdullah_v2 183.
Overlap with pool_v1: 33,256 texts (≈95% of pool_v1) also in v4 → **v4 adds ~38.7k new texts**. Not in v4 (but available): sharipov (750), kenandemirci (1,625), pardheev (2,977), trnhip/api, dedruck-newer, chosunghwan-current (v4 has 1,802 vs 2,331 now), ahmedberatozer quantized runs.

---

## 5. Notes on the top artifacts

**nishantdahal/ntl-two-stage-lora (our current base, 293 MB, CC-BY-SA-4.0).** Contains `pool_v1.parquet` (34,940 rows, 0 train-text overlaps) *and* the 8B/14B two-stage checkpoints + heads that the whole ≥0.59 lineage reuses. W1 dissection stands: it is the 0.592 base. Any pool upgrade means retraining the same architecture on a bigger pool and re-merging, not swapping files.

**ahmedberatozer/ntl-pool-v4 (27.5 MB parquet, CC-BY-SA-4.0).** Fastest single-file upgrade: +0.036 transfer over pool_v1 in my probe, and it is a strict superset of the 15-source "public pool" lineage plus 15 extra sources. Two cautions: (1) **arxitekton/arxitekton_v2 = 15,433 rows (21%) with no traceable public origin** and low probe fidelity (0.199/0.535 and 0.180/0.358); (2) it carries maniceet/willrice/tadeuhayashida bulk that the highes-quality builders excluded — although my ablation shows removing low-fidelity sources wholesale is *worse* for the probe (it deletes whole classes like GLM/kimi which live only in those sources). Practical recipe: keep v4 whole, add new3, dedup texts only.

**prvsiyan/ntl-public-in-style-pool-v1 (38,642 rows, CC-BY-SA-4.0).** The "in-style" idea: keep only passages whose typography the model actually uses in the competition (per-model veto on em/en dash, curly quotes, ellipsis), single-paragraph, 30–300 words, no train overlap. Transfer 0.3682 (pool_v1 0.3623) with fewer rows — the filtering is real. This is the curated pool the 0.604 notebook trains stage 1 on; **the cheapest high-confidence upgrade if we don't want the v4 provenance question.** Model counts are fairly balanced: min gemini-3.1-pro 1,231 / qwen3.8-max 1,294 / claude-fable 1,535; max gemini-3.8-flash 3,316.

**chosunghwan/ntl-gemini-new-prompts (CC BY 4.0, 10-21).** 2,331 rows across the four Gemini labels, 580 prompts; highest-fidelity external source measured (0.879 family). Gemini is our second-worst lab in the current probe (gemini-3.1 recall 0.49 → 0.60 with v4). Cheap add.

**pardheev / sharipov7ar (CC BY 4.0, 10-21).** GPT-heavy proxy data (pardheev 2,977 incl. 2,184 GPT-5.6 rows; sharipov 750 across 12 labels). Additive in the probe (+0.008 over v4 with chosunghwan), but they are proxy-served — pardheev records `model_served`; spot-check served==requested before trusting.

**kenandemirci/ntl-glm-gemini-generations (CC BY 4.0, 10-21).** Fills the weakest class family (GLM) with same-prompt-bank rows and its `api_model` matches labels, but probe family-agreement is 0.24 — either the NVIDIA-served GLM builds differ stylistically from competition GLM or the rows are atypical (median 94 words). Include only behind a per-source gate, or exclude like the current "new3+ke" result suggests.

**zbigniewkirchner in-style LoRA + claude-head (Apache-2.0).** The only *members* with published held-out member-level deltas (0.456→0.516; 0.4838→0.4932). Drop-in to any sklearn readout stack (adapters infer through the same Qwen3-8B family). Highest value-per-byte of the model artifacts.

**josephadamski91 atlas (adapter `other`, code Apache, weights per file CC-BY-SA).** W1 already established ATLAS 10% as the most-replicated single gain (+0.007, 7/7). Note the license page says "other" — the per-file licenses (LICENSE-WEIGHTS-CC-BY-SA-4.0.txt etc.) govern; attribute accordingly.

**pjmathematician/ultrabert-380m-ntl (Apache-2.0).** OOF BA 0.408 over 5 prompt-grouped folds, fp16, with transformers wheels for offline install — a genuinely orthogonal member (DeBERTa-family), already used in the 0.563 notebook. Good diversity for our blend.

**lightningv08 artifacts + swaroopmeher mirrors.** The 0.595 notebook's OLMo2 + smollm3 + tabular assets (license "other" — check per-file) and the CC-BY-SA mirrors of lora24-s7 fivefold outputs / in-style / tokattn. Mirrors are the pragmatic way to freeze kernel outputs that could otherwise vanish.

**olegzholobov/name-that-llm-response-corpus (394 MB, license "other").** Biggest independent corpus (8 parquet files, wide 18-model layout, `external_*` + a quarantined luna slice — suggests the author already audited some luna rows). Not downloaded here (page size vs 3.4k downloads suggests real content); rows unverified; license non-standard → cross-check only; do not build the submission path on it without reading README/validation.json inside.

**illiadolenko/ntl-merged-corpus-v1 (124.7 MB CSV, license unknown).** Self-describes as a merge with dirtiness diagnostics (6,506 dups / 1,103 groups in pool_v1). Only 448 of its rows are in pool_v4. Unknown license blocks public reuse; useful as an audit reference at most.

---

## 6. Ranked "what to attach" (value / cost / risk)

1. **prvsiyan/ntl-public-in-style-pool-v1** — retrain stage 1 on it (or on prvsiyan+pool_v1). *Value:* measured 0.3682 transfer (> our 0.3623), curated typography, powers the 0.604 lineage. *Cost:* attach (16 MB) + one retrain run. *Risk:* low (CC-BY-SA-4.0).
2. **ahmedberatozer/ntl-pool-v4 (+ new3: chosunghwan + pardheev + sharipov)** — best measured pool recipe (BA 0.4068, +0.045 over current). *Cost:* attach (≤30 MB total) + retrain. *Risk:* low licence-wise (CC-BY-SA + CC BY) but **filter/quarantine `arxitekton*` if you want traceable provenance**; validate pardheev/sharipov served-model ids first.
3. **zbigniewkirchner/ntl-qwen3-8b-lora-instyle** (+ claude-head) — append as members with weight ≈0.15–0.17 + twin, per the 0.574 notebook. *Cost:* adapter inference on T4; small files. *Risk:* low (Apache-2.0), measured member deltas exist.
4. **josephadamski91/ntl-atlas-qwen3-8b-v1** — blend ATLAS at 10% pre-Sinkhorn. *Cost:* 463 MB + one inference pass. *Risk:* low–medium (licenses per file; W1 replication strong).
5. **pjmathematician/ultrabert-380m-ntl** — add as an orthogonal member (OOF 0.408). *Cost:* GPU inference (380M model is fast); Apache-2.0. *Risk:* low.
6. **swaroopmeher/ntl-public-heads** — frozen mirrors of fivefold lora24-s7/readouts + in-style + tokattn for stacking experiments without depending on kernel outputs. *Cost:* attach. *Risk:* low (CC-BY-SA-4.0).
7. **lightningv08/ntl-olmo2-native24-final-inference-assets** — OLMo2 readout at small coefficient (0.03), exact-inference manifests. *Cost:* medium. *Risk:* medium ("other" license, per-file check).
8. **chosunghwan/pardheev/sharipov as *generation targets*** — if we ever generate more data ourselves, copy their prompt-bank protocol (same prompt answered by many labels, per-model finish-reason gates) — the measured fidelity ranking endorses it.
9. **pjmathematician/ntl-train-prompts-reconstructed + lwolwo prompt bank** — for prompt-cluster inference / synthetic in-style generation. *Risk:* only train prompts; never as test features.
10. **Conditional spares:** zibowang819/ntl-v1-artifacts (CC0, 3 Qwen3-4B adapters + pool_v2), huangkx666 ELECTRA, rudrabhaskar heads, kingofeverything f0 — unverified quality; keep as ensemble fodder if time allows.

**Explicitly not recommended:** axelst corpora (different task), torpidoff package (generic base models), cosmorin sample (180 rows), bbucxi (unknown license), verracodeguacas atlas as a *data* source (train rows only — fine for diagnostics), illiadolenko merged (unknown license) and olegzholobov (non-standard license + unverified) for the critical path.

---

## 7. Unverified / open items

- **Rows/content not locally verified:** olegzholobov corpus (README/validation.json inside; wide format), illiadolenko merged (53,860 claim), prvsiyan adapters v1 (page size null), zibowang819 pool_v2 rows, dhruvtibarewal adapters + its "0.611" claim, ahmedberatozer quantized-run extras, verracodeguacas 8B extraction policy (page itself flags "no drop-in hidden-test inference").
- **Provenance unresolved:** pool_v4's `arxitekton`, `arxitekton_v2`, `jadkutbi`, `zakir`, `tadeuhayashida` subsets — no matching public datasets found in searches; treated as opaque.
- **Not verified by construction:** all LB claims (hidden labels); transfer deltas are from a linear-SVM proxy for stage 1 and may compress or vanish for the LoRA; fidelity probe is fit-on-train (relative comparison only).
- **Rate-limit debt:** a handful of dataset file-listings were fetched late (listing pages show stale `size 0` for several working datasets — page-size metadata is unreliable; trust file listings/rows).
- **License caveats:** "other"/"unknown" refs (olegzholobov, lightningv08, joseph atlas page, swaroopmeher encoders, illiadolenko merged, bbucxi) need per-file checks before a public notebook uses them; for a private submission path the operational risk is attribution/compliance, not function.
