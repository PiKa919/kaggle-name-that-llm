# Scan of YouTube `@code4AI` ("Discover AI") for competition clues

Date: 2026-10-28 20:40 UTC. Request: "check the videos from this page… also check his yt posts".

## What was actually searched

| Method | Coverage | Result |
| :--- | :--- | :--- |
| Channel RSS (`feeds/videos.xml?channel_id=UCfOvNb3xj28SNqPQ_JIbumg`) | latest 15 uploads, 2026-10-11 → 2026-10-28 | no competition video |
| Channel search `@code4AI/search?query=` parsed from `ytInitialData` | 8 queries: `attribution`, `name that llm`, `stylometry`, `fingerprint`, `kaggle`, `persona`, `which model wrote`, `identify the model`, `AI text detection`, `18 models` | 1 relevant hit (`stylometry` = 0 hits) |
| Web search (Exa) `site:youtube.com code4AI "Name That LLM"` | whole web index | nothing |

Channel identity: 1,561 videos, 91.7K subs, paper-explainer format — one paper per video,
author list and abstract read off the slide, no code walkthroughs of competitions.

**Conclusion: the channel has no video about the "Name That LLM" Kaggle competition.**
It is paper commentary; the models it covers (GLM 5.3, Kimi K3, Claude Opus 5, GPT-6 sol,
Qwen 3.8) overlap with our 18 classes only because both track the same frontier.

## The one adjacent paper — and why it does not transfer

`fTtliP3bV3A` (2026-09-27, 2,228 views) — *"Who Is Behind the Harness? Fingerprinting
LLMs through Agentic Behavior"*, Chuyi Wang et al., Tsinghua. Transcript pulled in full
(3,110 words).

- **Task**: identify which LLM is behind a coding-agent harness (silent model substitution).
- **Method ("Lighter")**: 3 A/B coding probes (`verify` / `recover` / `resolve`) → 6 full
  execution traces → **958 features** = 756 trace-level (decision, outcome, ordering,
  transition, revisit, run length, graph density, backtracking) + 202 distribution-level
  (mean and std of **101 descriptors** over the 6 traces).
- **Scoring**: per feature, estimate the reference state distribution of each of 36 known
  model endpoints, score a query by **−average KL(query ‖ model)** with **Jeffreys ½-count
  smoothing** to stop unobserved states from zeroing a class.
- **Reported**: 100 % (mini-swe-agent) / 94.4 % (opencode) top-1; 80 % cumulative evidence
  needs **43–45 distinct mechanisms**, i.e. no single feature identifies a model.

### Transfer check against our codebase

| Lighter component | Do we already have it? | Evidence |
| :--- | :--- | :--- |
| Discrete per-class state distribution + smoothing, scored by likelihood | **yes** | `src/typo_prior.py`: 16-state categorical over the 4 typography indicators, Dirichlet-smoothed, contamination mixture `(1−ρ)·Cat + ρ·U`, winsorised nats, fused in `src/fusion.py`. The docstring there already lists the same three failure modes (wrong constants, non-independence, floored Bernoulli) that Lighter's Jeffreys smoothing is meant to fix. |
| Bayesian typography readout in the frontier pipeline | **yes** | the 0.654 output ships `candidate_typobayes_full.parquet` ("prompt-heldout validated Bayesian typography") plus the hard `GUARDS` table in `/tmp/pub654/infer.py:431`. |
| 101 descriptors → mean/std distribution features | **yes** | our stylometry block: `F_tr` descriptors + punctuation + stop-word rates, plus the 159-feature forensics engine (`src/features.py`). |
| Many weak features, none decisive, summed in log space | **yes** | the whole log-opinion pool (`W = {'stylegram': 1.516, …}`, 16 members). |

**So the transferable idea is already implemented, and it is already in the 0.654 base.**
There is no new lever here for v20.

### Why the *headline* method is unusable even if it were new

Our test rows are single paragraphs with no harness, no tool calls, no traces. Lighter's
entire signal is the action sequence of an agent loop; a paragraph has no sequence of
decisions to observe. The analogy that does hold — *"what the harness does not change is
the model's own signature"*, i.e. style survives the prompt — is exactly the premise of
our prompt-centred twins and the shared-prompt Gaussian head, both already in v14/v19.

## Action taken

None. No code change, no experiment queued from this. Recorded so the search is not
repeated.
