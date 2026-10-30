# W2 — Rules, Meta, and Forum Intelligence

**Competition:** [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)  
**Snapshot:** 2026-10-21 (competition day 7)  
**Author:** research worker W2  
**Scope:** official rules, submission/runtime constraints, external-data and checkpoint policy, leaderboard meta, full discussion-forum digest.

---

## 0. Executive summary

- **Task:** 18-way authorship attribution over 40–250-word LLM passages from 6 labs; metric is **balanced accuracy** (per-class recall averaged over 18 classes, chance = 0.0556).
- **Data:** 2,699 train passages / 150 prompts (149 prompts × 18 models + `p002` × 17, missing `claude-fable-5`); hidden test ≈ 4,000 passages from **unseen prompts**; interactive `test.csv` has 3 example rows.
- **Leaderboard split:** public LB = **approximately 40%** of test; private LB = **the other 60%** (official Leaderboard page). This contradicts the 20–30% / 70–80% assumption in `AGENTS.md`; use 40/60.
- **Limits:** **5 submissions/day**, **2 Final Submissions**, **max team size 5**, **9-hour** notebook runtime, **internet disabled** at scoring.
- **External data:** allowed if publicly available and equally accessible at no cost, **or** satisfying the host's "Reasonableness"/"minimal cost" standard. Public pre-trained models are explicitly allowed. Paid-API synthetic data is a **gray zone** with no host ruling as of this snapshot.
- **Our position:** rank **293** of **2,359 teams**, best public LB **0.600**; leader **0.688**; 10th **0.630**; top-200 cutoff ≈ **0.604**. We need **+0.004** for top 200 and **+0.030** for top 10.
- **Forum consensus on leverage:** external/public training data > prompt-grouped CV > 7–14B LoRA backbone > test-time clustering tricks. The clustering/Hungarian path is consistently a CV trap that loses on LB — matching our v8/v9 regressions from 0.600.

---

## 1. Official competition facts

| Item | Value | Source |
|---|---|---|
| Host / sponsor | Kaggle / Google LLC | Rules §1.1–1.2 |
| Start | 2026-10-14 | Overview → Timeline |
| Entry + team-merger deadline | 2027-01-06, 11:59 PM UTC | Overview → Timeline |
| Final submission deadline | 2027-01-13, 11:59 PM UTC | Overview → Timeline |
| Prize pool | $50,000; $5,000 each for 1st–10th | Rules §1.5 |
| Participation (snapshot) | 7,858 entrants, 2,474 participants, 2,359 teams, 14,831 submissions | Overview |
| Data license | CC BY-SA 4.0 | Rules §1.7; Data page |
| Winner license | CC-BY 4.0 | Rules §1.6 |
| Metric | Balanced Accuracy, K = 18 | Overview → Evaluation |
| Test size | ~4,000 hidden passages; 3 example rows interactively | Data page |
| Public / private split | ~40% / ~60% | Leaderboard page |
| Max team size | 5 | Rules §2.1.a |
| Submissions | 5/day; up to 2 Final Submissions | Rules §2.2 |
| Runtime | CPU ≤ 9 h; GPU ≤ 9 h | Overview → Code Requirements |
| Internet at scoring | Disabled | Overview → Code Requirements |
| Submission file | `submission.parquet`, columns `id,model` | Overview; Rules |

**Valid labels (18):** `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`, `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`, `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`, `qwen3.8-max`, `glm-5.3`, `glm-5.3-flash`, `kimi-k3`.

**Dataset facts:** every train prompt is answered exactly once by each model except `p002`, which lacks `claude-fable-5`. Test prompts do not appear in train. Only light formatting cleanup (e.g. stripping a whole-passage `**bold**` wrapper) was applied. Test per-model counts are unequal and the public/private model mix differs.

**Minor source inconsistency:** Overview says interactive `test.csv` has "a single example row"; Data page says "three example rows"; the actual file has 3 rows. Treat 3 as correct.

---

## 2. Rules that constrain data, checkpoints, and submissions

All quotes below are verbatim from the official Rules page (retrieved 2026-10-21). Section numbers are Kaggle's.

### 2.1 Submission limits (the number)

> **§2.2 SUBMISSION LIMITS**
>
> a. You may submit a maximum of five (5) Submissions per day.
>
> b. You may select up to two (2) Final Submissions for judging.

**Answer: 5 submissions per day; 2 Final Submissions.** Only the Final Submissions are judged on the private LB; if you do not select them, Kaggle selects automatically (§3.18.c). Plan the last two carefully.

### 2.2 Team limits

> **§2.1 TEAM LIMITS**
>
> a. The maximum Team size is five (5).
>
> b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline.

### 2.3 External data and tools — the critical clause

> **§2.6 EXTERNAL DATA AND TOOLS**
>
> a. You may use data other than the Competition Data ("External Data") to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b below. The ability to use External Data under this Section does not limit your other obligations under these Competition Rules, including but not limited to Section 2.8 (Winners Obligations).
>
> b. The use of external data and models is acceptable unless specifically prohibited by the Host. Because of the potential costs or restrictions (e.g., "geo restrictions") associated with obtaining rights to use external data or certain software and associated tools, their use must be "reasonably accessible to all" and of "minimal cost". Also, regardless of the cost challenges as they might affect all Participants during the course of the competition, the costs of potentially procuring a license for software used to generate a Submission, must also be considered. The Host will employ an assessment of whether or not the following criteria can exclude the use of the particular LLM, data set(s), or tool(s):
>
> 1. Are Participants being excluded from a competition because of the "excessive" costs for access to certain LLMs, external data, or tools that might be used by other Participants. The Host will assess the excessive cost concern by applying a "Reasonableness" standard (the "Reasonableness Standard"). The Reasonableness Standard will be determined and applied by the Host in light of things like cost thresholds and accessibility.
>
> 2. By way of example only, a small subscription charge to use additional elements of a large language model such as Gemini Advanced are acceptable if meeting the Reasonableness Standard of Sec. 8.2. Purchasing a license to use a proprietary dataset that exceeds the cost of a prize in the competition would not be considered reasonable.

**Key reading:** the rule does **not** say external data must be free to *you*. It says it must be free/equally accessible to *other participants*, **or** pass a host-administered "minimal cost / reasonably accessible" test. A small paid-API spend is not per se prohibited; a cost exceeding a prize is. The host has discretion and has not published a threshold.

### 2.4 Code requirements (competition-specific, permissive on models)

> **Code Requirements**
>
> - CPU Notebook <= 9 hours run-time
> - GPU Notebook <= 9 hours run-time
> - Internet access disabled
> - Freely & publicly available external data is allowed, including pre-trained models
> - Submission file must be named `submission.parquet`

This is the clearest statement that **public pre-trained checkpoints are allowed**. It is permissive, not a restriction to "only free data".

### 2.5 Prohibited conduct: test/validation labels

> **§3.4 COMPETITION ENTRY**
>
> b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.

This bans **human labeling / human prediction** of validation or test records. It does **not** explicitly ban unsupervised use of unlabeled test text (clustering, transductive centering, pseudo-labeling by a model). Several forum threads ask the hosts to confirm this; none has a host ruling (see §5: 750379, 750051, 750768, 750213). Treat hard test-set label fitting and LB probing as high-risk.

### 2.6 Private vs public sharing

> **§3.5.d Private Sharing.** No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.
>
> **§3.6.a Private Code Sharing.** … you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data … This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs.
>
> **§3.6.b Public Code Sharing.** You are permitted to publicly share Competition Code … By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license … that in no event limits commercial use …
>
> **§3.6.c Use of Open Source.** … if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license …

**Consequence:** using another team's **publicly shared** notebook/weights is permitted and the share is deemed OSI-licensed. Using the same weights via DM, Discord, or a private repo is prohibited. Public notebook blending is therefore legal; private weight trading is not.

### 2.7 Competition data security

> **§2.4.b Data Security.** … You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition.

Do not publish train/test rows, even inside otherwise-public synthetic datasets. Synthetic data generated from your own prompts is fine; competition text is not.

### 2.8 Winner obligations (affects data/license choices now)

> **§2.8.a** … Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation … must be capable of generating the winning Submission … include training code, inference code, and a description of the required computational environment.
>
> **§2.5.a.3** In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s).

You must be able to reproduce the run and open-source your own code. Third-party data/models with incompatible licenses do not have to be re-licensed. Keep a reproducible training/inference path from now.

---

## 3. External checkpoints and data: what is allowed, what is risky

| Practice | Status | Basis / notes |
|---|---|---|
| Public pre-trained backbone (Qwen, OLMo, DeBERTa, etc.) | **Allowed** | Code Requirements: "including pre-trained models"; §2.6. |
| Publicly shared Kaggle notebook weights / LoRA adapters | **Allowed** | §3.6.b deems public shares OSI-licensed; must be public to all. |
| Public external corpus (Oleg 18-model corpus, helper datasets, extra Claude passages) | **Allowed if public and equally accessible** | §2.6.a; verify license and that no competition text is redistributed. |
| Paid-API synthetic passages, kept private | **Gray / risky** | §2.6.b allows "minimal cost" under host discretion, but the data is not "available at no cost" to others. No host ruling. |
| Paid-API synthetic passages, published as a public Kaggle dataset | **Gray, but strongest defensive position** | Becomes free/equally accessible to all under §2.6.a; still document cost and protocol. This is what 749629 / 750267 / 750947 did. |
| Free Kaggle Benchmarks / Model Proxy quota generations | **Likely allowed, unconfirmed** | Community says yes (750038); hosts have not answered 750216/751011. Every verified participant gets the same quota, which supports "equally accessible". |
| Private sharing of weights/data outside the team | **Prohibited** | §3.5.d, §3.6.a. |
| Redistributing train/test rows | **Prohibited** | §2.4.b. |
| Hand-labeling or human-predicting test/validation rows | **Prohibited** | §3.4.b. |
| Unsupervised clustering / transductive centering on unlabeled test text | **Not explicitly banned; unresolved** | §3.4.b bans human labels, not unsupervised test features. Hosts silent (750379, 750051, 750768). Public LB shows these tricks underperform anyway. |
| Fitting to public-LB feedback / probing labels | **High-risk** | Arguably "information from … human prediction"; also poor generalization (40% public vs 60% private). Avoid. |
| Training on hidden test labels | **Prohibited / impossible** | Labels are not provided. |

**Practical policy recommendation:** build on public, OSI-compatible backbones and the already-public external corpora. If we generate paid data, publish it publicly with prompt protocol, license, and cost disclosure, and keep a "competition-data-only" score in the writeup. Do not build the core solution on an unreviewed interpretation of §2.6.

---

## 4. Submission and runtime checklist

- [ ] Output `submission.parquet` in the notebook working directory, columns exactly `id,model`.
- [ ] Every test `id` exactly once; no nulls; all labels in the 18-name set.
- [ ] Notebook completes in **< 9 h** on CPU or GPU; leave ≥ 30 min margin.
- [ ] No internet calls at scoring time; all weights/data attached as Kaggle Datasets or Models.
- [ ] Copy frequently-read weights from `/kaggle/input` to `/tmp` (or attach as a Kaggle Model) to avoid the slow-input-read failure mode (750924, 751268).
- [ ] No assertions that depend on interactive test size (3 rows vs ~4,000); make the hidden rerun unable to throw (750553).
- [ ] FP16 on T4 (no bfloat16); merge LoRA before submission to cut load time.
- [ ] Sort test passages by token length for efficient batching; batch 8–16.
- [ ] Keep **2 Final Submissions** for the best private-LB candidates; only 5 new submissions/day.
- [ ] Record CV by `prompt_id` group, per-class recall, and confusion among siblings for every candidate.

---

## 5. Discussion forum digest (47 threads, 2026-10-14 → 2026-10-21)

Votes/comments are as of the snapshot. "OP" author is the thread author; Kaggle's API returns empty author fields for top-level posts, so a few authors are marked `n/a`.

| ID | Date | Votes | Comments | Author | Title / key takeaway |
|---|---|---:|---:|---|---|
| [749382](https://www.kaggle.com/competitions/name-that-llm/discussion/749382) | 10-14 | 16 | 4 | Ashley Oldacre | **Pinned getting-started + official Discord.** Newbie hub; Discord link questions. |
| [749447](https://www.kaggle.com/competitions/name-that-llm/discussion/749447) | 10-14 | **110** | **46** | CoreyJamesLevinson | **"Will the winner just be whoever can get the most extra training data?"** The central fairness debate. Top replies: "Pay That LLM"; 1M vs 100K examples per model; paid generation as the real lever. |
| [749455](https://www.kaggle.com/competitions/name-that-llm/discussion/749455) | 10-15 | 18 | 0 | pjmathematician | **Reconstructed all 150 train prompts** by shuffling each prompt's 18 passages, stripping labels, and asking Claude Sonnet 5 for the likely instruction + tags. Public dataset + method. |
| [749494](https://www.kaggle.com/competitions/name-that-llm/discussion/749494) | 10-15 | 4 | 1 | Zacchaeus | Joke thread ("Guess who I am?"). No technical signal. |
| [749511](https://www.kaggle.com/competitions/name-that-llm/discussion/749511) | 10-15 | 7 | 0 | YePing Lin | **PersonalLLM dataset** (~10k prompts × 8 models: GPT-4o, Claude 3 Opus/Sonnet, Gemini 1.5 Pro, Llama-3-70B, Mixtral). Old model set; not the 18 targets. |
| [749545](https://www.kaggle.com/competitions/name-that-llm/discussion/749545) | 10-15 | 1 | 1 | Maniceet Sahay | **"Training Data Generation Paid."** Asks whether small paid API spend or Codex/Claude subscriptions are allowed. Community: "open secret… augment with your own money." No host answer. |
| [749552](https://www.kaggle.com/competitions/name-that-llm/discussion/749552) | 10-15 | -7 | 0 | Anpeng Yuan | Compares to Jigsaw-Agile; body is garbled ("NaN.NaN…"). No signal. |
| [749593](https://www.kaggle.com/competitions/name-that-llm/discussion/749593) | 10-15 | 3 | 0 | Aleksey Osipenko | **Random-split CV 0.22 vs grouped-by-`prompt_id` 0.37.** Random splits leak the 17 same-prompt siblings; topic words seen only with other labels push predictions away. Mandatory grouped CV. |
| [749613](https://www.kaggle.com/competitions/name-that-llm/discussion/749613) | 10-15 | 5 | 1 | Efe Karamese | **Rules question on self-generated API data.** Paid candidate-model API passages under §2.6; public datasets `seyitkaangunes/ntl-extra-api-passages` and `lucasboesen/ntl-free-proxy-generations`. No host answer. |
| [749629](https://www.kaggle.com/competitions/name-that-llm/discussion/749629) | 10-15 | **25** | 5 | Oleg Zholobov | **18-model external corpus:** 135,171 unique responses; 70,980 wide rows; 24,015 40–250-word candidates; 13,642 wide; **1,902 prompt groups with all 18 models**; source URLs + license metadata. Public dataset/notebook. |
| [749664](https://www.kaggle.com/competitions/name-that-llm/discussion/749664) | 10-15 | 0 | 0 | Gray Robertson | Rules questions: mixed-license public data, Kaggle Benchmarks credits, test setup. No replies. |
| [749832](https://www.kaggle.com/competitions/name-that-llm/discussion/749832) | 10-16 | 0 | 1 | OsakanaTeishoku | Tried **Laya/ModernBERT** ("system 1") — poor CV; Qwen-based stacks better. |
| [749914](https://www.kaggle.com/competitions/name-that-llm/discussion/749914) | 10-16 | 8 | 2 | Yan Zhou | **Asks hosts for more official data** (2,699 rows "way too little"). Replies: doubt hosts will clarify paid data; enforcement is hard. |
| [749947](https://www.kaggle.com/competitions/name-that-llm/discussion/749947) | 10-16 | -8 | 0 | anooj pakvasa | "What's the difference"; garbled. No signal. |
| [750038](https://www.kaggle.com/competitions/name-that-llm/discussion/750038) | 10-16 | 2 | 2 | Zbigniew Kirchner | **Can Kaggle Benchmarks free AI quota generate training data?** Community says yes; no host confirmation. Model Proxy includes some candidate models. |
| [750051](https://www.kaggle.com/competitions/name-that-llm/discussion/750051) | 10-16 | 0 | 2 | Emirhan Kadan | **"Clustering test?"** Is unsupervised test clustering + assignment legal? Replies: tried, "doesn't work the trivial way"; LB worse than simulation; depends on answers/prompt. |
| [750059](https://www.kaggle.com/competitions/name-that-llm/discussion/750059) | 10-16 | 0 | 0 | Alariun | Requests organizer-approved provenance docs / independent evaluation prompts. No answer. |
| [750065](https://www.kaggle.com/competitions/name-that-llm/discussion/750065) | 10-16 | 0 | 4 | Serhii Hrynko | **Asks for partial credit for correct family.** Community: no; would be gamed; families are not equally hard (Qwen/Kimi easy, GPT 5-way blind). |
| [750200](https://www.kaggle.com/competitions/name-that-llm/discussion/750200) | 10-17 | 0 | 0 | water joe | Link to "Kaggle synth-prompts" notebook. No content. |
| [750213](https://www.kaggle.com/competitions/name-that-llm/discussion/750213) | 10-17 | 2 | 1 | Karthikeya Thatipamula | **Test-time prompt grouping simulation.** Plain ~0.35; with true prompt groups and 18 answers/prompt gains ~+0.10–0.117; at 1–3 answers/prompt ≈ 0. Public LB gain much smaller than CV. Asks hosts about test layout. |
| [750216](https://www.kaggle.com/competitions/name-that-llm/discussion/750216) | 10-17 | 3 | 0 | Balram Srinivasan | Asks hosts to confirm free Kaggle Benchmarks/Model Proxy quota under §2.6, with public dataset + request logs. No answer. |
| [750224](https://www.kaggle.com/competitions/name-that-llm/discussion/750224) | 10-17 | 5 | 3 | Seyit kaan Gunes | **[Dataset] ~5,000 extra Claude passages**, 1,000 new prompts, official API, no competition text. Best used for a **within-Claude 5-way head**, not the 18-way classifier. Qwen3-8B LoRA with 5-way Claude head: 0.484 vs 0.456 plain, 0.493 with extra. |
| [750230](https://www.kaggle.com/competitions/name-that-llm/discussion/750230) | 10-17 | 0 | 6 | Raheel Abbas | Learning journey: TF-IDF 0.322 → LoRA ensemble 0.541. Advice: improve the training pipeline, not the ensemble; genre-shift collapse; balance data. |
| [750267](https://www.kaggle.com/competitions/name-that-llm/discussion/750267) | 10-17 | **36** | 11 | prvsiyan | **"Useful helper dataset"** built with free Claude/OpenRouter credits; explicitly framed as a public, free counter to pay-to-win. License field ambiguity: CC BY-SA 4.0 text vs CC BY 4.0 field. |
| [750365](https://www.kaggle.com/competitions/name-that-llm/discussion/750365) | 10-17 | 0 | 0 | SeaGoat | Points to authorship attribution/verification paper arXiv 2209.06869. No discussion. |
| [750379](https://www.kaggle.com/competitions/name-that-llm/discussion/750379) | 10-18 | 0 | 0 | Chandramouli | Asks whether submissions may cluster unlabeled hidden `test.csv` and use cluster-centered embeddings, without human labels or exporting hidden data. **No answer.** |
| [750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390) | 10-18 | 12 | 2 | Adisak Sukul | **The single most useful technical thread.** Grouped CV 0.217 vs 0.369; CPU TF-IDF LB 0.353. Clustering + Hungarian: 0.353 → 0.344; per-cluster log-prob centering → 0.329. Qwen3-14B-Base LoRA r16, 3 epochs, max_len 384: fold-0 CV 0.443, public LB 0.466. 2×T4 tips: fp16, `device_map="auto"`, ~14 GiB max_memory/GPU, merge LoRA, sort by length. Short passages hardest: ≤60 words 0.270 vs >180 words 0.419. **Truncation augmentation did not help** (overall 0.343 → 0.335; ≤60-word bucket 0.270 → 0.262). |
| [750408](https://www.kaggle.com/competitions/name-that-llm/discussion/750408) | 10-18 | 2 | 0 | ln (nazarov) | Asks for the exact formatting procedure used on train passages. No answer. |
| [750421](https://www.kaggle.com/competitions/name-that-llm/discussion/750421) | 10-18 | -9 | 0 | n/a | Claude Code project writeup; local gains vanished on LB; external data lowered CV. Heavily downvoted. |
| [750447](https://www.kaggle.com/competitions/name-that-llm/discussion/750447) | 10-18 | 0 | 1 | n/a | **ATLAS 10% blend** on public notebooks: +0.002 to +0.004. Two-stage LoRA v2 + 10% ATLAS + 10% early layers: 0.592 → 0.596. ATLAS adapter = Qwen3-8B-Base trained on the official 2,699 rows. |
| [750462](https://www.kaggle.com/competitions/name-that-llm/discussion/750462) | 10-18 | 4 | 2 | Aleksey Osipenko | **Early Qwen layers beat late layers:** frozen Qwen3-4B (36 layers), mean pooling + logistic regression per layer, grouped CV. Early ≈ 0.39 vs last ≈ 0.31. CV consistently ~0.015 above public LB. |
| [750512](https://www.kaggle.com/competitions/name-that-llm/discussion/750512) | 10-18 | 1 | 1 | Lucky Raj | Clarification request on public API-generated datasets, Benchmarks proxy data, §2.5/§2.8. Only a community "I'm unsure too" reply. |
| [750553](https://www.kaggle.com/competitions/name-that-llm/discussion/750553) | 10-18 | 0 | 4 | n/a | Hidden-rerun exception logs are not exposed. Advice: make the rerun unable to throw; test on a full-size fake test; remove assertions tied to interactive size. |
| [750619](https://www.kaggle.com/competitions/name-that-llm/discussion/750619) | 10-19 | 1 | 1 | n/a | "Why ensemble?" Advice: focus on your own pipeline; public blends move fast and can distract; single strong model first. |
| [750664](https://www.kaggle.com/competitions/name-that-llm/discussion/750664) | 10-19 | 4 | 4 | Masood Ahmed | **Punctuation audit.** Claude and GLM: em dashes + straight quotes. GPT and Gemini: curly apostrophes/quotes. Gemini most separable by simple logistic regression; Claude ↔ GLM most confused. Caution: external proxy data can flip em-dash rates. |
| [750675](https://www.kaggle.com/competitions/name-that-llm/discussion/750675) | 10-19 | 0 | 0 | n/a | Proposes multi-layer transformer representations + dimensionality reduction + nonlinear classification to recover fingerprints lost by mean pooling/linear heads. No replies. |
| [750706](https://www.kaggle.com/competitions/name-that-llm/discussion/750706) | 10-19 | 2 | 1 | Jean Brandicourt | Train structure: 150 full 18-model rounds; `p002` missing `claude-fable-5`. Clustering + Hungarian (threshold 0.9) raised grouped CV 0.34 → 0.39–0.40 but **LB fell 0.345 → 0.343**. Hidden test likely has few passages/prompt, so the one-per-model constraint forces errors. |
| [750768](https://www.kaggle.com/competitions/name-that-llm/discussion/750768) | 10-19 | 1 | 0 | n/a | Asks hosts: do several hidden-test passages share a prompt? At most one answer per model per prompt? **No answer.** |
| [750844](https://www.kaggle.com/competitions/name-that-llm/discussion/750844) | 10-19 | -8 | 2 | HALX2158 | "Pay to win? Five proposals": hosts clarify generated-data rules + formatting protocol; participants publish generated data with protocol/license, list dependencies, report competition-data-only score. Controversial; downvoted. |
| [750908](https://www.kaggle.com/competitions/name-that-llm/discussion/750908) | 10-20 | -3 | 1 | n/a | GPU compute question; reply links a 70B-on-2×T4 RAG notebook (quantized/offload). |
| [750924](https://www.kaggle.com/competitions/name-that-llm/discussion/750924) | 10-20 | 0 | **10** | XZXZZX | **Extremely slow `/kaggle/input` reads** on some GPU sessions (hours). Kaggle staff: dataset-cache changes + queue; workarounds: attach models as Kaggle Models, copy to `/tmp`, avoid random reads from `/kaggle/input`. |
| [750947](https://www.kaggle.com/competitions/name-that-llm/discussion/750947) | 10-20 | 8 | 1 | Maniceet Sahay | Public **OpenRouter + OpenAI generations** dataset; reports meaningful gains. Free to all. |
| [751011](https://www.kaggle.com/competitions/name-that-llm/discussion/751011) | 10-20 | 4 | 0 | Adisak Sukul | Consolidated §2.6 questions: candidate-model-generated data, public dataset publication, cost thresholds. **No host answer.** |
| [751054](https://www.kaggle.com/competitions/name-that-llm/discussion/751054) | 10-20 | 2 | 2 | Soham Acharya | Submission ran 2–3 h locally but exceeded 9 h in rerun. Points to 750924; another user confirms. |
| [751268](https://www.kaggle.com/competitions/name-that-llm/discussion/751268) | 10-21 | 1 | 5 | HALX2158 | 0.600+ forked notebook hits 9-h limit due to **Qwen3.5-9B weight load from `/kaggle/input`**, not the token head. Fix: copy weights to `/tmp` / use Kaggle Models. |
| [751294](https://www.kaggle.com/competitions/name-that-llm/discussion/751294) | 10-21 | 0 | 0 | Andres Carmona | §2.6: paid-API synthetic data published as a public dataset — acceptable? Exact cost/protocol described. **No answer.** |
| [751370](https://www.kaggle.com/competitions/name-that-llm/discussion/751370) | 10-21 | 1 | 0 | Darragh1Walsh | **Gemini-3.7 Flash requests routed to Gemini-3.8 Flash** after deprecation; consistent 3.7→3.8 misclassification. Implies label noise / version aliasing in the Gemini sibling pair. |

**Thread-level count:** 47 topics, 14,831 total submissions at snapshot. The rules-question cluster (749613, 749664, 750038, 750216, 750379, 750512, 750768, 751011, 751294) has **zero host replies**; the only staff reply in the corpus is in 750924 (dataset cache).

---

## 6. What the forum says actually moves the needle

Ranked by strength of evidence (votes + measured CV/LB deltas), not by popularity alone.

### 1. More and better external training data is the dominant lever
- 749447 has **110 votes / 46 comments** and is entirely about whether the winner will be whoever buys the most generations. No other thread has close to that engagement.
- 749629 publishes **1,902 prompt groups with all 18 models** (24,015 candidates in the 40–250-word band), explicitly built to close the data gap.
- 750267 (36 votes), 750947 (8), 750224 (5) are all public data drops; 750224 reports a within-Claude head gain of **0.484 vs 0.456**, and 0.493 with extra data.
- Caveat: no host has confirmed paid-generation legality. The safe version of this lever is **using or publishing public, free, equally accessible data**, not private paid data.

### 2. Grouped CV by `prompt_id` is non-negotiable
- 749593: random split **0.22** vs grouped **0.37** — the direction is counterintuitive because same-prompt siblings leak topic words with other labels.
- 750390: random **0.217** vs grouped **0.369**; CPU TF-IDF baseline LB **0.353**.
- Every credible technical post uses grouped CV. Our own pipeline must report per-class recall and sibling confusion on grouped folds only.

### 3. A fine-tuned 7–14B backbone + LoRA is the strongest single-model family
- 750390: Qwen3-14B-Base, LoRA r16, 3 epochs, max_len 384 → fold-0 CV **0.443**, public LB **0.466** on 2×T4 (fp16, ~14 GiB/GPU, merge LoRA first).
- 750447: two-stage LoRA + ATLAS + early-layer blend reaches **0.596** public; ATLAS alone adds +0.002–0.004.
- 750230: TF-IDF **0.322** → LoRA ensemble **0.541**.
- 750462: frozen Qwen3-4B early layers **0.39** vs last layer **0.31**; CV ≈ LB + 0.015.
- 749832: ModernBERT/Laya underperforms Qwen-based stacks.
- Public top is 0.688, so the remaining gap is data + calibration + sibling discrimination, not a different architecture family.

### 4. Test-time clustering / Hungarian assignment is a CV trap at realistic test density
- 750390: TF-IDF **0.353 → 0.344** with clustering + Hungarian; **0.329** with per-cluster log-prob centering.
- 750706: grouped CV **0.34 → 0.39–0.40**, but LB **0.345 → 0.343**.
- 750213 simulation: with true prompt groups and **18 answers/prompt**, gains are +0.10–0.117; at **1–3 answers/prompt**, gains are ≈ 0. The hidden test is subsampled and not a balanced 18-round.
- 750051: independent team reports the same simulation-vs-LB gap.
- This is exactly the pattern behind our v8/v9 regressions from 0.600. **Do not ship hard one-per-model constraints.**

### 5. Sibling discrimination and label noise, not lab discrimination, is the remaining error mass
- 750664: lab is nearly readable from punctuation (Claude/GLM straight quotes + em dashes; GPT/Gemini curly quotes), but **Claude ↔ GLM** and the Gemini siblings are the confusions. Gemini is the most separable lab by simple linear models.
- 751370: Gemini-3.7 Flash requests routed to Gemini-3.8 Flash — a real label-noise/aliasing risk in the 3.7/3.8 pair.
- 750224: a **within-Claude 5-way head** on top of a Qwen3-8B LoRA is the right pattern for sibling errors; 18-way head trained only on original rows.
- 750462 / 750675: early/multi-layer representations retain more author signal than the final layer; mean pooling + linear head leaves signal on the table.
- 750065: no partial credit for family; a GPT family call is still a blind 5-way choice. Balanced accuracy punishes any collapsed sibling recall.

### 6. Operational failures are real and cost entire submissions
- 750924 (10 comments): `/kaggle/input` reads can take hours; copy to `/tmp` or attach as Kaggle Models.
- 751268: a 0.600+ fork blew the 9-h limit on Qwen3.5-9B weight loading.
- 751054: 2–3 h locally → >9 h in rerun; same root cause.
- 750553: hidden-rerun logs are not visible; remove assertions that depend on the 3-row interactive test and make the rerun unable to throw.

### 7. Ensembling early is low ROI
- 750230: "Improving ensemble method is not so effective. Try to improve your training pipeline."
- 750619: public blends move fast; a single strong model plus a focused pipeline beats premature stacking.
- 750447's careful blends add only +0.002–0.004 each; they are finish-line moves, not a path from 0.60 to 0.68.

---

## 7. Leaderboard meta (snapshot 2026-10-21)

**Top 15 public LB**

| Rank | Team | Score | Last submission |
|---:|---|---:|---|
| 1 | Hod Carriers | 0.688 | 2026-10-21 03:59 |
| 2 | Vishak K Bhat | 0.667 | 2026-10-21 11:19 |
| 3 | Adisak Sukul | 0.658 | 2026-10-21 02:11 |
| 4 | Harveen Singh Chadha | 0.644 | 2026-10-19 14:59 |
| 5 | Le Viet | 0.637 | 2026-10-21 08:38 |
| 6 | Maniceet Sahay | 0.634 | 2026-10-20 00:04 |
| 7 | Søren Ravn Andersen | 0.634 | 2026-10-21 06:11 |
| 8 | Aarush112006 | 0.633 | 2026-10-21 12:59 |
| 9 | TF | 0.631 | 2026-10-21 00:09 |
| 10 | kulshekhar | 0.630 | 2026-10-21 03:25 |
| 11 | Horikita Saku | 0.629 | 2026-10-20 20:19 |
| 12 | VitalyR | 0.628 | 2026-10-21 08:33 |
| 13 | Kevin Arvai | 0.627 | 2026-10-21 01:17 |
| 14 | Komil Parmar | 0.626 | 2026-10-21 09:53 |
| 15 | Eyas Taifour | 0.624 | 2026-10-21 05:29 |

- Rank 30 = 0.615; rank 200 ≈ 0.603–0.604. Our 0.600 sits around rank 293.
- The 0.688 leader is 0.021 above #2 and 0.030 above #10 — a large gap for day 7, consistent with an external-data/backbone advantage rather than a small blend.
- Public is 40% of test; the private 60% can reshuffle the top. Two Final Submissions matter.

**Our submissions**

| Version | Score | Note |
|---|---:|---|
| v5 | **0.600** | Best; dual LoRA 8B + 14B |
| v6 | 0.593 | |
| v8 | 0.593 | Triple-trunk |
| v9 | 0.591 | Current |
| v3 | 0.532 | |
| v2 | 0.539 | |
| v4 | 0.528 | |
| early | 0.355 / 0.315 | Baselines |

---

## 8. Gaps, contradictions, and open questions

1. **Public/private split:** official Leaderboard says **40% / 60%**; `AGENTS.md` says 20–30% / 70–80%. Use the official 40/60. This materially changes how much to trust public-LB deltas.
2. **Test row count:** Overview says 1 example row; Data page says 3; file has 3. Assume 3 interactively, ~4,000 hidden.
3. **Paid API data:** §2.6.b permits "minimal cost" at host discretion, but the hosts have not answered any of 749613, 749664, 750216, 750512, 751011, 751294. The only safe general policy is public data + public release of anything you generate.
4. **Unsupervised test-set use:** §3.4.b bans human labeling/prediction, not clustering of unlabeled test text. 750379/750051/750768 ask; no host answer. The LB evidence says the hard version does not work anyway.
5. **Prompt reconstruction:** 749455 reconstructed all 150 train prompts with an LLM. Useful for EDA and synthetic-data prompt diversity, but do not assume test prompts are from the same distribution.
6. **Label noise:** 751370 documents Gemini 3.7→3.8 routing. If the hidden labels follow the same aliasing, some "errors" may be unlearnable; within-family calibration matters more than forcing a boundary.
7. **Truncation augmentation:** 750390's measured result contradicts the `AGENTS.md` mitigation claim — it slightly hurt both overall and the short-passage bucket. Do not adopt it without re-testing.
8. **External corpus licensing:** 750267's license field (CC BY 4.0) conflicts with its README/data card (CC BY-SA 4.0). Verify before relying on it for a winning solution.

---

## 9. Recommended actions for our team

**Immediate (next 24 h)**
1. Freeze a grouped-CV harness: `StratifiedGroupKFold(5, shuffle=True, random_state=42)` on `prompt_id`, per-class recall + sibling confusion, and a length-bucket report (≤60, 61–120, 121–180, >180 words).
2. Reproduce the 0.600 v5 pipeline under this harness and record per-class recalls. Identify which of the 18 classes are dragging balanced accuracy (likely Claude/GLM siblings and Gemini 3.7/3.8).
3. Stop shipping test-time Hungarian/centering variants. If kept, gate them behind a simulation that matches 1–3 answers/prompt and require a public-LB gain before inclusion.

**Data (highest ROI)**
4. Evaluate the public corpora in this order: Oleg 18-model corpus (749629), `prvsiyan` helper dataset (750267), Maniceet OpenRouter/OpenAI (750947), extra Claude passages (750224). Check licenses and overlap with competition text before training.
5. For sibling errors, train a **within-family head** (5-way Claude, 5-way GPT, 2-way GLM, 2-way Gemini) on the original rows plus family-matched external rows, and blend with the 18-way head rather than replacing it.
6. If generating new data, use the free Kaggle Benchmarks/Model Proxy quota where possible, publish the rows with prompt protocol + license + cost, and keep a competition-data-only ablation. Do not rely on an unreviewed paid-data interpretation.

**Model**
7. Prioritize a single strong 7–14B Qwen-family LoRA with early/multi-layer pooling (750462) over adding more ensemble members. Merge LoRA before submission.
8. Add length-stratified sampling or loss weighting rather than truncation augmentation; 750390 showed truncation did not help.
9. Calibrate per class (temperature / prior correction) to protect minority-class recall; balanced accuracy rewards fixing the worst class, not the average.

**Submission**
10. Keep one submission slot per day for a controlled public-LB probe; reserve the final two for the best private-LB candidates. With 5/day and a 40/60 split, avoid overfitting the public 40%.
11. Run a full-size fake test (~4,000 rows) in the visible notebook to catch runtime/assertion failures; copy weights to `/tmp`; target ≤ 7 h to leave margin.
12. Validate `submission.parquet` with the exact schema/label/null/duplicate checks before every commit.

---

## 10. Source notes

- Rules: https://www.kaggle.com/competitions/name-that-llm/rules (retrieved 2026-10-21)
- Overview: https://www.kaggle.com/competitions/name-that-llm/overview
- Data: https://www.kaggle.com/competitions/name-that-llm/data
- Leaderboard: https://www.kaggle.com/competitions/name-that-llm/leaderboard
- Discussion index: https://www.kaggle.com/competitions/name-that-llm/discussion
- Raw forum dump: `scratch/w2/forum_full.json`; rendered: `scratch/w2/forum_full.md`; digest: `scratch/w2/forum_digest.txt`
- Leaderboard snapshots: `scratch/w2/lb_top.csv`, `scratch/w2/lb_30.csv`, `scratch/w2/lb_200.csv`; our history: `scratch/w2/our_subs.csv`
