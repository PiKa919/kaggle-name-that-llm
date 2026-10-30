"""
DeepSeek Brutal Teacher Interface via OpenCode CLI.
Acts as a ruthless, world-class Kaggle Grandmaster mentor who tears apart weak approaches
and prescribes concrete mathematical, architectural, and data solutions to dominate the competition.

Now integrated with FULL ANTIGRAVITY MEMORY, CODEBASE VISIBILITY, AND EXPERIMENT LOGS.
DeepSeek has full access to inspect any file in /Users/pika/Documents/Projects/kaggle-llm
and Antigravity's persistent artifacts.
"""

import os
import subprocess
import sys

REPO_ROOT = "/Users/pika/Documents/Projects/kaggle-llm"
BRAIN_REPORT = "/Users/pika/.gemini/antigravity/brain/21e7e884-a3e4-4f1f-8bee-9b95161ae969/grandmaster_solution_report.md"

SYSTEM_PROMPT = """You are the BRUTAL KAGGLE GRANDMASTER TEACHER for the 'Name That LLM' competition.
Your persona: Ruthless, mathematically rigorous, allergic to shallow heuristics and wishful thinking, and relentlessly focused on finding high-leverage algorithmic breakthroughs.

Evaluation Metric: Balanced Accuracy across 18 candidate LLM classes. Baseline random guessing is 0.0556.
Current Best Public Solution on LB: 0.607 (Two-Stage LoRA on Qwen3-8B/14B + 4-View n-grams + Shared-Prompt Gaussian Cavity EM + Sinkhorn-Knopp marginal balancing).
Our Initial Baseline on LB: 0.355 (TF-IDF + Stylometry LightGBM).
Our Current Best Public LB: 0.539 (ref 57439844, Grandmaster LoRA + 4-View N-Grams).

WORKSPACE VISIBILITY:
You are running directly inside the local workspace at:
  /Users/pika/Documents/Projects/kaggle-llm
You have access to the entire codebase, experiments, and documentation:
  - AGENTS.md: Competition specification, constraints, Dual T4 specs.
  - docs/superpowers/plans/2026-10-20-road-to-0.70-sota-plan.md: Active 5-task roadmap to 0.70+.
  - .superpowers/sdd/2026-10-20-road-to-0.70-sota-plan/progress.md: Verified SDD execution ledger.
  - experiments/results.tsv: Historical benchmark scores.
  - experiments/14_watermark_and_artifact_probe.py: Typographic & watermark empirical probe results.
  - notebooks/grandmaster_sub/submission_grandmaster.py: The code that scored 0.539 on Public LB.
  - Antigravity Solution Report: /Users/pika/.gemini/antigravity/brain/21e7e884-a3e4-4f1f-8bee-9b95161ae969/grandmaster_solution_report.md

When presented with queries or experiments:
1. Use your tools to read any relevant files in the repo before making claims.
2. Brutally tear apart flaws: leakages, over-smoothing, topic-confounding, naive heuristics, and uncalibrated blends.
3. Demand proof and evidence over hand-wavy claims.
4. Prescribe concrete, executable architectural innovations:
   - Tokenization & lexical representations (e.g. Qwen3 intermediate layers, shape/function grammars).
   - Prompt-topic confound isolation (e.g. Contra-X style vs topic contrastive loss, Domain-Adversarial GRL).
   - Calibration & post-processing (e.g. Sinkhorn-Knopp, Dirichlet calibration, Cavity EM).
Be concise, incisive, and merciless in your pursuit of top-1 on the leaderboard.
"""

def consult_deepseek(query: str, model: str = "openrouter/openrouter/free", session_id: str = "ses_kaggle_deepseek") -> str:
    """
    Send a query to DeepSeek via opencode CLI with full workspace context and persistent session memory.
    """
    # Build prompt with attached memory pointers
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== OPERATOR QUERY & CURRENT STATE ===\n"
        f"{query}\n\n"
        f"=== REPOSITORY & MEMORY ARTIFACTS AT YOUR DISPOSAL ===\n"
        f"You can view any of these files to inspect current code:\n"
        f"- AGENTS.md\n"
        f"- docs/superpowers/plans/2026-10-20-road-to-0.70-sota-plan.md\n"
        f"- notebooks/grandmaster_sub/submission_grandmaster.py\n"
        f"- experiments/results.tsv\n"
        f"- {BRAIN_REPORT}\n"
    )
    
    cmd = [
        "/Users/pika/.opencode/bin/opencode",
        "run",
        "-m", model,
        "--session", session_id,
        "--auto",
        full_prompt
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180
        )
        out = proc.stdout.strip()
        # Clean opencode header if present
        if "> build" in out:
            lines = out.split("\n")
            out = "\n".join([l for l in lines if not l.startswith("> build") and not l.startswith("Loaded plugin")])
        if not out and proc.stderr:
            out = f"[Stderr output: {proc.stderr.strip()}]"
        return out.strip()
    except subprocess.TimeoutExpired:
        if model != "openrouter/deepseek/deepseek-chat":
            print(f"Warning: {model} timed out, falling back to openrouter/deepseek/deepseek-chat...", file=sys.stderr)
            return consult_deepseek(query, model="openrouter/deepseek/deepseek-chat", session_id=session_id)
        return "[ERROR: DeepSeek consultation timed out after 180s]"
    except Exception as e:
        if model != "openrouter/deepseek/deepseek-chat":
            return consult_deepseek(query, model="openrouter/deepseek/deepseek-chat", session_id=session_id)
        return f"[ERROR querying DeepSeek: {e}]"


if __name__ == "__main__":
    test_query = "Critique our latest findings from experiments/14_watermark_and_artifact_probe.py and advise on next steps."
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    print(consult_deepseek(test_query))
