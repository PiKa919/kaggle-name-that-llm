"""
Adversarial Invalidation & Stress-Test Engine for SOTA Master Ensemble.
Invokes OpenCode with deep thinking blocks to tear apart and invalidate
the new architecture across mathematical, empirical, and systemic vulnerabilities.
"""

import os
import subprocess
import sys

REPO_ROOT = "/Users/pika/Documents/Projects/kaggle-llm"
FINDINGS_PATH = os.path.join(REPO_ROOT, "experiments/invalidation_findings.md")

QUERY = """
### ADVERSARIAL INVALIDATION ASSIGNMENT: TEAR APART SOTA MASTER ENSEMBLE (v2)

You are tasked with an AGGRESSIVE, UNCOMPROMISING INVALIDATION REVIEW of our latest SOTA Master Ensemble for the Kaggle "Name That LLM" competition.

Current Public Leaderboard Standing:
- Best Score: 0.539 (ref 57439844)
- Target: 0.70+ Balanced Accuracy
- Commit Version 2 of `notebooks/master_sota/submission_sota.py` successfully completed execution on Kaggle Dual T4 in 289.8s.

INSPECT THE CODEBASE IN /Users/pika/Documents/Projects/kaggle-llm:
1. notebooks/master_sota/submission_sota.py (The complete submission script)
2. src/sibling_discriminators.py (Pairwise sibling discriminators)
3. src/hierarchical_classifier.py (Dynamic length-gated typographic prior)
4. src/contrastive_loss.py (Contra-X style vs topic loss)
5. src/multi_trunk.py (Multi-trunk sequential engine)
6. src/eval_stats.py (Length-stratified metrics)

EXAMINE AND BRUTALLY TEAR APART THESE 5 SPECIFIC VULNERABILITIES:

1. THE STEP-FUNCTION LENGTH GATE:
   In `submission_sota.py`, we use:
     is_short = (word_counts_test < 80)
     w_typo = np.where(is_short, 0.45, 0.15)
   Critique the discontinuity at 79 vs 81 words. Does a sharp step-function cause variance spikes on test passages hovering around 75-85 words? What is the mathematically sound smooth transition function (e.g. generalized sigmoid)?

2. CONDORCET CYCLES & INDEPENDENT PAIRWISE DELTAS:
   In `submission_sota.py`, we fit 8 pairwise classifiers independently on N-gram features and add `+ 0.10 * delta` and `- 0.10 * delta` directly to class logits:
     log_base[:, idx2] += 0.10 * delta
     log_base[:, idx1] -= 0.10 * delta
   Does unconstrained sum of pairwise log-odds violate log-probability coherence? Can circular preferences (A > B, B > C, C > A) occur, distorting posterior probabilities? How should pairwise margins be projected onto a consistent potential vector (e.g. HodgeRank or Bradley-Terry-Luce projection)?

3. OUT-OF-DISTRIBUTION TYPOGRAPHIC FAILURE MODES:
   `TYPO_PROFILES` hardcodes Bernoulli rates:
     gemini: 0.01 em-dash rate
     claude: 0.01 curly quote rate
   What happens if unseen test prompts (e.g. Python code blocks with straight quotes, or fiction dialogue with em-dashes) force Gemini to emit dashes, or force Claude to emit quotes? Does `obs * log(p) + (1-obs) * log(1-p)` heavily penalize the true author if a single em-dash appears in Gemini? How should robust likelihood clipping or Dirichlet priors be tuned?

4. N-GRAM REGULARIZATION VS LORA CALIBRATION:
   We blend `0.85 * log_p_lora / 0.86 + 0.15 * log_p_ng / 0.80`.
   Is a static 85/15 scalar blend optimal across all 18 classes, or do some classes (like Qwen and Kimi) have stronger N-gram signatures than others?

5. HIDDEN TEST SET RUNTIME ON 4,000 ROWS:
   In `submission_sota.py`, we fit 5-fold StratifiedGroupKFold on train (2,699 rows) and predict on test.
   Then we fit 8 pairwise logistic regressions.
   Does this scale cleanly to 4,000 hidden test rows within the Kaggle time limit?

DELIVERABLE:
Provide:
1. Detailed critique and proof of each flaw.
2. The exact mathematical formulation for fixes.
3. Drop-in Python code patches for `submission_sota.py` and `src/`.
"""

def main():
    print("=== STARTING AGGRESSIVE INVALIDATION LOOP VIA OPENCODE WITH THINKING ===")
    cmd = [
        "/Users/pika/.opencode/bin/opencode",
        "run",
        "-m", "opencode/deepseek-v4.1-flash-free",
        "--thinking",
        QUERY
    ]
    
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300
    )
    
    output = proc.stdout.strip()
    if "> build" in output:
        lines = output.split("\n")
        output = "\n".join([l for l in lines if not l.startswith("> build") and not l.startswith("Loaded plugin")])
        
    print("\n=== RAW ADVERSARIAL FINDINGS RECEIVED ===")
    print(output[:1000] + "\n...[truncated]...")
    
    with open(FINDINGS_PATH, "w") as f:
        f.write("# Adversarial Invalidation Findings & Mathematical Critiques\n\n")
        f.write(output)
        
    print(f"\nSaved full findings to {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
