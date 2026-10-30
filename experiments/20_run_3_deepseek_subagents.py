"""
Multi-Agent Orchestrator: Dispatches 3 parallel DeepSeek subagents via OpenCode
to author specialized modules for:
1. Sibling Stylometric & Token-Boundary Discrimination (src/sibling_features.py)
2. Test-Time Probability Calibration & Optimal Transport (src/test_time_calibration.py)
3. Short-Passage Specialization & Entropy Gating (src/short_passage_engine.py)
"""
import concurrent.futures
import os
import re
import subprocess
import sys
import time

REPO_ROOT = "/Users/pika/Documents/Projects/kaggle-llm"

AGENTS = [
    {
        "name": "Subagent 1: Sibling Features",
        "prompt_file": "experiments/prompts/subagent1_sibling_features.txt",
        "output_file": "src/sibling_features.py",
    },
    {
        "name": "Subagent 2: Test-Time Calibration & OT",
        "prompt_file": "experiments/prompts/subagent2_calibration_ot.txt",
        "output_file": "src/test_time_calibration.py",
    },
    {
        "name": "Subagent 3: Short-Passage Specialization",
        "prompt_file": "experiments/prompts/subagent3_short_passage.txt",
        "output_file": "src/short_passage_engine.py",
    },
]


def extract_python_code(raw_output: str) -> str:
    """Extracts python code blocks from LLM markdown response."""
    # Look for ```python ... ```
    matches = re.findall(r"```python\s*(.*?)\s*```", raw_output, flags=re.DOTALL)
    if matches:
        # Return the longest code block
        return max(matches, key=len).strip()
    
    # Fallback to any ``` ... ``` block
    matches = re.findall(r"```\s*(.*?)\s*```", raw_output, flags=re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    
    # If no blocks found, return cleaned output
    lines = [l for l in raw_output.split("\n") if not l.startswith("> build")]
    return "\n".join(lines).strip()


def run_agent(agent_spec: dict) -> dict:
    name = agent_spec["name"]
    prompt_path = os.path.join(REPO_ROOT, agent_spec["prompt_file"])
    output_path = os.path.join(REPO_ROOT, agent_spec["output_file"])
    
    print(f"[{name}] Starting execution...")
    t0 = time.time()
    
    with open(prompt_path, "r") as f:
        prompt_text = f.read()
        
    cmd = [
        "/Users/pika/.opencode/bin/opencode",
        "run",
        "-m", "opencode/deepseek-v4.1-flash-free",
        "--thinking",
        prompt_text
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
        elapsed = time.time() - t0
        
        if proc.returncode != 0:
            print(f"[{name}] FAILED with code {proc.returncode} in {elapsed:.1f}s. Stderr: {proc.stderr[:200]}")
            return {"name": name, "success": False, "error": proc.stderr}
            
        code = extract_python_code(proc.stdout)
        if not code or len(code) < 100:
            print(f"[{name}] FAILED: empty or too short output ({len(code)} chars) in {elapsed:.1f}s")
            return {"name": name, "success": False, "error": "Empty code extracted"}
            
        with open(output_path, "w") as f:
            f.write(code + "\n")
            
        print(f"[{name}] SUCCESS: Wrote {len(code.splitlines())} lines to {output_path} in {elapsed:.1f}s")
        return {"name": name, "success": True, "output_path": output_path, "lines": len(code.splitlines()), "elapsed": elapsed}
        
    except subprocess.TimeoutExpired:
        print(f"[{name}] TIMED OUT after 180s")
        return {"name": name, "success": False, "error": "Timeout"}


def main():
    print(f"=== DISPATCHING 3 PARALLEL DEEPSEEK SUBAGENTS ===")
    t_start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_agent, agent): agent["name"] for agent in AGENTS}
        results = []
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            
    print(f"\n=== ALL SUBAGENTS FINISHED IN {time.time() - t_start:.1f}s ===")
    for r in results:
        status = "PASSED" if r["success"] else f"FAILED: {r.get('error', 'Unknown')}"
        print(f"- {r['name']}: {status}")


if __name__ == "__main__":
    main()
