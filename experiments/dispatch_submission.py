"""
Kaggle Submission Dispatch & LB Scoring Monitor
Dispatches the verified submission.parquet to the competition once quota resets (00:00 UTC).
"""

import datetime
import subprocess
import sys
import time

COMPETITION = "name-that-llm"
KERNEL_ID = "pikaaa/name-that-llm-sota-master-ensemble"
SUBMISSION_FILE = "submission.parquet"

def get_utc_time():
    return datetime.datetime.now(datetime.timezone.utc)

def main():
    now_utc = get_utc_time()
    print(f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Check recent submissions
    print("\n--- Recent Kaggle Submissions ---")
    sub_res = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMPETITION], capture_output=True, text=True)
    print(sub_res.stdout)

    print(f"\nSubmitting from kernel {KERNEL_ID}...")
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", COMPETITION,
        "-k", KERNEL_ID,
        "-f", SUBMISSION_FILE,
        "-m", "v5: Dual Two-Stage LoRA (8B+14B) + Pool N-Grams + Centered Twin + HodgeRank Sibling + Length Gate"
    ]
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

    if res.returncode == 0:
        print("\nMonitoring submission processing...")
        for _ in range(30):
            time.sleep(10)
            status_check = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMPETITION], capture_output=True, text=True)
            lines = status_check.stdout.strip().split("\n")
            if len(lines) >= 3:
                latest = lines[2]
                print(f"Latest submission line: {latest}")
                if "COMPLETE" in latest:
                    print("\nSubmission processing COMPLETE!")
                    break

if __name__ == "__main__":
    main()
