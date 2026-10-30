#!/bin/bash
# v12-fast public-run gate, then submission.  Mirrors the v13 job, but for the
# 0.80 public-twin / 0.20 claude-twin centre weighting.  It waits for the weekly
# GPU quota reset, pushes the kernel, validates the public 3-row artifacts and
# the v10 runtime envelope, and only then submits.  The literal v11/v12 sources
# are NOT used: they inherit the 16,076 s v11 legacy and would time out.
cd /Users/pika/Documents/Projects/kaggle-llm || exit 1
LOG=experiments/v12_fast_gate.log
mkdir -p experiments
exec >> "$LOG" 2>&1
echo "================================================================"
echo "V12FAST_JOB_START $(date -u)"
echo "================================================================"

# 0. Prove which weights this kernel carries before spending GPU time.
python3 - <<'PY'
import hashlib, pathlib, re
p = pathlib.Path('notebooks/v12_fast/script.py')
s = p.read_text()
block = re.search(r"EXTRA_LORA24 = \{[^}]+\}", s).group(0)
assert "'lora24_public_twin': 0.80" in block, block
assert "'lora24_claude_twin': 0.20" in block, block
assert "'lora24_public_twin': 0.70" not in block
print('V12FAST_WEIGHTS_OK', block.replace('\n', ' '))
print('V12FAST_SCRIPT_SHA256', hashlib.sha256(s.encode()).hexdigest())
print('V12FAST_SCRIPT_BYTES', len(s))
PY

# 1. Wait for the weekly GPU quota reset unless explicitly skipped.
if [ "${NTL_SKIP_WAIT:-0}" = "1" ]; then
  echo "WAIT_SKIPPED NTL_SKIP_WAIT=1 now=$(date -u)"
else
python3 - <<'PY'
import datetime, time
now = datetime.datetime.now(datetime.timezone.utc)
target = now.replace(hour=0, minute=1, second=0, microsecond=0)
days = (5 - target.weekday()) % 7
if days == 0 and target <= now:
    days = 7
target += datetime.timedelta(days=days)
wait = (target - now).total_seconds()
print(f"WAIT now={now.isoformat()} target={target.isoformat()} hours={wait/3600:.3f}", flush=True)
time.sleep(max(0.0, wait))
print(f"WAIT_DONE {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
PY
fi

# 2. Push the patched v12-fast and capture the version number.
PUSH_OUT=$(kaggle kernels push -p notebooks/v12_fast 2>&1)
echo "$PUSH_OUT"
VER=$(echo "$PUSH_OUT" | sed -n 's/.*[Kk]ernel version \([0-9][0-9]*\).*/\1/p' | tail -1)
VER=${VER:-2}
echo "V12FAST_VERSION=$VER"

# 3. Poll the public 3-row run.
FAIL=0
STATUS=""
for i in $(seq 1 300); do
  STATUS=$(kaggle kernels status pikaaa/name-that-llm-v12-fast 2>&1 | tail -1)
  echo "PUBLIC_POLL $(date -u) $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCELLED*|*CANCELED*) FAIL=1; break ;;
  esac
  sleep 120
done
if [ "$FAIL" = "1" ]; then
  echo "V12FAST_PUBLIC_FAILED $STATUS"
  exit 1
fi
if ! echo "$STATUS" | grep -q COMPLETE; then
  echo "V12FAST_PUBLIC_NOT_COMPLETE $STATUS"
  exit 1
fi

# 4. Download and gate the public artifacts.
rm -rf /tmp/v12fastpublic
mkdir -p /tmp/v12fastpublic
kaggle kernels output pikaaa/name-that-llm-v12-fast -p /tmp/v12fastpublic 2>&1 | tail -5
LOG_FILE=$(ls /tmp/v12fastpublic/*.log 2>/dev/null | head -1)
echo "LOG_FILE=$LOG_FILE"
grep -E "n-gram model done|lora8b|BROAD|legacy|BLEND|REWEIGHT|ATLAS predictions done|done in|curated pool|dropped train_regen|member matrix" "$LOG_FILE" | tail -50
TOTAL=$(grep -o 'done in [0-9][0-9]*s' "$LOG_FILE" 2>/dev/null | tail -1 | grep -o '[0-9][0-9]*')
echo "PUBLIC_TOTAL_SECONDS=$TOTAL"
if [ -z "$TOTAL" ] || [ "$TOTAL" -gt 11880 ]; then
  echo "V12FAST_RUNTIME_GATE_FAILED total=$TOTAL limit=11880"
  exit 1
fi
if ! grep -q "REWEIGHT lora24_public +0.10" "$LOG_FILE"; then
  echo "V12FAST_REWEIGHT_MISSING lora24_public"
  exit 1
fi
if ! grep -q "REWEIGHT lora24_claude +0.05" "$LOG_FILE"; then
  echo "V12FAST_REWEIGHT_MISSING lora24_claude"
  exit 1
fi
if grep -q "REWEIGHT lora24_public_twin" "$LOG_FILE"; then
  grep -q "REWEIGHT lora24_public_twin +0.80" "$LOG_FILE" || { echo "V12FAST_TWIN_WEIGHT_WRONG public"; exit 1; }
  grep -q "REWEIGHT lora24_claude_twin +0.20" "$LOG_FILE" || { echo "V12FAST_TWIN_WEIGHT_WRONG claude"; exit 1; }
  echo "V12FAST_TWINS_EXERCISED_PUBLIC"
else
  echo "V12FAST_TWINS_ABSENT_PUBLIC expected on a 3-row pool without prompt groups"
fi

python3 - <<'PY'
import numpy as np
z = np.load('/tmp/v12fastpublic/members_test.npz')
assert 'base' in z.files and 'atlas' in z.files and 'lora24_public_prob' in z.files, sorted(z.files)
assert z['base'].shape[1] == 18 and z['atlas'].shape[1] == 18
print('MEMBERS_OK', sorted(z.files))
print('base', z['base'].shape, 'atlas', z['atlas'].shape,
      'lora24_public', z['lora24_public_prob'].shape)
PY

python3 - <<'PY'
import pandas as pd
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
sub = pd.read_parquet('/tmp/v12fastpublic/submission.parquet')
v10 = pd.read_parquet('/tmp/v10/submission.parquet')
assert sub.shape == (3, 2), sub.shape
assert list(sub.columns) == ['id', 'model'], list(sub.columns)
assert set(sub.id.astype(str)) == set(v10.id.astype(str))
assert not sub.isnull().any().any()
assert sub.model.isin(LABELS).all()
print('SUB_OK')
print(sub.to_string(index=False))
PY

# 5. Submit the gated version.  Regular first-of-day submission, not a Final.
echo "V12FAST_SUBMIT $(date -u)"
SUBMIT_OUT=$(kaggle competitions submit -c name-that-llm -k pikaaa/name-that-llm-v12-fast -v "$VER" \
  -f submission.parquet -m "v12 fast: lora24 twins 0.80/0.20, v10 runtime envelope" 2>&1)
echo "$SUBMIT_OUT"

# 6. Poll the hidden rerun until it scores or errors.
python3 - <<'PY'
import time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
s = None
for i in range(120):
    subs = api.competition_submissions('name-that-llm')
    matches = [x for x in subs if 'v12 fast' in str(getattr(x, 'description', ''))]
    s = matches[0] if matches else subs[0]
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), s.ref, s.status,
          s.public_score, s.private_score, s.error_description, flush=True)
    if 'COMPLETE' in str(s.status) or 'ERROR' in str(s.status):
        break
    time.sleep(300)
print('V12FAST_FINAL', s.ref, s.status, s.public_score, s.private_score, s.error_description, flush=True)
PY

echo "V12FAST_JOB_END $(date -u)"
