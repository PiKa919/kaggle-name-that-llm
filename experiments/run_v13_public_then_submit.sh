#!/bin/bash
# v13 public-run gate, then submission.  Launched in the background until the GPU
# quota resets at Saturday 00:00 UTC.  It never submits unless the public 3-row run
# is COMPLETE, within the v10 runtime envelope, and its parquet/member payload pass.
cd /Users/pika/Documents/Projects/kaggle-llm || exit 1
LOG=experiments/v13_run.log
mkdir -p experiments
exec >> "$LOG" 2>&1
echo "================================================================"
echo "V13_JOB_START $(date -u)"
echo "================================================================"

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

# 2. Push v13 and capture the version number.
PUSH_OUT=$(kaggle kernels push -p notebooks/v13 2>&1)
echo "$PUSH_OUT"
VER=$(echo "$PUSH_OUT" | sed -n 's/.*[Kk]ernel version \([0-9][0-9]*\).*/\1/p' | tail -1)
VER=${VER:-1}
echo "V13_VERSION=$VER"

# 3. Poll the public 3-row run.
FAIL=0
STATUS=""
for i in $(seq 1 300); do
  STATUS=$(kaggle kernels status pikaaa/name-that-llm-v13 2>&1 | tail -1)
  echo "PUBLIC_POLL $(date -u) $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCELLED*|*CANCELED*) FAIL=1; break ;;
  esac
  sleep 120
done
if [ "$FAIL" = "1" ]; then
  echo "V13_PUBLIC_FAILED $STATUS"
  exit 1
fi
if ! echo "$STATUS" | grep -q COMPLETE; then
  echo "V13_PUBLIC_NOT_COMPLETE $STATUS"
  exit 1
fi

# 4. Download and gate the public artifacts.
rm -rf /tmp/v13public
mkdir -p /tmp/v13public
kaggle kernels output pikaaa/name-that-llm-v13 -p /tmp/v13public 2>&1 | tail -5
LOG_FILE=$(ls /tmp/v13public/*.log 2>/dev/null | head -1)
echo "LOG_FILE=$LOG_FILE"
grep -E "n-gram model done|lora8b|BROAD|legacy|BLEND|REWEIGHT|ATLAS predictions done|done in|curated pool|dropped train_regen|member matrix" "$LOG_FILE" | tail -40
TOTAL=$(grep -o 'done in [0-9][0-9]*s' "$LOG_FILE" 2>/dev/null | tail -1 | grep -o '[0-9][0-9]*')
echo "PUBLIC_TOTAL_SECONDS=$TOTAL"
if [ -z "$TOTAL" ] || [ "$TOTAL" -gt 11880 ]; then
  echo "V13_RUNTIME_GATE_FAILED total=$TOTAL limit=11880"
  exit 1
fi

python3 - <<'PY'
import numpy as np
z = np.load('/tmp/v13public/members_test.npz')
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
sub = pd.read_parquet('/tmp/v13public/submission.parquet')
v10 = pd.read_parquet('/tmp/v10/submission.parquet')
assert sub.shape == (3, 2), sub.shape
assert list(sub.columns) == ['id', 'model'], list(sub.columns)
assert set(sub.id.astype(str)) == set(v10.id.astype(str))
assert not sub.isnull().any().any()
assert sub.model.isin(LABELS).all()
print('SUB_OK')
print(sub.to_string(index=False))
PY

# 5. Submit the gated version.  This is a regular first-of-day submission, not a
#    Final Submission; the v10 fallback remains untouched.
echo "V13_SUBMIT $(date -u)"
SUBMIT_OUT=$(kaggle competitions submit -c name-that-llm -k pikaaa/name-that-llm-v13 -v "$VER" \
  -f submission.parquet -m "v13 curated-pool SVC + lora24 prompt-centre" 2>&1)
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
    s = subs[0]
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), s.ref, s.status,
          s.public_score, s.private_score, s.error_description, flush=True)
    if 'COMPLETE' in str(s.status) or 'ERROR' in str(s.status):
        break
    time.sleep(300)
print('V13_FINAL', s.ref, s.status, s.public_score, s.private_score, s.error_description, flush=True)
PY

echo "V13_JOB_END $(date -u)"
