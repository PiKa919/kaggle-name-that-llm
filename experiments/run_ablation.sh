#!/bin/bash
# Generic public-run gate + submission runner for a single-change ablation.
#   usage: run_ablation.sh <dir> <kernel-id> <log> <description>
# It never submits unless the public 3-row run is COMPLETE, inside the v10 runtime
# envelope, and its parquet + member payload pass the integrity gate.
set -u
cd /Users/pika/Documents/Projects/kaggle-llm || exit 1
DIR="$1"
KID="$2"
LOG="$3"
DESC="$4"
TAG=$(echo "$DIR" | tr '[:lower:]' '[:upper:]')
mkdir -p experiments
exec >> "$LOG" 2>&1
echo "================================================================"
echo "${TAG}_JOB_START $(date -u)"
echo "================================================================"

SHA=$(shasum -a 256 "notebooks/$DIR/script.py" | cut -d' ' -f1)
echo "${TAG}_SCRIPT_SHA256 $SHA"
echo "${TAG}_SCRIPT_BYTES $(wc -c < "notebooks/$DIR/script.py")"

# 1. Push and capture the version number (skipped when the run is already in flight).
if [ "${NTL_SKIP_PUSH:-0}" = "1" ]; then
  VER=${NTL_VERSION:-1}
  echo "${TAG}_PUSH_SKIPPED NTL_SKIP_PUSH=1 version=$VER kernel=$KID"
else
PUSH_OUT=$(kaggle kernels push -p "notebooks/$DIR" 2>&1)
echo "$PUSH_OUT"
VER=$(echo "$PUSH_OUT" | sed -n 's/.*[Kk]ernel version \([0-9][0-9]*\).*/\1/p' | tail -1)
VER=${VER:-1}
echo "${TAG}_VERSION=$VER"
if ! echo "$PUSH_OUT" | grep -qi "successfully pushed"; then
  echo "${TAG}_PUSH_FAILED"
  exit 1
fi
fi

# 2. Poll the public 3-row run.
FAIL=0
STATUS=""
for i in $(seq 1 300); do
  STATUS=$(kaggle kernels status "$KID" 2>&1 | tail -1)
  echo "PUBLIC_POLL $(date -u) $STATUS"
  case "$STATUS" in
    *COMPLETE*) break ;;
    *ERROR*|*CANCELLED*|*CANCELED*) FAIL=1; break ;;
  esac
  sleep 120
done
if [ "$FAIL" = "1" ]; then
  echo "${TAG}_PUBLIC_FAILED $STATUS"
  exit 1
fi
if ! echo "$STATUS" | grep -q COMPLETE; then
  echo "${TAG}_PUBLIC_NOT_COMPLETE $STATUS"
  exit 1
fi

# 3. Download and gate the public artifacts.
OUT="/tmp/${DIR}public"
rm -rf "$OUT"
mkdir -p "$OUT"
kaggle kernels output "$KID" -p "$OUT" 2>&1 | tail -5
LOG_FILE=$(ls "$OUT"/*.log 2>/dev/null | head -1)
echo "LOG_FILE=$LOG_FILE"
grep -E "n-gram model done|lora8b|lora14b|BLEND|REWEIGHT|ATLAS predictions done|done in|curated pool|dropped train_regen|member matrix" "$LOG_FILE" | tail -40
TOTAL=$(grep -o 'done in [0-9][0-9]*s' "$LOG_FILE" 2>/dev/null | tail -1 | grep -o '[0-9][0-9]*')
echo "PUBLIC_TOTAL_SECONDS=$TOTAL"
if [ -z "$TOTAL" ] || [ "$TOTAL" -gt 11880 ]; then
  echo "${TAG}_RUNTIME_GATE_FAILED total=$TOTAL limit=11880"
  exit 1
fi

python3 - "$OUT" <<'PY'
import sys, numpy as np
out = sys.argv[1]
z = np.load(out + '/members_test.npz')
assert 'base' in z.files and 'atlas' in z.files, sorted(z.files)
assert z['base'].shape[1] == 18 and z['atlas'].shape[1] == 18
print('MEMBERS_OK', sorted(z.files))
print('base', z['base'].shape, 'atlas', z['atlas'].shape)
PY

python3 - "$OUT" <<'PY'
import sys, pandas as pd
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
out = sys.argv[1]
sub = pd.read_parquet(out + '/submission.parquet')
_ref = next(p for p in ('/tmp/v10/submission.parquet',
                        '/Users/pika/Documents/Projects/kaggle-llm/notebooks/v10/output/submission.parquet')
            if __import__('os').path.exists(p))
v10 = pd.read_parquet(_ref)
assert sub.shape == (3, 2), sub.shape
assert list(sub.columns) == ['id', 'model'], list(sub.columns)
assert set(sub.id.astype(str)) == set(v10.id.astype(str))
assert not sub.isnull().any().any()
assert sub.model.isin(LABELS).all()
print('SUB_OK')
print(sub.to_string(index=False))
PY

# 4. Submit the gated version.  Regular submission, not a Final Submission.
echo "${TAG}_SUBMIT $(date -u)"
SUBMIT_OUT=$(kaggle competitions submit -c name-that-llm -k "$KID" -v "$VER" \
  -f submission.parquet -m "$DESC" 2>&1)
echo "$SUBMIT_OUT"

# 5. Poll the hidden rerun until it scores or errors.
python3 - <<'PY'
import time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
s = None
for _ in range(120):
    subs = api.competition_submissions('name-that-llm')
    s = subs[0]
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), s.ref, s.status,
          s.public_score, s.private_score, s.error_description, flush=True)
    if 'COMPLETE' in str(s.status) or 'ERROR' in str(s.status):
        break
    time.sleep(300)
print('FINAL', s.ref, s.status, s.public_score, s.private_score, s.error_description, flush=True)
PY

echo "${TAG}_JOB_END $(date -u)"
