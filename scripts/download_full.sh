#!/usr/bin/env bash
# =============================================================================
# RareAI - Full dataset download (~85 GB, 8 FASTQs + Track 1 files)
# =============================================================================
# PREREQUISITES
#   1. Access granted to SageBio/mva-hackathon-2026-data on Hugging Face
#   2. HF_TOKEN set in .env
#   3. RAREAI_DATA_DIR in .env pointing at a disk with >= 110 GB free
#      (current: external disk - see .env)
#
# USAGE
#   Foreground (watch it run):
#     ./scripts/download_full.sh
#
#   Background (recommended; survives terminal close):
#     nohup ./scripts/download_full.sh >/dev/null 2>&1 &
#
# MONITOR (from another terminal)
#     uv run rareai status --watch 30     # live progress table
#     tail -f logs/download_full.log     # raw retry/attempt log
#
# BEHAVIOR
#   - Retries each file up to 12 times with growing backoff (configs/dataset.yaml)
#   - Partial transfers resume automatically between attempts
#   - Already-complete files are skipped, so the script is fully re-runnable
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

if ! grep -qs '^HF_TOKEN=..' .env; then
  echo "ERROR: HF_TOKEN missing in .env (see .env.example)" >&2
  exit 1
fi

DATA_DIR="$(grep -m1 '^RAREAI_DATA_DIR=' .env | cut -d= -f2)"
if [[ -n "${DATA_DIR}" ]]; then
  AVAIL_GB=$(df --output=avail -BG "${DATA_DIR}" | tail -1 | tr -dc '0-9')
  echo "Data dir : ${DATA_DIR}"
  echo "Free     : ${AVAIL_GB} GB (need >= ~110 GB for the full set)"
  if (( AVAIL_GB < 110 )); then
    echo "WARNING: low disk space - download may fail partway." >&2
  fi
fi

mkdir -p logs
echo "==> Full dataset download started at $(date)" | tee -a logs/download_full.log
if uv run rareai download --full 2>&1 | tee -a logs/download_full.log; then
  echo "==> SUCCESS at $(date)" | tee -a logs/download_full.log
else
  echo "==> FAILED at $(date) - re-run this script; completed files are skipped." | tee -a logs/download_full.log
  exit 1
fi
