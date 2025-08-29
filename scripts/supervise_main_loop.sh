#!/usr/bin/env bash
set -euo pipefail

# Simple supervisor for the main CLI loop. Restarts on exit with backoff.
# PIT-safe: does not change trading logic; only ensures the orchestrator stays up.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Export .env if available
set -a
source .env 2>/dev/null || true
set +a

PY_BIN="python3"
command -v python3 >/dev/null 2>&1 || PY_BIN="python"

RETRY_MAX_TRIES=${RETRY_MAX_TRIES:-0}             # 0 = infinite
RETRY_BACKOFF_S=${RETRY_BACKOFF_S:-2}
RETRY_BACKOFF_MAX_S=${RETRY_BACKOFF_MAX_S:-30}

tries=0
while :; do
  # Propagate redis retry knobs to the child process
  export REDIS_RETRY_ON_START=${REDIS_RETRY_ON_START:-1}
  export REDIS_RETRY_TRIES=${REDIS_RETRY_TRIES:-5}
  export REDIS_RETRY_BACKOFF_S=${REDIS_RETRY_BACKOFF_S:-1}

  echo "[supervisor] starting main loop (try=$tries)" | tee -a loop.supervisor.log
  set +e
  $PY_BIN -m v26meme.cli loop >> loop.out 2>> loop.err
  ec=$?
  set -e
  echo "[supervisor] main loop exited ec=$ec" | tee -a loop.supervisor.log

  tries=$((tries+1))
  if [[ "$RETRY_MAX_TRIES" -gt 0 && "$tries" -ge "$RETRY_MAX_TRIES" ]]; then
    echo "[supervisor] reached max tries ($RETRY_MAX_TRIES); exiting" | tee -a loop.supervisor.log
    exit 1
  fi
  # backoff with cap
  sleep "${RETRY_BACKOFF_S}"
  if [[ "$RETRY_BACKOFF_S" -lt "$RETRY_BACKOFF_MAX_S" ]]; then
    RETRY_BACKOFF_S=$(( RETRY_BACKOFF_S * 2 ))
    if [[ "$RETRY_BACKOFF_S" -gt "$RETRY_BACKOFF_MAX_S" ]]; then
      RETRY_BACKOFF_S=$RETRY_BACKOFF_MAX_S
    fi
  fi
done

