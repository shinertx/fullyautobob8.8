#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(pwd)"

# export .env if present
set -a
[ -f .env ] && source .env || true
set +a

# ensure Redis (portable: only attempt systemctl if available)
if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-units --type=service | grep -q redis; then
    sudo systemctl enable redis-server || true
    sudo systemctl start redis-server || true
  fi
fi

if ! redis-cli PING >/dev/null 2>&1; then
  echo "Redis not reachable (ensure redis-server running)" >&2
  exit 1
fi

# venv + deps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

: "${DASHBOARD_PORT:=8601}"

# launch in tmux
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed; please install to auto-launch sessions" >&2
  exit 1
fi

tmux has-session -t trading_session 2>/dev/null || tmux new-session -d -s trading_session "bash -lc 'source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; python -m v26meme.cli loop'"
tmux has-session -t dashboard_session 2>/dev/null || tmux new-session -d -s dashboard_session "bash -lc 'source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; streamlit run dashboard/app.py --server.fileWatcherType=none --server.port=${DASHBOARD_PORT}'"
tmux has-session -t hyper_lab 2>/dev/null || tmux new-session -d -s hyper_lab "bash -lc 'source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; python -m v26meme.labs.hyper_lab run'"

echo "✅ v4.7.5 launched (dynamic mapping enabled) in tmux sessions: trading_session, dashboard_session, hyper_lab"
