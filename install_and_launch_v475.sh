#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(pwd)"

# Detect python executable (prefer python3) 
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "No python interpreter found (need python3)" >&2
  exit 1
fi

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
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

: "${DASHBOARD_PORT:=8601}"
: "${HYPER_LAB_PORT:=8610}"

# launch in tmux
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed; please install to auto-launch sessions" >&2
  exit 1
fi

echo "--- Stopping any existing tmux sessions ---"
tmux kill-session -t trading_session 2>/dev/null || true
tmux kill-session -t dashboard_session 2>/dev/null || true
tmux kill-session -t hyper_lab 2>/dev/null || true
tmux kill-session -t hyper_lab_dashboard 2>/dev/null || true
echo "Old sessions cleared."

echo "--- Launching new tmux sessions ---"
# Create a new session for the main loop, separate from hyper_lab
if ! tmux has-session -t main_loop 2>/dev/null; then
  tmux new-session -d -s main_loop "bash -lc 'source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; $PYTHON_BIN -m v26meme.cli loop > loop.out 2> loop.err'"
fi
if ! tmux has-session -t dashboard_session 2>/dev/null; then
  tmux new-session -d -s dashboard_session "bash -lc 'source .venv/bin/activate; set -a; source .env 2>/dev/null || true; set +a; streamlit run dashboard/app.py --server.fileWatcherType=none --server.port=${DASHBOARD_PORT}'"
fi
# Hyper lab now runs the main loop as well.
if ! tmux has-session -t hyper_lab 2>/dev/null; then
  tmux new-session -d -s hyper_lab "bash -lc 'source .venv/bin/activate; set -a; export HYPER_LAB_PORT=${HYPER_LAB_PORT}; source .env 2>/dev/null || true; set +a; $PYTHON_BIN -m v26meme.labs.hyper_lab main --config=configs/config.yaml > hyper_lab.out 2> hyper_lab.err'"
fi
if ! tmux has-session -t hyper_lab_dashboard 2>/dev/null; then
  tmux new-session -d -s hyper_lab_dashboard "bash -lc 'source .venv/bin/activate; streamlit run dashboard/hyper_lab_app.py --server.fileWatcherType=none --server.port=${HYPER_LAB_PORT}'"
fi

echo "✅ v4.7.5 launched (sessions: dashboard_session (:${DASHBOARD_PORT}), hyper_lab (main loop), hyper_lab_dashboard (:${HYPER_LAB_PORT}))"
