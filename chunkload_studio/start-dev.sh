#!/usr/bin/env bash
set -euo pipefail

# repo root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
LOG_DIR="$ROOT/.ui-meta"
PID_FILE="$ROOT/.ui-meta/pids"

mkdir -p "$LOG_DIR"

# Configuration (override via env)
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
BACKEND_APP="${BACKEND_APP:-chunkbench.api:app}"   # e.g. chunkbench.api.app:app if your app module lives there
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Prevent duplicate runs
if [[ -f "$PID_FILE" ]]; then
  echo "Found $PID_FILE; previous dev processes may be running. Run scripts/stop-dev.sh first." >&2
  exit 1
fi

# Start backend
echo "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT} (app=${BACKEND_APP}) ..."
nohup python -m uvicorn --app-dir "$ROOT/src" "$BACKEND_APP" \
  --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload \
  > "$LOG_DIR/backend.log" 2>&1 &

BACKEND_PID=$!
sleep 0.5

# Start frontend
echo "Starting frontend on http://localhost:${FRONTEND_PORT} ..."
pushd "$ROOT/chunkbench_studio" >/dev/null
nohup npm run dev -- --port "$FRONTEND_PORT" \
  > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
popd >/dev/null
sleep 0.5

# Save PIDs
{
  echo "BACKEND_PID=$BACKEND_PID"
  echo "FRONTEND_PID=$FRONTEND_PID"
} > "$PID_FILE"

echo "Started. PIDs saved to $PID_FILE"
echo "Logs:"
echo "  Backend:  $LOG_DIR/backend.log"
echo "  Frontend: $LOG_DIR/frontend.log"
