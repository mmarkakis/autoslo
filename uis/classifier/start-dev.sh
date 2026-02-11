#!/usr/bin/env bash
set -euo pipefail

# Repository root (two levels up from uis/classifier)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$ROOT/.classifier-ui-meta"
PID_FILE="$LOG_DIR/pids"

mkdir -p "$LOG_DIR"

# Configuration (override via env)
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
BACKEND_APP="${BACKEND_APP:-autoslo.api.main:app}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-1998}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"

# Prevent duplicate runs
if [[ -f "$PID_FILE" ]]; then
  echo "Found $PID_FILE; previous dev processes may be running. Run stop-dev.sh first." >&2
  exit 1
fi

# Start backend
echo "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT} (app=${BACKEND_APP}) ..."
nohup python -m uvicorn --app-dir "$ROOT/src" "$BACKEND_APP" \
  --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload \
  > "$LOG_DIR/backend.log" 2>&1 &

BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 2

# Check if backend is running
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  echo "Backend failed to start. Check $LOG_DIR/backend.log for details."
  exit 1
fi

# Start frontend
echo "Starting frontend on http://localhost:${FRONTEND_PORT} ..."
cd "$SCRIPT_DIR"

# Install dependencies if needed
if [[ ! -d "node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm install
fi

nohup npm run dev -- --port "$FRONTEND_PORT" > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait a moment for frontend to start
sleep 2

# Check if frontend is running
if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
  echo "Frontend failed to start. Check $LOG_DIR/frontend.log for details."
  kill "$BACKEND_PID" 2>/dev/null || true
  exit 1
fi

# Save PIDs
cat > "$PID_FILE" <<EOF
BACKEND_PID=$BACKEND_PID
FRONTEND_PID=$FRONTEND_PID
EOF

echo ""
echo "========================================="
echo "Arrival Classifier UI is running!"
echo "========================================="
echo ""
echo "Frontend: http://localhost:${FRONTEND_PORT}"
echo "Backend:  http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "API docs: http://${BACKEND_HOST}:${BACKEND_PORT}/docs"
echo ""
echo "Logs:"
echo "  Backend:  $LOG_DIR/backend.log"
echo "  Frontend: $LOG_DIR/frontend.log"
echo ""
echo "To stop, run: $SCRIPT_DIR/stop-dev.sh"
