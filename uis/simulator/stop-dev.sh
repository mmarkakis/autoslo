#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_FILE="$ROOT/.simulator-ui-meta/pids"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE. Nothing to stop."
  exit 0
fi

# Load PIDs
source "$PID_FILE"

stop_pid() {
  local name="$1"
  local pid="${2:-}"
  if [[ -z "$pid" ]]; then
    echo "No PID for $name."
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $name (pid=$pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    # Wait up to ~4s
    for i in $(seq 1 20); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.2
    done
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
      echo "Force killing $name (pid=$pid)..."
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "$name stopped."
  else
    echo "$name (pid=$pid) is not running."
  fi
}

stop_pid "Backend" "${BACKEND_PID:-}"
stop_pid "Frontend" "${FRONTEND_PID:-}"

rm -f "$PID_FILE"
echo "Cleanup complete."
