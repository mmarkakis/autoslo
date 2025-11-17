#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PID_FILE="$ROOT/.ui-meta/pids"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE. Nothing to stop."
  exit 0
fi

# shellcheck disable=SC1090
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
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name did not exit, forcing..."
      kill -KILL "$pid" 2>/dev/null || true
    fi
  else
    echo "$name (pid=$pid) is not running."
  fi
}

stop_pid "backend" "${BACKEND_PID:-}"
stop_pid "frontend" "${FRONTEND_PID:-}"

rm -f "$PID_FILE"
echo "Stopped. Removed $PID_FILE."

rm -rf "$ROOT/.ui-meta"
