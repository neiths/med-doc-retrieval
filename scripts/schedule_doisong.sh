#!/usr/bin/env bash
# ==============================================================================
# Background manager for crawling Sức Khỏe Đời Sống (Bộ Y Tế - 78k URLs)
# Runs in batches with intervals to prevent IP bans.
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

PID_FILE="$REPO_ROOT/logs/crawl_doisong.pid"
LOG_FILE="$REPO_ROOT/logs/crawl_doisong.log"
mkdir -p "$REPO_ROOT/logs"

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Crawler is already running with PID $(cat "$PID_FILE")."
            exit 0
        fi

        echo "Starting continuous crawler for Sức Khỏe Đời Sống..."
        nohup "$PYTHON_BIN" scripts/crawl_vietnamese.py \
            --group doisong \
            --concurrency 15 \
            --domain-concurrency 3 \
            --cooldown-seconds 45 \
            >> "$LOG_FILE" 2>&1 &

        echo $! > "$PID_FILE"
        echo "Crawler started in background (PID: $!)."
        echo "Log file: $LOG_FILE"
        echo "To view live logs: ./scripts/schedule_doisong.sh logs"
        ;;

    stop)
        if [ -f "$PID_FILE" ]; then
            PID="$(cat "$PID_FILE")"
            if kill -0 "$PID" 2>/dev/null; then
                echo "Stopping crawler (PID: $PID)..."
                kill "$PID"
                rm -f "$PID_FILE"
                echo "Crawler stopped."
            else
                rm -f "$PID_FILE"
            fi
        fi
        pkill -f "crawl_vietnamese.*doisong" 2>/dev/null || true
        ;;

    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Crawler is RUNNING (PID: $(cat "$PID_FILE"))."
            echo "Recent log:"
            tail -n 5 "$LOG_FILE" 2>/dev/null || true
        else
            echo "Crawler is NOT running."
        fi
        ;;

    logs)
        tail -f "$LOG_FILE"
        ;;

    *)
        echo "Usage: $0 {start|stop|status|logs}"
        exit 1
        ;;
esac
