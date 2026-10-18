#!/usr/bin/env bash
# ==============================================================================
# Helper script to manage scheduled crawl of nhathuoclongchau.com.vn in background
# Runs 100 URLs every 8 minutes with auto-resume and rate limiting.
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

PID_FILE="$REPO_ROOT/logs/crawl_longchau.pid"
LOG_FILE="$REPO_ROOT/logs/crawl_longchau.log"
mkdir -p "$REPO_ROOT/logs"

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Crawler is already running with PID $(cat "$PID_FILE")."
            exit 0
        fi

        echo "Starting continuous adaptive crawler for nhathuoclongchau.com.vn..."
        nohup "$PYTHON_BIN" scripts/crawl_missing_urls.py \
            --hosts nhathuoclongchau.com.vn \
            --concurrency 10 \
            --domain-concurrency 2 \
            --domain-interval 0.5 \
            --cooldown-seconds 60 \
            --max-retries 3 \
            >> "$LOG_FILE" 2>&1 &

        echo $! > "$PID_FILE"
        echo "Crawler started in background (PID: $!)."
        echo "Log file: $LOG_FILE"
        echo "To view live logs: ./scripts/schedule_longchau.sh logs"
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
                echo "No active process found for PID $PID. Cleaning up stale PID file."
                rm -f "$PID_FILE"
            fi
        else
            echo "No crawler PID file found. Checking for running process..."
            pkill -f "scripts/crawl_missing_urls.py.*nhathuoclongchau" || echo "No crawler process running."
        fi
        ;;

    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Crawler is RUNNING (PID: $(cat "$PID_FILE"))."
            echo "Last 5 lines of log:"
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
        echo ""
        echo "Commands:"
        echo "  start   - Run crawler in background (100 URLs every 8 mins)"
        echo "  stop    - Stop the running background crawler"
        echo "  status  - Check if crawler is running"
        echo "  logs    - Follow live crawler logs"
        exit 1
        ;;
esac
