#!/usr/bin/env bash
set -euo pipefail

# Seed DB on first run
if [ ! -f "$BILLY_DB" ]; then
    mkdir -p "$(dirname "$BILLY_DB")"
    echo "Seeding Billy DB at $BILLY_DB"
    python -c "import app.db"   # creates schema via init_db()
    python reset_db.py
fi

# REST API in background on :8766
uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT:-8766}" &
REST_PID=$!

# MCP server (SSE) in foreground on :8765
echo "Billy MCP server listening on http://0.0.0.0:${MCP_PORT:-8765}/sse"
echo "Billy REST API listening on http://0.0.0.0:${API_PORT:-8766}/docs"
exec python -m app.main_noauth --http

# Cleanup REST on exit
trap "kill $REST_PID 2>/dev/null" EXIT
