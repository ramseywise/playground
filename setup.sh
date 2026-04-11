#!/usr/bin/env bash
set -euo pipefail

# ── Prerequisites ─────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[ERROR] uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[setup] Created .env from .env.example — set ANTHROPIC_API_KEY before running."
else
    echo "[setup] .env already exists, skipping."
fi

# ── Dependencies ──────────────────────────────────────────────────────────────
uv sync --extra librarian --extra api --extra mcp

# ── Local data directories ────────────────────────────────────────────────────
# Mirror LibrarySettings defaults: duckdb_path=.duckdb/librarian.db, chroma_persist_dir=.chroma
mkdir -p .duckdb .chroma logs

echo ""
echo "[setup] Done. Next steps:"
echo "  1. Edit .env and set ANTHROPIC_API_KEY"
echo "  2. Run: make eval-unit              # smoke-test the install"
echo "  3. Run: docker compose -f infra/docker/docker-compose.yml up --build"
