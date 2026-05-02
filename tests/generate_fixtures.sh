#!/usr/bin/env bash
# generate_fixtures.sh — generate binary test fixtures that are not checked in.
#
# Run from the workspace root:
#   bash tests/generate_fixtures.sh
#
# Requirements: ffmpeg on PATH.
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURES_DIR="${WORKSPACE_ROOT}/fixtures"

# ── sample.mp3 ────────────────────────────────────────────────────────────────
SAMPLE_MP3="${FIXTURES_DIR}/sample.mp3"

if [[ -f "${SAMPLE_MP3}" ]]; then
    echo "[fixtures] ${SAMPLE_MP3} already exists, skipping."
else
    if ! command -v ffmpeg &>/dev/null; then
        echo "[fixtures] ERROR: ffmpeg not found on PATH." >&2
        echo "           Install with: brew install ffmpeg" >&2
        exit 1
    fi
    echo "[fixtures] Generating ${SAMPLE_MP3} ..."
    ffmpeg -loglevel error \
        -f lavfi \
        -i sine=frequency=1000:duration=3 \
        -ar 16000 \
        -ac 1 \
        "${SAMPLE_MP3}"
    echo "[fixtures] Done: ${SAMPLE_MP3}"
fi
