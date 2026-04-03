#!/bin/bash
# /pre_start.sh
# mein_llm pre-start hook
# Runs before supervisord launches services
# Called from Docker ENTRYPOINT override or On Start Script in QuickPod

set -e

echo "########################################"
echo "  mein_llm pre_start running..."
echo "########################################"

# ── Create /workspace folder structure ───────────────────────────────────────
echo "[INFO] Creating /workspace folder structure..."
mkdir -p \
    /workspace/ollama/models \
    /workspace/chroma \
    /workspace/maxential \
    /workspace/rag/workflows \
    /workspace/logs

# ── Auto-configure rclone B2 from QuickPod env vars ──────────────────────────
if [ -n "${B2_KEY_ID}" ] && [ -n "${B2_APPLICATION_KEY}" ]; then
    export RCLONE_CONFIG_B2_TYPE=b2
    export RCLONE_CONFIG_B2_ACCOUNT="${B2_KEY_ID}"
    export RCLONE_CONFIG_B2_KEY="${B2_APPLICATION_KEY}"
    echo "[INFO] rclone B2 auto-configured from environment"
    echo "[INFO] B2 bucket: ${B2_BUCKET:-<not set>}"
else
    echo "[WARN] B2_KEY_ID / B2_APPLICATION_KEY not set - rclone B2 not configured"
fi

# ── Restore webui.db from /workspace if present ──────────────────────────────
WEBUI_DB_SRC="/workspace/webui.db"
WEBUI_DB_DST="/usr/local/lib/python3.12/dist-packages/open_webui/data/webui.db"
if [ -f "$WEBUI_DB_SRC" ]; then
    echo "[INFO] Restoring webui.db from /workspace..."
    mkdir -p "$(dirname $WEBUI_DB_DST)"
    cp "$WEBUI_DB_SRC" "$WEBUI_DB_DST"
    echo "[INFO] webui.db restored"
else
    echo "[WARN] No webui.db found at $WEBUI_DB_SRC - Open WebUI will start fresh"
fi

# ── Restore config.json from /workspace if present ───────────────────────────
CONFIG_SRC="/workspace/config.json"
CONFIG_DST="/usr/local/lib/python3.12/dist-packages/open_webui/data/config.json"
if [ -f "$CONFIG_SRC" ]; then
    echo "[INFO] Restoring config.json from /workspace..."
    cp "$CONFIG_SRC" "$CONFIG_DST"
    echo "[INFO] config.json restored"
else
    echo "[WARN] No config.json found at $CONFIG_SRC - skipping"
fi

echo "########################################"
echo "  mein_llm pre_start complete."
echo "  Launching supervisord..."
echo "########################################"
