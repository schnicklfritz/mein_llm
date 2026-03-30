FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# ── Ollama ────────────────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# ── App files ─────────────────────────────────────────────────────────────────
COPY scripts/ /scripts/
RUN chmod +x /scripts/*.sh

COPY supervisor/ /etc/supervisor/conf.d/

# ── Ports ─────────────────────────────────────────────────────────────────────
# 11434 = Ollama
# 8000  = ChromaDB
# 3000  = MCP server
EXPOSE 11434 8000 3000

# ── Runtime env ───────────────────────────────────────────────────────────────
ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_NUM_PARALLEL=2 \
    CHROMA_HOST=0.0.0.0 \
    CHROMA_PORT=8000 \
    CHROMA_DATA_DIR=/workspace/chroma \
    MCP_PORT=3000 \
    DEEPSEEK_API_KEY="" \
    DEEPSEEK_BASE_URL="https://api.deepseek.com" \
    OLLAMA_MODEL="comfy-llm"

ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
