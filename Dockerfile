FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git aria2 rclone pigz unzip \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    supervisor \
    zstd \
    netcat-openbsd \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Rclone
RUN curl -fsSL https://rclone.org/install.sh | bash

# Uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx    

# ── Ollama ────────────────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# ── App files ─────────────────────────────────────────────────────────────────
COPY scripts/ /scripts/

COPY supervisor/ /etc/supervisor/conf.d/

COPY scripts/pre_start.sh /pre_start.sh
RUN chmod +x /pre_start.sh

# ── Ports ─────────────────────────────────────────────────────────────────────
# 11434 = Ollama
# 8000  = ChromaDB
# 3000  = MCP server
EXPOSE 11434 8000 3000 8080 8001

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
    OLLAMA_MODEL="comfy-llm" \
    OPENWEBUI_PORT=8080 \
    OLLAMA_BASE_URL="http://localhost:11434"

ENTRYPOINT ["/bin/bash", "-c", "/pre_start.sh && /usr/bin/supervisord -c /etc/supervisor/supervisord.conf"]

