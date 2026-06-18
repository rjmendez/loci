FROM python:3.11-slim

WORKDIR /app

# Install build deps for some C extensions (fastembed, sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first — layer cache friendly
COPY requirements.txt pyproject.toml ./
COPY memcheck/ ./memcheck/
RUN pip install --no-cache-dir -e .

COPY server.py .env.example ./
COPY tests/ ./tests/

# fastembed downloads ONNX models on first embed; mount a cache volume to
# avoid re-downloading across restarts.
ENV FASTEMBED_CACHE_PATH=/data/fastembed-cache
ENV HERMES_MEMORY_DIR=/data/memory-sessions

# Embedding + vector store — point at your external services via env or .env
ENV QDRANT_URL=http://localhost:6333
ENV QDRANT_API_KEY=
ENV OLLAMA_BASE_URL=http://localhost:11434
ENV EMBED_MODEL=nomic-embed-text

# Transport: "stdio" for Claude Code subprocess, "sse" or "streamable-http" for HTTP
ENV HERMES_MCP_TRANSPORT=sse
ENV HERMES_MCP_HOST=0.0.0.0
ENV HERMES_MCP_PORT=8000

EXPOSE 8000

# /data holds fastembed model cache + JSONL session fallback files
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request,os; urllib.request.urlopen('http://localhost:' + os.environ.get('HERMES_MCP_PORT','8000') + '/health')" || exit 1

CMD ["python", "server.py"]
