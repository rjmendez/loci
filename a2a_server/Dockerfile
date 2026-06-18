FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.133.1" \
    "uvicorn[standard]==0.41.0" \
    "starlette==1.0.1" \
    "pydantic==2.13.4" \
    "aiohttp==3.13.4" \
    "pyotp==2.9.0"

COPY server.py client.py ./

# MRPINK_A2A_TOKEN is required — server exits at startup if unset.
# Provide it via .env file or environment injection.
# Generate: python3 -c "import secrets;print(secrets.token_hex(32))"

# Mnemosyne SQLite path — mount a volume here for persistence across restarts
ENV MNEMOSYNE_DATA_DIR=/data/mnemosyne
# A2A identity
ENV MRPINK_A2A_HOST=0.0.0.0
ENV MRPINK_A2A_PORT=8201
ENV MRPINK_A2A_URL=http://localhost:8201
ENV HERMES_AGENT_ID=mrpink
# External services
ENV QDRANT_URL=http://localhost:6333
ENV QDRANT_API_KEY=
ENV MNEMOSYNE_EMBEDDING_API_URL=http://localhost:11434/v1
ENV MNEMOSYNE_EMBEDDING_MODEL=nomic-embed-text

EXPOSE 8201

# /data holds Mnemosyne SQLite (memory store for recall/remember skills)
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8201/health')" || exit 1

CMD ["python", "server.py"]
