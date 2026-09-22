# MCP + FastAPI service image (Sude).

FROM python:3.11-slim

# Unbuffered so a container that dies still tells you why.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency metadata and source before the rest, so editing a config or an eval
# file does not invalidate the install layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/
# CPU-only torch first: the default wheel adds ~2 GB of CUDA libraries the
# service never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[serve,agents,judge,eval]"

COPY configs/ ./configs/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

# Non-root. The serving path only ever reads the corpus.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

# Readiness, not liveness. The process answers immediately; it can only serve a
# query once a backend is bound, and that difference is what stops a deploy going
# green with no index. The start period must outlast the backend warm-up (loading
# the embedding model and index took over 100 s on a cold machine): failures
# inside it do not count, so a container that is still loading is not killed.
# `degraded` passes here — the stub is a legitimate state to
# run in — and the deploy workflow decides whether to ship it.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import json,sys,urllib.request; \
        s=json.load(urllib.request.urlopen('http://localhost:8000/ready'))['status']; \
        sys.exit(0 if s in ('ok','degraded') else 1)"

CMD ["python", "scripts/serve_S.py", "api", "--host", "0.0.0.0", "--port", "8000"]
