# ── Build stage ───────────────────────────────────────────────────────────────
# We use a slim Python image. "slim" omits docs, test tools, and build headers
# that are present in the full image — final image is ~200MB instead of ~900MB.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies into a separate prefix so we can copy just them later.
# --no-cache-dir avoids storing the pip download cache in the image layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# CONCEPT: non-root user
#   Running as root inside a container means a container escape gives an
#   attacker root on the host. We create a dedicated user with no home dir
#   and no shell for least-privilege.
RUN adduser --system --no-create-home --shell /bin/false gateway
USER gateway

# Expose the port uvicorn listens on (documented for docker-compose / k8s)
EXPOSE 8000

# CONCEPT: exec form CMD
#   CMD ["uvicorn", ...] uses exec form — the process becomes PID 1 directly,
#   so Docker's SIGTERM on `docker stop` reaches it immediately.
#   Shell form (CMD uvicorn ...) spawns a shell as PID 1 that may not forward
#   signals, causing a 10-second grace-period timeout on every stop.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
