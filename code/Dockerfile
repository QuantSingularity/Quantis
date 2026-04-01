# ---- Base stage ----
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies stage ----
FROM base AS deps

COPY api/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Runtime stage ----
FROM base AS runtime

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r quantis && useradd -r -g quantis quantis

# Create required directories
RUN mkdir -p /app/models /app/data && chown -R quantis:quantis /app

# Copy application code
COPY --chown=quantis:quantis api/ ./api/
COPY --chown=quantis:quantis data/ ./data/
COPY --chown=quantis:quantis models/ ./models/

USER quantis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
