# NeuroMCP-Agent Docker Image
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY responsible_ai/ ./responsible_ai/
COPY agents/ ./agents/
COPY mcp/ ./mcp/
COPY data/ ./data/
COPY databases/ ./databases/

# Create non-root user for security
RUN useradd -m -u 1000 neuromcp && \
    chown -R neuromcp:neuromcp /app
USER neuromcp

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "scripts/train.py", "--help"]

# Labels
LABEL maintainer="NeuroMCP-Agent Team"
LABEL version="2.5.0"
LABEL description="Trustworthy Multi-Agent Deep Learning for EEG-Based Neurological Disease Detection"
