# ── Stage 1: Build ────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application
COPY . .

# Expose ports
EXPOSE 8501 8000

# Default: run dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
