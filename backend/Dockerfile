# ===============================================
# Army AI Platform - Backend Dockerfile (GPU)
# FastAPI + NVIDIA NGC PyTorch Container
# ===============================================

# Use NVIDIA NGC PyTorch container
# Includes: PyTorch, CUDA, cuDNN, Python 3.10
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
# NGC container already includes PyTorch, CUDA, cuDNN
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenMMLab packages using mim
# This ensures proper CUDA/PyTorch compatibility
# Fix numpy/pandas compatibility issue first

# Install mmcv with prebuilt wheel to avoid long compilation
# Using CUDA 12.1 + PyTorch 2.4 prebuilt wheel

# Copy application code
COPY app ./app
# Create necessary directories
RUN mkdir -p /app/storage /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
