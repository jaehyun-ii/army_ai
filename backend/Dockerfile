# ===============================================
# Army AI Platform - Backend Dockerfile (GPU)
# FastAPI + Custom CUDA/PyTorch Stack
# ===============================================
# Ubuntu: 22.04
# CUDA Toolkit: 13.0.2
# PyTorch: 2.9.0+
# Python: 3.10
# ===============================================

# 1. CUDA 13.0.2, Ubuntu 22.04 기반 이미지 사용
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

# 2. Python 3.10 및 필수 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# 3. pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 4. 최신 PyTorch(2.9.0+, CUDA 12/13 지원) 설치
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 5. TensorRT 설치
RUN pip install tensorrt

# 6. 애플리케이션 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -U openmim
RUN mim install mmcv==2.1.0
RUN mim install mmengine==0.10.7
RUN mim install mmdet==3.3.0

# 7. 애플리케이션 코드 복사
COPY app ./app

# 8. 필요한 디렉토리 생성
RUN mkdir -p /app/storage /app/logs

# 9. 포트 노출
EXPOSE 8000

# 10. Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 11. FastAPI 서버 시작
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
