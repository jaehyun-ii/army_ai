# ===============================================
# Army AI Platform - Backend Dockerfile (GPU)
# FastAPI + Custom CUDA/PyTorch Stack
# ===============================================
# Ubuntu: 22.04
# CUDA Toolkit: 12.6
# PyTorch: 2.5.0+
# Python: 3.10
# ===============================================

# 1. CUDA 12.6, Ubuntu 22.04 기반 이미지 사용
FROM nvcr.io/nvidia/pytorch:25.10-py3

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

# 2. 필수 시스템 패키지 설치 (NGC 이미지에 없는 것만)
# NOTE: build-essential, cmake, git, wget, curl, ca-certificates,
#       OpenCV dependencies, video codecs are already in NGC image
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 3. pip 업그레이드 (Python 3.12 호환 setuptools 설치)
RUN pip install --upgrade pip setuptools>=75.0.0 wheel

# 6. 애플리케이션 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install mmengine (required before mmcv)
RUN pip install mmengine==0.10.7

# Install mmcv from source (v2.1.0)
RUN git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    git checkout v2.1.0 && \
    pip install -r requirements/optional.txt && \
    export MMCV_WITH_OPS=1 && \
    export FORCE_CUDA=1 && \
    pip install . --no-build-isolation -v && \
    cd ..



# Install mmdet
RUN git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && \
    export FORCE_CUDA=1 && \
    pip install . --no-build-isolation -v  && \
    cd ..


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
