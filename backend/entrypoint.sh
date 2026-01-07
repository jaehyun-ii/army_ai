#!/bin/bash
set -e

echo "==================================="
echo "Army AI Backend - Starting..."
echo "==================================="

# OverlayFS 마운트 (복사 없이 레이어링)
if [ -d "/storage_init" ] && [ "$(ls -A /storage_init 2>/dev/null)" ]; then
    if ! mountpoint -q /storage 2>/dev/null; then
        echo "Setting up OverlayFS (no copy, instant)..."

        # 디렉토리 생성
        mkdir -p /storage_rw/upper /storage_rw/work /storage

        # OverlayFS 마운트
        # lowerdir: 읽기 전용 (storage_init)
        # upperdir: 읽기/쓰기 (새 파일, 수정사항)
        # workdir: overlay 작업용
        mount -t overlay overlay \
            -o lowerdir=/storage_init,upperdir=/storage_rw/upper,workdir=/storage_rw/work \
            /storage

        echo "✅ OverlayFS mounted: /storage"
        echo "   Lower (RO): /storage_init"
        echo "   Upper (RW): /storage_rw/upper"
    else
        echo "OverlayFS already mounted at /storage"
    fi
else
    echo "No storage_init found, using regular storage"
    mkdir -p /storage
fi

# Ensure required directories exist
mkdir -p /storage/2d/datasets /storage/2d/patches
mkdir -p /storage/3d/datasets /storage/3d/patches
mkdir -p /storage/model /storage/evaluate
mkdir -p /app/logs

echo "Storage ready"
echo "==================================="

# Execute the main command
exec "$@"
