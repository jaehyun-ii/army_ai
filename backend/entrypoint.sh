#!/bin/bash
set -e

echo "==================================="
echo "Army AI Backend - Starting..."
echo "==================================="

# OverlayFS 마운트 (복사 없이 레이어링)
if [ -d "/storage_init" ] && [ "$(ls -A /storage_init 2>/dev/null)" ]; then
    if ! mountpoint -q /storage 2>/dev/null; then
        echo "Setting up OverlayFS (no copy, instant)..."

        # 호스트 ./storage에 직접 저장하도록 설정
        # workdir는 upperdir와 같은 마운트 내에 있어야 함
        mkdir -p /storage_overlay/.overlayfs_work /storage

        # OverlayFS 마운트
        # lowerdir: 읽기 전용 (storage_init)
        # upperdir: 읽기/쓰기 (호스트 ./storage에 저장)
        # workdir: overlay 작업용 (upperdir와 같은 마운트 내)
        mount -t overlay overlay \
            -o lowerdir=/storage_init,upperdir=/storage_overlay,workdir=/storage_overlay/.overlayfs_work \
            /storage

        echo "✅ OverlayFS mounted: /storage"
        echo "   Lower (RO): /storage_init"
        echo "   Upper (RW): /storage_overlay → 호스트 ./storage"
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
