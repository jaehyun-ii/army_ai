#!/bin/bash
set -e

echo "==================================="
echo "CARLA Client Backend - Starting..."
echo "==================================="

# OverlayFS 마운트 (복사 없이 레이어링)
# storage_init (읽기 전용) + storage_rw (호스트 ./storage/3d) → /storage_merged
if [ -d "/storage_init" ] && [ "$(ls -A /storage_init 2>/dev/null)" ]; then
    if ! mountpoint -q /storage_merged 2>/dev/null; then
        echo "Setting up OverlayFS (no copy, instant)..."

        # 호스트 ./storage/3d에 직접 저장하도록 설정
        mkdir -p /storage_upper /storage_work /storage_merged

        # OverlayFS 마운트
        # lowerdir: 읽기 전용 (storage_init)
        # upperdir: 읽기/쓰기 (호스트 ./storage/3d에 저장)
        # workdir: overlay 작업용 (upperdir와 분리된 디렉토리)
        mount -t overlay overlay \
            -o lowerdir=/storage_init,upperdir=/storage_upper,workdir=/storage_work \
            /storage_merged

        echo "✅ OverlayFS mounted: /storage_merged"
        echo "   Lower (RO): /storage_init"
        echo "   Upper (RW): /storage_upper → 호스트 ./storage/3d"
    else
        echo "OverlayFS already mounted at /storage_merged"
    fi

    # Symlink workspace to storage_merged
    if [ ! -L /workspace ]; then
        ln -sf /storage_merged /workspace
        echo "Workspace linked to /storage_merged"
    fi
else
    echo "No storage_init found, using regular workspace"
    mkdir -p /workspace
fi

# Ensure required directories exist
mkdir -p /workspace/exports /workspace/imports /workspace/logs /workspace/storage

echo "Workspace ready"
echo "==================================="

# Execute the main command
exec "$@"
