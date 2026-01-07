#!/bin/bash
set -e

echo "==================================="
echo "CARLA Client Backend - Starting..."
echo "==================================="

# OverlayFS 마운트 (복사 없이 레이어링)
# storage_init (읽기 전용) + storage (읽기/쓰기) → /storage_merged
if [ -d "/storage_init" ] && [ "$(ls -A /storage_init 2>/dev/null)" ]; then
    if ! mountpoint -q /storage_merged 2>/dev/null; then
        echo "Setting up OverlayFS (no copy, instant)..."

        # 디렉토리 생성
        mkdir -p /storage_rw/upper /storage_rw/work /storage_merged

        # OverlayFS 마운트
        # lowerdir: 읽기 전용 (storage_init)
        # upperdir: 읽기/쓰기 (storage에서 마운트된 실제 데이터)
        # workdir: overlay 작업용

        # storage가 있으면 2계층 overlay (storage_init < storage < 새 변경사항)
        if [ -d "/storage" ] && [ "$(ls -A /storage 2>/dev/null)" ]; then
            echo "Using 2-layer overlay: storage_init (base) + storage (middle) + upper (new)"
            mount -t overlay overlay \
                -o lowerdir=/storage:/storage_init,upperdir=/storage_rw/upper,workdir=/storage_rw/work \
                /storage_merged
        else
            echo "Using single overlay: storage_init (base) + upper (new)"
            mount -t overlay overlay \
                -o lowerdir=/storage_init,upperdir=/storage_rw/upper,workdir=/storage_rw/work \
                /storage_merged
        fi

        echo "✅ OverlayFS mounted: /storage_merged"
        echo "   Lower: /storage_init (+ /storage if exists)"
        echo "   Upper: /storage_rw/upper"
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
