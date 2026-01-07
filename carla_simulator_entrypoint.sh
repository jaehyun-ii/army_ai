#!/bin/bash
set -e

echo "==================================="
echo "CARLA Simulator - Starting..."
echo "==================================="

# CARLA Vehicles blueprint path
CARLA_VEHICLES_PATH="/home/carla/CarlaUE4/Content/Carla/Blueprints/Vehicles"

# OverlayFS 마운트 for Vehicles (복사 없이 레이어링)
if [ -d "/storage_init/Vehicles" ] && [ "$(ls -A /storage_init/Vehicles 2>/dev/null)" ]; then
    if ! mountpoint -q "$CARLA_VEHICLES_PATH" 2>/dev/null; then
        echo "Setting up OverlayFS for Vehicles (no copy, instant)..."

        # Ensure CARLA vehicles directory exists
        mkdir -p "$CARLA_VEHICLES_PATH"

        # Create storage_rw with proper permissions
        mkdir -p /storage_rw
        chmod 777 /storage_rw  # Allow container user to create subdirs
        mkdir -p /storage_rw/upper /storage_rw/work

        # OverlayFS 마운트
        # lowerdir: 읽기 전용 (storage_init/Vehicles)
        # upperdir: 읽기/쓰기 (새 차량, 수정사항)
        # workdir: overlay 작업용

        # storage/Vehicles가 있으면 2계층 overlay
        if [ -d "/storage/Vehicles" ] && [ "$(ls -A /storage/Vehicles 2>/dev/null)" ]; then
            echo "Using 2-layer overlay: storage_init (base) + storage (custom)"
            mount -t overlay overlay \
                -o lowerdir=/storage/Vehicles:/storage_init/Vehicles,upperdir=/storage_rw/upper,workdir=/storage_rw/work \
                "$CARLA_VEHICLES_PATH"
        else
            echo "Using single overlay: storage_init (base)"
            mount -t overlay overlay \
                -o lowerdir=/storage_init/Vehicles,upperdir=/storage_rw/upper,workdir=/storage_rw/work \
                "$CARLA_VEHICLES_PATH"
        fi

        echo "✅ OverlayFS mounted: $CARLA_VEHICLES_PATH"
        echo "   Lower: /storage_init/Vehicles (+ /storage/Vehicles if exists)"
        echo "   Upper: /storage_rw/upper"
    else
        echo "OverlayFS already mounted at $CARLA_VEHICLES_PATH"
    fi
else
    echo "No storage_init vehicles found, using default CARLA vehicles"
    mkdir -p "$CARLA_VEHICLES_PATH"
fi

echo "Vehicles ready"
echo "==================================="

# Execute CARLA
exec /bin/bash CarlaUE4.sh "$@"
