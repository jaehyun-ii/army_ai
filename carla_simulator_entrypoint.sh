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

        # 호스트 ./storage/simulator에 직접 저장하도록 설정
        # workdir는 upperdir와 같은 마운트 내에 있어야 함
        mkdir -p /storage_overlay/.overlayfs_work
        chmod -R 777 /storage_overlay /storage_overlay/.overlayfs_work

        # OverlayFS 마운트
        # lowerdir: 읽기 전용 (storage_init/Vehicles)
        # upperdir: 읽기/쓰기 (호스트 ./storage/simulator에 저장)
        # workdir: overlay 작업용 (upperdir와 같은 마운트 내)
        mount -t overlay overlay \
            -o lowerdir=/storage_init/Vehicles,upperdir=/storage_overlay,workdir=/storage_overlay/.overlayfs_work \
            "$CARLA_VEHICLES_PATH"

        echo "✅ OverlayFS mounted: $CARLA_VEHICLES_PATH"
        echo "   Lower (RO): /storage_init/Vehicles"
        echo "   Upper (RW): /storage_overlay → 호스트 ./storage/simulator"
    else
        echo "OverlayFS already mounted at $CARLA_VEHICLES_PATH"
    fi
else
    echo "No storage_init vehicles found, using default CARLA vehicles"
    mkdir -p "$CARLA_VEHICLES_PATH"
fi

echo "Vehicles ready"
echo "==================================="

# Execute CARLA as carla user (we started as root to fix permissions)
echo "Switching to carla user and starting CARLA..."
exec su -s /bin/bash carla -c "/bin/bash CarlaUE4.sh $*"
