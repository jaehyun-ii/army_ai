#!/bin/bash
set -e

echo "==================================="
echo "CARLA Simulator - Starting..."
echo "==================================="

# CARLA Vehicles blueprint path (now mounted from host OverlayFS)
CARLA_VEHICLES_PATH="/home/carla/CarlaUE4/Content/Carla/Blueprints/Vehicles"

# Vehicles are now OverlayFS mounted on host, just verify
echo "Checking vehicles directory..."
if [ -d "$CARLA_VEHICLES_PATH" ]; then
    echo "✅ Vehicles ready (OverlayFS mounted on host)"
    echo "   Path: $CARLA_VEHICLES_PATH"
else
    echo "⚠ Warning: Vehicles directory not found at $CARLA_VEHICLES_PATH"
    mkdir -p "$CARLA_VEHICLES_PATH"
fi

echo "==================================="

# Execute CARLA as carla user (we started as root to fix permissions)
echo "Switching to carla user and starting CARLA..."
exec su -s /bin/bash carla -c "/bin/bash CarlaUE4.sh $*"
