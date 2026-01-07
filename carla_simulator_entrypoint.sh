#!/bin/bash
set -e

echo "==================================="
echo "CARLA Simulator - Starting..."
echo "==================================="

# CARLA Vehicles blueprint path
CARLA_VEHICLES_PATH="/home/carla/CarlaUE4/Content/Carla/Blueprints/Vehicles"
INIT_FLAG="/storage/.vehicles_initialized"

# Initialize vehicles from storage_init if needed
if [ -d "/storage_init/Vehicles" ] && [ "$(ls -A /storage_init/Vehicles 2>/dev/null)" ]; then
    if [ -f "$INIT_FLAG" ]; then
        echo "Vehicles already initialized (found $INIT_FLAG), skipping..."
    else
        echo "First-time vehicle initialization using rsync..."

        # Ensure CARLA vehicles directory exists
        mkdir -p "$CARLA_VEHICLES_PATH"

        # Sync vehicles from storage_init using rsync (only new files)
        echo "Syncing vehicles from storage_init..."
        rsync -av --ignore-existing /storage_init/Vehicles/ "$CARLA_VEHICLES_PATH/"

        # Sync custom vehicles from storage (these override storage_init)
        if [ -d "/storage/Vehicles" ] && [ "$(ls -A /storage/Vehicles 2>/dev/null)" ]; then
            echo "Loading custom vehicles from storage (overwrite)..."
            rsync -av /storage/Vehicles/ "$CARLA_VEHICLES_PATH/"
        fi

        # Create flag file
        mkdir -p /storage
        touch "$INIT_FLAG"
        echo "Vehicle initialization complete via rsync"
    fi
else
    echo "No storage_init vehicles found, using default CARLA vehicles"
fi

echo "==================================="

# Execute CARLA
exec /bin/bash CarlaUE4.sh "$@"
