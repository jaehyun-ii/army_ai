#!/bin/bash
set -e

echo "==================================="
echo "CARLA Simulator - Starting..."
echo "==================================="

# CARLA Vehicles blueprint path
CARLA_VEHICLES_PATH="/home/carla/CarlaUE4/Content/Carla/Blueprints/Vehicles"

# Initialize vehicles from storage_init if needed
if [ -d "/storage_init/Vehicles" ] && [ "$(ls -A /storage_init/Vehicles 2>/dev/null)" ]; then
    echo "Initializing vehicles from storage_init..."

    # Install rsync if not available
    if ! command -v rsync &> /dev/null; then
        echo "Installing rsync..."
        apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1
    fi

    # Ensure CARLA vehicles directory exists
    mkdir -p "$CARLA_VEHICLES_PATH"

    # Copy storage_init vehicles to CARLA path (only if files don't exist)
    rsync -av --ignore-existing /storage_init/Vehicles/ "$CARLA_VEHICLES_PATH/"

    # Copy existing custom vehicles from storage to CARLA path
    if [ -d "/storage/Vehicles" ] && [ "$(ls -A /storage/Vehicles 2>/dev/null)" ]; then
        echo "Loading custom vehicles from storage..."
        rsync -av /storage/Vehicles/ "$CARLA_VEHICLES_PATH/"
    fi

    echo "Vehicle initialization complete"
else
    echo "No storage_init vehicles found, using default CARLA vehicles"
fi

echo "==================================="

# Execute CARLA
exec /bin/bash CarlaUE4.sh "$@"
