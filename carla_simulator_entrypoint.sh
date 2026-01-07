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
        echo "First-time vehicle initialization using hardlinks (instant)..."

        # Ensure CARLA vehicles directory exists
        mkdir -p "$CARLA_VEHICLES_PATH"

        echo "Creating hardlinks from storage_init/Vehicles..."
        file_count=0

        # Hardlink all files from storage_init/Vehicles
        find /storage_init/Vehicles -type f 2>/dev/null | while read src_file; do
            filename="$(basename "$src_file")"
            dest_file="$CARLA_VEHICLES_PATH/$filename"

            if [ ! -e "$dest_file" ]; then
                if ln "$src_file" "$dest_file" 2>/dev/null; then
                    file_count=$((file_count + 1))
                else
                    # Fallback to copy if hardlink fails
                    cp "$src_file" "$dest_file"
                fi
            fi
        done

        # Copy existing custom vehicles from storage (these override storage_init)
        if [ -d "/storage/Vehicles" ] && [ "$(ls -A /storage/Vehicles 2>/dev/null)" ]; then
            echo "Loading custom vehicles from storage (overwrite)..."
            find /storage/Vehicles -type f 2>/dev/null | while read src_file; do
                filename="$(basename "$src_file")"
                dest_file="$CARLA_VEHICLES_PATH/$filename"
                # Always copy/overwrite from storage
                cp -f "$src_file" "$dest_file"
            done
        fi

        # Create flag file
        mkdir -p /storage
        touch "$INIT_FLAG"
        echo "Vehicle initialization complete: $file_count files from storage_init"
    fi
else
    echo "No storage_init vehicles found, using default CARLA vehicles"
fi

echo "==================================="

# Execute CARLA
exec /bin/bash CarlaUE4.sh "$@"
