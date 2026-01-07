#!/bin/bash
set -e

echo "==================================="
echo "Army AI Backend - Starting..."
echo "==================================="

# Initialize storage from storage_init if needed
INIT_FLAG="/storage/.initialized"

if [ -d "/storage_init" ] && [ "$(ls -A /storage_init 2>/dev/null)" ]; then
    if [ -f "$INIT_FLAG" ]; then
        echo "Storage already initialized (found $INIT_FLAG), skipping..."
    else
        echo "First-time initialization from storage_init using rsync..."

        # Ensure storage directories exist
        mkdir -p /storage/2d/datasets /storage/2d/patches
        mkdir -p /storage/3d/datasets /storage/3d/patches
        mkdir -p /storage/model /storage/evaluate

        # Use rsync for efficient synchronization
        # -a: archive mode (preserves permissions, timestamps, etc.)
        # --ignore-existing: only copy files that don't exist in destination
        # --info=progress2: show overall progress instead of file-by-file
        echo "Syncing files from storage_init..."
        rsync -a --ignore-existing --info=progress2 /storage_init/ /storage/

        # Create flag file
        touch "$INIT_FLAG"
        echo "Initialization complete via rsync"
    fi
else
    echo "No storage_init found or empty, skipping initialization"
fi

# Ensure storage directories exist
mkdir -p /storage/2d/datasets /storage/2d/patches
mkdir -p /storage/3d/datasets /storage/3d/patches
mkdir -p /storage/model /storage/evaluate
mkdir -p /app/logs

echo "Storage ready"
echo "==================================="

# Execute the main command
exec "$@"
