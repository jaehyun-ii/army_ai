#!/bin/bash
set -e

echo "==================================="
echo "CARLA Client Backend - Starting..."
echo "==================================="

# Initialize workspace from storage_init if needed
INIT_FLAG="/workspace/.initialized"

if [ -d "/storage_init" ] && [ "$(ls -A /storage_init 2>/dev/null)" ]; then
    if [ -f "$INIT_FLAG" ]; then
        echo "Workspace already initialized (found $INIT_FLAG), skipping..."
    else
        echo "First-time workspace initialization using rsync..."

        # Ensure workspace directories exist
        mkdir -p /workspace/exports /workspace/imports /workspace/logs /workspace/storage

        # Sync from storage_init using rsync (only new files)
        echo "Syncing from storage_init..."
        rsync -av --ignore-existing /storage_init/ /workspace/

        # Now sync from /storage (read-write storage) - these override storage_init
        if [ -d "/storage" ]; then
            echo "Loading from storage (overwrite mode)..."
            rsync -av /storage/ /workspace/
        fi

        # Create flag file
        touch "$INIT_FLAG"
        echo "Workspace initialization complete via rsync"
    fi
else
    echo "No storage_init found, using storage only"
    mkdir -p /workspace/exports /workspace/imports /workspace/logs /workspace/storage
fi

echo "==================================="

# Execute the main command
exec "$@"
