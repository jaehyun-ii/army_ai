#!/bin/bash
set -e

echo "==================================="
echo "Army AI Backend - Starting..."
echo "==================================="

# Initialize storage from storage_init if needed
if [ -d "/storage_init" ] && [ "$(ls -A /storage_init 2>/dev/null)" ]; then
    echo "Initializing storage from storage_init..."

    # Copy storage_init contents to storage (only if files don't exist)
    # This preserves existing files in storage while adding missing ones from storage_init
    rsync -av --ignore-existing /storage_init/ /storage/

    echo "Storage initialization complete"
else
    echo "No storage_init found or empty, skipping initialization"
fi

# Ensure storage directories exist
mkdir -p /storage/2d/datasets /storage/2d/patches
mkdir -p /storage/3d/datasets /storage/3d/patches
mkdir -p /storage/model
mkdir -p /storage/evaluate
mkdir -p /app/logs

echo "Storage directories ready"
echo "==================================="

# Execute the main command
exec "$@"
