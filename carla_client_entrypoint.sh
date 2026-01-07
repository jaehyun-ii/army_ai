#!/bin/bash
set -e

echo "==================================="
echo "CARLA Client Backend - Starting..."
echo "==================================="

# Initialize workspace from storage_init if needed
if [ -d "/storage_init" ] && [ "$(ls -A /storage_init 2>/dev/null)" ]; then
    echo "Initializing workspace from storage_init..."

    # Install rsync if not available
    if ! command -v rsync &> /dev/null; then
        echo "Installing rsync..."
        apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1
    fi

    # Ensure workspace directories exist
    mkdir -p /workspace/exports /workspace/imports /workspace/logs /workspace/storage

    # Copy storage_init to workspace (only if files don't exist)
    # exports
    if [ -d "/storage_init/exports" ]; then
        rsync -av --ignore-existing /storage_init/exports/ /workspace/exports/
    fi

    # imports
    if [ -d "/storage_init/imports" ]; then
        rsync -av --ignore-existing /storage_init/imports/ /workspace/imports/
    fi

    # logs
    if [ -d "/storage_init/logs" ]; then
        rsync -av --ignore-existing /storage_init/logs/ /workspace/logs/
    fi

    # storage
    if [ -d "/storage_init/storage" ]; then
        rsync -av --ignore-existing /storage_init/storage/ /workspace/storage/
    fi

    # Now copy from /storage (read-write storage) - these override storage_init
    if [ -d "/storage" ]; then
        if [ -d "/storage/exports" ]; then
            rsync -av /storage/exports/ /workspace/exports/
        fi
        if [ -d "/storage/imports" ]; then
            rsync -av /storage/imports/ /workspace/imports/
        fi
        if [ -d "/storage/logs" ]; then
            rsync -av /storage/logs/ /workspace/logs/
        fi
        if [ -d "/storage/storage" ]; then
            rsync -av /storage/storage/ /workspace/storage/
        fi
    fi

    echo "Workspace initialization complete"
else
    echo "No storage_init found, using storage only"
    mkdir -p /workspace/exports /workspace/imports /workspace/logs /workspace/storage
fi

echo "==================================="

# Execute the main command
exec "$@"
