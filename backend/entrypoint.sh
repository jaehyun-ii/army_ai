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
        echo "First-time initialization from storage_init using hardlinks (instant)..."

        # Ensure storage directories exist
        mkdir -p /storage/2d/datasets /storage/2d/patches
        mkdir -p /storage/3d/datasets /storage/3d/patches
        mkdir -p /storage/model /storage/evaluate

        # Create hardlinks instead of copying
        # Hardlinks: 즉시 완료, 디스크 공간 절약, 같은 inode 공유
        # 안전: storage_init은 ro 마운트라서 수정 불가

        echo "Creating hardlinks from storage_init..."
        file_count=0

        # Find all files in storage_init and create hardlinks
        find /storage_init -type f 2>/dev/null | while read src_file; do
            # Get relative path
            rel_path="${src_file#/storage_init/}"
            dest_file="/storage/$rel_path"
            dest_dir="$(dirname "$dest_file")"

            # Create destination directory if needed
            mkdir -p "$dest_dir"

            # Create hardlink only if destination doesn't exist
            if [ ! -e "$dest_file" ]; then
                if ln "$src_file" "$dest_file" 2>/dev/null; then
                    file_count=$((file_count + 1))
                else
                    # If hardlink fails (different filesystem), fallback to copy
                    cp "$src_file" "$dest_file"
                    echo "  Warning: hardlink failed for $rel_path, copied instead"
                fi
            fi
        done

        # Create flag file
        touch "$INIT_FLAG"
        echo "Initialization complete: $file_count files linked"
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
