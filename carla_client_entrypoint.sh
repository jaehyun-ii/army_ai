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
        echo "First-time workspace initialization using hardlinks (instant)..."

        # Ensure workspace directories exist
        mkdir -p /workspace/exports /workspace/imports /workspace/logs /workspace/storage

        echo "Creating hardlinks from storage_init..."
        file_count=0

        # Hardlink files from storage_init subdirectories
        for subdir in exports imports logs storage; do
            if [ -d "/storage_init/$subdir" ] && [ "$(ls -A /storage_init/$subdir 2>/dev/null)" ]; then
                find "/storage_init/$subdir" -type f 2>/dev/null | while read src_file; do
                    rel_path="${src_file#/storage_init/$subdir/}"
                    dest_file="/workspace/$subdir/$rel_path"
                    dest_dir="$(dirname "$dest_file")"

                    mkdir -p "$dest_dir"

                    if [ ! -e "$dest_file" ]; then
                        if ln "$src_file" "$dest_file" 2>/dev/null; then
                            file_count=$((file_count + 1))
                        else
                            # Fallback to copy
                            cp "$src_file" "$dest_file"
                        fi
                    fi
                done
            fi
        done

        # Now copy from /storage (read-write storage) - these override storage_init
        if [ -d "/storage" ]; then
            echo "Loading from storage (overwrite mode)..."
            for subdir in exports imports logs storage; do
                if [ -d "/storage/$subdir" ] && [ "$(ls -A /storage/$subdir 2>/dev/null)" ]; then
                    find "/storage/$subdir" -type f 2>/dev/null | while read src_file; do
                        rel_path="${src_file#/storage/$subdir/}"
                        dest_file="/workspace/$subdir/$rel_path"
                        dest_dir="$(dirname "$dest_file")"

                        mkdir -p "$dest_dir"
                        cp -f "$src_file" "$dest_file"
                    done
                fi
            done
        fi

        # Create flag file
        touch "$INIT_FLAG"
        echo "Workspace initialization complete: $file_count files linked from storage_init"
    fi
else
    echo "No storage_init found, using storage only"
    mkdir -p /workspace/exports /workspace/imports /workspace/logs /workspace/storage
fi

echo "==================================="

# Execute the main command
exec "$@"
