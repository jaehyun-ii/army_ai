#!/bin/bash
set -e

echo "==================================="
echo "CARLA Client Backend - Starting..."
echo "==================================="

# Workspace is now OverlayFS mounted on host (directly to /workspace)
# Just ensure required directories exist
echo "Setting up workspace directories..."
mkdir -p /workspace/exports /workspace/imports /workspace/logs /workspace/storage

echo "✅ Workspace ready (OverlayFS mounted on host)"
echo "==================================="

# Execute the main command
exec "$@"
