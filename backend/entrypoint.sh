#!/bin/bash
set -e

echo "==================================="
echo "Army AI Backend - Starting..."
echo "==================================="

# Storage is now OverlayFS mounted on host
# Just ensure required directories exist
echo "Setting up storage directories..."
mkdir -p /storage/2d/datasets /storage/2d/patches
mkdir -p /storage/3d/datasets /storage/3d/patches
mkdir -p /storage/model /storage/evaluate
mkdir -p /storage/simulator/Vehicles
mkdir -p /app/logs

echo "✅ Storage ready (OverlayFS mounted on host)"
echo "==================================="

# Execute the main command
exec "$@"
