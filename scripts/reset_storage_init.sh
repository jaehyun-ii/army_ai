#!/bin/bash
# Reset storage initialization flags
# Use this when you want to re-initialize from storage_init

echo "Resetting storage initialization flags..."

# Backend storage
if [ -f "./storage/.initialized" ]; then
    rm ./storage/.initialized
    echo "✅ Removed backend storage flag"
fi

# CARLA simulator vehicles
if [ -f "./storage/simulator/.vehicles_initialized" ]; then
    rm ./storage/simulator/.vehicles_initialized
    echo "✅ Removed CARLA vehicles flag"
fi

# CARLA client workspace (inside container, use docker exec)
echo "Note: CARLA client workspace flag is inside container at /workspace/.initialized"
echo "To reset it, run: docker exec army_ai_carla_backend_dev rm -f /workspace/.initialized"

echo ""
echo "Next container restart will re-initialize from storage_init"
