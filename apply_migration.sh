#!/bin/bash
# Apply database migration for image_3d annotations support

echo "Applying database migration: add_image_3d_annotations.sql"

# Run migration inside Docker container
docker exec army_ai_postgres psql -U admin -d armydb -f /database/migrations/add_image_3d_annotations.sql

if [ $? -eq 0 ]; then
    echo "✅ Migration applied successfully"
else
    echo "❌ Migration failed"
    exit 1
fi
