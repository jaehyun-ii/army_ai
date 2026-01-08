-- Create vehicle_models table for CARLA 3D model management
-- Created: 2026-01-07

-- Create ENUM types
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vehiclecategory') THEN
        CREATE TYPE vehiclecategory AS ENUM ('군용차량', '항공기', '개인장비', '건축물', '기타');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'modelformat') THEN
        CREATE TYPE modelformat AS ENUM ('fbx', 'obj', 'gltf', 'glb', 'dae', 'blend');
    END IF;
END$$;

-- Create vehicle_models table
CREATE TABLE IF NOT EXISTS vehicle_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL UNIQUE,
    display_name VARCHAR(200),
    category vehiclecategory NOT NULL DEFAULT '기타',
    format modelformat NOT NULL,

    -- File information
    storage_key TEXT NOT NULL,
    file_name VARCHAR(1024) NOT NULL,
    file_size BIGINT,

    -- Model metadata
    polygons INTEGER,
    vertices INTEGER,
    textures JSONB,
    materials JSONB,

    -- Metadata
    description TEXT,
    metadata JSONB,

    -- Status
    status VARCHAR(50) DEFAULT 'ready',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_vehicle_models_name ON vehicle_models(name);
CREATE INDEX IF NOT EXISTS idx_vehicle_models_category ON vehicle_models(category);
CREATE INDEX IF NOT EXISTS idx_vehicle_models_format ON vehicle_models(format);
CREATE INDEX IF NOT EXISTS idx_vehicle_models_deleted_at ON vehicle_models(deleted_at);
CREATE INDEX IF NOT EXISTS idx_vehicle_models_created_at ON vehicle_models(created_at DESC);

-- Add comments
COMMENT ON TABLE vehicle_models IS 'CARLA simulator 3D vehicle models';
COMMENT ON COLUMN vehicle_models.name IS 'Model identifier (alphanumeric, -, _)';
COMMENT ON COLUMN vehicle_models.display_name IS 'Korean display name';
COMMENT ON COLUMN vehicle_models.storage_key IS 'Storage path: simulator/Vehicle/{name}/{file}';
COMMENT ON COLUMN vehicle_models.deleted_at IS 'Soft delete timestamp';
