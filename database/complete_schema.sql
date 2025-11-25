BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role_enum') THEN
  CREATE TYPE user_role_enum AS ENUM ('user', 'admin');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'attack_type_enum') THEN
  CREATE TYPE attack_type_enum AS ENUM ('patch', 'noise');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_framework_enum') THEN
  CREATE TYPE model_framework_enum AS ENUM ('pytorch', 'tensorflow', 'onnx', 'tensorrt', 'openvino', 'custom');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_framework_enum') THEN
  CREATE TYPE model_framework_enum AS ENUM ('pytorch', 'tensorflow', 'onnx', 'tensorrt', 'openvino', 'custom');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_stage_enum') THEN
  CREATE TYPE model_stage_enum AS ENUM ('draft', 'staging', 'production', 'archived');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_stage_enum') THEN
  CREATE TYPE model_stage_enum AS ENUM ('draft', 'staging', 'production', 'archived');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'artifact_type_enum') THEN
  CREATE TYPE artifact_type_enum AS ENUM ('model','weights','config','labelmap','tokenizer','calibration','support','other');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'eval_status_enum') THEN
  CREATE TYPE eval_status_enum AS ENUM ('queued','running','completed','failed','aborted');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'eval_phase_enum') THEN
  CREATE TYPE eval_phase_enum AS ENUM ('pre_attack','post_attack');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'rt_run_status_enum') THEN
  CREATE TYPE rt_run_status_enum AS ENUM ('running','completed','failed','aborted');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'annotation_type_enum') THEN
  CREATE TYPE annotation_type_enum AS ENUM ('bbox','polygon','keypoint','segmentation');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'experiment_status_enum') THEN
  CREATE TYPE experiment_status_enum AS ENUM ('draft','running','completed','failed','archived');
END IF;
END$$;

DO $$BEGIN
IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'eval_dataset_type_enum') THEN
  CREATE TYPE eval_dataset_type_enum AS ENUM ('base','attack');
END IF;
END$$;

CREATE OR REPLACE FUNCTION trg_set_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION trg_rt_runs_validate_completion()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
  frame_cnt integer;
BEGIN
  IF (TG_OP = 'UPDATE'
      AND NEW.status = 'completed'
      AND (OLD.status IS DISTINCT FROM NEW.status)
      AND NEW.frames_expected IS NOT NULL
      AND NEW.frames_expected > 0) THEN
    SELECT count(*) INTO frame_cnt
    FROM rt_frames
    WHERE run_id = NEW.id
      AND deleted_at IS NULL;
    IF frame_cnt <> NEW.frames_expected THEN
      RAISE EXCEPTION 'Cannot mark run % as completed. frames_expected=%, stored_frames=%',
        NEW.id, NEW.frames_expected, frame_cnt;
    END IF;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TABLE users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  username varchar(100) NOT NULL,
  email varchar(255),
  password_hash varchar(255) NOT NULL,
  role user_role_enum NOT NULL DEFAULT 'user',
  is_active boolean NOT NULL DEFAULT true,
  -- Security fields for login attempt tracking and session management
  failed_login_attempts integer NOT NULL DEFAULT 0,
  locked_until timestamptz,
  current_session_id varchar(255),
  last_login_at timestamptz,
  -- Timestamps
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz
);

CREATE UNIQUE INDEX uq_users_username_active
ON users (lower(username))
WHERE deleted_at IS NULL;

CREATE UNIQUE INDEX uq_users_email_active
ON users (lower(email))
WHERE deleted_at IS NULL AND email IS NOT NULL;

-- Index for session lookups (single login enforcement)
CREATE INDEX idx_users_current_session_id
ON users(current_session_id)
WHERE current_session_id IS NOT NULL;

-- Index for locked accounts (performance optimization)
CREATE INDEX idx_users_locked_until
ON users(locked_until)
WHERE locked_until IS NOT NULL;

CREATE TABLE experiments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(200) NOT NULL,
  description text,
  objective text,
  hypothesis text,
  status experiment_status_enum NOT NULL DEFAULT 'draft',
  started_at timestamptz,
  ended_at timestamptz,
  config jsonb,
  results_summary jsonb,
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_experiment_name CHECK (char_length(name) > 0),
  CONSTRAINT chk_experiment_status_time CHECK (
    (status = 'draft' AND started_at IS NULL AND ended_at IS NULL) OR
    (status = 'running' AND started_at IS NOT NULL AND ended_at IS NULL) OR
    (status IN ('completed','failed','archived') AND started_at IS NOT NULL AND ended_at IS NOT NULL AND ended_at >= started_at)
  ),
  CONSTRAINT chk_experiment_config CHECK (config IS NULL OR jsonb_typeof(config) = 'object'),
  CONSTRAINT chk_experiment_results CHECK (results_summary IS NULL OR jsonb_typeof(results_summary) = 'object')
);

CREATE TABLE experiment_tags (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(100) NOT NULL UNIQUE,
  color varchar(20),
  description text,
  created_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_tag_name CHECK (char_length(name) > 0)
);

CREATE UNIQUE INDEX uq_experiment_tags_name_active
ON experiment_tags (lower(name))
WHERE deleted_at IS NULL;

CREATE TABLE experiment_tag_assignments (
  experiment_id uuid REFERENCES experiments(id) ON DELETE CASCADE,
  tag_id uuid REFERENCES experiment_tags(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (experiment_id, tag_id)
);

CREATE INDEX idx_tag_assignments_tag ON experiment_tag_assignments(tag_id);
CREATE INDEX idx_tag_assignments_experiment ON experiment_tag_assignments(experiment_id);

CREATE TABLE od_models (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(200) NOT NULL,
  task varchar(50) NOT NULL DEFAULT 'object-detection',
  owner_id uuid REFERENCES users(id) ON DELETE SET NULL,
  description text,
  version varchar(64) NOT NULL,
  framework model_framework_enum NOT NULL,
  framework_version varchar(64),
  input_spec jsonb,
  labelmap jsonb,
  inference_params jsonb,
  stage model_stage_enum NOT NULL DEFAULT 'draft',
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  published_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_od_models_task CHECK (char_length(task) > 0),
  CONSTRAINT chk_od_models_input CHECK (input_spec IS NULL OR jsonb_typeof(input_spec) = 'object'),
  CONSTRAINT chk_od_models_labelmap CHECK (labelmap IS NULL OR jsonb_typeof(labelmap) = 'object'),
  CONSTRAINT chk_od_models_inference CHECK (inference_params IS NULL OR jsonb_typeof(inference_params) = 'object')
);

CREATE UNIQUE INDEX uq_od_models_name_version_active
ON od_models (lower(name), version)
WHERE deleted_at IS NULL;

CREATE TABLE model_artifacts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id uuid NOT NULL REFERENCES od_models(id) ON DELETE CASCADE,
  artifact_type artifact_type_enum NOT NULL,
  storage_key text NOT NULL,
  file_name varchar(1024) NOT NULL,
  size_bytes bigint,
  sha256 varchar(64),
  content_type varchar(200),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_artifact_size CHECK (size_bytes IS NULL OR size_bytes >= 0)
);

CREATE INDEX idx_model_artifacts_model_active
ON model_artifacts (model_id)
WHERE deleted_at IS NULL;

CREATE TABLE datasets_2d (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(200) NOT NULL,
  description text,
  owner_id uuid REFERENCES users(id) ON DELETE SET NULL,
  storage_path text NOT NULL,
  metadata jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_dataset_name CHECK (char_length(name) > 0)
);

CREATE UNIQUE INDEX uq_datasets_name_active
ON datasets_2d (lower(name))
WHERE deleted_at IS NULL;

CREATE TABLE images_2d (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id uuid NOT NULL REFERENCES datasets_2d(id) ON DELETE CASCADE,
  file_name varchar(1024) NOT NULL,
  storage_key text NOT NULL,
  width integer,
  height integer,
  mime_type varchar(100),
  metadata jsonb,
  uploaded_by uuid REFERENCES users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_images_dimensions CHECK ((width IS NULL AND height IS NULL) OR (width > 0 AND height > 0)),
  CONSTRAINT chk_images_file_name CHECK (char_length(file_name) > 0)
);

CREATE INDEX idx_images_dataset_active
ON images_2d (dataset_id)
WHERE deleted_at IS NULL;

CREATE TABLE patches_2d (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(200) NOT NULL,
  description text,
  target_model_id uuid REFERENCES od_models(id) ON DELETE RESTRICT,
  source_dataset_id uuid REFERENCES datasets_2d(id) ON DELETE SET NULL,
  target_class varchar(200),
  method varchar(200),
  hyperparameters jsonb,
  patch_metadata jsonb,
  storage_key text,
  file_name varchar(1024),
  size_bytes integer,
  sha256 varchar(64),
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_patch_name CHECK (char_length(name) > 0),
  CONSTRAINT chk_patch_hparams CHECK (hyperparameters IS NULL OR jsonb_typeof(hyperparameters)='object')
);

CREATE UNIQUE INDEX uq_patches_name_active
ON patches_2d (lower(name))
WHERE deleted_at IS NULL;

CREATE TABLE attack_datasets_2d (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(200) NOT NULL,
  description text,
  attack_type attack_type_enum NOT NULL,
  target_model_id uuid REFERENCES od_models(id) ON DELETE RESTRICT,
  base_dataset_id uuid NOT NULL REFERENCES datasets_2d(id) ON DELETE RESTRICT,
  output_dataset_id uuid REFERENCES datasets_2d(id) ON DELETE SET NULL,
  target_class varchar(200),
  patch_id uuid REFERENCES patches_2d(id) ON DELETE RESTRICT,
  parameters jsonb,
  experiment_id uuid REFERENCES experiments(id) ON DELETE SET NULL,
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_attack_name CHECK (char_length(name) > 0),
  CONSTRAINT chk_attack_parameters CHECK (parameters IS NULL OR jsonb_typeof(parameters) = 'object'),
  CONSTRAINT chk_attack_patch CHECK (
    (attack_type = 'patch' AND patch_id IS NOT NULL) OR
    (attack_type <> 'patch' AND patch_id IS NULL)
  )
);

CREATE UNIQUE INDEX uq_attack_datasets_name_active
ON attack_datasets_2d (lower(name))
WHERE deleted_at IS NULL;

CREATE TABLE audit_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_id uuid REFERENCES users(id) ON DELETE SET NULL,
  action varchar(200) NOT NULL,
  target_type varchar(100),
  target_id uuid,
  details jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT chk_audit_action CHECK (char_length(action) > 0),
  CONSTRAINT chk_audit_target_type CHECK (target_type IS NULL OR char_length(target_type) > 0)
);

CREATE TABLE eval_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(200) NOT NULL,
  description text,
  phase eval_phase_enum NOT NULL,
  model_id uuid NOT NULL REFERENCES od_models(id) ON DELETE RESTRICT,
  base_dataset_id uuid REFERENCES datasets_2d(id) ON DELETE RESTRICT,
  attack_dataset_id uuid REFERENCES attack_datasets_2d(id) ON DELETE RESTRICT,
  experiment_id uuid REFERENCES experiments(id) ON DELETE SET NULL,
  params jsonb,
  metrics_summary jsonb,
  started_at timestamptz,
  ended_at timestamptz,
  status eval_status_enum NOT NULL DEFAULT 'queued',
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_eval_name CHECK (char_length(name) > 0),
  CONSTRAINT chk_eval_params CHECK (params IS NULL OR jsonb_typeof(params)='object'),
  CONSTRAINT chk_eval_metrics CHECK (metrics_summary IS NULL OR jsonb_typeof(metrics_summary)='object'),
  CONSTRAINT chk_eval_phase_requirements CHECK (
    (phase = 'pre_attack' AND base_dataset_id IS NOT NULL AND attack_dataset_id IS NULL) OR
    (phase = 'post_attack' AND base_dataset_id IS NOT NULL AND attack_dataset_id IS NOT NULL)
  ),
  CONSTRAINT chk_eval_time_range CHECK (ended_at IS NULL OR started_at IS NULL OR ended_at >= started_at)
);

CREATE INDEX idx_eval_runs_phase ON eval_runs(phase);
CREATE INDEX idx_eval_runs_status ON eval_runs(status);
CREATE INDEX idx_eval_runs_model ON eval_runs(model_id);
CREATE INDEX idx_eval_runs_base ON eval_runs(base_dataset_id);
CREATE INDEX idx_eval_runs_attack ON eval_runs(attack_dataset_id);
CREATE INDEX idx_eval_runs_created_at ON eval_runs(created_at);

CREATE TABLE eval_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  image_2d_id uuid REFERENCES images_2d(id) ON DELETE SET NULL,
  dataset_type eval_dataset_type_enum,
  ground_truth jsonb,
  prediction jsonb,
  metrics jsonb,
  notes text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_eval_item_gt CHECK (ground_truth IS NULL OR jsonb_typeof(ground_truth) IN ('object','array')),
  CONSTRAINT chk_eval_item_pred CHECK (prediction IS NULL OR jsonb_typeof(prediction) IN ('object','array')),
  CONSTRAINT chk_eval_item_metrics CHECK (metrics IS NULL OR jsonb_typeof(metrics)='object')
);

CREATE INDEX idx_eval_items_run ON eval_items(run_id);
CREATE INDEX idx_eval_items_image ON eval_items(image_2d_id);
CREATE INDEX idx_eval_items_dataset_type ON eval_items(dataset_type);
CREATE INDEX idx_eval_items_run_type ON eval_items(run_id, dataset_type);

CREATE TABLE eval_lists (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(200) NOT NULL,
  description text,
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_eval_lists_name CHECK (char_length(name) > 0)
);

CREATE TABLE eval_list_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  list_id uuid NOT NULL REFERENCES eval_lists(id) ON DELETE CASCADE,
  run_id uuid NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  sort_order integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz
);

CREATE UNIQUE INDEX uq_eval_list_items_list_run
ON eval_list_items (list_id, run_id)
WHERE deleted_at IS NULL;

CREATE TABLE rt_capture_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id uuid REFERENCES od_models(id) ON DELETE RESTRICT,
  window_seconds integer NOT NULL DEFAULT 5,
  frames_expected integer NOT NULL DEFAULT 10,
  fps_target numeric(6,3),
  started_at timestamptz NOT NULL DEFAULT now(),
  ended_at timestamptz,
  status rt_run_status_enum NOT NULL DEFAULT 'running',
  notes text,
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_rt_window CHECK (window_seconds > 0),
  CONSTRAINT chk_rt_frames_expected CHECK (frames_expected > 0),
  CONSTRAINT chk_rt_time_range CHECK (ended_at IS NULL OR ended_at >= started_at),
  CONSTRAINT chk_rt_fps CHECK (fps_target IS NULL OR fps_target > 0)
);

CREATE TABLE rt_frames (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES rt_capture_runs(id) ON DELETE CASCADE,
  seq_no integer NOT NULL,
  captured_at timestamptz NOT NULL DEFAULT now(),
  storage_key text,
  width integer,
  height integer,
  mime_type varchar(100),
  metadata jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_rt_frames_seq CHECK (seq_no > 0),
  CONSTRAINT chk_rt_frames_dimensions CHECK ((width IS NULL AND height IS NULL) OR (width > 0 AND height > 0))
);

CREATE UNIQUE INDEX uq_rt_frames_run_seq
ON rt_frames (run_id, seq_no)
WHERE deleted_at IS NULL;

CREATE TABLE annotations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  image_2d_id uuid REFERENCES images_2d(id) ON DELETE CASCADE,
  rt_frame_id uuid REFERENCES rt_frames(id) ON DELETE CASCADE,
  annotation_type annotation_type_enum NOT NULL DEFAULT 'bbox',
  class_name varchar(200) NOT NULL,
  class_index integer,
  bbox_x numeric(10,2),
  bbox_y numeric(10,2),
  bbox_width numeric(10,2),
  bbox_height numeric(10,2),
  polygon_data jsonb,
  keypoints jsonb,
  confidence numeric(5,4) DEFAULT 1.0,
  is_crowd boolean DEFAULT false,
  area numeric(12,2),
  metadata jsonb,
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_annotation_class_name CHECK (char_length(class_name) > 0),
  CONSTRAINT chk_annotation_class_index CHECK (class_index IS NULL OR class_index >= 0),
  CONSTRAINT chk_annotation_confidence CHECK (confidence >= 0 AND confidence <= 1),
  CONSTRAINT chk_annotation_bbox CHECK (annotation_type <> 'bbox' OR (bbox_x IS NOT NULL AND bbox_y IS NOT NULL AND bbox_width > 0 AND bbox_height > 0)),
  CONSTRAINT chk_annotation_polygon CHECK (annotation_type <> 'polygon' OR (polygon_data IS NOT NULL AND jsonb_typeof(polygon_data) = 'array')),
  CONSTRAINT chk_annotation_keypoints CHECK (annotation_type <> 'keypoint' OR (keypoints IS NOT NULL AND jsonb_typeof(keypoints) = 'array')),
  CONSTRAINT chk_annotation_area_nonneg CHECK (area IS NULL OR area >= 0),
  CONSTRAINT chk_annotation_image_xor CHECK (
    (image_2d_id IS NOT NULL AND rt_frame_id IS NULL) OR
    (image_2d_id IS NULL AND rt_frame_id IS NOT NULL)
  )
);

CREATE INDEX idx_annotations_image_2d ON annotations(image_2d_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_annotations_rt_frame ON annotations(rt_frame_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_annotations_class ON annotations(class_name);
CREATE INDEX idx_annotations_class_index ON annotations(class_index);
CREATE INDEX idx_annotations_type ON annotations(annotation_type);
CREATE INDEX idx_annotations_polygon_gin ON annotations USING gin (polygon_data) WHERE polygon_data IS NOT NULL;
CREATE INDEX idx_annotations_keypoints_gin ON annotations USING gin (keypoints) WHERE keypoints IS NOT NULL;

CREATE TABLE experiment_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  experiment_id uuid NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
  run_type varchar(50) NOT NULL,
  parameters jsonb,
  status eval_status_enum NOT NULL DEFAULT 'queued',
  result_summary jsonb,
  created_by uuid REFERENCES users(id) ON DELETE SET NULL,
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_experiment_run_type CHECK (char_length(run_type) > 0),
  CONSTRAINT chk_experiment_run_params CHECK (parameters IS NULL OR jsonb_typeof(parameters)='object'),
  CONSTRAINT chk_experiment_run_results CHECK (result_summary IS NULL OR jsonb_typeof(result_summary)='object')
);

CREATE TABLE experiment_run_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  experiment_run_id uuid NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
  log_level varchar(20),
  message text NOT NULL,
  metadata jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE MATERIALIZED VIEW experiment_summary AS
SELECT
  e.id,
  e.name,
  e.status,
  e.started_at,
  e.ended_at,
  COUNT(DISTINCT a2d.id) AS attack_2d_count,
  COUNT(DISTINCT er.id) AS eval_run_count,
  e.created_at,
  e.created_by
FROM experiments e
LEFT JOIN attack_datasets_2d a2d ON e.id = a2d.experiment_id AND a2d.deleted_at IS NULL
LEFT JOIN eval_runs er ON e.id = er.experiment_id AND er.deleted_at IS NULL
WHERE e.deleted_at IS NULL
GROUP BY e.id, e.name, e.status, e.started_at, e.ended_at, e.created_at, e.created_by;

CREATE UNIQUE INDEX experiment_summary_pkey ON experiment_summary(id);

CREATE VIEW evaluation_overview AS
SELECT
  er.id AS eval_run_id,
  er.phase,
  er.status,
  er.created_at,
  d.name AS base_dataset_name,
  ad.name AS attack_dataset_name,
  m.name AS model_name,
  er.metrics_summary
FROM eval_runs er
LEFT JOIN datasets_2d d ON er.base_dataset_id = d.id
LEFT JOIN attack_datasets_2d ad ON er.attack_dataset_id = ad.id
LEFT JOIN od_models m ON er.model_id = m.id
WHERE er.deleted_at IS NULL;

CREATE VIEW dataset_statistics AS
SELECT
  d.id,
  d.name,
  COUNT(i.id) FILTER (WHERE i.deleted_at IS NULL) AS image_count,
  jsonb_build_object(
    'total_annotations',
    COALESCE((
      SELECT COUNT(*)
      FROM annotations a
      JOIN images_2d ai ON ai.id = a.image_2d_id
      WHERE ai.dataset_id = d.id
        AND a.deleted_at IS NULL
    ), 0)
  ) AS statistics
FROM datasets_2d d
LEFT JOIN images_2d i ON d.id = i.dataset_id AND i.deleted_at IS NULL
WHERE d.deleted_at IS NULL
GROUP BY d.id, d.name;

CREATE VIEW eval_run_pairs AS
SELECT
  pre.id  AS pre_run_id,
  post.id AS post_run_id,
  pre.model_id,
  pre.base_dataset_id,
  post.attack_dataset_id,
  pre.metrics_summary AS pre_metrics,
  post.metrics_summary AS post_metrics,
  pre.created_at AS pre_created_at,
  post.created_at AS post_created_at
FROM eval_runs pre
JOIN eval_runs post
  ON post.phase = 'post_attack'
 AND pre.phase = 'pre_attack'
 AND pre.model_id = post.model_id
 AND pre.base_dataset_id = post.base_dataset_id
WHERE pre.deleted_at IS NULL
  AND post.deleted_at IS NULL;

CREATE VIEW eval_run_pairs_delta AS
SELECT
  pre_run_id,
  post_run_id,
  model_id,
  base_dataset_id,
  attack_dataset_id,
  (pre_metrics->>'mAP')::numeric AS pre_map,
  (post_metrics->>'mAP')::numeric AS post_map,
  ((post_metrics->>'mAP')::numeric - (pre_metrics->>'mAP')::numeric) AS delta_map
FROM eval_run_pairs;

CREATE TRIGGER annotations_set_updated_at_trg
BEFORE UPDATE ON annotations
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER rt_capture_runs_set_updated_at_trg
BEFORE UPDATE ON rt_capture_runs
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER rt_frames_set_updated_at_trg
BEFORE UPDATE ON rt_frames
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER eval_runs_set_updated_at_trg
BEFORE UPDATE ON eval_runs
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER eval_items_set_updated_at_trg
BEFORE UPDATE ON eval_items
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER od_models_set_updated_at_trg
BEFORE UPDATE ON od_models
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER datasets_2d_set_updated_at_trg
BEFORE UPDATE ON datasets_2d
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER patches_2d_set_updated_at_trg
BEFORE UPDATE ON patches_2d
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER attack_datasets_2d_set_updated_at_trg
BEFORE UPDATE ON attack_datasets_2d
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER experiment_runs_set_updated_at_trg
BEFORE UPDATE ON experiment_runs
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER rt_capture_runs_validate_completion_trg
BEFORE UPDATE ON rt_capture_runs
FOR EACH ROW EXECUTE FUNCTION trg_rt_runs_validate_completion();

-- ============================================================================
-- PHASE 1 IMPROVEMENTS (Added 2025-01-14)
-- Quick wins: Missing indexes, triggers, and data quality fixes
-- ============================================================================

-- ----------------------------------------------------------------------------
-- PART 1: MISSING INDEXES ON FOREIGN KEYS
-- ----------------------------------------------------------------------------

-- attack_datasets_2d foreign key indexes
CREATE INDEX idx_attack_datasets_experiment ON attack_datasets_2d(experiment_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_attack_datasets_output ON attack_datasets_2d(output_dataset_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_attack_datasets_base ON attack_datasets_2d(base_dataset_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_attack_datasets_target_model ON attack_datasets_2d(target_model_id) WHERE deleted_at IS NULL;

-- eval_runs foreign key and composite indexes
CREATE INDEX idx_eval_runs_experiment ON eval_runs(experiment_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_eval_runs_model_status ON eval_runs(model_id, status) WHERE deleted_at IS NULL;
CREATE INDEX idx_eval_runs_experiment_phase ON eval_runs(experiment_id, phase) WHERE deleted_at IS NULL;

-- annotations foreign key indexes (image and frame indexes already exist at lines 470-471)
CREATE INDEX idx_annotations_created_by ON annotations(created_by) WHERE deleted_at IS NULL;

-- images_2d foreign key indexes
CREATE INDEX idx_images_uploaded_by ON images_2d(uploaded_by) WHERE deleted_at IS NULL;
CREATE INDEX idx_images_filename ON images_2d(file_name) WHERE deleted_at IS NULL;

-- od_models foreign key indexes
CREATE INDEX idx_od_models_owner ON od_models(owner_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_od_models_created_by ON od_models(created_by) WHERE deleted_at IS NULL;

-- datasets_2d foreign key indexes
CREATE INDEX idx_datasets_owner ON datasets_2d(owner_id) WHERE deleted_at IS NULL;

-- patches_2d foreign key indexes
CREATE INDEX idx_patches_target_model ON patches_2d(target_model_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_patches_source_dataset ON patches_2d(source_dataset_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_patches_created_by ON patches_2d(created_by) WHERE deleted_at IS NULL;

-- experiments foreign key indexes
CREATE INDEX idx_experiments_created_by ON experiments(created_by) WHERE deleted_at IS NULL;
CREATE INDEX idx_experiments_status ON experiments(status) WHERE deleted_at IS NULL;

-- ----------------------------------------------------------------------------
-- PART 2: JSONB GIN INDEXES FOR FASTER JSONB QUERIES
-- ----------------------------------------------------------------------------

CREATE INDEX idx_datasets_metadata_classes_gin ON datasets_2d USING gin((metadata->'classes')) WHERE deleted_at IS NULL;
CREATE INDEX idx_attack_datasets_parameters_gin ON attack_datasets_2d USING gin(parameters) WHERE deleted_at IS NULL;
CREATE INDEX idx_eval_runs_params_gin ON eval_runs USING gin(params) WHERE deleted_at IS NULL;
CREATE INDEX idx_experiments_config_gin ON experiments USING gin(config) WHERE deleted_at IS NULL;

-- ----------------------------------------------------------------------------
-- PART 3: MISSING UPDATED_AT TRIGGERS
-- ----------------------------------------------------------------------------

CREATE TRIGGER users_set_updated_at_trg
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER experiments_set_updated_at_trg
BEFORE UPDATE ON experiments
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER images_2d_set_updated_at_trg
BEFORE UPDATE ON images_2d
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER model_artifacts_set_updated_at_trg
BEFORE UPDATE ON model_artifacts
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER eval_lists_set_updated_at_trg
BEFORE UPDATE ON eval_lists
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE TRIGGER eval_list_items_set_updated_at_trg
BEFORE UPDATE ON eval_list_items
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

-- ============================================================================
-- PHASE 3 IMPROVEMENTS (Added 2025-01-14)
-- Structural improvements: Storage normalization and class definitions
-- NOTE: These are ADDITIVE changes for new databases. Existing tables kept for compatibility.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- ISSUE #1: Storage Objects Normalization
-- ----------------------------------------------------------------------------
-- Centralizes storage metadata from 7 tables into single source of truth

CREATE TABLE storage_objects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  storage_key text NOT NULL,
  file_name varchar(1024) NOT NULL,
  size_bytes bigint,
  sha256 varchar(64),
  content_type varchar(200),
  -- Image-specific (nullable for non-images)
  width integer,
  height integer,
  -- Generic metadata
  metadata jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_storage_dimensions CHECK (
    (width IS NULL AND height IS NULL) OR (width > 0 AND height > 0)
  ),
  CONSTRAINT chk_storage_size CHECK (size_bytes IS NULL OR size_bytes >= 0)
);

CREATE UNIQUE INDEX uq_storage_key_active ON storage_objects(storage_key) WHERE deleted_at IS NULL;
CREATE INDEX idx_storage_sha256 ON storage_objects(sha256) WHERE deleted_at IS NULL;
CREATE INDEX idx_storage_content_type ON storage_objects(content_type) WHERE deleted_at IS NULL;

-- ----------------------------------------------------------------------------
-- ISSUE #5: Class Definitions Normalization
-- ----------------------------------------------------------------------------
-- Replaces od_models.labelmap JSONB with normalized tables

CREATE TABLE class_definitions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(200) NOT NULL,
  category varchar(100),
  description text,
  created_at timestamptz NOT NULL DEFAULT now(),
  deleted_at timestamptz,
  CONSTRAINT chk_class_name CHECK (char_length(name) > 0)
);

CREATE UNIQUE INDEX uq_class_name_active ON class_definitions(lower(name)) WHERE deleted_at IS NULL;
CREATE INDEX idx_class_category ON class_definitions(category) WHERE deleted_at IS NULL;

-- Model-Class mapping (replaces labelmap)
CREATE TABLE model_classes (
  model_id uuid REFERENCES od_models(id) ON DELETE CASCADE,
  class_id uuid REFERENCES class_definitions(id) ON DELETE CASCADE,
  class_index integer NOT NULL,
  PRIMARY KEY (model_id, class_index),
  UNIQUE (model_id, class_id)
);

CREATE INDEX idx_model_classes_class ON model_classes(class_id);

-- Dataset-Class mapping (replaces metadata.classes)
CREATE TABLE dataset_classes (
  dataset_id uuid REFERENCES datasets_2d(id) ON DELETE CASCADE,
  class_id uuid REFERENCES class_definitions(id) ON DELETE CASCADE,
  sort_order integer NOT NULL DEFAULT 0,
  PRIMARY KEY (dataset_id, class_id)
);

CREATE INDEX idx_dataset_classes_class ON dataset_classes(class_id);

-- ----------------------------------------------------------------------------
-- ISSUE #7: Predictions Normalization
-- ----------------------------------------------------------------------------
-- Replaces eval_items.prediction JSONB with normalized table

CREATE TABLE predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  eval_item_id uuid NOT NULL REFERENCES eval_items(id) ON DELETE CASCADE,
  class_id uuid REFERENCES class_definitions(id) ON DELETE SET NULL,
  class_name varchar(200) NOT NULL,
  class_index integer,
  -- Normalized bbox coordinates (0.0 - 1.0)
  bbox_x1 numeric(10, 6) NOT NULL,
  bbox_y1 numeric(10, 6) NOT NULL,
  bbox_x2 numeric(10, 6) NOT NULL,
  bbox_y2 numeric(10, 6) NOT NULL,
  confidence numeric(5, 4) NOT NULL,
  -- Additional prediction data
  metadata jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT chk_class_name CHECK (char_length(class_name) > 0),
  CONSTRAINT chk_class_index CHECK (class_index IS NULL OR class_index >= 0),
  CONSTRAINT chk_bbox_x1 CHECK (bbox_x1 >= 0 AND bbox_x1 <= 1),
  CONSTRAINT chk_bbox_y1 CHECK (bbox_y1 >= 0 AND bbox_y1 <= 1),
  CONSTRAINT chk_bbox_x2 CHECK (bbox_x2 >= 0 AND bbox_x2 <= 1),
  CONSTRAINT chk_bbox_y2 CHECK (bbox_y2 >= 0 AND bbox_y2 <= 1),
  CONSTRAINT chk_bbox_valid CHECK (bbox_x2 > bbox_x1 AND bbox_y2 > bbox_y1),
  CONSTRAINT chk_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX idx_predictions_eval_item ON predictions(eval_item_id);
CREATE INDEX idx_predictions_class_name ON predictions(class_name);
CREATE INDEX idx_predictions_class_id ON predictions(class_id);
CREATE INDEX idx_predictions_confidence ON predictions(confidence);

-- ============================================================================
-- PHASE 2 IMPROVEMENTS (Added 2025-01-14)
-- Data quality and type safety improvements
-- ============================================================================

-- ----------------------------------------------------------------------------
-- ISSUE #9: dataset_type VARCHAR → ENUM (Already Applied)
-- ----------------------------------------------------------------------------
-- eval_dataset_type_enum was already defined at line 78-80
-- eval_items.dataset_type already uses eval_dataset_type_enum type

-- ----------------------------------------------------------------------------
-- ISSUE #2: Remove Duplicate Columns from eval_items
-- ----------------------------------------------------------------------------
-- Removed: file_name, storage_key (duplicates of images_2d columns)
-- Impact: Queries must now JOIN with images_2d to get file metadata
-- Benefit: Single source of truth, no data inconsistency risk

-- ----------------------------------------------------------------------------
-- ISSUE #11: Experiment Tags Normalization
-- ----------------------------------------------------------------------------
-- Created: experiment_tags (tag definitions)
-- Created: experiment_tag_assignments (many-to-many mapping)
-- Removed: experiments.tags JSONB column
-- Benefit: Better search performance, tag management, autocomplete support

-- ============================================================================

CREATE TABLE alembic_version (
  version_num varchar(32) PRIMARY KEY
);

INSERT INTO alembic_version (version_num) VALUES ('restore_annotations');

COMMIT;
