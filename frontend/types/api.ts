/**
 * API Response Types
 * 백엔드 API 응답 타입 정의
 */

// ============================================
// Common Types
// ============================================

export interface ApiResponse<T> {
  data?: T
  error?: string
  message?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

// ============================================
// User & Auth Types
// ============================================

export interface User {
  id: string
  username: string
  email: string
  name?: string
  rank?: string
  unit?: string
  role: 'admin' | 'user' | 'viewer'
  created_at: string
  updated_at: string
  last_login_at?: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface LoginResponse {
  success: boolean
  token: string
  user: User
}

export interface RegisterRequest {
  email: string
  password: string
  name?: string
}

// ============================================
// Model Types
// ============================================

export interface Model {
  id: string
  name: string
  model_type: string
  version: string
  framework: 'pytorch' | 'tensorflow' | 'onnx' | 'tensorrt'
  task: 'detection' | 'classification' | 'segmentation'
  file_path?: string
  config?: Record<string, any>
  metadata?: {
    input_size?: number[]
    num_classes?: number
    class_names?: string[]
    [key: string]: any
  }
  created_at: string
  updated_at: string
}

// ============================================
// Dataset Types
// ============================================

export interface Dataset {
  id: string
  name: string
  description?: string
  type: '2D_IMAGE' | '3D_POINT_CLOUD'
  source?: 'CUSTOM' | 'COCO' | 'VOC' | 'IMAGENET'
  size: number
  image_count?: number
  storage_path?: string
  storage_location?: string
  metadata?: {
    total_size_bytes?: number
    model?: string
    class_distribution?: Record<string, number>
    [key: string]: any
  }
  created_at: string
  updated_at: string
}

export interface ImageFile {
  id: string
  dataset_id: string
  filename: string
  file_name?: string
  data?: string // Base64 encoded
  width?: number
  height?: number
  format: string
  mime_type?: string
  mimeType?: string
  storage_key?: string
  metadata?: any
  created_at: string
  detections?: ImageDetection[]
  visualization?: string
}

export interface ImageDetection {
  bbox: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
  class: string
  class_id: number
  confidence: number
  isGroundTruth?: boolean
}

export interface DatasetStats {
  total_images: number
  total_annotations: number
  class_distribution: Record<string, number>
  avg_annotations_per_image: number
}

// ============================================
// Attack Dataset Types
// ============================================

export interface AttackDataset {
  id: string
  name: string
  attack_type: 'patch' | 'noise' | 'fgsm' | 'pgd'
  base_dataset_id: string
  model_id?: string
  patch_id?: string
  description?: string
  metadata?: {
    epsilon?: number
    iterations?: number
    patch_scale?: number
    [key: string]: any
  }
  created_at: string
  updated_at: string
  is_attack_dataset?: boolean
}

// ============================================
// Adversarial Patch Types
// ============================================

export interface AdversarialPatch {
  id: string
  patch_name: string
  attack_method: string
  model_id: string
  source_dataset_id: string
  target_class: string
  patch_size: number
  learning_rate?: number
  iterations?: number
  status: 'pending' | 'running' | 'completed' | 'failed'
  patch_path?: string
  metadata?: Record<string, any>
  created_at: string
  updated_at: string
}

export interface GeneratePatchRequest {
  patch_name: string
  model_id: string
  dataset_id: string
  target_class: string
  plugin_name?: string
  patch_size?: number
  area_ratio?: number
  epsilon?: number
  alpha?: number
  iterations?: number
  batch_size?: number
  description?: string
  created_by?: string
  session_id?: string
}

// ============================================
// Evaluation Types
// ============================================

export type EvaluationPhase = 'clean' | 'adversarial'
export type EvaluationStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface EvaluationRun {
  id: string
  run_name?: string
  phase: EvaluationPhase
  model_id: string
  base_dataset_id: string
  attack_dataset_id?: string
  status: EvaluationStatus
  conf_threshold?: number
  iou_threshold?: number
  metrics?: {
    map?: number
    map_50?: number
    map_75?: number
    precision?: number
    recall?: number
    f1_score?: number
    [key: string]: number | undefined
  }
  class_metrics?: Record<string, any>
  pr_curve_data?: any
  created_at: string
  updated_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
}

export interface CreateEvaluationRunRequest {
  run_name?: string
  phase: EvaluationPhase
  model_id: string
  base_dataset_id: string
  attack_dataset_id?: string
  conf_threshold?: number
  iou_threshold?: number
  description?: string
  created_by?: string
}

export interface EvaluationItem {
  id: string
  run_id: string
  image_id: string
  dataset_type: 'base' | 'attack'
  predictions: any[]
  ground_truth: any[]
  metrics: Record<string, number>
}

export interface CompareRobustnessRequest {
  clean_run_id: string
  adv_run_id: string
}

export interface CompareRobustnessResponse {
  clean_metrics: Record<string, number>
  adversarial_metrics: Record<string, number>
  robustness_drop: Record<string, number>
  summary: {
    overall_robustness: number
    map_drop: number
    precision_drop: number
    recall_drop: number
  }
}

// ============================================
// Annotation Types
// ============================================

export interface Annotation {
  id: string
  image_id: string
  annotation_type: 'bbox' | 'polygon' | 'keypoint' | 'segmentation'
  class_name: string
  class_index?: number
  bbox_x?: number
  bbox_y?: number
  bbox_width?: number
  bbox_height?: number
  x1?: number
  y1?: number
  x2?: number
  y2?: number
  polygon_data?: any[]
  keypoints?: any[]
  confidence?: number
  is_crowd?: boolean
  is_ground_truth?: boolean
  metadata?: Record<string, any>
  created_at: string
}

export interface CreateAnnotationRequest {
  annotation_type: 'bbox' | 'polygon' | 'keypoint' | 'segmentation'
  class_name: string
  class_index?: number
  bbox_x?: number
  bbox_y?: number
  bbox_width?: number
  bbox_height?: number
  polygon_data?: any[]
  keypoints?: any[]
  confidence?: number
  is_crowd?: boolean
  metadata?: Record<string, any>
}

// ============================================
// Experiment & Tag Types
// ============================================

export interface Experiment {
  id: string
  name: string
  description?: string
  status: 'active' | 'completed' | 'archived'
  tags?: string[]
  metadata?: Record<string, any>
  created_at: string
  updated_at: string
}

export interface Tag {
  id: string
  name: string
  color?: string
  description?: string
  created_at: string
  updated_at: string
}

// ============================================
// Storage Types
// ============================================

export interface StorageInfo {
  total_size: number
  used_size: number
  available_size: number
  datasets_count: number
  models_count: number
}

export interface StorageFile {
  name: string
  path: string
  size: number
  type: 'file' | 'directory'
  modified_at: string
}
