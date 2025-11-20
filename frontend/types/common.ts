/**
 * 공통 타입 정의
 */

// 사용자 관련 타입
export interface User {
  id: string
  name: string
  email: string
  role: UserRole
  department?: string
  rank?: string
  status: UserStatus
  createdAt: Date
  updatedAt?: Date
  lastLogin?: Date
}

export type UserRole = 'admin' | 'researcher' | 'operator' | 'viewer'
export type UserStatus = 'active' | 'inactive' | 'suspended'

// 데이터셋 관련 타입
export interface Dataset {
  id: string
  name: string
  description?: string
  size: number
  fileCount: number
  type: DatasetType
  status: DatasetStatus
  createdAt: Date
  updatedAt?: Date
  createdBy: string
  tags?: string[]
}

export type DatasetType = '2d-image' | '3d-model' | 'video' | 'mixed'
export type DatasetStatus = 'pending' | 'processing' | 'ready' | 'error'

// 모델 관련 타입
export interface AIModel {
  id: string
  name: string
  version: string
  type: ModelType
  architecture: string
  accuracy?: number
  performance?: ModelPerformance
  status: ModelStatus
  createdAt: Date
  trainedAt?: Date
  evaluatedAt?: Date
}

export type ModelType = 'detection' | 'classification' | 'segmentation' | 'tracking'
export type ModelStatus = 'training' | 'trained' | 'evaluating' | 'deployed' | 'archived'

export interface ModelPerformance {
  accuracy: number
  precision: number
  recall: number
  f1Score: number
  latency?: number
  throughput?: number
}

// 평가 관련 타입
export interface Evaluation {
  id: string
  modelId: string
  datasetId: string
  type: EvaluationType
  status: EvaluationStatus
  results?: EvaluationResults
  startedAt: Date
  completedAt?: Date
  performedBy: string
}

export type EvaluationType = 'reliability' | 'adversarial' | 'performance' | 'comprehensive'
export type EvaluationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface EvaluationResults {
  overallScore: number
  metrics: Record<string, number>
  details?: any
  recommendations?: string[]
}

// 적대적 공격 관련 타입
export interface AdversarialAttack {
  id: string
  name: string
  type: AttackType
  targetModel: string
  parameters: AttackParameters
  status: AttackStatus
  successRate?: number
  createdAt: Date
}

export type AttackType = 'fgsm' | 'pgd' | 'carlini-wagner' | 'patch' | 'universal' | 'universal_noise' | 'noise_osfd'
export type AttackStatus = 'configuring' | 'generating' | 'testing' | 'completed'

export interface AttackParameters {
  epsilon?: number
  iterations?: number
  stepSize?: number
  confidence?: number
  targetClass?: string
  [key: string]: any
}

// 작업 관련 타입
export interface Task {
  id: string
  name: string
  type: TaskType
  status: TaskStatus
  progress: number
  priority: TaskPriority
  assignedTo?: string
  startedAt?: Date
  completedAt?: Date
  estimatedTime?: number
  result?: any
}

export type TaskType = 'training' | 'evaluation' | 'generation' | 'analysis' | 'export'
export type TaskStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
export type TaskPriority = 'low' | 'medium' | 'high' | 'critical'

// 시스템 상태 관련 타입
export interface SystemStatus {
  status: 'normal' | 'warning' | 'error' | 'maintenance'
  cpu: number
  memory: number
  gpu?: number
  storage: StorageInfo
  activeUsers: number
  runningTasks: number
  timestamp: Date
}

export interface StorageInfo {
  used: number
  total: number
  percentage: number
}

// API 응답 타입
export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: ApiError
  message?: string
  timestamp: Date
}

export interface ApiError {
  code: string
  message: string
  details?: any
}

// 페이지네이션 타입
export interface PaginationParams {
  page: number
  pageSize: number
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  totalPages: number
  hasNext: boolean
  hasPrevious: boolean
}

// 필터 타입
export interface FilterOptions {
  search?: string
  status?: string[]
  type?: string[]
  dateFrom?: Date
  dateTo?: Date
  tags?: string[]
  [key: string]: any
}

// 파일 업로드 타입
export interface FileUpload {
  file: File
  progress: number
  status: 'pending' | 'uploading' | 'completed' | 'error'
  error?: string
}

// 어노테이션 관련 타입
export type AnnotationType = 'bbox' | 'polygon' | 'keypoint' | 'segmentation'

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Annotation {
  id: string
  image_2d_id?: string
  image_3d_id?: string
  rt_frame_id?: string
  annotation_type: AnnotationType
  class_name: string
  class_index?: number

  // Bounding box fields (normalized)
  bbox_x?: number
  bbox_y?: number
  bbox_width?: number
  bbox_height?: number

  // Bounding box fields (pixel)
  x1?: number
  y1?: number
  x2?: number
  y2?: number

  // Polygon data
  polygon_data?: any

  // Keypoints
  keypoints?: any

  confidence?: number
  is_crowd?: boolean
  area?: number
  metadata?: Record<string, any>
  created_at: string
}

export interface AnnotationCreate {
  annotation_type: AnnotationType
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

export interface DatasetAnnotationSummary {
  dataset_id: string
  dataset_name: string
  total_images: number
  total_annotations: number
  images_with_annotations: number
  images_without_annotations: number
  avg_annotations_per_image: number
  class_distribution: Record<string, number>
  avg_confidence_per_class: Record<string, number>
  unique_classes: number
}