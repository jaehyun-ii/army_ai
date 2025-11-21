/**
 * API Client for FastAPI Backend (Proxy Pattern)
 * All requests go through Next.js API routes (/api/*)
 */

import type {
  User,
  Model,
  Dataset,
  ImageFile,
  DatasetStats,
  AttackDataset,
  AdversarialPatch,
  GeneratePatchRequest,
  EvaluationRun,
  CreateEvaluationRunRequest,
  EvaluationItem,
  CompareRobustnessRequest,
  CompareRobustnessResponse,
  Annotation,
  CreateAnnotationRequest,
  Experiment,
  Tag,
  StorageInfo,
  StorageFile,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
} from '@/types/api'

interface APIResponse<T> {
  data?: T
  error?: string
  message?: string
}

class APIClient {
  private baseURL: string

  constructor(baseURL: string = '') {
    // Use empty baseURL to make relative requests to Next.js API routes
    this.baseURL = baseURL
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`

    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      credentials: 'include',
    }

    try {
      const response = await fetch(url, config)

      // 204 No Content는 빈 응답 반환
      if (response.status === 204) {
        return {} as T
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        // Handle both string and object detail formats
        let errorMessage = errorData.detail || `HTTP ${response.status}: ${response.statusText}`

        // If detail is an array (validation errors), format it
        if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map((err: any) =>
            `${err.loc?.join('.')} - ${err.msg}`
          ).join(', ')
        } else if (typeof errorData.detail === 'object') {
          errorMessage = JSON.stringify(errorData.detail)
        }

        throw new Error(errorMessage)
      }

      const contentType = response.headers.get('content-type')
      if (contentType && contentType.includes('application/json')) {
        return await response.json()
      }

      return {} as T
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error)
      throw error
    }
  }

  // ============================================
  // Authentication Endpoints
  // ============================================

  async register(data: RegisterRequest): Promise<User> {
    return this.request<User>(`/api/auth/register`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async login(data: LoginRequest): Promise<LoginResponse> {
    return this.request<LoginResponse>(`/api/auth/login`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  // ============================================
  // Dataset CRUD Endpoints (datasets-2d)
  // ============================================

  async getDatasets(skip = 0, limit = 100): Promise<Dataset[]> {
    return this.request<Dataset[]>(`/api/datasets?skip=${skip}&limit=${limit}`)
  }

  async getDataset(id: string): Promise<Dataset> {
    return this.request<Dataset>(`/api/datasets/${id}`)
  }

  async createDataset(data: Partial<Dataset>): Promise<Dataset> {
    return this.request<Dataset>(`/api/datasets`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async updateDataset(id: string, data: Partial<Dataset>): Promise<Dataset> {
    return this.request<Dataset>(`/api/datasets/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    })
  }

  async deleteDataset(id: string): Promise<void> {
    return this.request<void>(`/api/datasets/${id}`, {
      method: 'DELETE',
    })
  }

  async getDatasetImages(datasetId: string, skip = 0, limit = 100): Promise<{ images: ImageFile[], total: number }> {
    return this.request<{ images: ImageFile[], total: number }>(`/api/datasets/${datasetId}/images?skip=${skip}&limit=${limit}`)
  }

  async deleteImage(imageId: string): Promise<void> {
    return this.request<void>(`/api/datasets/images/${imageId}`, {
      method: 'DELETE',
    })
  }

  // ============================================
  // Dataset Service Endpoints (upload, stats)
  // ============================================

  async uploadDatasetFolder(data: {
    source_folder: string
    dataset_name: string
    description?: string
    owner_id?: string
    inference_metadata_path?: string
  }) {
    // Note: This endpoint uses FormData in the actual route
    // Keeping JSON format here for backward compatibility
    return this.request(`/api/datasets/upload-folder`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getDatasetStats(id: string) {
    // TODO: Create /api/datasets/[id]/stats proxy route
    return this.request(`/api/datasets/${id}/stats`)
  }

  async getDatasetDetectionStats(id: string, data: {
    detection_model_id: string  // Frontend parameter name for backward compatibility
    conf_threshold?: number
  }) {
    // TODO: Create /api/datasets/[id]/detection-stats proxy route
    return this.request(`/api/datasets/${id}/detection-stats`, {
      method: 'POST',
      body: JSON.stringify({
        model_id: data.detection_model_id,
        conf_threshold: data.conf_threshold,
      }),
    })
  }

  async deleteDatasetWithFiles(id: string) {
    // TODO: Create /api/datasets/[id] DELETE or use existing route
    return this.request(`/api/datasets/${id}`, {
      method: 'DELETE',
    })
  }

  // ============================================
  // Models Endpoints
  // ============================================

  async getModels(skip = 0, limit = 100): Promise<Model[]> {
    return this.request<Model[]>(`/api/models?skip=${skip}&limit=${limit}`)
  }

  async getModel(id: string): Promise<Model> {
    return this.request<Model>(`/api/models/${id}`)
  }

  async createModel(data: Partial<Model>): Promise<Model> {
    return this.request<Model>(`/api/models`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  // ============================================
  // Adversarial Patch Endpoints
  // ============================================

  async generateAdversarialPatch(data: GeneratePatchRequest): Promise<AdversarialPatch> {
    // Use existing /api/adversarial-patch/generate proxy route
    return this.request<AdversarialPatch>(`/api/adversarial-patch/generate`, {
      method: 'POST',
      body: JSON.stringify({
        patch_name: data.patch_name,
        attack_method: data.plugin_name || 'robust_dpatch',
        source_dataset_id: data.dataset_id,
        model_id: data.model_id,
        target_class: data.target_class,
        patch_size: data.patch_size ?? 100,
        learning_rate: 5.0,
        iterations: data.iterations ?? 100,
        session_id: data.session_id,
      }),
    })
  }

  async getPatch(patchId: string): Promise<AdversarialPatch> {
    // Use existing /api/adversarial-patches proxy route
    return this.request<AdversarialPatch>(`/api/adversarial-patches/${patchId}`)
  }

  async getPatchImage(patchId: string): string {
    // Return relative URL for proxy pattern
    return `/api/adversarial-patches/${patchId}/image`
  }

  async downloadPatch(patchId: string): Promise<void> {
    // Use existing /api/adversarial-patches/[id]/download proxy route
    const url = `/api/adversarial-patches/${patchId}/download`
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = `patch_${patchId}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  }

  async listPatches(skip = 0, limit = 100, targetClass?: string): Promise<AdversarialPatch[]> {
    // Use existing /api/adversarial-patches proxy route
    let endpoint = `/api/adversarial-patches?skip=${skip}&limit=${limit}`
    if (targetClass) {
      endpoint += `&target_class=${targetClass}`
    }
    return this.request<AdversarialPatch[]>(endpoint)
  }

  async generateAttackDataset(data: {
    attack_dataset_name: string
    model_id: string
    base_dataset_id: string
    patch_id: string
    target_class: string
    patch_scale?: number
    description?: string
    created_by?: string
  }) {
    // TODO: Create /api/attack-datasets/patch proxy route
    return this.request(`/api/attack-datasets/patch`, {
      method: 'POST',
      body: JSON.stringify({
        attack_name: data.attack_dataset_name,
        patch_id: data.patch_id,
        base_dataset_id: data.base_dataset_id,
        patch_scale: (data.patch_scale ?? 0.3) * 100, // Convert ratio to percentage
        session_id: data.created_by, // Note: using created_by as session_id if needed
      }),
    })
  }

  async getAttackDataset(attackId: string): Promise<AttackDataset> {
    return this.request<AttackDataset>(`/api/attack-datasets/${attackId}`)
  }

  async downloadAttackDataset(attackId: string): Promise<void> {
    // TODO: Create /api/attack-datasets/[id]/download proxy route
    const url = `/api/attack-datasets/${attackId}/download`
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = `attack_dataset_${attackId}.zip`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(downloadUrl)
  }

  async listAttackDatasets(skip = 0, limit = 100, attackType?: string): Promise<AttackDataset[]> {
    let endpoint = `/api/attack-datasets?skip=${skip}&limit=${limit}`
    if (attackType) {
      endpoint += `&attack_type=${attackType}`
    }
    return this.request<AttackDataset[]>(endpoint)
  }

  // ============================================
  // Noise Attack Endpoints
  // ============================================

  async generateFGSMAttack(data: {
    attack_dataset_name: string
    detection_model_id: string
    base_dataset_id: string
    epsilon?: number
    targeted?: boolean
    target_class?: string
    description?: string
    created_by?: string
    session_id?: string
  }) {
    // TODO: Create /api/attack-datasets/noise proxy route
    return this.request(`/api/attack-datasets/noise`, {
      method: 'POST',
      body: JSON.stringify({
        attack_name: data.attack_dataset_name,
        attack_method: 'fgsm',
        base_dataset_id: data.base_dataset_id,
        model_id: data.detection_model_id,
        epsilon: data.epsilon ?? 8.0,
        session_id: data.session_id,
      }),
    })
  }

  async generatePGDAttack(data: {
    attack_dataset_name: string
    detection_model_id: string
    base_dataset_id: string
    epsilon?: number
    alpha?: number
    iterations?: number
    targeted?: boolean
    target_class?: string
    description?: string
    created_by?: string
    session_id?: string
  }) {
    // TODO: Create /api/attack-datasets/noise proxy route
    return this.request(`/api/attack-datasets/noise`, {
      method: 'POST',
      body: JSON.stringify({
        attack_name: data.attack_dataset_name,
        attack_method: 'pgd',
        base_dataset_id: data.base_dataset_id,
        model_id: data.detection_model_id,
        epsilon: data.epsilon ?? 8.0,
        alpha: data.alpha ?? 2.0,
        iterations: data.iterations ?? 10,
        session_id: data.session_id,
      }),
    })
  }

  // Removed: Gaussian, Uniform, and Iterative Gradient noise attacks
  // These are not supported by the backend and won't be implemented

  // ============================================
  // Evaluation Endpoints
  // ============================================

  async createEvaluationRun(data: CreateEvaluationRunRequest): Promise<EvaluationRun> {
    // Remove null/undefined values to avoid database constraint violations
    const cleanData = Object.fromEntries(
      Object.entries(data).filter(([_, v]) => v != null)
    )
    return this.request<EvaluationRun>(`/api/evaluations`, {
      method: 'POST',
      body: JSON.stringify(cleanData),
    })
  }

  async getEvaluationRun(runId: string): Promise<EvaluationRun> {
    return this.request<EvaluationRun>(`/api/evaluations/${runId}`)
  }

  async listEvaluationRuns(params: {
    page?: number
    page_size?: number
    phase?: string
    status?: string
    model_id?: string
    base_dataset_id?: string
    attack_dataset_id?: string
  }): Promise<EvaluationRun[]> {
    const queryParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, String(value))
      }
    })
    return this.request<EvaluationRun[]>(`/api/evaluations?${queryParams.toString()}`)
  }

  async updateEvaluationRun(runId: string, data: Partial<EvaluationRun>): Promise<EvaluationRun> {
    return this.request<EvaluationRun>(`/api/evaluations/${runId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    })
  }

  async deleteEvaluationRun(runId: string): Promise<void> {
    return this.request<void>(`/api/evaluations/${runId}`, {
      method: 'DELETE',
    })
  }

  async executeEvaluationRun(
    runId: string,
    params: {
      conf_threshold?: number
      iou_threshold?: number
      session_id?: string
    } = {}
  ) {
    const queryParams = new URLSearchParams()
    if (params.conf_threshold !== undefined) {
      queryParams.append('conf_threshold', String(params.conf_threshold))
    }
    if (params.iou_threshold !== undefined) {
      queryParams.append('iou_threshold', String(params.iou_threshold))
    }

    // Build request body - only include session_id if provided
    const body = params.session_id ? { session_id: params.session_id } : {}

    return this.request(`/api/evaluations/${runId}/execute?${queryParams.toString()}`, {
      method: 'POST',
      body: JSON.stringify(body),
    })
  }

  async compareRobustness(data: {
    clean_run_id: string
    adv_run_id: string
  }) {
    // TODO: Create /api/evaluations/compare-robustness proxy route
    return this.request(`/api/evaluations/compare-robustness`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getEvaluationItems(runId: string, page = 1, pageSize = 100, datasetType?: 'base' | 'attack') {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
    })
    if (datasetType) {
      params.append('dataset_type', datasetType)
    }
    // TODO: Create /api/evaluations/[id]/items proxy route
    return this.request(`/api/evaluations/${runId}/items?${params.toString()}`)
  }

  async getEvaluationPRCurveData(runId: string) {
    return this.request(`/api/evaluations/${runId}/pr-curve-data`)
  }

  async getEvaluationImagesWithPredictions(runId: string, page = 1, pageSize = 20, datasetType?: 'base' | 'attack') {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
    })
    if (datasetType) {
      params.append('dataset_type', datasetType)
    }
    // TODO: Create /api/evaluations/[id]/images-with-predictions proxy route
    return this.request(`/api/evaluations/${runId}/images-with-predictions?${params.toString()}`)
  }

  async compareEvaluationImages(cleanRunId: string, advRunId: string, page = 1, pageSize = 20) {
    // TODO: Create /api/evaluations/compare-images proxy route
    return this.request(`/api/evaluations/compare-images?clean_run_id=${cleanRunId}&adv_run_id=${advRunId}&page=${page}&page_size=${pageSize}`)
  }

  // ============================================
  // Experiments Endpoints
  // ============================================

  async createExperiment(data: any) {
    // TODO: Create /api/experiments proxy route
    return this.request(`/api/experiments`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getExperiment(id: string) {
    // TODO: Create /api/experiments/[id] proxy route
    return this.request(`/api/experiments/${id}`)
  }

  async listExperiments(skip = 0, limit = 100) {
    // TODO: Create /api/experiments proxy route
    return this.request(`/api/experiments?skip=${skip}&limit=${limit}`)
  }

  async updateExperiment(id: string, data: any) {
    // TODO: Create /api/experiments/[id] proxy route
    return this.request(`/api/experiments/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    })
  }

  async deleteExperiment(id: string) {
    // TODO: Create /api/experiments/[id] proxy route
    return this.request(`/api/experiments/${id}`, {
      method: 'DELETE',
    })
  }

  // ============================================
  // Experiment Tags Endpoints (PHASE 2 NEW)
  // ============================================

  async listTags(options?: { skip?: number; limit?: number; search?: string }) {
    let endpoint = `/api/tags`
    const params = new URLSearchParams()

    if (options?.skip !== undefined) params.append('skip', options.skip.toString())
    if (options?.limit !== undefined) params.append('limit', options.limit.toString())
    if (options?.search) params.append('search', options.search)

    if (params.toString()) endpoint += `?${params.toString()}`

    // TODO: Create /api/tags proxy route
    return this.request(endpoint)
  }

  async createTag(data: { name: string; color?: string; description?: string }) {
    // TODO: Create /api/tags proxy route
    return this.request(`/api/tags`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getTag(id: string) {
    // TODO: Create /api/tags/[id] proxy route
    return this.request(`/api/tags/${id}`)
  }

  async updateTag(id: string, data: { name?: string; color?: string; description?: string }) {
    // TODO: Create /api/tags/[id] proxy route
    return this.request(`/api/tags/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
  }

  async deleteTag(id: string) {
    // TODO: Create /api/tags/[id] proxy route
    return this.request(`/api/tags/${id}`, {
      method: 'DELETE',
    })
  }

  // ============================================
  // Annotations Endpoints
  // ============================================

  async getImageAnnotations(
    imageId: string,
    options?: {
      annotation_type?: 'bbox' | 'polygon' | 'keypoint' | 'segmentation'
      min_confidence?: number
    }
  ): Promise<Annotation[]> {
    let endpoint = `/api/annotations/image/${imageId}`
    const params = new URLSearchParams()

    if (options?.annotation_type) {
      params.append('annotation_type', options.annotation_type)
    }
    if (options?.min_confidence !== undefined) {
      params.append('min_confidence', options.min_confidence.toString())
    }

    if (params.toString()) {
      endpoint += `?${params.toString()}`
    }

    return this.request<Annotation[]>(endpoint)
  }

  async getDatasetAnnotationsSummary(
    datasetId: string,
    minConfidence?: number
  ) {
    let endpoint = `/api/annotations/dataset/${datasetId}`
    if (minConfidence !== undefined) {
      endpoint += `?min_confidence=${minConfidence}`
    }
    return this.request(endpoint)
  }

  async createAnnotationsBulk(
    imageId: string,
    annotations: Array<{
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
    }>
  ) {
    // TODO: Create /api/annotations/bulk proxy route
    return this.request(`/api/annotations/bulk?image_id=${imageId}`, {
      method: 'POST',
      body: JSON.stringify(annotations),
    })
  }

  async deleteImageAnnotations(
    imageId: string,
    annotationType?: 'bbox' | 'polygon' | 'keypoint' | 'segmentation'
  ) {
    let endpoint = `/api/annotations/image/${imageId}`
    if (annotationType) {
      endpoint += `?annotation_type=${annotationType}`
    }
    // Use existing /api/annotations/image/[id] proxy route
    return this.request(endpoint, {
      method: 'DELETE',
    })
  }

  // ============================================
  // Storage Endpoints
  // ============================================

  async getStorageInfo() {
    // TODO: Create /api/storage/info proxy route
    return this.request(`/api/storage/info`)
  }

  async listStorageFiles(path?: string) {
    const endpoint = path
      ? `/api/storage/list?path=${encodeURIComponent(path)}`
      : `/api/storage/list`
    // TODO: Create /api/storage/list proxy route
    return this.request(endpoint)
  }
}

// Export singleton instance
export const apiClient = new APIClient()

// Export class for custom instances
export default APIClient
