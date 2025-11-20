/**
 * API Client for FastAPI Backend
 * 새로운 백엔드 API: /home/jaehyun/army/army_backend/backend
 */

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
const API_V1 = '/api/v1'

interface APIResponse<T> {
  data?: T
  error?: string
  message?: string
}

class APIClient {
  private baseURL: string

  constructor(baseURL: string = BACKEND_API_URL) {
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

  async register(data: { email: string; password: string; name?: string }) {
    return this.request(`${API_V1}/auth/register`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async login(data: { email: string; password: string }) {
    return this.request(`${API_V1}/auth/login-json`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  // ============================================
  // Dataset CRUD Endpoints (datasets-2d)
  // ============================================

  async getDatasets(skip = 0, limit = 100) {
    return this.request(`${API_V1}/datasets-2d?skip=${skip}&limit=${limit}`)
  }

  async getDataset(id: string) {
    return this.request(`${API_V1}/datasets-2d/${id}`)
  }

  async createDataset(data: any) {
    return this.request(`${API_V1}/datasets-2d/`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async updateDataset(id: string, data: any) {
    return this.request(`${API_V1}/datasets-2d/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    })
  }

  async deleteDataset(id: string) {
    return this.request(`${API_V1}/datasets-2d/${id}`, {
      method: 'DELETE',
    })
  }

  async getDatasetImages(datasetId: string, skip = 0, limit = 100) {
    return this.request(`${API_V1}/datasets-2d/${datasetId}/images?skip=${skip}&limit=${limit}`)
  }

  async deleteImage(imageId: string) {
    return this.request(`${API_V1}/datasets-2d/images/${imageId}`, {
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
    return this.request(`${API_V1}/dataset-service/upload-folder`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getDatasetStats(id: string) {
    return this.request(`${API_V1}/dataset-service/${id}/stats`)
  }

  async getDatasetDetectionStats(id: string, data: {
    detection_model_id: string  // Frontend parameter name for backward compatibility
    conf_threshold?: number
  }) {
    // Fixed: Backend expects 'model_id', not 'detection_model_id'
    return this.request(`${API_V1}/dataset-service/${id}/detection-stats`, {
      method: 'POST',
      body: JSON.stringify({
        model_id: data.detection_model_id,
        conf_threshold: data.conf_threshold,
      }),
    })
  }

  async deleteDatasetWithFiles(id: string) {
    return this.request(`${API_V1}/dataset-service/${id}`, {
      method: 'DELETE',
    })
  }

  // ============================================
  // Models Endpoints
  // ============================================

  async getModels(skip = 0, limit = 100) {
    return this.request(`${API_V1}/models?skip=${skip}&limit=${limit}`)
  }

  async getModel(id: string) {
    return this.request(`${API_V1}/models/${id}`)
  }

  async createModel(data: any) {
    return this.request(`${API_V1}/models/`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  // ============================================
  // Adversarial Patch Endpoints
  // ============================================

  async generateAdversarialPatch(data: {
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
  }) {
    // Fixed: Use correct backend endpoint /api/v1/patches/generate
    return this.request(`${API_V1}/patches/generate`, {
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

  async getPatch(patchId: string) {
    // Fixed: Use correct backend endpoint /api/v1/patches/{id}
    return this.request(`${API_V1}/patches/${patchId}`)
  }

  async getPatchImage(patchId: string) {
    // Fixed: Use correct backend endpoint
    const url = `${this.baseURL}${API_V1}/patches/${patchId}/image`
    return url // 이미지 URL 반환
  }

  async downloadPatch(patchId: string): Promise<void> {
    // Fixed: Use correct backend endpoint
    const url = `${this.baseURL}${API_V1}/patches/${patchId}/download`
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

  async listPatches(skip = 0, limit = 100, targetClass?: string) {
    // Fixed: Use correct backend endpoint
    let endpoint = `${API_V1}/patches?skip=${skip}&limit=${limit}`
    if (targetClass) {
      endpoint += `&target_class=${targetClass}`
    }
    return this.request(endpoint)
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
    // Fixed: Use correct backend endpoint /api/v1/attack-datasets/patch
    return this.request(`${API_V1}/attack-datasets/patch`, {
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

  async getAttackDataset(attackId: string) {
    return this.request(`${API_V1}/attack-datasets/${attackId}`)
  }

  async downloadAttackDataset(attackId: string): Promise<void> {
    // Fixed: Use correct backend endpoint
    const url = `${this.baseURL}${API_V1}/attack-datasets/${attackId}/download`
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

  async listAttackDatasets(skip = 0, limit = 100, attackType?: string) {
    let endpoint = `${API_V1}/attack-datasets?skip=${skip}&limit=${limit}`
    if (attackType) {
      endpoint += `&attack_type=${attackType}`
    }
    return this.request(endpoint)
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
    // Fixed: Use unified noise attack endpoint with attack_method parameter
    return this.request(`${API_V1}/attack-datasets/noise`, {
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
    // Fixed: Use unified noise attack endpoint with attack_method parameter
    return this.request(`${API_V1}/attack-datasets/noise`, {
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

  async createEvaluationRun(data: any) {
    // Remove null/undefined values to avoid database constraint violations
    const cleanData = Object.fromEntries(
      Object.entries(data).filter(([_, v]) => v != null)
    )
    return this.request(`${API_V1}/evaluation/runs`, {
      method: 'POST',
      body: JSON.stringify(cleanData),
    })
  }

  async getEvaluationRun(runId: string) {
    return this.request(`${API_V1}/evaluation/runs/${runId}`)
  }

  async listEvaluationRuns(params: {
    page?: number
    page_size?: number
    phase?: string
    status?: string
    model_id?: string
    base_dataset_id?: string
    attack_dataset_id?: string
  }) {
    const queryParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        queryParams.append(key, String(value))
      }
    })
    return this.request(`${API_V1}/evaluation/runs?${queryParams.toString()}`)
  }

  async updateEvaluationRun(runId: string, data: any) {
    return this.request(`${API_V1}/evaluation/runs/${runId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    })
  }

  async deleteEvaluationRun(runId: string) {
    return this.request(`${API_V1}/evaluation/runs/${runId}`, {
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

    return this.request(`${API_V1}/evaluation/runs/${runId}/execute?${queryParams.toString()}`, {
      method: 'POST',
      body: JSON.stringify(body),
    })
  }

  async compareRobustness(data: {
    clean_run_id: string
    adv_run_id: string
  }) {
    return this.request(`${API_V1}/evaluation/runs/compare-robustness`, {
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
    return this.request(`${API_V1}/evaluation/runs/${runId}/items?${params.toString()}`)
  }

  async getEvaluationPRCurveData(runId: string) {
    return this.request(`${API_V1}/evaluation/runs/${runId}/pr-curve-data`)
  }

  async getEvaluationImagesWithPredictions(runId: string, page = 1, pageSize = 20, datasetType?: 'base' | 'attack') {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
    })
    if (datasetType) {
      params.append('dataset_type', datasetType)
    }
    return this.request(`${API_V1}/evaluation/runs/${runId}/images-with-predictions?${params.toString()}`)
  }

  async compareEvaluationImages(cleanRunId: string, advRunId: string, page = 1, pageSize = 20) {
    return this.request(`${API_V1}/evaluation/runs/compare-images?clean_run_id=${cleanRunId}&adv_run_id=${advRunId}&page=${page}&page_size=${pageSize}`)
  }

  // ============================================
  // Experiments Endpoints
  // ============================================

  async createExperiment(data: any) {
    return this.request(`${API_V1}/experiments/`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getExperiment(id: string) {
    return this.request(`${API_V1}/experiments/${id}`)
  }

  async listExperiments(skip = 0, limit = 100) {
    return this.request(`${API_V1}/experiments?skip=${skip}&limit=${limit}`)
  }

  async updateExperiment(id: string, data: any) {
    return this.request(`${API_V1}/experiments/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    })
  }

  async deleteExperiment(id: string) {
    return this.request(`${API_V1}/experiments/${id}`, {
      method: 'DELETE',
    })
  }

  // ============================================
  // Experiment Tags Endpoints (PHASE 2 NEW)
  // ============================================

  async listTags(options?: { skip?: number; limit?: number; search?: string }) {
    let endpoint = `${API_V1}/tags`
    const params = new URLSearchParams()

    if (options?.skip !== undefined) params.append('skip', options.skip.toString())
    if (options?.limit !== undefined) params.append('limit', options.limit.toString())
    if (options?.search) params.append('search', options.search)

    if (params.toString()) endpoint += `?${params.toString()}`

    return this.request(endpoint)
  }

  async createTag(data: { name: string; color?: string; description?: string }) {
    return this.request(`${API_V1}/tags`, {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  async getTag(id: string) {
    return this.request(`${API_V1}/tags/${id}`)
  }

  async updateTag(id: string, data: { name?: string; color?: string; description?: string }) {
    return this.request(`${API_V1}/tags/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
  }

  async deleteTag(id: string) {
    return this.request(`${API_V1}/tags/${id}`, {
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
  ) {
    let endpoint = `${API_V1}/annotations/image/${imageId}`
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

    return this.request(endpoint)
  }

  async getDatasetAnnotationsSummary(
    datasetId: string,
    minConfidence?: number
  ) {
    let endpoint = `${API_V1}/annotations/dataset/${datasetId}`
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
    return this.request(`${API_V1}/annotations/bulk?image_id=${imageId}`, {
      method: 'POST',
      body: JSON.stringify(annotations),
    })
  }

  async deleteImageAnnotations(
    imageId: string,
    annotationType?: 'bbox' | 'polygon' | 'keypoint' | 'segmentation'
  ) {
    let endpoint = `${API_V1}/annotations/image/${imageId}`
    if (annotationType) {
      endpoint += `?annotation_type=${annotationType}`
    }
    return this.request(endpoint, {
      method: 'DELETE',
    })
  }

  // ============================================
  // Storage Endpoints
  // ============================================

  async getStorageInfo() {
    return this.request(`${API_V1}/storage/info`)
  }

  async listStorageFiles(path?: string) {
    const endpoint = path
      ? `${API_V1}/storage/list?path=${encodeURIComponent(path)}`
      : `${API_V1}/storage/list`
    return this.request(endpoint)
  }
}

// Export singleton instance
export const apiClient = new APIClient()

// Export class for custom instances
export default APIClient
