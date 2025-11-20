/**
 * Adversarial API Client
 * 새로운 FastAPI 백엔드: /home/jaehyun/army/army_backend/backend
 */

import { getBackendUrl } from './backend-url'

const BACKEND_API_URL = getBackendUrl()
const API_V1 = '/api/v1'

// ============================================
// Type Definitions
// ============================================

export interface BackendDataset {
  id: string
  name: string
  type?: string
  description?: string
  created_at: string
  metadata?: any
}

export interface YoloModel {
  id: string
  name: string
  version?: string
  framework?: string
  path?: string
}

export interface TrainingConfig {
  patch_name: string
  dataset_id: string
  target_class: string
  model_path: string // model_id로 사용
  attack_method?: "patch" | "dpatch" | "robust_dpatch" // Added: attack method selection
  iterations: number
  patch_size: number
  learning_rate?: number // Added: learning rate parameter
  session_id: string  // Added: Pre-generated session ID for SSE connection
}

export interface TrainingResponse {
  training_id: string | number
  session_id: string  // Added: SSE session ID for progress tracking
  status: string
  message: string
  sse_url?: string
}

export interface TrainingResult {
  id: number | string
  patch_name?: string
  dataset_name?: string
  target_class: string
  model_path: string
  iterations: number
  status: string
  best_score?: number
  patch_file_path?: string
  created_at: string
}

export interface TrainingLog {
  type: 'status' | 'info' | 'progress' | 'complete' | 'error' | 'success' | 'warning'
  message: string
  iteration?: number
  total_iterations?: number
  avg_loss?: number
  detected_count?: number
  total_count?: number
  training_id?: number | string
  patch_path?: string
  patch_file?: string
  best_score?: number
}

export interface AdversarialDataConfig {
  dataset_name: string
  source_dataset_id: string
  training_id?: string | number
  attack_type: 'patch' | 'noise'
  patch_scale?: number
  noise_method?: "pgd" | "fgsm" // Added: noise attack method
  noise_epsilon?: number // Renamed from noise_intensity
  noise_alpha?: number // Added: PGD alpha parameter
  noise_iterations?: number // Added: PGD iterations parameter
  target_class?: string
  image_indices?: number[]
  model_id?: string
  session_id?: string  // Added for SSE support
}

export interface AdversarialDataResponse {
  generation_id: string
  output_dataset_id?: string | null
  message: string
}

export interface AdversarialDataLog {
  type: 'status' | 'info' | 'progress' | 'complete' | 'error' | 'warning' | 'success'
  message: string
  progress?: number
  processed?: number
  total?: number
  successful?: number
  failed?: number
  current_image?: string
  dataset_id?: string
  output_dir?: string
  attack_dataset_id?: string
  storage_path?: string
  processed_images?: number
  skipped_images?: number
  failed_images?: number
  avg_noise_magnitude?: number
}

// ============================================
// API Functions
// ============================================

export async function fetchBackendDatasets(): Promise<BackendDataset[]> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/datasets-2d`
    const response = await fetch(endpoint)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()
    return data || []
  } catch (error) {
    console.error('Failed to fetch backend datasets:', error)
    throw error
  }
}

export async function fetchYoloModels(): Promise<YoloModel[]> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/models/versions`
    const response = await fetch(endpoint)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()
    // 응답 형식 변환
    return data.map((model: any) => ({
      id: model.id,
      name: model.name || 'Unknown Model',
      version: model.version,
      framework: model.framework,
      path: model.id // ID를 path로 사용
    })) || []
  } catch (error) {
    console.error('Failed to fetch YOLO models:', error)
    throw error
  }
}

export async function startTraining(config: TrainingConfig): Promise<TrainingResponse> {
  try {
    // Updated endpoint to match backend test files
    const endpoint = `${BACKEND_API_URL}${API_V1}/patches/generate`

    // CRITICAL FIX: Use the pre-generated session_id from config instead of generating a new one
    const sessionId = config.session_id
    console.log('[API] startTraining called with session_id:', sessionId)

    // Updated request body to match backend test_patch_only.py
    const requestBody = {
      patch_name: config.patch_name,
      attack_method: config.attack_method || "robust_dpatch", // Default to robust_dpatch as it's the most robust
      source_dataset_id: config.dataset_id,
      model_id: config.model_path,
      target_class: config.target_class,
      patch_size: config.patch_size,
      learning_rate: config.learning_rate || 5.0, // Default from test file
      iterations: config.iterations,
      session_id: sessionId
    }

    console.log('[API] Sending training request with session_id:', sessionId)
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()

    return {
      training_id: data.patch?.id || sessionId,
      session_id: sessionId,  // Return the same session ID
      status: 'completed',
      message: data.message || 'Patch generation started',
      sse_url: `${BACKEND_API_URL}${API_V1}/patches/sse/${sessionId}`  // Fixed: correct SSE endpoint
    }
  } catch (error) {
    console.error('Failed to start training:', error)
    throw error
  }
}

export async function fetchTrainingResult(trainingId: number | string): Promise<TrainingResult> {
  try {
    // Fixed: Use correct backend endpoint
    const endpoint = `${BACKEND_API_URL}${API_V1}/patches/${trainingId}`
    const response = await fetch(endpoint)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()

    return {
      id: typeof trainingId === 'number' ? trainingId : trainingId,
      patch_name: data.name,
      dataset_name: data.dataset_id,
      target_class: data.target_class,
      model_path: data.model_id,
      iterations: data.patch_metadata?.iterations || 0,
      status: 'completed',
      best_score: data.patch_metadata?.best_score,
      patch_file_path: data.patch_metadata?.patch_file,
      created_at: data.created_at
    }
  } catch (error) {
    console.error('Failed to fetch training result:', error)
    throw error
  }
}

export async function fetchTrainingResults(trainingId: number | string): Promise<any> {
  try {
    // Fixed: Use correct backend endpoint
    const endpoint = `${BACKEND_API_URL}${API_V1}/patches/${trainingId}`
    const response = await fetch(endpoint)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()

    return {
      results: data.patch_metadata,
      patch: data
    }
  } catch (error) {
    console.error('Failed to fetch training results:', error)
    throw error
  }
}

export function connectPatchGenerationSSE(sessionId: string, callbacks: {
  onOpen?: () => void
  onMessage?: (data: TrainingLog) => void
  onError?: (error: Event) => void
  onClose?: () => void
}): EventSource | null {
  try {
    // Updated endpoint to match backend
    const sseUrl = `${BACKEND_API_URL}${API_V1}/patches/sse/${sessionId}`
    console.log('[SSE] Connecting to patch generation:', sseUrl)
    console.log('[SSE] Session ID:', sessionId)

    // Use the session ID directly - no need to transform it
    const eventSource = new EventSource(sseUrl)

    eventSource.onopen = () => {
      console.log('[SSE] ✅ Patch generation connected for session:', sessionId)
      callbacks.onOpen?.()
    }

    eventSource.onmessage = (event) => {
      console.log('[SSE] 📨 Patch generation message:', event.data)
      try {
        const data = JSON.parse(event.data) as TrainingLog
        console.log('[SSE] 📦 Parsed patch data:', data)
        callbacks.onMessage?.(data)
      } catch (error) {
        console.error('[SSE] ❌ Failed to parse SSE message:', error, 'Raw data:', event.data)
      }
    }

    eventSource.onerror = (error) => {
      console.error('[SSE] ❌ Patch generation error:', error)
      console.error('[SSE] ReadyState:', eventSource.readyState)
      callbacks.onError?.(error as Event)
      eventSource.close()
    }

    return eventSource
  } catch (error) {
    console.error('[SSE] ❌ Failed to connect to patch generation:', error)
    return null
  }
}

export async function downloadPatch(trainingId: number | string, patchName?: string, storageKey?: string): Promise<void> {
  try {
    // Fixed: Use correct backend endpoint
    const endpoint = storageKey
      ? `${BACKEND_API_URL}${API_V1}/storage/${storageKey}`
      : `${BACKEND_API_URL}${API_V1}/patches/${trainingId}/download`

    const response = await fetch(endpoint)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = patchName || `patch_${trainingId}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  } catch (error) {
    console.error('Failed to download patch:', error)
    throw error
  }
}

export async function startAdversarialDataGeneration(config: AdversarialDataConfig): Promise<AdversarialDataResponse> {
  try {
    let endpoint = ''
    let requestBody: any = {}

    if (config.attack_type === 'patch') {
      // Adversarial Patch 적용 - Updated to match test_patch_only.py
      endpoint = `${BACKEND_API_URL}${API_V1}/attack-datasets/patch`
      requestBody = {
        attack_name: config.dataset_name,
        patch_id: config.training_id,
        base_dataset_id: config.source_dataset_id,
        patch_scale: config.patch_scale ? config.patch_scale * 100 : 30.0, // Convert ratio to percentage
        session_id: config.session_id  // Added for SSE support
      }
    } else if (config.attack_type === 'noise') {
      // Noise Attack 적용 - Updated to match test_noise_only.py
      endpoint = `${BACKEND_API_URL}${API_V1}/attack-datasets/noise`
      requestBody = {
        attack_name: config.dataset_name,
        attack_method: config.noise_method || "pgd", // or "fgsm"
        base_dataset_id: config.source_dataset_id,
        model_id: config.model_id,
        epsilon: config.noise_epsilon || 8.0,
        alpha: config.noise_alpha || 2.0,
        iterations: config.noise_iterations || 10,
        target_class: config.target_class,  // Added: target class for attack
        session_id: config.session_id  // Added for SSE support
      }
    }

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()

    const outputDatasetId =
      data.attack_dataset?.parameters?.output_dataset_id ??
      data.parameters?.output_dataset_id ??
      data.output_dataset_id ??
      null

    return {
      generation_id: data.attack_dataset?.id || data.id || '',
      output_dataset_id: outputDatasetId,
      message: data.message || 'Success'
    }
  } catch (error) {
    console.error('Failed to start adversarial data generation:', error)
    throw error
  }
}

export function connectAdversarialDataSSE(
  sessionId: string,
  attackType: 'patch' | 'noise',
  callbacks: {
    onOpen?: () => void
    onMessage?: (data: AdversarialDataLog) => void
    onError?: (error: Event) => void
    onClose?: () => void
  }
): EventSource | null {
  try {
    // Updated SSE endpoints to match backend - both use /attack-datasets/sse
    const sseUrl = `${BACKEND_API_URL}${API_V1}/attack-datasets/sse/${sessionId}`

    console.log(`[SSE] Connecting to ${attackType} attack dataset:`, sseUrl)

    const eventSource = new EventSource(sseUrl)

    eventSource.onopen = () => {
      console.log(`[SSE] ✅ ${attackType} attack dataset connected`)
      callbacks.onOpen?.()
    }

    eventSource.onmessage = (event) => {
      console.log(`[SSE] 📨 ${attackType} attack message:`, event.data)
      try {
        const data = JSON.parse(event.data) as AdversarialDataLog
        callbacks.onMessage?.(data)
      } catch (error) {
        console.error('[SSE] Failed to parse SSE message:', error)
      }
    }

    eventSource.onerror = (error) => {
      console.error(`[SSE] ❌ ${attackType} attack error:`, error)
      callbacks.onError?.(error as Event)
      eventSource.close()
    }

    return eventSource
  } catch (error) {
    console.error(`[SSE] ❌ Failed to connect to ${attackType} attack:`, error)
    return null
  }
}

export async function fetchAdversarialDatasetImages(datasetId: string, limit: number = 10): Promise<any[]> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/datasets-2d/${datasetId}/images?limit=${limit}`
    console.log('[fetchAdversarialDatasetImages] Fetching from:', endpoint)
    const response = await fetch(endpoint)
    if (!response.ok) {
      console.error('[fetchAdversarialDatasetImages] HTTP error:', response.status)
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    const data = await response.json()
    console.log('[fetchAdversarialDatasetImages] Response data:', data)

    // Backend returns array directly, not wrapped in items
    const images = Array.isArray(data) ? data : []
    return images
  } catch (error) {
    console.error('[fetchAdversarialDatasetImages] Failed to fetch adversarial dataset images:', error)
    throw error
  }
}

export async function downloadAdversarialDataset(datasetId: string, datasetName?: string): Promise<void> {
  try {
    // Fixed: Use correct backend endpoint
    const endpoint = `${BACKEND_API_URL}${API_V1}/attack-datasets/${datasetId}/download`
    const response = await fetch(endpoint)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = datasetName || `adversarial_dataset_${datasetId}.zip`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  } catch (error) {
    console.error('Failed to download adversarial dataset:', error)
    throw error
  }
}

export interface PatchPreviewRequest {
  patch_id: string
  image_id: string
  model_id: string
  target_class: string
  patch_scale: number
}

export interface PatchPreviewResponse {
  image_data: string // Base64 encoded image
  image_mime_type: string
  detections_count: number
  patch_applied: boolean
}

export async function previewPatchOnImage(request: PatchPreviewRequest): Promise<PatchPreviewResponse> {
  try {
    // Fixed: Use correct backend endpoint
    const endpoint = `${BACKEND_API_URL}${API_V1}/patches/${request.patch_id}/preview`
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_id: request.image_id,
        model_id: request.model_id,
        target_class: request.target_class,
        patch_scale: request.patch_scale,
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to preview patch on image:', error)
    throw error
  }
}

// ============================================
// Asset Management API Functions
// ============================================

export interface PatchAsset {
  id: string
  name: string
  target_class: string
  source_dataset_id: string
  target_model_id?: string
  created_at: string
  patch_metadata?: {
    iterations?: number
    patch_size?: number
    num_training_samples?: number
    best_score?: number
    patch_file?: string
  }
}

export interface AttackDatasetAsset {
  id: string
  name: string
  attack_type: 'patch' | 'noise'
  base_dataset_id: string
  target_class?: string
  target_model_id?: string
  created_at: string
  parameters?: {
    processed_images?: number
    output_dataset_id?: string
    attack_method?: string
  }
}

export async function fetchPatches(params?: {
  skip?: number
  limit?: number
  target_class?: string
}): Promise<PatchAsset[]> {
  try {
    const queryParams = new URLSearchParams()
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString())
    if (params?.target_class) queryParams.append('target_class', params.target_class)

    const endpoint = `${BACKEND_API_URL}${API_V1}/patches?${queryParams.toString()}`
    const response = await fetch(endpoint)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data || []
  } catch (error) {
    console.error('Failed to fetch patches:', error)
    throw error
  }
}

export async function fetchPatch(patchId: string): Promise<PatchAsset> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/patches/${patchId}`
    const response = await fetch(endpoint)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to fetch patch:', error)
    throw error
  }
}

export async function deletePatch(patchId: string): Promise<void> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/patches/${patchId}`
    const response = await fetch(endpoint, {
      method: 'DELETE',
    })

    if (!response.ok && response.status !== 204) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }
  } catch (error) {
    console.error('Failed to delete patch:', error)
    throw error
  }
}

export async function fetchAttackDatasets(params?: {
  skip?: number
  limit?: number
  target_class?: string
}): Promise<AttackDatasetAsset[]> {
  try {
    const queryParams = new URLSearchParams()
    if (params?.skip !== undefined) queryParams.append('skip', params.skip.toString())
    if (params?.limit !== undefined) queryParams.append('limit', params.limit.toString())
    if (params?.target_class) queryParams.append('target_class', params.target_class)

    const endpoint = `${BACKEND_API_URL}${API_V1}/attack-datasets?${queryParams.toString()}`
    const response = await fetch(endpoint)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data || []
  } catch (error) {
    console.error('Failed to fetch attack datasets:', error)
    throw error
  }
}

export async function fetchAttackDataset(datasetId: string): Promise<AttackDatasetAsset> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/attack-datasets/${datasetId}`
    const response = await fetch(endpoint)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to fetch attack dataset:', error)
    throw error
  }
}

export async function deleteAttackDataset(datasetId: string): Promise<void> {
  try {
    // Fixed: Use correct backend endpoint
    const endpoint = `${BACKEND_API_URL}${API_V1}/attack-datasets/${datasetId}`
    const response = await fetch(endpoint, {
      method: 'DELETE',
    })

    if (!response.ok && response.status !== 204) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }
  } catch (error) {
    console.error('Failed to delete attack dataset:', error)
    throw error
  }
}

/**
 * Get URL for patch image preview
 */
export function getPatchImageUrl(patchId: string): string {
  // Fixed: Use correct backend endpoint
  return `${BACKEND_API_URL}${API_V1}/patches/${patchId}/image`
}

/**
 * Get URL for dataset image by storage key
 */
export function getImageUrlByStorageKey(storageKey: string): string {
  if (!storageKey) {
    console.warn('[getImageUrlByStorageKey] Empty storage key provided')
    return ''
  }

  // Remove leading "storage/" if present to avoid duplication in URL
  const cleanedKey = storageKey.startsWith('storage/')
    ? storageKey.substring('storage/'.length)
    : storageKey

  const url = `${BACKEND_API_URL}${API_V1}/storage/${cleanedKey}`
  console.log('[getImageUrlByStorageKey] Generated URL:', url, 'for key:', storageKey)
  return url
}

/**
 * Fetch images from a dataset
 */
export async function fetchDatasetImages(datasetId: string, limit: number = 4): Promise<any[]> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/datasets-2d/${datasetId}/images?limit=${limit}`
    console.log('[fetchDatasetImages] Fetching from:', endpoint)
    const response = await fetch(endpoint)

    if (!response.ok) {
      console.error('[fetchDatasetImages] HTTP error:', response.status)
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    console.log('[fetchDatasetImages] Response data:', data)

    // Backend returns array directly, not wrapped in items
    const images = Array.isArray(data) ? data : []
    console.log('[fetchDatasetImages] Parsed images:', images)
    return images
  } catch (error) {
    console.error('[fetchDatasetImages] Failed to fetch dataset images:', error)
    return []
  }
}
