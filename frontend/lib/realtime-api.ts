/**
 * Real-time Camera & Inference API Client
 */

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
const API_V1 = '/api/v1'

// Re-export from adversarial-api for convenience
export { fetchYoloModels, type YoloModel } from './adversarial-api'

// ============================================
// Type Definitions
// ============================================

export interface Camera {
  device_id: string
  device_path: string
  name: string
  width: number
  height: number
  fps: number
  is_available: boolean
}

export interface RealtimeSession {
  run_id: string
  model_id: string
  device: string
  status: string
  message: string
  stream_url: string
  mjpeg_url: string
}

export interface Detection {
  bbox: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
  class_id: number
  class_name: string
  confidence: number
}

export interface RealtimeStats {
  run_id: string
  frame_count: number
  fps: number
  detections: number
  inference_time_ms: number
  last_update: number
}

// ============================================
// API Functions
// ============================================

export async function fetchAvailableCameras(): Promise<Camera[]> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/realtime/webcam/list`
    const response = await fetch(endpoint)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data.cameras || []
  } catch (error) {
    console.error('Failed to fetch cameras:', error)
    throw error
  }
}

export async function startRealtimeSession(config: {
  run_name: string
  device: string
  model_id: string
  fps_target?: number
  window_seconds?: number
  conf_threshold?: number
  iou_threshold?: number
}): Promise<RealtimeSession> {
  try {
    const params = new URLSearchParams({
      run_name: config.run_name,
      device: config.device,
      model_id: config.model_id,
      fps_target: (config.fps_target || 30).toString(),
      window_seconds: (config.window_seconds || 300).toString(),
      conf_threshold: (config.conf_threshold || 0.25).toString(),
      iou_threshold: (config.iou_threshold || 0.45).toString()
    })

    const endpoint = `${BACKEND_API_URL}${API_V1}/realtime/webcam/sessions/start?${params}`
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to start realtime session:', error)
    throw error
  }
}

export async function stopRealtimeSession(run_id: string): Promise<void> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/realtime/webcam/stop/${run_id}`
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }
  } catch (error) {
    console.error('Failed to stop realtime session:', error)
    throw error
  }
}

export async function fetchRealtimeStats(run_id: string): Promise<RealtimeStats> {
  try {
    const endpoint = `${BACKEND_API_URL}${API_V1}/realtime/webcam/stats/${run_id}`
    const response = await fetch(endpoint)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Failed to fetch realtime stats:', error)
    throw error
  }
}

export function getRealtimeMJPEGUrl(
  run_id: string,
  model_id: string,
  config?: {
    conf_threshold?: number
    draw_boxes?: boolean
  }
): string {
  const params = new URLSearchParams({
    model_id: model_id,
    conf_threshold: (config?.conf_threshold || 0.25).toString(),
    draw_boxes: (config?.draw_boxes !== false).toString()
  })

  return `${BACKEND_API_URL}${API_V1}/realtime/webcam/stream-mjpeg/${run_id}?${params}`
}
