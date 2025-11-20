/**
 * SSE (Server-Sent Events) client for real-time system statistics
 */

export interface SystemStats {
  timestamp: number
  sample_number?: number
  cpu: {
    usage_percent: number
    usage_per_core: number[]
    core_count: number
    thread_count: number
    frequency_mhz: number | null
    frequency_max_mhz: number | null
  }
  memory: {
    total_gb: number
    available_gb: number
    used_gb: number
    percent: number
    swap_total_gb: number
    swap_used_gb: number
    swap_percent: number
  }
  disk: {
    total_gb: number
    used_gb: number
    free_gb: number
    percent: number
    read_mb: number | null
    write_mb: number | null
  }
  gpu: {
    available: boolean
    count?: number
    gpus?: Array<{
      id: number
      name: string
      load_percent: number
      memory_total_mb: number
      memory_used_mb: number
      memory_free_mb: number
      memory_percent: number
      temperature_c: number
    }>
    error?: string
  } | null
  network: {
    bytes_sent_mb: number
    bytes_recv_mb: number
    packets_sent: number
    packets_recv: number
  }
  process: {
    pid: number
    cpu_percent: number
    memory_mb: number
    threads: number
    open_files: number
  }
}

export type StatsCallback = (stats: SystemStats) => void
export type ErrorCallback = (error: Error) => void
export type CloseCallback = () => void

export class SystemStatsSSE {
  private eventSource: EventSource | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private shouldReconnect = true

  constructor(
    private interval: number = 1.0,
    private onStats?: StatsCallback,
    private onError?: ErrorCallback,
    private onClose?: CloseCallback
  ) {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
    this.url = `${backendUrl}/api/v1/system/stats/stream?interval=${interval}`
  }

  connect(): void {
    try {
      this.eventSource = new EventSource(this.url)

      this.eventSource.onopen = () => {
        console.log('[SystemStatsSSE] Connected')
        this.reconnectAttempts = 0
      }

      this.eventSource.onmessage = (event) => {
        try {
          const stats: SystemStats = JSON.parse(event.data)
          this.onStats?.(stats)
        } catch (error) {
          console.error('[SystemStatsSSE] Failed to parse stats:', error)
          this.onError?.(error as Error)
        }
      }

      this.eventSource.onerror = (event) => {
        console.error('[SystemStatsSSE] SSE error:', event)
        this.onError?.(new Error('SSE connection error'))

        // Close and attempt to reconnect if enabled
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++
          console.log(
            `[SystemStatsSSE] Reconnecting (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`
          )
          this.eventSource?.close()
          setTimeout(() => this.connect(), this.reconnectDelay)
        } else {
          this.onClose?.()
        }
      }
    } catch (error) {
      console.error('[SystemStatsSSE] Failed to create EventSource:', error)
      this.onError?.(error as Error)
    }
  }

  disconnect(): void {
    this.shouldReconnect = false
    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }
  }

  isConnected(): boolean {
    return this.eventSource !== null && this.eventSource.readyState === EventSource.OPEN
  }
}

/**
 * React hook for system stats SSE
 */
export function useSystemStatsSSE(
  interval: number = 1.0,
  onStats?: StatsCallback,
  onError?: ErrorCallback
): {
  connect: () => void
  disconnect: () => void
  isConnected: boolean
} {
  let sseClient: SystemStatsSSE | null = null
  let connected = false

  const connect = () => {
    if (sseClient) {
      sseClient.disconnect()
    }

    sseClient = new SystemStatsSSE(
      interval,
      onStats,
      onError,
      () => {
        connected = false
      }
    )

    sseClient.connect()
    connected = true
  }

  const disconnect = () => {
    if (sseClient) {
      sseClient.disconnect()
      sseClient = null
      connected = false
    }
  }

  return {
    connect,
    disconnect,
    isConnected: connected,
  }
}

/**
 * Fetch system stats via HTTP (one-time request)
 */
export async function fetchSystemStats(): Promise<SystemStats> {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
  const response = await fetch(`${backendUrl}/api/v1/system/stats`)

  if (!response.ok) {
    throw new Error(`Failed to fetch system stats: ${response.statusText}`)
  }

  return response.json()
}
