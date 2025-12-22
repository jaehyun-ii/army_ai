'use client'

import { useState, useEffect, useRef, memo } from 'react'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import {
  Cpu,
  HardDrive,
  Activity,
  Server,
  Clock,
  AlertCircle,
  Zap
} from 'lucide-react'
import { SystemStats } from '@/lib/system-stats-ws'

function SystemStatusBarComponent() {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const sseRef = useRef<EventSource | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5
  const lastRenderTimeRef = useRef<number>(0)
  const renderIntervalMs = 1000 // 최대 1초에 한 번만 렌더
  const isCameraConnectedRef = useRef<boolean>(false)

  // Connect to SSE
  const connectSSE = () => {
    try {
      // Use Next.js API proxy route for SSE
      // Increased interval from 1.0s to 3.0s to reduce interference with camera SSE
      const eventSource = new EventSource(`/api/system-stats/stream?interval=3.0`)

      eventSource.onopen = () => {
        console.log('[SystemStatusBar] SSE connected')
        setError(null)
        reconnectAttemptsRef.current = 0
      }

      eventSource.onmessage = (event) => {
        try {
          if (isCameraConnectedRef.current) {
            // 카메라 연결 시에는 메시지를 읽기만 하고 렌더/상태 갱신을 건너뛰어 간섭을 줄인다
            return
          }
          const data: SystemStats = JSON.parse(event.data)
          const now = performance.now()
          if (now - lastRenderTimeRef.current >= renderIntervalMs) {
            lastRenderTimeRef.current = now
            // Use requestAnimationFrame to defer state updates and reduce blocking
            requestAnimationFrame(() => {
              setStats(data)
              setLastUpdate(new Date())
              setError(null)
            })
          }
        } catch (err) {
          console.error('[SystemStatusBar] Failed to parse stats:', err)
          setError('Data parse error')
        }
      }

      eventSource.onerror = (event) => {
        console.error('[SystemStatusBar] SSE error:', event)
        setError('Connection error')

        // Close and attempt to reconnect
        eventSource.close()

        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000)
          console.log(`[SystemStatusBar] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`)

          reconnectTimeoutRef.current = setTimeout(() => {
            connectSSE()
          }, delay)
        } else {
          setError('연결이 끊어졌습니다')
        }
      }

      sseRef.current = eventSource
    } catch (err) {
      console.error('[SystemStatusBar] Failed to create EventSource:', err)
      setError('SSE 생성 실패')
    }
  }

  // Set up SSE connection
  useEffect(() => {
    connectSSE()
    const handleCameraChange = (e: Event) => {
      const detail = (e as CustomEvent<{ isConnected: boolean }>).detail
      isCameraConnectedRef.current = detail?.isConnected ?? false
    }
    window.addEventListener('camera-connection-change', handleCameraChange)

    // Check online status
    const handleOnline = () => {
      if (!sseRef.current || sseRef.current.readyState !== EventSource.OPEN) {
        connectSSE()
      }
    }

    window.addEventListener('online', handleOnline)

    return () => {
      // Cleanup
      if (sseRef.current) {
        sseRef.current.close()
        sseRef.current = null
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('camera-connection-change', handleCameraChange)
    }
  }, [])

  // Status color thresholds
  const STATUS_THRESHOLDS = {
    warning: 75,
    critical: 90
  } as const

  // Get status color based on usage percentage
  const getStatusColor = (usage: number) => {
    if (usage < STATUS_THRESHOLDS.warning) return 'text-tertiary'
    if (usage < STATUS_THRESHOLDS.critical) return 'text-warning'
    return 'text-error'
  }

  if (!stats && !error) {
    return (
      <div className="w-full border-t border-border bg-[var(--surface)]/90 backdrop-blur-xl shadow-[0_-8px_30px_-12px_rgba(0,0,0,0.25)]">
        <div className="flex items-center justify-center h-20 text-muted-foreground">
          <Activity className="w-5 h-5 mr-2 animate-pulse" />
          <span className="text-base">시스템 상태 로딩중...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full border-t border-border bg-[var(--surface)]/90 backdrop-blur-xl shadow-[0_-8px_30px_-12px_rgba(0,0,0,0.25)]">
      <div className="px-6 py-4 max-w-6xl mx-auto">
        <div className="flex items-center justify-between gap-6">
          {/* System Info */}
          <div className="flex items-center space-x-2">
            <Server className="w-5 h-5 text-primary" />
            <span className="text-sm text-on-surface-variant font-mono">
              System Monitor
            </span>
          </div>

          {/* CPU */}
          <div className="flex items-center space-x-2">
            <Cpu className={`w-5 h-5 ${stats ? getStatusColor(stats.cpu.usage_percent) : 'text-muted'}`} />
            <div className="flex flex-col">
              <div className="flex items-center space-x-1">
                <span className="text-sm text-muted">CPU</span>
                <span className={`text-sm font-bold ${stats ? getStatusColor(stats.cpu.usage_percent) : 'text-muted'}`}>
                  {stats?.cpu.usage_percent || 0}%
                </span>
              </div>
              <div className="w-24">
                <Progress value={stats?.cpu.usage_percent || 0} className="h-2" />
              </div>
            </div>
            <span className="text-sm text-muted">
              {stats?.cpu.core_count || 'N/A'} Core{(stats?.cpu.core_count || 0) > 1 ? 's' : ''}
            </span>
          </div>

          {/* Memory */}
          <div className="flex items-center space-x-2">
            <HardDrive className={`w-5 h-5 ${stats ? getStatusColor(stats.memory.percent) : 'text-muted'}`} />
            <div className="flex flex-col">
              <div className="flex items-center space-x-1">
                <span className="text-sm text-muted">RAM</span>
                <span className={`text-sm font-bold ${stats ? getStatusColor(stats.memory.percent) : 'text-muted'}`}>
                  {stats?.memory.percent || 0}%
                </span>
              </div>
              <div className="w-24">
                <Progress value={stats?.memory.percent || 0} className="h-2" />
              </div>
            </div>
            <span className="text-sm text-muted">
              {stats?.memory.used_gb || '0'}/{stats?.memory.total_gb || '0'}G
            </span>
          </div>

          {/* GPU Section - Backend limits to 4 GPUs */}
          {stats?.gpu && stats.gpu.available && stats.gpu.gpus && stats.gpu.gpus.length > 0 && (
            <div className="flex items-center space-x-2 border-l border-border pl-4">
              {stats.gpu.gpus.map((gpu) => {
                // GPU 메모리 사용률 계산 (연산 가동률 대신 메모리 사용률 표시)
                const hasMemory = gpu.memory_total_mb != null && gpu.memory_used_mb != null
                const memoryPercent = hasMemory
                  ? (gpu.memory_used_mb! / gpu.memory_total_mb!) * 100
                  : 0
                const memoryDisplay = hasMemory
                  ? `${(gpu.memory_used_mb! / 1024).toFixed(1)}/${(gpu.memory_total_mb! / 1024).toFixed(1)}G`
                  : 'N/A'

                return (
                  <div key={gpu.id} className="flex items-center space-x-1.5">
                    <Zap className={`w-4 h-4 ${getStatusColor(memoryPercent)}`} />
                    <div className="flex flex-col">
                      <div className="flex items-center space-x-1">
                        <span className="text-xs text-muted">GPU{gpu.id}</span>
                        <span className={`text-xs font-bold ${getStatusColor(memoryPercent)}`}>
                          {memoryPercent.toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-16">
                        <Progress value={memoryPercent} className="h-1.5" />
                      </div>
                      <div className="text-xs text-muted">
                        {memoryDisplay}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          {/* Network & Status */}
          <div className="flex items-center space-x-4 border-l border-border pl-4">


            {/* Last Update */}
            <div className="flex items-center space-x-2">
              <Clock className={`w-5 h-5 ${error ? 'text-error' : 'text-tertiary animate-pulse'}`} />
              <span className="text-sm text-muted">
                {lastUpdate.toLocaleTimeString('ko-KR')}
              </span>
            </div>

            {/* Error Indicator */}
            {error && (
              <div className="flex items-center space-x-1">
                <AlertCircle className="w-5 h-5 text-error" />
                <span className="text-sm text-error">{error}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// Memoize to prevent unnecessary re-renders
export const SystemStatusBar = memo(SystemStatusBarComponent)
