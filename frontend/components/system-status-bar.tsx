'use client'

import { useState, useEffect, useRef } from 'react'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import {
  Cpu,
  HardDrive,
  Activity,
  Server,
  Clock,
  AlertCircle,
  Zap,
  Database
} from 'lucide-react'
import { SystemStats } from '@/lib/system-stats-ws'

export function SystemStatusBar() {
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const sseRef = useRef<EventSource | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5

  // Connect to SSE
  const connectSSE = () => {
    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
      const eventSource = new EventSource(`${backendUrl}/api/v1/system/stats/stream?interval=1.0`)

      eventSource.onopen = () => {
        console.log('[SystemStatusBar] SSE connected')
        setError(null)
        reconnectAttemptsRef.current = 0
      }

      eventSource.onmessage = (event) => {
        try {
          const data: SystemStats = JSON.parse(event.data)
          setStats(data)
          setLastUpdate(new Date())
          setError(null)
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
    }
  }, [])

  // Get status color
  const getStatusColor = (usage: number) => {
    if (usage < 50) return 'text-green-400'
    if (usage < 75) return 'text-yellow-400'
    return 'text-red-400'
  }

  if (!stats && !error) {
    return (
      <div className="w-full bg-slate-900/95 backdrop-blur-sm border-t border-white/10">
        <div className="flex items-center justify-center h-20 text-slate-400">
          <Activity className="w-5 h-5 mr-2 animate-pulse" />
          <span className="text-base">시스템 상태 로딩중...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full bg-slate-900/95 backdrop-blur-sm border-t border-white/10">
      <div className="px-6 py-5">
        <div className="flex items-center justify-between space-x-6">
          {/* System Info */}
          <div className="flex items-center space-x-2">
            <Server className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-300 font-mono">
              System Monitor
            </span>
          </div>

          {/* CPU */}
          <div className="flex items-center space-x-2">
            <Cpu className={`w-5 h-5 ${stats ? getStatusColor(stats.cpu.usage_percent) : 'text-slate-400'}`} />
            <div className="flex flex-col">
              <div className="flex items-center space-x-1">
                <span className="text-sm text-slate-300">CPU</span>
                <span className={`text-sm font-bold ${stats ? getStatusColor(stats.cpu.usage_percent) : 'text-slate-400'}`}>
                  {stats?.cpu.usage_percent.toFixed(1) || 0}%
                </span>
              </div>
              <div className="w-24">
                <Progress value={stats?.cpu.usage_percent || 0} className="h-2" />
              </div>
            </div>
            <span className="text-sm text-slate-500">
              {stats?.cpu.frequency_mhz ? `${(stats.cpu.frequency_mhz / 1000).toFixed(1)}GHz` : 'N/A'}
            </span>
          </div>

          {/* Memory */}
          <div className="flex items-center space-x-2">
            <HardDrive className={`w-5 h-5 ${stats ? getStatusColor(stats.memory.percent) : 'text-slate-400'}`} />
            <div className="flex flex-col">
              <div className="flex items-center space-x-1">
                <span className="text-sm text-slate-300">RAM</span>
                <span className={`text-sm font-bold ${stats ? getStatusColor(stats.memory.percent) : 'text-slate-400'}`}>
                  {stats?.memory.percent.toFixed(1) || 0}%
                </span>
              </div>
              <div className="w-24">
                <Progress value={stats?.memory.percent || 0} className="h-2" />
              </div>
            </div>
            <span className="text-sm text-slate-500">
              {stats?.memory.used_gb.toFixed(1) || '0'}/{stats?.memory.total_gb.toFixed(1) || '0'}G
            </span>
          </div>

          {/* Disk */}
          <div className="flex items-center space-x-2">
            <Database className={`w-5 h-5 ${stats ? getStatusColor(stats.disk.percent) : 'text-slate-400'}`} />
            <div className="flex flex-col">
              <div className="flex items-center space-x-1">
                <span className="text-sm text-slate-300">Disk</span>
                <span className={`text-sm font-bold ${stats ? getStatusColor(stats.disk.percent) : 'text-slate-400'}`}>
                  {stats?.disk.percent.toFixed(1) || 0}%
                </span>
              </div>
              <div className="w-24">
                <Progress value={stats?.disk.percent || 0} className="h-2" />
              </div>
            </div>
            <span className="text-sm text-slate-500">
              {stats?.disk.used_gb.toFixed(0) || '0'}/{stats?.disk.total_gb.toFixed(0) || '0'}G
            </span>
          </div>

          {/* GPU Section - Supports multiple GPUs (up to 4) */}
          {stats?.gpu && stats.gpu.available && stats.gpu.gpus && (
            <div className="flex items-center space-x-3 border-l border-white/10 pl-4">
              {stats.gpu.gpus.slice(0, 4).map((gpu) => (
                <div key={gpu.id} className="flex items-center space-x-2">
                  <Zap className={`w-4 h-4 ${getStatusColor(gpu.load_percent)}`} />
                  <div className="flex flex-col">
                    <div className="flex items-center space-x-1">
                      <span className="text-xs text-slate-300">GPU{gpu.id}</span>
                      <span className={`text-xs font-bold ${getStatusColor(gpu.load_percent)}`}>
                        {gpu.load_percent.toFixed(0)}%
                      </span>
                      {gpu.temperature_c > 0 && (
                        <span className="text-xs text-slate-500">
                          {gpu.temperature_c.toFixed(0)}°
                        </span>
                      )}
                    </div>
                    <div className="w-20">
                      <Progress value={gpu.load_percent} className="h-1.5" />
                    </div>
                    <div className="text-xs text-slate-500">
                      {(gpu.memory_used_mb / 1024).toFixed(1)}/{(gpu.memory_total_mb / 1024).toFixed(0)}G
                    </div>
                  </div>
                </div>
              ))}
              {stats.gpu.gpus.length > 4 && (
                <Badge variant="outline" className="text-xs">
                  +{stats.gpu.gpus.length - 4} more
                </Badge>
              )}
            </div>
          )}

          {/* Network & Status */}
          <div className="flex items-center space-x-4 border-l border-white/10 pl-4">


            {/* Last Update */}
            <div className="flex items-center space-x-2">
              <Clock className={`w-5 h-5 ${error ? 'text-red-400' : 'text-green-400 animate-pulse'}`} />
              <span className="text-sm text-slate-500">
                {lastUpdate.toLocaleTimeString('ko-KR')}
              </span>
            </div>

            {/* Error Indicator */}
            {error && (
              <div className="flex items-center space-x-1">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <span className="text-sm text-red-400">{error}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}