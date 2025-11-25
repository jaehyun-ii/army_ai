"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ScrollText, Filter, Download, RefreshCw, AlertCircle, Info, AlertTriangle, XCircle, Loader2 } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

interface LogEntry {
  id: string
  timestamp: string
  log_level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
  username: string | null
  action: string
  ip_address: string | null
  message: string
  module: string | null
}

interface LogStats {
  total: number
  debug: number
  info: number
  warning: number
  error: number
  critical: number
}

export function LogManagement() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [stats, setStats] = useState<LogStats>({
    total: 0,
    debug: 0,
    info: 0,
    warning: 0,
    error: 0,
    critical: 0,
  })
  const [loading, setLoading] = useState(false)
  const [filterLevel, setFilterLevel] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [timeRange, setTimeRange] = useState<string>('24h')
  const { toast } = useToast()

  // Fetch logs from backend
  const fetchLogs = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

      // Calculate hours based on time range
      const hoursMap: Record<string, number> = {
        '1h': 1,
        '24h': 24,
        '7d': 168,
        '30d': 720,
      }
      const hours = hoursMap[timeRange] || 24

      // Build query parameters
      const params = new URLSearchParams({
        hours: hours.toString(),
        skip: '0',
        limit: '100',
      })

      if (filterLevel !== 'all') {
        params.append('log_level', filterLevel.toUpperCase())
      }

      if (searchTerm) {
        params.append('search_term', searchTerm)
      }

      const response = await fetch(`${backendUrl}/api/v1/system-logs?${params}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error('Failed to fetch logs')
      }

      const data = await response.json()
      setLogs(data.logs || [])
    } catch (error) {
      console.error('Error fetching logs:', error)
      toast({
        title: "오류",
        description: "로그를 불러오는데 실패했습니다.",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  // Fetch statistics from backend
  const fetchStats = async () => {
    try {
      const token = localStorage.getItem('token')
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

      const hoursMap: Record<string, number> = {
        '1h': 1,
        '24h': 24,
        '7d': 168,
        '30d': 720,
      }
      const hours = hoursMap[timeRange] || 24

      const response = await fetch(
        `${backendUrl}/api/v1/system-logs/statistics?hours=${hours}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        }
      )

      if (!response.ok) {
        throw new Error('Failed to fetch statistics')
      }

      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Error fetching statistics:', error)
    }
  }

  // Load data on mount and when filters change
  useEffect(() => {
    fetchLogs()
    fetchStats()
  }, [filterLevel, timeRange])

  // Handle search with debouncing
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchTerm !== undefined) {
        fetchLogs()
      }
    }, 500)

    return () => clearTimeout(timer)
  }, [searchTerm])

  const getLevelIcon = (level: string) => {
    const levelUpper = level.toUpperCase()
    switch (levelUpper) {
      case 'INFO':
        return <Info className="w-4 h-4" />
      case 'WARNING':
        return <AlertTriangle className="w-4 h-4" />
      case 'ERROR':
      case 'CRITICAL':
        return <XCircle className="w-4 h-4" />
      case 'DEBUG':
        return <AlertCircle className="w-4 h-4" />
      default:
        return null
    }
  }

  const getLevelBadge = (level: string) => {
    const levelUpper = level.toUpperCase()
    const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      INFO: 'default',
      WARNING: 'secondary',
      ERROR: 'destructive',
      CRITICAL: 'destructive',
      DEBUG: 'outline'
    }

    return (
      <Badge variant={variants[levelUpper] || 'default'} className="flex items-center gap-1">
        {getLevelIcon(level)}
        {levelUpper}
      </Badge>
    )
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }

  const handleRefresh = () => {
    fetchLogs()
    fetchStats()
    toast({
      title: "새로고침 완료",
      description: "로그 데이터가 업데이트되었습니다.",
    })
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card className="bg-gradient-to-br from-slate-900/20 to-slate-800/20 border-slate-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <ScrollText className="w-5 h-5" />
              전체 로그
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.total}</div>
            <p className="text-sm text-muted-foreground">
              {timeRange === '1h' ? '최근 1시간' :
               timeRange === '24h' ? '최근 24시간' :
               timeRange === '7d' ? '최근 7일' : '최근 30일'}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 border-blue-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Info className="w-5 h-5" />
              정보
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.info}</div>
            <p className="text-sm text-muted-foreground">정상 활동</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-yellow-900/20 to-yellow-800/20 border-yellow-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              경고
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.warning}</div>
            <p className="text-sm text-muted-foreground">주의 필요</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-red-900/20 to-red-800/20 border-red-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <XCircle className="w-5 h-5" />
              오류
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.error}</div>
            <p className="text-sm text-muted-foreground">즉시 확인</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 border-purple-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              치명적
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.critical}</div>
            <p className="text-sm text-muted-foreground">긴급 조치</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>시스템 로그</CardTitle>
              <CardDescription>시스템 활동 및 오류 로그를 관리합니다</CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" disabled>
                <Download className="w-4 h-4 mr-2" />
                내보내기
              </Button>
              <Button size="sm" onClick={handleRefresh} disabled={loading}>
                {loading ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4 mr-2" />
                )}
                새로고침
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex gap-4">
            <Input
              placeholder="검색 (사용자, 작업, 메시지...)"
              className="max-w-sm"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Select value={filterLevel} onValueChange={setFilterLevel}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="로그 레벨" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체</SelectItem>
                <SelectItem value="info">정보</SelectItem>
                <SelectItem value="warning">경고</SelectItem>
                <SelectItem value="error">오류</SelectItem>
                <SelectItem value="critical">치명적</SelectItem>
                <SelectItem value="debug">디버그</SelectItem>
              </SelectContent>
            </Select>
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="기간" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">최근 1시간</SelectItem>
                <SelectItem value="24h">최근 24시간</SelectItem>
                <SelectItem value="7d">최근 7일</SelectItem>
                <SelectItem value="30d">최근 30일</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <>
              <div className="rounded-lg border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[180px]">시간</TableHead>
                      <TableHead className="w-[100px]">레벨</TableHead>
                      <TableHead className="w-[120px]">사용자</TableHead>
                      <TableHead className="w-[150px]">작업</TableHead>
                      <TableHead className="w-[130px]">IP 주소</TableHead>
                      <TableHead>메시지</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {logs.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={6} className="text-center text-muted-foreground py-8">
                          로그가 없습니다
                        </TableCell>
                      </TableRow>
                    ) : (
                      logs.map((log) => (
                        <TableRow key={log.id}>
                          <TableCell className="font-mono text-sm">
                            {formatTimestamp(log.timestamp)}
                          </TableCell>
                          <TableCell>{getLevelBadge(log.log_level)}</TableCell>
                          <TableCell>{log.username || '-'}</TableCell>
                          <TableCell>{log.action}</TableCell>
                          <TableCell className="font-mono text-sm">
                            {log.ip_address || '-'}
                          </TableCell>
                          <TableCell className="max-w-[400px] truncate" title={log.message}>
                            {log.message}
                          </TableCell>
                        </TableRow>
                      ))
                    )}
                  </TableBody>
                </Table>
              </div>

              <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
                <p>총 {logs.length}개의 로그 항목</p>
                <p>마지막 업데이트: {new Date().toLocaleString('ko-KR')}</p>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
