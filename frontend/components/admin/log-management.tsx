"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from "@/components/ui/scroll-area"
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
import { ScrollText, Filter, Download, RefreshCw, AlertCircle, Info, AlertTriangle, XCircle, Loader2, FileText } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"

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
  const [currentPage, setCurrentPage] = useState(1)
  const [totalLogs, setTotalLogs] = useState(0)
  const [pageSize, setPageSize] = useState(50)
  const { toast } = useToast()

  // Fetch logs from backend
  const fetchLogs = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('token')

      // Calculate hours based on time range
      const hoursMap: Record<string, number> = {
        '1h': 1,
        '24h': 24,
        '7d': 168,
        '30d': 720,
      }
      const hours = hoursMap[timeRange] || 24

      // Build query parameters with pagination
      const skip = (currentPage - 1) * pageSize
      const params = new URLSearchParams({
        hours: hours.toString(),
        skip: skip.toString(),
        limit: pageSize.toString(),
      })

      if (filterLevel !== 'all') {
        params.append('log_level', filterLevel.toUpperCase())
      }

      if (searchTerm) {
        params.append('search_term', searchTerm)
      }

      const response = await fetch(`/api/system-logs?${params}`, {
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
      setTotalLogs(data.total || 0)
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
      
      const hoursMap: Record<string, number> = {
        '1h': 1,
        '24h': 24,
        '7d': 168,
        '30d': 720,
      }
      const hours = hoursMap[timeRange] || 24

      const response = await fetch(
        `/api/system-logs/statistics?hours=${hours}`,
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

  // Load data on mount and when filters or page changes
  useEffect(() => {
    fetchLogs()
  }, [filterLevel, timeRange, currentPage, pageSize])

  // Fetch stats separately (doesn't need pagination)
  useEffect(() => {
    fetchStats()
  }, [timeRange])

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [filterLevel, timeRange, searchTerm])

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

  const getMethodBadge = (action: string) => {
    const method = action.split(' ')[0]
    const methodColors: Record<string, { bg: string; text: string }> = {
      GET: { bg: 'bg-tertiary-container', text: 'text-on-tertiary-container' },
      POST: { bg: 'bg-primary-container', text: 'text-on-primary-container' },
      PUT: { bg: 'bg-secondary-container', text: 'text-on-secondary-container' },
      DELETE: { bg: 'bg-error-container', text: 'text-on-error-container' },
      PATCH: { bg: 'bg-secondary-container', text: 'text-on-secondary-container' },
    }

    const colors = methodColors[method] || { bg: 'bg-surface-container-high', text: 'text-on-surface-variant' }

    return (
      <span className={`text-xs font-mono font-semibold px-2 py-0.5 rounded ${colors.bg} ${colors.text}`}>
        {method}
      </span>
    )
  }

  const formatAction = (action: string, message: string): string => {
    const parts = action.split(' ')
    const method = parts[0]
    let path = parts[1] || ''

    // 메시지에서 실제 경로 추출 (여러 형식 지원)
    // 형식 1: "Request to /api/v1/xxx completed with status 200"
    // 형식 2: "Request to /api/v1/xxx completed"
    // 형식 3: "/api/v1/xxx" (경로만 포함된 경우)
    let pathMatch = message.match(/Request\s+to\s+(\/api\/v\d+\/\S+?)(?:\s|$)/)
    if (!pathMatch) {
      // 더 간단한 패턴으로 재시도
      pathMatch = message.match(/(\/api\/v\d+\/[^\s"'\)]+)/)
    }

    if (pathMatch && pathMatch[1]) {
      // /api/v1/users/ -> /api/users
      // /api/v2/users -> /api/users
      path = pathMatch[1].replace(/\/v\d+/, '')
    }

    // 경로 끝의 / 제거 (정규화)
    path = path.replace(/\/+$/, '')

    // 정확한 패턴 매칭 (HTTP 메서드 + 경로)
    const exactMatch: Record<string, string> = {
      // 인증 관련
      'POST /api/auth/login': '인증/로그인',
      'POST /api/auth/login-json': '인증/로그인',
      'POST /api/auth/logout': '인증/로그아웃',
      'POST /api/auth/register': '인증/회원가입',
      'GET /api/auth/me': '인증/운용자 정보 조회',

      // 2D 데이터셋
      'GET /api/datasets-2d': '2D 데이터셋/목록 조회',
      'POST /api/datasets-2d': '2D 데이터셋/생성',
      'GET /api/datasets-2d/upload-history': '2D 데이터셋/업로드 히스토리',

      // 3D 데이터셋
      'GET /api/datasets-3d': '3D 데이터셋/목록 조회',
      'POST /api/datasets-3d': '3D 데이터셋/생성',
      'GET /api/datasets-3d/upload-history': '3D 데이터셋/업로드 히스토리',

      // 적대적 공격
      'POST /api/attacks/evasion/run': '적대적 공격/회피 공격 실행',
      'POST /api/attacks/poisoning/run': '적대적 공격/중독 공격 실행',
      'GET /api/attacks/results': '적대적 공격/결과 조회',
      'GET /api/attacks/history': '적대적 공격/히스토리 조회',

      // 적대적 패치
      'GET /api/adversarial-patches': '적대적 패치/목록 조회',
      'POST /api/adversarial-patches': '적대적 패치/생성',
      'GET /api/adversarial-patches/history': '적대적 패치/히스토리 조회',

      // 평가
      'GET /api/evaluations': '평가/목록 조회',
      'POST /api/evaluations': '평가/실행',
      'GET /api/evaluations/records': '평가/기록 조회',
      'GET /api/evaluations/history': '평가/히스토리 조회',

      // AI 모델
      'GET /api/ai-models': 'AI 모델/목록 조회',
      'POST /api/ai-models': 'AI 모델/등록',
      'GET /api/ai-models/details': 'AI 모델/상세 조회',

      // 3D 객체
      'GET /api/3d-objects': '3D 객체/목록 조회',
      'POST /api/3d-objects': '3D 객체/등록',

      // 관리자 - 운용자
      'GET /api/admin/users': '운용자 관리/목록 조회',
      'GET /api/users': '운용자 관리/목록 조회',
      'POST /api/admin/users': '운용자 관리/계정 생성',
      'POST /api/users': '운용자 관리/계정 생성',
      'GET /api/admin/users/statistics': '운용자 관리/통계 조회',

      // 관리자 - 백업
      'GET /api/admin/backups': '백업 관리/목록 조회',
      'POST /api/admin/backups': '백업 관리/백업 생성',

      // 시스템 로그
      'GET /api/system-logs': '시스템 로그/조회',
      'GET /api/system-logs/statistics': '시스템 로그/통계 조회',

      // 시스템 상태
      'GET /api/system/stats/stream': '시스템/실시간 통계',
      'GET /api/system/stats': '시스템/통계 조회',
      'GET /api/health': '시스템/상태 확인',
    }

    // HTTP 메서드와 경로 결합
    const fullAction = `${method} ${path}`

    // 정확히 일치하는 매핑 찾기
    if (exactMatch[fullAction]) {
      return exactMatch[fullAction]
    }

    // 동적 경로 패턴 매칭 (ID가 포함된 경로)
    // 백업 관리
    if (method === 'GET' && path.match(/^\/api\/admin\/backups\/[^/]+$/)) {
      return '백업 관리/상세 조회'
    }
    if (method === 'DELETE' && path.match(/^\/api\/admin\/backups\/[^/]+$/)) {
      return '백업 관리/백업 삭제'
    }
    if (method === 'POST' && path.includes('/backups/') && path.includes('/restore')) {
      return '백업 관리/복구 수행'
    }
    if (method === 'POST' && path.includes('/backups/') && path.includes('/export')) {
      return '백업 관리/Export'
    }
    if (method === 'POST' && path.includes('/backups/import')) {
      return '백업 관리/Import'
    }

    // 운용자 관리
    if (method === 'GET' && (path.match(/^\/api\/admin\/users\/[^/]+$/) || path.match(/^\/api\/users\/[^/]+$/))) {
      return '운용자 관리/상세 조회'
    }
    if (method === 'PUT' && (path.match(/^\/api\/admin\/users\/[^/]+$/) || path.match(/^\/api\/users\/[^/]+$/))) {
      return '운용자 관리/계정 수정'
    }
    if (method === 'DELETE' && (path.match(/^\/api\/admin\/users\/[^/]+$/) || path.match(/^\/api\/users\/[^/]+$/))) {
      return '운용자 관리/계정 삭제'
    }

    // 2D 데이터셋
    if (method === 'GET' && path.match(/^\/api\/datasets-2d\/[^/]+$/)) {
      return '2D 데이터셋/상세 조회'
    }
    if (method === 'PUT' && path.match(/^\/api\/datasets-2d\/[^/]+$/)) {
      return '2D 데이터셋/수정'
    }
    if (method === 'DELETE' && path.match(/^\/api\/datasets-2d\/[^/]+$/)) {
      return '2D 데이터셋/삭제'
    }
    if (method === 'POST' && path.includes('/datasets-2d/') && path.includes('/upload')) {
      return '2D 데이터셋/이미지 업로드'
    }

    // 3D 데이터셋
    if (method === 'GET' && path.match(/^\/api\/datasets-3d\/[^/]+$/)) {
      return '3D 데이터셋/상세 조회'
    }
    if (method === 'DELETE' && path.match(/^\/api\/datasets-3d\/[^/]+$/)) {
      return '3D 데이터셋/삭제'
    }
    if (method === 'POST' && path.includes('/datasets-3d/') && path.includes('/upload')) {
      return '3D 데이터셋/모델 업로드'
    }

    // 적대적 패치
    if (method === 'GET' && path.match(/^\/api\/adversarial-patches\/[^/]+$/)) {
      return '적대적 패치/상세 조회'
    }
    if (method === 'DELETE' && path.match(/^\/api\/adversarial-patches\/[^/]+$/)) {
      return '적대적 패치/삭제'
    }
    if (method === 'GET' && path.includes('/adversarial-patches/') && path.includes('/download')) {
      return '적대적 패치/다운로드'
    }

    // 평가
    if (method === 'GET' && path.match(/^\/api\/evaluations\/[^/]+$/)) {
      return '평가/결과 상세 조회'
    }
    if (method === 'DELETE' && path.match(/^\/api\/evaluations\/[^/]+$/)) {
      return '평가/삭제'
    }

    // AI 모델
    if (method === 'GET' && path.match(/^\/api\/ai-models\/[^/]+$/)) {
      return 'AI 모델/상세 조회'
    }
    if (method === 'DELETE' && path.match(/^\/api\/ai-models\/[^/]+$/)) {
      return 'AI 모델/삭제'
    }
    if (method === 'GET' && path.includes('/ai-models/') && path.includes('/download')) {
      return 'AI 모델/다운로드'
    }

    // 3D 객체
    if (method === 'GET' && path.match(/^\/api\/3d-objects\/[^/]+$/)) {
      return '3D 객체/상세 조회'
    }
    if (method === 'DELETE' && path.match(/^\/api\/3d-objects\/[^/]+$/)) {
      return '3D 객체/삭제'
    }

    // 기타 일반 패턴 (우선순위가 낮은 매칭)
    if (path.includes('/auth')) return '인증/기타'
    if (path.includes('/datasets-2d')) return '2D 데이터셋/기타'
    if (path.includes('/datasets-3d')) return '3D 데이터셋/기타'
    if (path.includes('/attacks')) return '적대적 공격/기타'
    if (path.includes('/adversarial-patches')) return '적대적 패치/기타'
    if (path.includes('/evaluations')) return '평가/기타'
    if (path.includes('/ai-models')) return 'AI 모델/기타'
    if (path.includes('/3d-objects')) return '3D 객체/기타'
    if (path.includes('/users')) return '운용자 관리/기타'
    if (path.includes('/backups')) return '백업 관리/기타'
    if (path.includes('/system-logs')) return '시스템 로그/기타'
    if (path.includes('/system')) return '시스템/기타'

    // 매칭되지 않은 경우
    // path가 비어있거나 /api만 있는 경우, 원본 message에서 힌트 찾기
    if (!path || path === '/api' || path === '') {
      // message에서 키워드로 추정
      const lowerMessage = message.toLowerCase()
      if (lowerMessage.includes('user')) return '운용자 관리/알 수 없음'
      if (lowerMessage.includes('auth') || lowerMessage.includes('login') || lowerMessage.includes('register')) return '인증/알 수 없음'
      if (lowerMessage.includes('backup')) return '백업 관리/알 수 없음'
      if (lowerMessage.includes('system') || lowerMessage.includes('stats')) return '시스템/알 수 없음'
      if (lowerMessage.includes('dataset')) return '데이터셋/알 수 없음'
      if (lowerMessage.includes('evaluation')) return '평가/알 수 없음'
      if (lowerMessage.includes('model')) return 'AI 모델/알 수 없음'
    }

    // 최종 fallback
    return `${method} ${path || action.split(' ')[1] || action}`
  }

  const handleRefresh = () => {
    fetchLogs()
    fetchStats()
    toast({
      title: "새로고침 완료",
      description: "로그 데이터가 업데이트되었습니다.",
    })
  }

  const handleExportCSV = () => {
    try {
      // CSV 헤더
      const headers = ['시간', '레벨', '운용자', '작업', 'IP 주소', '메시지', '상태 코드', '응답 시간(ms)']

      // CSV 데이터 행
      const rows = logs.map(log => [
        formatTimestamp(log.timestamp),
        log.log_level,
        log.username || '-',
        log.action,
        log.ip_address || '-',
        `"${log.message.replace(/"/g, '""')}"`, // 따옴표 이스케이프
        log.id,
        log.module || '-',
      ])

      // CSV 문자열 생성
      const csvContent = [
        headers.join(','),
        ...rows.map(row => row.join(','))
      ].join('\n')

      // BOM 추가 (한글 깨짐 방지)
      const BOM = '\uFEFF'
      const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8;' })

      // 다운로드
      const link = document.createElement('a')
      const url = URL.createObjectURL(blob)
      link.setAttribute('href', url)
      link.setAttribute('download', `system-logs-${new Date().toISOString().split('T')[0]}.csv`)
      link.style.visibility = 'hidden'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      toast({
        title: "내보내기 완료",
        description: `${logs.length}개의 로그를 CSV 파일로 저장했습니다.`,
      })
    } catch (error) {
      console.error('Error exporting CSV:', error)
      toast({
        title: "오류",
        description: "CSV 내보내기에 실패했습니다.",
        variant: "destructive",
      })
    }
  }

  const totalPages = Math.ceil(totalLogs / pageSize)
  const startIndex = (currentPage - 1) * pageSize + 1
  const endIndex = Math.min(currentPage * pageSize, totalLogs)

  return (
    <AdversarialToolLayout
      title="로그 관리"
      description="시스템 활동 및 오류 로그를 관리합니다"
      icon={ScrollText}
      headerStats={
        <Button
          onClick={handleRefresh}
          disabled={loading}
          className="ds-btn-outline"
        >
          {loading ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4 mr-2" />
          )}
          새로고침
        </Button>
      }
      rightPanel={{
        title: "시스템 로그",
        icon: FileText,
        description: "로그 목록 및 통계",
        children: (
          <div className="space-y-6">
            {/* 통계 카드 */}
            <Card className="bg-surface-container/50 border-border">
              <CardContent className="pt-6">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div className="bg-primary-container rounded-xl p-4 border border-primary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <ScrollText className="w-4 h-4 text-primary" />
                      </div>
                      <span className="text-on-primary-container/70 text-sm font-medium">전체 로그</span>
                    </div>
                    <div className="text-3xl font-bold text-on-primary-container">{stats.total}</div>
                    <p className="text-xs text-on-primary-container/60">
                      {timeRange === '1h' ? '최근 1시간' :
                       timeRange === '24h' ? '최근 24시간' :
                       timeRange === '7d' ? '최근 7일' : '최근 30일'}
                    </p>
                  </div>
                  <div className="bg-tertiary-container rounded-xl p-4 border border-tertiary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-tertiary/10 rounded-lg">
                        <Info className="w-4 h-4 text-tertiary" />
                      </div>
                      <span className="text-on-tertiary-container/70 text-sm font-medium">정보</span>
                    </div>
                    <div className="text-3xl font-bold text-on-tertiary-container">{stats.info}</div>
                    <p className="text-xs text-on-tertiary-container/60">정상 활동</p>
                  </div>
                  <div className="bg-secondary-container rounded-xl p-4 border border-secondary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-secondary/10 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-secondary" />
                      </div>
                      <span className="text-on-secondary-container/70 text-sm font-medium">경고</span>
                    </div>
                    <div className="text-3xl font-bold text-on-secondary-container">{stats.warning}</div>
                    <p className="text-xs text-on-secondary-container/60">주의 필요</p>
                  </div>
                  <div className="bg-error-container rounded-xl p-4 border border-error/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-error/10 rounded-lg">
                        <XCircle className="w-4 h-4 text-error" />
                      </div>
                      <span className="text-on-error-container/70 text-sm font-medium">오류</span>
                    </div>
                    <div className="text-3xl font-bold text-on-error-container">{stats.error}</div>
                    <p className="text-xs text-on-error-container/60">즉시 확인</p>
                  </div>
                  <div className="bg-error-container rounded-xl p-4 border border-error/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-error/10 rounded-lg">
                        <AlertCircle className="w-4 h-4 text-error" />
                      </div>
                      <span className="text-on-error-container/70 text-sm font-medium">치명적</span>
                    </div>
                    <div className="text-3xl font-bold text-on-error-container">{stats.critical}</div>
                    <p className="text-xs text-on-error-container/60">긴급 조치</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 로그 목록 카드 */}
            <Card className="bg-surface-container/50 border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-primary">로그 목록</CardTitle>
                <CardDescription className="text-muted">시스템 활동 로그 기록</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-4 flex gap-4">
                  <Input
                    placeholder="검색 (운용자, 작업, 메시지...)"
                    className="max-w-sm bg-surface-container-high border-outline text-foreground"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                  <Select value={filterLevel} onValueChange={setFilterLevel}>
                    <SelectTrigger className="w-[180px] bg-surface-container-high border-outline text-foreground">
                      <SelectValue placeholder="로그 레벨" />
                    </SelectTrigger>
                    <SelectContent className="bg-surface-container border-outline">
                      <SelectItem value="all" className="text-foreground">전체</SelectItem>
                      <SelectItem value="info" className="text-foreground">정보</SelectItem>
                      <SelectItem value="warning" className="text-foreground">경고</SelectItem>
                      <SelectItem value="error" className="text-foreground">오류</SelectItem>
                      <SelectItem value="critical" className="text-foreground">치명적</SelectItem>
                      <SelectItem value="debug" className="text-foreground">디버그</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={timeRange} onValueChange={setTimeRange}>
                    <SelectTrigger className="w-[180px] bg-surface-container-high border-outline text-foreground">
                      <SelectValue placeholder="기간" />
                    </SelectTrigger>
                    <SelectContent className="bg-surface-container border-outline">
                      <SelectItem value="1h" className="text-foreground">최근 1시간</SelectItem>
                      <SelectItem value="24h" className="text-foreground">최근 24시간</SelectItem>
                      <SelectItem value="7d" className="text-foreground">최근 7일</SelectItem>
                      <SelectItem value="30d" className="text-foreground">최근 30일</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {loading ? (
                  <div className="flex items-center justify-center h-64">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                  </div>
                ) : (
                  <>
                    <ScrollArea className="h-[600px]">
                      <Table>
                    <TableHeader className="sticky top-0 bg-surface-container z-10">
                      <TableRow className="border-border">
                        <TableHead className="w-[180px] bg-surface-container-high text-muted">시간</TableHead>
                        <TableHead className="w-[100px] bg-surface-container-high text-muted">레벨</TableHead>
                        <TableHead className="w-[120px] bg-surface-container-high text-muted">운용자</TableHead>
                        <TableHead className="w-[220px] bg-surface-container-high text-muted">작업</TableHead>
                        <TableHead className="w-[130px] bg-surface-container-high text-muted">IP 주소</TableHead>
                        <TableHead className="bg-surface-container-high text-muted">메시지</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {logs.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={6} className="text-center text-muted py-8">
                            로그가 없습니다
                          </TableCell>
                        </TableRow>
                      ) : (
                        logs.map((log) => (
                          <TableRow key={log.id} className="border-border">
                            <TableCell className="font-mono text-sm text-muted">
                              {formatTimestamp(log.timestamp)}
                            </TableCell>
                            <TableCell>{getLevelBadge(log.log_level)}</TableCell>
                            <TableCell className="text-muted">{log.username || '-'}</TableCell>
                            <TableCell className="text-foreground" title={`원본: ${log.action}\n메시지: ${log.message}`}>
                              <div className="flex items-center gap-2">
                                {getMethodBadge(log.action)}
                                <span className="font-medium">
                                  {formatAction(log.action, log.message).includes('/') ? (
                                    <>
                                      <span className="text-muted">{formatAction(log.action, log.message).split('/')[0]}</span>
                                      <span className="text-muted mx-1">/</span>
                                      <span className="text-foreground">{formatAction(log.action, log.message).split('/')[1]}</span>
                                    </>
                                  ) : (
                                    formatAction(log.action, log.message)
                                  )}
                                </span>
                              </div>
                            </TableCell>
                            <TableCell className="font-mono text-sm text-muted">
                              {log.ip_address || '-'}
                            </TableCell>
                            <TableCell className="max-w-[400px] truncate text-foreground" title={log.message}>
                              {log.message}
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                    </ScrollArea>

                    {/* Pagination */}
                    <div className="mt-4 flex items-center justify-between">
                <div className="text-sm text-muted">
                  <p>전체 {totalLogs.toLocaleString()}개 중 {startIndex.toLocaleString()}-{endIndex.toLocaleString()}개 표시</p>
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    onClick={() => setCurrentPage(1)}
                    disabled={currentPage === 1}
                    className="ds-btn-outline"
                  >
                    처음
                  </Button>
                  <Button
                    onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                    disabled={currentPage === 1}
                    className="ds-btn-outline"
                  >
                    이전
                  </Button>

                  <span className="text-sm text-muted px-3">
                    페이지 {currentPage} / {totalPages}
                  </span>

                  <Button
                    onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                    disabled={currentPage === totalPages}
                    className="ds-btn-outline"
                  >
                    다음
                  </Button>
                  <Button
                    onClick={() => setCurrentPage(totalPages)}
                    disabled={currentPage === totalPages}
                    className="ds-btn-outline"
                  >
                    마지막
                  </Button>

                  <Select value={pageSize.toString()} onValueChange={(val) => setPageSize(Number(val))}>
                    <SelectTrigger className="w-[100px] bg-surface-container-high border-outline text-foreground">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-surface-container border-outline">
                      <SelectItem value="10" className="text-foreground">10개</SelectItem>
                      <SelectItem value="25" className="text-foreground">25개</SelectItem>
                      <SelectItem value="50" className="text-foreground">50개</SelectItem>
                      <SelectItem value="100" className="text-foreground">100개</SelectItem>
                    </SelectContent>
                  </Select>
                    </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        )
      }}
    />
  )
}
