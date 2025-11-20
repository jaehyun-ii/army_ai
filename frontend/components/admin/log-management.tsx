"use client"

import { useState } from 'react'
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
import { ScrollText, Filter, Download, RefreshCw, AlertCircle, Info, AlertTriangle, XCircle } from 'lucide-react'

interface LogEntry {
  id: string
  timestamp: string
  level: 'info' | 'warning' | 'error' | 'debug'
  user: string
  action: string
  ip: string
  details: string
}

export function LogManagement() {
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: '1',
      timestamp: '2024-01-20 10:35:22',
      level: 'info',
      user: 'admin',
      action: '로그인',
      ip: '192.168.1.100',
      details: '관리자 계정 로그인 성공'
    },
    {
      id: '2',
      timestamp: '2024-01-20 10:34:15',
      level: 'warning',
      user: 'user123',
      action: '모델 학습',
      ip: '192.168.1.105',
      details: 'GPU 메모리 사용량 90% 초과'
    },
    {
      id: '3',
      timestamp: '2024-01-20 10:33:08',
      level: 'error',
      user: 'system',
      action: '백업 실패',
      ip: 'localhost',
      details: '디스크 공간 부족으로 백업 실패'
    },
    {
      id: '4',
      timestamp: '2024-01-20 10:32:45',
      level: 'info',
      user: 'user456',
      action: '데이터 업로드',
      ip: '192.168.1.112',
      details: '2D 이미지 데이터셋 업로드 완료 (500개)'
    },
    {
      id: '5',
      timestamp: '2024-01-20 10:31:20',
      level: 'debug',
      user: 'developer',
      action: 'API 호출',
      ip: '192.168.1.120',
      details: 'GET /api/models - 응답시간: 245ms'
    },
    {
      id: '6',
      timestamp: '2024-01-20 10:30:10',
      level: 'warning',
      user: 'user789',
      action: '로그인 실패',
      ip: '192.168.1.130',
      details: '비밀번호 3회 연속 실패'
    }
  ])

  const [filterLevel, setFilterLevel] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'info':
        return <Info className="w-4 h-4" />
      case 'warning':
        return <AlertTriangle className="w-4 h-4" />
      case 'error':
        return <XCircle className="w-4 h-4" />
      case 'debug':
        return <AlertCircle className="w-4 h-4" />
      default:
        return null
    }
  }

  const getLevelBadge = (level: string) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      info: 'default',
      warning: 'secondary',
      error: 'destructive',
      debug: 'outline'
    }

    return (
      <Badge variant={variants[level] || 'default'} className="flex items-center gap-1">
        {getLevelIcon(level)}
        {level.toUpperCase()}
      </Badge>
    )
  }

  const filteredLogs = logs.filter(log => {
    if (filterLevel !== 'all' && log.level !== filterLevel) return false
    if (searchTerm && !Object.values(log).some(val =>
      val.toLowerCase().includes(searchTerm.toLowerCase())
    )) return false
    return true
  })

  const logStats = {
    total: logs.length,
    info: logs.filter(l => l.level === 'info').length,
    warning: logs.filter(l => l.level === 'warning').length,
    error: logs.filter(l => l.level === 'error').length
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-slate-900/20 to-slate-800/20 border-slate-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <ScrollText className="w-5 h-5" />
              전체 로그
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{logStats.total}</div>
            <p className="text-sm text-muted-foreground">최근 24시간</p>
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
            <div className="text-3xl font-bold">{logStats.info}</div>
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
            <div className="text-3xl font-bold">{logStats.warning}</div>
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
            <div className="text-3xl font-bold">{logStats.error}</div>
            <p className="text-sm text-muted-foreground">즉시 확인</p>
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
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                내보내기
              </Button>
              <Button size="sm">
                <RefreshCw className="w-4 h-4 mr-2" />
                새로고침
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex gap-4">
            <Input
              placeholder="검색 (사용자, 작업, IP...)"
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
                <SelectItem value="debug">디버그</SelectItem>
              </SelectContent>
            </Select>
            <Select defaultValue="24h">
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

          <div className="rounded-lg border overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[150px]">시간</TableHead>
                  <TableHead className="w-[100px]">레벨</TableHead>
                  <TableHead>사용자</TableHead>
                  <TableHead>작업</TableHead>
                  <TableHead>IP 주소</TableHead>
                  <TableHead>상세 정보</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredLogs.map((log) => (
                  <TableRow key={log.id}>
                    <TableCell className="font-mono text-sm">{log.timestamp}</TableCell>
                    <TableCell>{getLevelBadge(log.level)}</TableCell>
                    <TableCell>{log.user}</TableCell>
                    <TableCell>{log.action}</TableCell>
                    <TableCell className="font-mono text-sm">{log.ip}</TableCell>
                    <TableCell className="max-w-[300px] truncate" title={log.details}>
                      {log.details}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
            <p>총 {filteredLogs.length}개의 로그 항목</p>
            <p>마지막 업데이트: 2024-01-20 10:35:30</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}