"use client"

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Database, Download, Upload, RefreshCw, AlertTriangle, CheckCircle, Clock, HardDrive } from 'lucide-react'
import { Progress } from '@/components/ui/progress'

interface Backup {
  id: string
  name: string
  size: string
  date: string
  type: 'manual' | 'scheduled'
  status: 'success' | 'failed' | 'in-progress'
}

export function DBBackupRestore() {
  const [backups, setBackups] = useState<Backup[]>([
    {
      id: '1',
      name: 'backup_2024_01_20_0300.sql',
      size: '256 MB',
      date: '2024-01-20 03:00',
      type: 'scheduled',
      status: 'success'
    },
    {
      id: '2',
      name: 'backup_2024_01_19_1500.sql',
      size: '248 MB',
      date: '2024-01-19 15:00',
      type: 'manual',
      status: 'success'
    },
    {
      id: '3',
      name: 'backup_2024_01_19_0300.sql',
      size: '245 MB',
      date: '2024-01-19 03:00',
      type: 'scheduled',
      status: 'success'
    },
    {
      id: '4',
      name: 'backup_2024_01_18_0300.sql',
      size: '242 MB',
      date: '2024-01-18 03:00',
      type: 'scheduled',
      status: 'failed'
    }
  ])

  const [isBackingUp, setIsBackingUp] = useState(false)
  const [backupProgress, setBackupProgress] = useState(0)

  const getStatusBadge = (status: string) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive'> = {
      success: 'default',
      'in-progress': 'secondary',
      failed: 'destructive'
    }
    const labels: Record<string, string> = {
      success: '성공',
      'in-progress': '진행중',
      failed: '실패'
    }
    return <Badge variant={variants[status] || 'default'}>{labels[status] || status}</Badge>
  }

  const getTypeBadge = (type: string) => {
    const labels: Record<string, string> = {
      manual: '수동',
      scheduled: '자동'
    }
    return <Badge variant="outline">{labels[type] || type}</Badge>
  }

  const handleBackup = () => {
    setIsBackingUp(true)
    setBackupProgress(0)

    const interval = setInterval(() => {
      setBackupProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsBackingUp(false)
          return 100
        }
        return prev + 10
      })
    }, 500)
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-blue-900/20 to-blue-800/20 border-blue-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Database className="w-5 h-5" />
              총 백업
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{backups.length}</div>
            <p className="text-sm text-muted-foreground">저장된 백업</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-900/20 to-green-800/20 border-green-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <CheckCircle className="w-5 h-5" />
              성공
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {backups.filter(b => b.status === 'success').length}
            </div>
            <p className="text-sm text-muted-foreground">정상 백업</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-orange-900/20 to-orange-800/20 border-orange-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <HardDrive className="w-5 h-5" />
              사용량
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">991</div>
            <p className="text-sm text-muted-foreground">MB</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 border-purple-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Clock className="w-5 h-5" />
              다음 백업
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">03:00</div>
            <p className="text-sm text-muted-foreground">매일 자동</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>데이터베이스 백업/복구</CardTitle>
              <CardDescription>시스템 데이터베이스를 백업하고 복구합니다</CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={handleBackup}
                disabled={isBackingUp}
              >
                {isBackingUp ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    백업 진행중...
                  </>
                ) : (
                  <>
                    <Database className="w-4 h-4 mr-2" />
                    즉시 백업
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isBackingUp && (
            <Alert className="mb-4">
              <AlertTitle>백업 진행중</AlertTitle>
              <AlertDescription>
                <Progress value={backupProgress} className="mt-2" />
                <p className="text-sm mt-2">{backupProgress}% 완료</p>
              </AlertDescription>
            </Alert>
          )}

          <div className="mb-4">
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>자동 백업 설정</AlertTitle>
              <AlertDescription>
                매일 오전 3시에 자동으로 백업이 실행됩니다. 최근 7일간의 백업이 보관됩니다.
              </AlertDescription>
            </Alert>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>백업 파일</TableHead>
                <TableHead>크기</TableHead>
                <TableHead>날짜/시간</TableHead>
                <TableHead>타입</TableHead>
                <TableHead>상태</TableHead>
                <TableHead className="text-right">작업</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {backups.map((backup) => (
                <TableRow key={backup.id}>
                  <TableCell className="font-medium">{backup.name}</TableCell>
                  <TableCell>{backup.size}</TableCell>
                  <TableCell>{backup.date}</TableCell>
                  <TableCell>{getTypeBadge(backup.type)}</TableCell>
                  <TableCell>{getStatusBadge(backup.status)}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-1">
                      <Button variant="ghost" size="sm">
                        <Download className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={backup.status !== 'success'}
                      >
                        <Upload className="w-4 h-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          <div className="mt-6 p-4 bg-muted/50 rounded-lg">
            <h3 className="font-medium mb-2">복구 주의사항</h3>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• 백업 복구 시 현재 데이터베이스의 모든 데이터가 백업 시점으로 되돌아갑니다</li>
              <li>• 복구 전 현재 데이터베이스를 백업하는 것을 권장합니다</li>
              <li>• 복구 작업은 시스템 사용이 적은 시간에 수행하시기 바랍니다</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}