"use client"

import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { useToast } from '@/hooks/use-toast'
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Database, Download, RefreshCw, AlertTriangle, CheckCircle, Clock, HardDrive, Trash2, RotateCcw, Upload, FileArchive } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import { useOperation } from '@/contexts/OperationContext'

interface Backup {
  id: string
  name: string
  size: string
  date: string
  type: 'manual' | 'scheduled'
  status: 'success' | 'failed' | 'in-progress'
  total_size_bytes?: number
  restored_at?: string
}

interface BackupListResponse {
  backups: Backup[]
  total: number
  total_size_bytes: number
  total_size: string
}

export function DBBackupRestore() {
  const { toast } = useToast()
  const { setOperationInProgress: setGlobalOperation } = useOperation()
  const [backups, setBackups] = useState<Backup[]>([])
  const [totalSize, setTotalSize] = useState<string>('0 MB')
  const [isBackingUp, setIsBackingUp] = useState(false)
  const [isRestoring, setIsRestoring] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [isImportDialogOpen, setIsImportDialogOpen] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [zipFile, setZipFile] = useState<File | null>(null)
  const [importBackupName, setImportBackupName] = useState('')
  const backupPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Disable all interactions when backing up or restoring
  const isOperationInProgress = isBackingUp || isRestoring

  // Update global operation state whenever local state changes
  useEffect(() => {
    if (isBackingUp) {
      setGlobalOperation(true, 'DB 백업')
    } else if (isRestoring) {
      setGlobalOperation(true, 'DB 복구')
    } else {
      setGlobalOperation(false)
    }
  }, [isBackingUp, isRestoring, setGlobalOperation])

  // Load backups on mount
  useEffect(() => {
    loadBackups()
  }, [])

  // Auto-refresh backups every 5 seconds when backing up
  useEffect(() => {
    if (!isBackingUp) return

    const interval = setInterval(() => {
      loadBackups()
    }, 5000)

    return () => clearInterval(interval)
  }, [isBackingUp])

  const loadBackups = async (): Promise<{ ok: boolean; inProgress: boolean }> => {
    try {
      const response = await fetch('/api/admin/backups')
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to load backups')
      }

      const data: BackupListResponse = await response.json()
      setBackups(data.backups || [])
      setTotalSize(data.total_size || '0 MB')

      // Check if any backup is in progress
      const inProgress = data.backups.some(b => b.status === 'in-progress')
      setIsBackingUp(inProgress)
      return { ok: true, inProgress }
    } catch (error) {
      console.error('Failed to load backups:', error)
      toast({
        variant: "destructive",
        title: "로딩 실패",
        description: `백업 목록을 불러오는데 실패했습니다: ${error}`
      })
      return { ok: false, inProgress: false }
    } finally {
      setIsLoading(false)
    }
  }

  const handleBackup = async () => {
    try {
      setIsBackingUp(true)
      if (backupPollRef.current) {
        clearInterval(backupPollRef.current)
        backupPollRef.current = null
      }

      const response = await fetch('/api/admin/backups', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ type: 'manual' })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to create backup')
      }

      toast({
        title: "백업 시작",
        description: "백업이 시작되었습니다. 완료까지 수 분이 소요될 수 있습니다."
      })

      // Reload backups to show the new in-progress backup
      await loadBackups()

      // Poll until 백업 상태가 in-progress에서 벗어날 때까지 리스트 갱신
      backupPollRef.current = setInterval(async () => {
        const { ok, inProgress } = await loadBackups()
        if (!ok) return

        if (!inProgress) {
          if (backupPollRef.current) {
            clearInterval(backupPollRef.current)
            backupPollRef.current = null
          }
          setIsBackingUp(false)
        }
      }, 3000)
    } catch (error) {
      console.error('Failed to create backup:', error)
      toast({
        variant: "destructive",
        title: "백업 실패",
        description: `백업 생성에 실패했습니다: ${error}`
      })
      setIsBackingUp(false)
    }
  }

  const handleRestore = async (backupId: string) => {
    const confirmed = confirm(
      '⚠️ 경고: 복구를 진행하면 다음과 같은 변경이 발생합니다:\n\n' +
      '1. 데이터베이스가 백업 시점으로 완전히 되돌아갑니다\n' +
      '2. 기존 스토리지(/storage)가 삭제되고 백업 시점의 스토리지로 대체됩니다\n' +
      '3. 복구 과정은 되돌릴 수 없으며, 현재 데이터는 영구적으로 손실됩니다\n\n' +
      '복구 전에 현재 데이터베이스를 백업하는 것을 권장합니다.\n\n' +
      '정말로 복구하시겠습니까?'
    )

    if (!confirmed) return

    try {
      setIsRestoring(true)

      const response = await fetch(`/api/admin/backups/${backupId}/restore`, {
        method: 'POST'
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to restore backup')
      }

      toast({
        title: "복구 시작",
        description: "백업 복구가 시작되었습니다. 완료까지 수 분이 소요될 수 있습니다."
      })

      // Poll for restore completion (check every 3 seconds for 5 minutes max)
      let attempts = 0
      const maxAttempts = 100 // 5 minutes
      const pollInterval = setInterval(async () => {
        attempts++

        try {
          // Try to fetch backups to check if restore is complete
          const checkResponse = await fetch('/api/admin/backups')
          if (checkResponse.ok) {
            // Restore seems complete - reload the page
            clearInterval(pollInterval)
            toast({
              title: "복구 완료",
              description: "백업 복구가 완료되었습니다. 페이지를 새로고침합니다."
            })
            setTimeout(() => {
              window.location.reload()
            }, 1000)
          }
        } catch (error) {
          // Backend might be restarting - continue polling
          if (attempts >= maxAttempts) {
            clearInterval(pollInterval)
            setIsRestoring(false)
            toast({
              variant: "destructive",
              title: "복구 타임아웃",
              description: "복구 완료를 확인할 수 없습니다. 수동으로 페이지를 새로고침하세요."
            })
          }
        }
      }, 3000)
    } catch (error) {
      console.error('Failed to restore backup:', error)
      toast({
        variant: "destructive",
        title: "복구 실패",
        description: `백업 복구에 실패했습니다: ${error}`
      })
      setIsRestoring(false)
    }
  }

  const handleDelete = async (backupId: string) => {
    const confirmed = confirm('이 백업을 삭제하시겠습니까?')
    if (!confirmed) return

    try {
      const response = await fetch(`/api/admin/backups/${backupId}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to delete backup')
      }

      toast({
        title: "삭제 완료",
        description: "백업이 성공적으로 삭제되었습니다."
      })

      await loadBackups()
    } catch (error) {
      console.error('Failed to delete backup:', error)
      toast({
        variant: "destructive",
        title: "삭제 실패",
        description: `백업 삭제에 실패했습니다: ${error}`
      })
    }
  }

  const handleExport = async (backupId: string, backupName: string) => {
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        throw new Error('인증 토큰이 없습니다. 다시 로그인 후 시도해 주세요.')
      }

      toast({
        title: "Export 시작",
        description: "백업 파일을 다운로드하는 중..."
      })

      const response = await fetch(`/api/admin/backups/${backupId}/export`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })

      if (!response.ok) {
        const error = await response.json().catch(async () => {
          // try text for debugging
          const text = await response.text()
          return { detail: text || 'Export failed' }
        })
        throw new Error(error.detail || error.error || 'Failed to export backup')
      }

      // Get filename from Content-Disposition header
      const contentDisposition = response.headers.get('content-disposition')
      let filename = `backup_export_${backupName}.zip`

      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?([^"]+)"?/)
        if (match && match[1]) {
          filename = match[1]
        }
      }

      // Download the file
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.style.display = 'none'

      document.body.appendChild(link)
      link.click()

      // Cleanup
      setTimeout(() => {
        window.URL.revokeObjectURL(url)
        document.body.removeChild(link)
      }, 100)

      toast({
        title: "Export 완료",
        description: `백업 파일이 다운로드되었습니다: ${filename}`
      })
    } catch (error) {
      console.error('Failed to export backup:', error)
      toast({
        variant: "destructive",
        title: "Export 실패",
        description: `백업 Export에 실패했습니다: ${error instanceof Error ? error.message : String(error)}`
      })
    }
  }

  const handleImport = async () => {
    if (!zipFile) {
      toast({
        variant: "destructive",
        title: "파일 누락",
        description: "ZIP 파일을 선택해주세요."
      })
      return
    }

    try {
      setIsImporting(true)

      const formData = new FormData()
      formData.append('zip_file', zipFile)
      if (importBackupName) {
        formData.append('backup_name', importBackupName)
      }

      toast({
        title: "Import 시작",
        description: "백업 파일을 업로드하는 중..."
      })

      const response = await fetch('/api/admin/backups/import', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || 'Failed to import backup')
      }

      const data = await response.json()

      toast({
        title: "Import 완료",
        description: `백업이 '${data.backup_name}'으로 성공적으로 Import되었습니다.`
      })

      // Reset form
      setZipFile(null)
      setImportBackupName('')
      setIsImportDialogOpen(false)

      // Reload backups
      await loadBackups()
    } catch (error) {
      console.error('Failed to import backup:', error)
      toast({
        variant: "destructive",
        title: "Import 실패",
        description: `백업 Import에 실패했습니다: ${error}`
      })
    } finally {
      setIsImporting(false)
    }
  }
  // Cleanup poller and global state on unmount
  useEffect(() => {
    return () => {
      if (backupPollRef.current) clearInterval(backupPollRef.current)
      setGlobalOperation(false) // Clear global state on unmount
    }
  }, [setGlobalOperation])

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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-6 h-6 animate-spin mr-2" />
        <span>로딩 중...</span>
      </div>
    )
  }

  const successBackups = backups.filter(b => b.status === 'success')
  const restoredBackups = backups.filter(b => b.restored_at).sort((a, b) => {
    if (!a.restored_at || !b.restored_at) return 0
    return new Date(b.restored_at).getTime() - new Date(a.restored_at).getTime()
  })
  const latestRestore = restoredBackups.length > 0 ? restoredBackups[0] : null

  return (
    <AdversarialToolLayout
      title="DB 백업 및 복구"
      description="시스템 데이터베이스와 스토리지를 백업하고 복구합니다"
      icon={Database}
      headerStats={
        <div className="flex gap-2">
          <Button
            onClick={loadBackups}
            disabled={isOperationInProgress}
            className="ds-btn-outline"
          >
            <RefreshCw className={`w-4 h-4 ${isOperationInProgress ? 'animate-spin' : ''}`} />
            새로고침
          </Button>
          <Button
            onClick={handleBackup}
            disabled={isOperationInProgress}
            className="ds-btn-primary"
          >
            {isBackingUp ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                백업 진행중...
              </>
            ) : isRestoring ? (
              <>
                <RotateCcw className="w-4 h-4 mr-2 animate-pulse" />
                복구 진행중...
              </>
            ) : (
              <>
                <Database className="w-4 h-4 mr-2" />
                즉시 백업
              </>
            )}
          </Button>
        </div>
      }
      rightPanel={{
        title: "백업 관리",
        icon: HardDrive,
        description: "백업 목록 및 복구 관리",
        children: (
          <div className="space-y-6">
            {/* 통계 카드 */}
            <Card className="bg-surface-container/50 border-border">
              <CardContent className="pt-6">
                <div className={`grid grid-cols-2 md:grid-cols-5 gap-4 ${isOperationInProgress ? 'opacity-70' : ''} transition-opacity`}>
                  <div className="bg-primary-container rounded-xl p-4 border border-primary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <Database className="w-4 h-4 text-primary" />
                      </div>
                      <span className="text-on-primary-container/70 text-sm font-medium">총 백업</span>
                    </div>
                    <div className="text-3xl font-bold text-on-primary-container">{backups.length}</div>
                    <p className="text-xs text-on-primary-container/60">저장된 백업</p>
                  </div>
                  <div className="bg-tertiary-container rounded-xl p-4 border border-tertiary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-tertiary/10 rounded-lg">
                        <CheckCircle className="w-4 h-4 text-tertiary" />
                      </div>
                      <span className="text-on-tertiary-container/70 text-sm font-medium">성공</span>
                    </div>
                    <div className="text-3xl font-bold text-on-tertiary-container">{successBackups.length}</div>
                    <p className="text-xs text-on-tertiary-container/60">정상 백업</p>
                  </div>
                  <div className="bg-secondary-container rounded-xl p-4 border border-secondary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-secondary/10 rounded-lg">
                        <HardDrive className="w-4 h-4 text-secondary" />
                      </div>
                      <span className="text-on-secondary-container/70 text-sm font-medium">사용량</span>
                    </div>
                    <div className="text-3xl font-bold text-on-secondary-container">{totalSize}</div>
                    <p className="text-xs text-on-secondary-container/60">총 용량</p>
                  </div>
                  <div className="bg-primary-container rounded-xl p-4 border border-primary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <Clock className="w-4 h-4 text-primary" />
                      </div>
                      <span className="text-on-primary-container/70 text-sm font-medium">최근 백업</span>
                    </div>
                    <div className="text-2xl font-bold text-on-primary-container">
                      {successBackups.length > 0 ? successBackups[0].date.split(' ')[1] : 'N/A'}
                    </div>
                    <p className="text-xs text-on-primary-container/60">
                      {successBackups.length > 0 ? successBackups[0].date.split(' ')[0] : '백업 없음'}
                    </p>
                  </div>
                  <div className="bg-tertiary-container rounded-xl p-4 border border-tertiary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-tertiary/10 rounded-lg">
                        <RotateCcw className="w-4 h-4 text-tertiary" />
                      </div>
                      <span className="text-on-tertiary-container/70 text-sm font-medium">최근 복구</span>
                    </div>
                    <div className="text-lg font-bold truncate text-on-tertiary-container" title={latestRestore?.name || '복구 이력 없음'}>
                      {latestRestore?.name || 'N/A'}
                    </div>
                    <p className="text-xs text-on-tertiary-container/60">
                      {latestRestore?.restored_at
                        ? new Date(latestRestore.restored_at).toLocaleString('ko-KR', {
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            hour12: false
                          })
                        : '복구 이력 없음'}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 백업 목록 카드 */}
            <Card className="bg-surface-container/50 border-border">
              <CardHeader className="pb-3">
                <div className="flex justify-between items-center">
                  <div>
                    <CardTitle className="text-primary">백업 목록</CardTitle>
                    <CardDescription className="text-muted">데이터베이스 백업 파일 목록</CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Dialog open={isImportDialogOpen} onOpenChange={setIsImportDialogOpen}>
                      <DialogTrigger asChild>
                        <Button
                          disabled={isOperationInProgress}
                          className="ds-btn-outline"
                        >
                          <Upload className="w-4 h-4 mr-2" />
                          Import
                        </Button>
                      </DialogTrigger>
                <DialogContent className="sm:max-w-[525px]">
                  <DialogHeader>
                    <DialogTitle>백업 Import</DialogTitle>
                    <DialogDescription>
                      Export된 ZIP 백업 파일을 업로드하여 시스템에 추가합니다.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="grid gap-4 py-4">
                    <Alert>
                      <FileArchive className="h-4 w-4" />
                      <AlertTitle>ZIP 파일 형식</AlertTitle>
                      <AlertDescription>
                        Export 버튼으로 다운로드한 ZIP 파일만 업로드할 수 있습니다.
                        ZIP 파일에는 데이터베이스와 스토리지 백업이 포함되어 있습니다.
                      </AlertDescription>
                    </Alert>
                    <div className="grid gap-2">
                      <Label htmlFor="backup-name">
                        백업 이름 (선택사항)
                      </Label>
                      <Input
                        id="backup-name"
                        placeholder="이름을 입력하지 않으면 원본 이름이 사용됩니다"
                        value={importBackupName}
                        onChange={(e) => setImportBackupName(e.target.value)}
                        disabled={isImporting}
                      />
                      <p className="text-sm text-muted-foreground">
                        자동으로 'export_' 접두사가 추가됩니다
                      </p>
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="zip-file">
                        백업 ZIP 파일 *
                      </Label>
                      <Input
                        id="zip-file"
                        type="file"
                        accept=".zip"
                        onChange={(e) => setZipFile(e.target.files?.[0] || null)}
                        disabled={isImporting}
                      />
                      {zipFile && (
                        <div className="p-3 bg-tertiary-container border border-tertiary rounded-md">
                          <p className="text-sm text-tertiary flex items-center gap-2">
                            <FileArchive className="w-4 h-4" />
                            <span className="font-medium">{zipFile.name}</span>
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            크기: {(zipFile.size / (1024 * 1024)).toFixed(2)} MB
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                  <DialogFooter>
                    <Button
                      onClick={() => setIsImportDialogOpen(false)}
                      disabled={isImporting}
                      className="ds-btn-outline"
                    >
                      취소
                    </Button>
                    <Button
                      onClick={handleImport}
                      disabled={isImporting || !zipFile}
                      className="ds-btn-primary"
                    >
                      {isImporting ? (
                        <>
                          <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                          Import 중...
                        </>
                      ) : (
                        <>
                          <Upload className="w-4 h-4 mr-2" />
                          Import
                        </>
                      )}
                    </Button>
                  </DialogFooter>
                    </DialogContent>
                    </Dialog>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {isBackingUp && (
                  <Alert className="mb-4">
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <AlertTitle>백업 진행중</AlertTitle>
                    <AlertDescription>
                      백업이 진행 중입니다. 완료까지 수 분이 소요될 수 있습니다. 모든 작업이 일시적으로 비활성화됩니다.
                    </AlertDescription>
                  </Alert>
                )}

                {isRestoring && (
                  <Alert className="mb-4 border-orange-500/50 bg-orange-900/20">
                    <RotateCcw className="h-4 w-4 animate-pulse" />
                    <AlertTitle>복구 진행중</AlertTitle>
                    <AlertDescription>
                      백업 복구가 진행 중입니다. 완료까지 수 분이 소요될 수 있습니다. 모든 작업이 일시적으로 비활성화됩니다.
                    </AlertDescription>
                  </Alert>
                )}

                <div className="mb-4">
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>백업 정보</AlertTitle>
                    <AlertDescription>
                      백업에는 데이터베이스와 스토리지(/storage) 디렉토리가 모두 포함됩니다.
                      복구 시 기존 스토리지가 완전히 삭제되고 백업 시점의 스토리지로 대체되니 주의하세요.
                    </AlertDescription>
                  </Alert>
                </div>

                <ScrollArea className="h-[400px]">
                  <div className={`${isOperationInProgress ? 'opacity-50 pointer-events-none' : ''} transition-opacity`}>
                    <Table>
              <TableHeader>
                <TableRow className="border-border">
                  <TableHead className="text-muted">백업 이름</TableHead>
                  <TableHead className="text-muted">크기</TableHead>
                  <TableHead className="text-muted">날짜/시간</TableHead>
                  <TableHead className="text-muted">타입</TableHead>
                  <TableHead className="text-muted">상태</TableHead>
                  <TableHead className="text-right text-muted">작업</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
              {backups.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted py-8">
                    백업이 없습니다. "즉시 백업" 버튼을 클릭하여 백업을 생성하세요.
                  </TableCell>
                </TableRow>
              ) : (
                backups.map((backup) => (
                  <TableRow key={backup.id} className="border-border">
                    <TableCell className="font-medium text-foreground">{backup.name}</TableCell>
                    <TableCell className="text-muted">{backup.size}</TableCell>
                    <TableCell className="text-muted">{backup.date}</TableCell>
                    <TableCell>{getTypeBadge(backup.type)}</TableCell>
                    <TableCell>{getStatusBadge(backup.status)}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          disabled={backup.status !== 'success' || isOperationInProgress}
                          onClick={() => handleExport(backup.id, backup.name)}
                          title={isOperationInProgress ? "작업 진행 중" : "Export (다운로드)"}
                          className="ds-btn-icon"
                        >
                          <Download className="w-4 h-4" />
                        </Button>
                        <Button
                          disabled={backup.status !== 'success' || isOperationInProgress}
                          onClick={() => handleRestore(backup.id)}
                          title={isOperationInProgress ? "작업 진행 중" : "복구"}
                          className="ds-btn-icon"
                        >
                          <RotateCcw className="w-4 h-4" />
                        </Button>
                        <Button
                          disabled={isOperationInProgress}
                          onClick={() => handleDelete(backup.id)}
                          title={isOperationInProgress ? "작업 진행 중" : "삭제"}
                          className="ds-btn-icon text-error hover:text-error"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
                    </TableBody>
                  </Table>
                  </div>
                </ScrollArea>

                <div className="mt-6 p-4 bg-muted/50 rounded-lg">
                  <h3 className="font-medium mb-2">복구 주의사항</h3>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• <strong className="text-tertiary">백업 복구 시 데이터베이스와 스토리지가 모두 백업 시점으로 되돌아갑니다</strong></li>
                    <li>• <strong className="text-tertiary">기존 스토리지는 완전히 삭제되고 백업 파일로 대체됩니다</strong></li>
                    <li>• 복구 과정은 되돌릴 수 없으므로, 복구 전 현재 데이터를 백업하는 것을 강력히 권장합니다</li>
                    <li>• 복구 작업은 시스템 사용이 적은 시간에 수행하시기 바랍니다</li>
                    <li>• 복구 중에는 시스템이 일시적으로 중단될 수 있습니다</li>
                    <li>• 복구 실패 시 자동으로 롤백되어 원래 상태로 복원됩니다</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        )
      }}
    />
  )
}
