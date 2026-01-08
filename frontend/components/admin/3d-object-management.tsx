"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from '@/components/ui/badge'
import { Box, Upload, RefreshCw, CheckCircle2, XCircle, AlertCircle, FolderUp, Power } from 'lucide-react'

interface VehicleInfo {
  name: string
}

// 업로드 다이얼로그 컴포넌트
function UploadDialog({
  open,
  onOpenChange,
  onSuccess
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess: () => void
}) {
  const [files, setFiles] = useState<FileList | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [uploadInfo, setUploadInfo] = useState<any>(null)

  const resetForm = () => {
    setFiles(null)
    setError(null)
    setSuccess(false)
    setUploadInfo(null)
  }

  const handleUpload = async () => {
    if (!files || files.length === 0) {
      setError('Vehicles 폴더의 파일들을 선택해주세요')
      return
    }

    setIsUploading(true)
    setError(null)

    try {
      const formData = new FormData()

      // 모든 파일 추가
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i])
      }

      const response = await fetch('/api/carla/vehicles/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || '업로드에 실패했습니다')
      }

      const data = await response.json()
      setUploadInfo(data.result)
      setSuccess(true)

      // 3초 후 자동 닫기 및 목록 새로고침
      setTimeout(() => {
        resetForm()
        onOpenChange(false)
        onSuccess()
      }, 5000)
    } catch (err) {
      setError(err instanceof Error ? err.message : '알 수 없는 오류가 발생했습니다')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-primary">CARLA Vehicles 폴더 업로드</DialogTitle>
          <DialogDescription className="text-muted">
            Vehicles 폴더의 모든 파일을 선택하여 업로드합니다. 기존 폴더를 덮어씁니다.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Important Warning */}
          <Alert className="bg-yellow-500/10 border-yellow-500/50">
            <AlertCircle className="h-4 w-4 text-yellow-500" />
            <AlertDescription className="text-yellow-700 dark:text-yellow-400">
              <strong>중요:</strong> 업로드 후 CARLA 시뮬레이터를 재시작해야 변경사항이 반영됩니다.
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <Label htmlFor="files" className="text-foreground">
              Vehicles 폴더 파일 <span className="text-error">*</span>
            </Label>
            <Input
              id="files"
              type="file"
              // @ts-ignore - webkitdirectory is not in standard types
              webkitdirectory="true"
              directory="true"
              multiple
              onChange={(e) => setFiles(e.target.files)}
              className="bg-surface-container-high border-border text-foreground"
              disabled={isUploading || success}
            />
            <p className="text-xs text-muted mt-1">
              전체 Vehicles 폴더를 선택하세요. 폴더 구조가 그대로 유지됩니다.
            </p>
            {files && files.length > 0 && (
              <p className="text-xs text-tertiary mt-1">
                {files.length}개 파일 선택됨
              </p>
            )}
          </div>

          <div className="bg-surface-container-high p-4 rounded-lg space-y-2">
            <p className="text-sm font-medium text-foreground">업로드 정보:</p>
            <ul className="text-xs text-muted space-y-1 list-disc list-inside">
              <li>기존 storage/simulator/Vehicle 폴더가 백업된 후 교체됩니다</li>
              <li>ZIP 파일로 업로드하거나 폴더 전체를 선택할 수 있습니다</li>
              <li>업로드 완료 후 CARLA를 재시작해야 합니다</li>
            </ul>
          </div>

          <Button
            onClick={handleUpload}
            disabled={isUploading || success || !files}
            className="w-full bg-primary hover:bg-primary/90 text-foreground"
            size="lg"
          >
            {isUploading ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                업로드 중...
              </>
            ) : success ? (
              <>
                <CheckCircle2 className="w-4 h-4 mr-2" />
                업로드 완료!
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Vehicles 폴더 업로드
              </>
            )}
          </Button>

          {error && (
            <Alert variant="destructive" className="bg-error-container border-error">
              <XCircle className="h-4 w-4 text-error" />
              <AlertDescription className="text-error">{error}</AlertDescription>
            </Alert>
          )}

          {success && uploadInfo && (
            <Alert className="bg-tertiary-container border-tertiary">
              <CheckCircle2 className="h-4 w-4 text-tertiary" />
              <AlertDescription className="text-tertiary space-y-2">
                <div className="font-semibold">업로드 완료!</div>
                <div className="text-xs space-y-1">
                  <div>파일 개수: {uploadInfo.file_count}개</div>
                  <div>총 크기: {(uploadInfo.total_size / 1024 / 1024).toFixed(2)} MB</div>
                  <div className="font-bold text-yellow-600 dark:text-yellow-400 mt-2 flex items-center gap-2">
                    <Power className="w-4 h-4" />
                    CARLA 시뮬레이터를 재시작해주세요!
                  </div>
                </div>
              </AlertDescription>
            </Alert>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function Object3DManagement() {
  const [vehicles, setVehicles] = useState<VehicleInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false)

  const loadVehicles = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/carla/sim_object_list')
      if (response.ok) {
        const data = await response.json()
        // CARLA API returns vehicle names in result array
        const vehicleList = (data.result || []).map((name: string) => ({ name }))
        setVehicles(vehicleList)
      } else {
        console.error('Failed to load vehicles')
      }
    } catch (error) {
      console.error('Error loading vehicles:', error)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadVehicles()
  }, [])

  return (
    <>
      <div className="space-y-6">
        {/* CARLA Restart Warning Banner */}
        <Alert className="bg-yellow-500/10 border-yellow-500/50">
          <Power className="h-4 w-4 text-yellow-500" />
          <AlertDescription className="text-yellow-700 dark:text-yellow-400">
            <strong>알림:</strong> 3D 모델 업로드 후에는 CARLA 시뮬레이터를 재시작해야 변경사항이 반영됩니다.
          </AlertDescription>
        </Alert>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-surface-container/50 border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2 text-primary">
                <Box className="w-5 h-5 text-primary" />
                전체 모델
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-foreground">{vehicles.length}</div>
              <p className="text-sm text-muted">CARLA 차량 모델</p>
            </CardContent>
          </Card>

          <Card className="bg-surface-container/50 border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2 text-primary">
                <FolderUp className="w-5 h-5 text-tertiary" />
                스토리지
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold text-foreground">storage/simulator/Vehicle</div>
              <p className="text-sm text-muted">모델 저장 경로</p>
            </CardContent>
          </Card>

          <Card className="bg-surface-container/50 border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg text-primary flex items-center gap-2">
                <Power className="w-5 h-5 text-error" />
                CARLA 상태
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xl font-bold text-tertiary">재시작 필요</div>
              <p className="text-sm text-muted">모델 변경 시</p>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-surface-container border-border">
          <CardHeader>
            <div className="flex justify-between items-center">
              <div>
                <CardTitle className="text-primary">CARLA 3D 차량 모델 관리</CardTitle>
                <CardDescription className="text-muted">
                  CARLA 시뮬레이터에서 사용 가능한 차량 모델 목록
                </CardDescription>
              </div>
              <div className="flex gap-2">
                <Button size="sm" onClick={loadVehicles} disabled={isLoading} className="ds-btn-outline">
                  <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                  새로고침
                </Button>
                <Button size="sm" className="ds-btn-primary" onClick={() => setIsUploadDialogOpen(true)}>
                  <Upload className="w-4 h-4 mr-2" />
                  Vehicles 폴더 업로드
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Alert className="mb-4 bg-blue-500/10 border-blue-500/50">
              <AlertCircle className="h-4 w-4 text-blue-500" />
              <AlertDescription className="text-blue-700 dark:text-blue-400 text-sm">
                이 목록은 CARLA 시뮬레이터에서 직접 가져옵니다. 새 모델을 추가한 후에는 CARLA를 재시작하고
                이 페이지를 새로고침하세요.
              </AlertDescription>
            </Alert>

            <Table>
              <TableHeader>
                <TableRow className="border-border">
                  <TableHead className="text-muted">번호</TableHead>
                  <TableHead className="text-muted">모델명</TableHead>
                  <TableHead className="text-muted">상태</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={3} className="text-center py-8">
                      <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-primary" />
                      <p className="text-sm text-muted">CARLA에서 모델 목록을 불러오는 중...</p>
                    </TableCell>
                  </TableRow>
                ) : vehicles.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={3} className="text-center py-8">
                      <p className="text-sm text-muted">등록된 모델이 없거나 CARLA에 연결할 수 없습니다.</p>
                      <p className="text-xs text-muted mt-1">CARLA 시뮬레이터가 실행 중인지 확인하세요.</p>
                    </TableCell>
                  </TableRow>
                ) : (
                  vehicles.map((vehicle, index) => (
                    <TableRow key={index} className="border-border">
                      <TableCell className="text-muted">{index + 1}</TableCell>
                      <TableCell className="font-medium text-foreground">{vehicle.name}</TableCell>
                      <TableCell>
                        <Badge variant="default" className="bg-tertiary-container text-tertiary">
                          사용 가능
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      <UploadDialog
        open={isUploadDialogOpen}
        onOpenChange={setIsUploadDialogOpen}
        onSuccess={loadVehicles}
      />
    </>
  )
}
