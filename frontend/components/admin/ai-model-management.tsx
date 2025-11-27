"use client"

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { useModels } from '@/hooks/api'
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
import { Badge } from '@/components/ui/badge'
import { Brain, Upload, Download, Trash2, RefreshCw, CheckCircle2, XCircle, AlertCircle } from 'lucide-react'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

interface AIModel {
  id: string
  name: string
  type: string
  framework: string
  status: 'loaded' | 'unloaded'
  is_loaded: boolean
  num_classes: number
  lastUpdated: string
}

interface AutoDetectedMetadata {
  auto_detected?: boolean
  model_type?: string
  class_names?: string[]
  num_classes?: number
  input_size?: number[]
  framework?: string
}

// Validate name: only alphanumeric, dash, underscore
const validateName = (name: string): boolean => {
  const validPattern = /^[a-zA-Z0-9_-]+$/
  return validPattern.test(name)
}

// Simple File Upload Component (for .pt + optional config file)
function SimpleModelUploadContent({ onSuccess }: { onSuccess?: () => void }) {
  const [weightsFile, setWeightsFile] = useState<File | null>(null)
  const [configFile, setConfigFile] = useState<File | null>(null)
  const [autoDetectedInfo, setAutoDetectedInfo] = useState<AutoDetectedMetadata | null>(null)
  const [modelName, setModelName] = useState<string>('')
  const [modelDescription, setModelDescription] = useState<string>('')
  const [isUploading, setIsUploading] = useState(false)
  const [isCompleted, setIsCompleted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const handleWeightsFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setWeightsFile(file)
      setAutoDetectedInfo(null) // 새 파일 선택 시 이전 자동 감지 정보 초기화
      setError(null) // 에러 초기화
      setSuccess(null) // 성공 메시지 초기화
      setIsCompleted(false) // 완료 상태 초기화
    }
  }

  const handleConfigFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setConfigFile(file)
      setAutoDetectedInfo(null) // 새 파일 선택 시 이전 자동 감지 정보 초기화
      setError(null) // 에러 초기화
      setSuccess(null) // 성공 메시지 초기화
      setIsCompleted(false) // 완료 상태 초기화
    }
  }

  const handleModelNameChange = (value: string) => {
    // Only allow alphanumeric, dash, underscore
    if (value === '' || validateName(value)) {
      setModelName(value)
      setError(null)
    } else {
      setError('모델 이름은 영문자, 숫자, - (대시), _ (언더스코어)만 사용할 수 있습니다')
    }
  }

  const uploadModel = async () => {
    if (!weightsFile) {
      setError('모델 가중치 파일(.pt)을 선택해주세요')
      return
    }

    if (!modelName.trim()) {
      setError('모델 이름을 입력해주세요')
      return
    }

    if (!validateName(modelName)) {
      setError('모델 이름은 영문자, 숫자, - (대시), _ (언더스코어)만 사용할 수 있습니다')
      return
    }

    setIsUploading(true)
    setError(null)
    setSuccess(null)

    try {
      const formData = new FormData()
      formData.append('weights_file', weightsFile)
      if (configFile) {
        formData.append('yaml_file', configFile)
      }

      // Add manual settings
      formData.append('name', modelName)
      if (modelDescription) formData.append('description', modelDescription)

      const response = await fetch('/api/models/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || '모델 업로드에 실패했습니다')
      }

      const data = await response.json()

      // Extract auto-detected metadata
      const metadata: AutoDetectedMetadata = {
        model_type: data.inference_params?.estimator_type || 'Unknown',
        class_names: data.labelmap ? Object.values(data.labelmap) : undefined,
        num_classes: data.labelmap ? Object.keys(data.labelmap).length : undefined,
        input_size: data.input_spec?.shape ? data.input_spec.shape.slice(0, 2) : undefined,
        framework: data.framework || 'Unknown',
        auto_detected: data.inference_params?.auto_detected || false,
      }
      setAutoDetectedInfo(metadata)

      setSuccess(`모델 "${data.name}"이 성공적으로 업로드되었습니다`)
      setIsCompleted(true) // 업로드 완료 상태로 설정

      // 폼 초기화는 하지 않고 모델 목록 새로고침만 수행
      onSuccess?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : '알 수 없는 오류가 발생했습니다')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6">
        {/* Model Name */}
        <div className="space-y-2">
          <Label htmlFor="model-name" className="text-white">
            모델 이름 <span className="text-red-400">*</span>
          </Label>
          <Input
            id="model-name"
            value={modelName}
            onChange={(e) => handleModelNameChange(e.target.value)}
            placeholder="예: yolov8_person_detector"
            className="bg-slate-700/50 border-white/10 text-white"
            required
            disabled={isUploading || isCompleted}
          />
          <p className="text-xs text-slate-400 mt-1">영문자, 숫자, - (대시), _ (언더스코어)만 사용 가능</p>
        </div>

        {/* File Upload Section */}
        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="weights-file" className="text-white">
              모델 가중치 (.pt) <span className="text-red-400">*</span>
            </Label>
            <Input
              id="weights-file"
              type="file"
              accept=".pt,.pth"
              onChange={handleWeightsFileChange}
              className="bg-slate-700/50 border-white/10 text-white"
              disabled={isUploading || isCompleted}
            />
            {weightsFile && (
              <p className="text-sm text-green-400">✓ {weightsFile.name}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="config-file" className="text-white">
              Config 파일
            </Label>
            <Input
              id="config-file"
              type="file"
              accept=".yaml,.yml,.py"
              onChange={handleConfigFileChange}
              className="bg-slate-700/50 border-white/10 text-white"
              disabled={isUploading || isCompleted}
            />
            {configFile && (
              <p className="text-sm text-green-400">✓ {configFile.name}</p>
            )}
          </div>
        </div>

        {/* Description */}
        <div className="space-y-2">
          <Label htmlFor="model-description" className="text-white">설명</Label>
          <Textarea
            id="model-description"
            value={modelDescription}
            onChange={(e) => setModelDescription(e.target.value)}
            placeholder="모델 설명 (선택)"
            rows={2}
            className="bg-slate-700/50 border-white/10 text-white"
            disabled={isUploading || isCompleted}
          />
        </div>

        <Button
          onClick={uploadModel}
          disabled={isUploading || isCompleted || !weightsFile}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white"
          size="lg"
        >
          {isUploading ? (
            <>
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
              업로드 중...
            </>
          ) : (
            <>
              <Upload className="w-4 h-4 mr-2" />
              모델 업로드
            </>
          )}
        </Button>

        {/* Auto-detected info display */}
        {autoDetectedInfo && (
          <Alert className="bg-green-900/20 border-green-500/30">
            <CheckCircle2 className="h-4 w-4 text-green-400" />
            <AlertDescription className="text-green-300">
              <div className="font-semibold mb-2">자동 감지된 정보</div>
              <div className="space-y-1 text-sm">
                {autoDetectedInfo.framework && (
                  <div>프레임워크: {autoDetectedInfo.framework}</div>
                )}
                {autoDetectedInfo.model_type && (
                  <div>모델 타입: {autoDetectedInfo.model_type.toUpperCase()}</div>
                )}
                {autoDetectedInfo.num_classes && (
                  <div>클래스 수: {autoDetectedInfo.num_classes}</div>
                )}
                {autoDetectedInfo.input_size && (
                  <div>입력 크기: {JSON.stringify(autoDetectedInfo.input_size)}</div>
                )}
              </div>
            </AlertDescription>
          </Alert>
        )}

        {/* Error/Success Messages */}
        {error && (
          <Alert variant="destructive" className="bg-red-900/20 border-red-500/30">
            <XCircle className="h-4 w-4 text-red-400" />
            <AlertDescription className="text-red-300">{error}</AlertDescription>
          </Alert>
        )}

        {success && !error && (
          <Alert className="bg-green-900/20 border-green-500/30">
            <CheckCircle2 className="h-4 w-4 text-green-400" />
            <AlertDescription className="text-green-300">{success}</AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  )
}

export function AIModelManagement() {
  // Use custom hook for models
  const { data: modelsData, isLoading, refetch } = useModels(0, 100)

  const [selectedModel, setSelectedModel] = useState<AIModel | null>(null)
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false)
  const [isTestInferenceDialogOpen, setIsTestInferenceDialogOpen] = useState(false)
  const [testImage, setTestImage] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<any>(null)
  const [isTesting, setIsTesting] = useState(false)
  const [imageNaturalSize, setImageNaturalSize] = useState<{ width: number; height: number } | null>(null)
  const [modelToDelete, setModelToDelete] = useState<AIModel | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)

  // Transform API response to component format
  const models: AIModel[] = modelsData?.map((model: any) => ({
    id: model.id,
    name: model.name,
    type: model.task === 'object-detection' ? '객체탐지' : '분류',
    framework: model.framework,
    status: 'unloaded' as const,
    is_loaded: false,
    num_classes: model.labelmap ? Object.keys(model.labelmap).length : 0,
    lastUpdated: model.created_at ? new Date(model.created_at).toLocaleDateString('ko-KR') : 'N/A'
  })) || []

  // Handle image upload for testing
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setTestImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  // Run test inference
  const runTestInference = async () => {
    if (!selectedModel || !testImage) return

    setIsTesting(true)
    setTestResults(null)

    try {
      const response = await fetch(
        `/api/models/${selectedModel.id}/predict`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_base64: testImage.split(',')[1],
            conf_threshold: 0.25,
            iou_threshold: 0.45
          })
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        const errorMsg = errorData.error || errorData.detail || '추론 실패'
        throw new Error(errorMsg)
      }

      const data = await response.json()
      console.log('Inference result:', data)
      setTestResults(data)
    } catch (error) {
      console.error('Test inference error:', error)
      const errorMsg = error instanceof Error ? error.message : '추론 중 오류가 발생했습니다'
      alert(`추론 오류:\n${errorMsg}`)
    } finally {
      setIsTesting(false)
    }
  }

  // Open test inference dialog
  const openTestInference = (model: AIModel) => {
    setSelectedModel(model)
    setTestImage(null)
    setTestResults(null)
    setIsTestInferenceDialogOpen(true)
  }

  // Download model
  const downloadModel = async (model: AIModel) => {
    try {
      const response = await fetch(`/api/models/${model.id}/download`, {
        method: 'GET',
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: '다운로드 실패' }))
        throw new Error(errorData.detail || '모델 다운로드에 실패했습니다')
      }

      // Get filename from Content-Disposition header
      const contentDisposition = response.headers.get('content-disposition')
      let filename = `${model.name}.zip`

      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?(.+?)"?$/)
        if (match) {
          filename = match[1]
        }
      }

      // Download the file
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Download error:', error)
      const errorMsg = error instanceof Error ? error.message : '다운로드 중 오류가 발생했습니다'
      alert(`다운로드 오류:\n${errorMsg}`)
    }
  }

  // Delete model
  const deleteModel = async () => {
    if (!modelToDelete) return

    setIsDeleting(true)

    try {
      const response = await fetch(`/api/models/${modelToDelete.id}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Delete error:', errorText)
        throw new Error('모델 삭제에 실패했습니다')
      }

      // Success - refresh model list
      await refetch()
      setModelToDelete(null)
    } catch (error) {
      console.error('Delete model error:', error)
      alert(error instanceof Error ? error.message : '모델 삭제 중 오류가 발생했습니다')
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>AI 모델 관리</CardTitle>
              <CardDescription>객체탐지 및 식별 AI 모델을 관리합니다</CardDescription>
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => refetch()} disabled={isLoading}>
                <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                새로고침
              </Button>
              <Dialog open={isUploadDialogOpen} onOpenChange={setIsUploadDialogOpen}>
                <DialogTrigger asChild>
                  <Button size="sm">
                    <Upload className="w-4 h-4 mr-2" />
                    모델 업로드
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
                  <DialogHeader>
                    <DialogTitle>모델 업로드</DialogTitle>
                    <DialogDescription>
                      .pt 파일과  config 파일 (.yaml 또는 .py)을 업로드합니다
                    </DialogDescription>
                  </DialogHeader>

                  <SimpleModelUploadContent onSuccess={() => refetch()} />
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex gap-4">
            <Input placeholder="모델명 검색..." className="max-w-sm" />
            <Select defaultValue="all">
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="모델 타입" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체</SelectItem>
                <SelectItem value="detection">객체탐지</SelectItem>
                <SelectItem value="classification">분류</SelectItem>
                <SelectItem value="segmentation">세그멘테이션</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>모델명</TableHead>
                <TableHead>타입</TableHead>
                <TableHead>프레임워크</TableHead>
                <TableHead>최종 업데이트</TableHead>
                <TableHead className="text-right">작업</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">모델 목록을 불러오는 중...</p>
                  </TableCell>
                </TableRow>
              ) : models.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8">
                    <p className="text-sm text-muted-foreground">업로드된 모델이 없습니다.</p>
                    <p className="text-xs text-muted-foreground mt-1">위의 "모델 업로드" 버튼을 클릭하여 모델을 업로드하세요.</p>
                  </TableCell>
                </TableRow>
              ) : (
                models.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell className="font-medium">{model.name}</TableCell>
                    <TableCell>{model.type}</TableCell>
                    <TableCell>
                      <span className="font-medium">{model.framework}</span>
                    </TableCell>
                    <TableCell>{model.lastUpdated}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          title="테스트 추론"
                          onClick={() => openTestInference(model)}
                        >
                          <Brain className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          title="다운로드"
                          onClick={() => downloadModel(model)}
                        >
                          <Download className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-500 hover:text-red-700"
                          title="삭제"
                          onClick={() => setModelToDelete(model)}
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
        </CardContent>
      </Card>

      {/* Test Inference Dialog */}
      <Dialog open={isTestInferenceDialogOpen} onOpenChange={setIsTestInferenceDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>모델 테스트 추론</DialogTitle>
            <DialogDescription>
              {selectedModel ? `${selectedModel.name}` : '모델을 테스트합니다'}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6">
            {/* Image Upload */}
            <div className="space-y-4">
              <div>
                <Label htmlFor="test-image">테스트 이미지 업로드</Label>
                <Input
                  id="test-image"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="mt-2"
                />
              </div>

              {/* Preview Image */}
              {testImage && (
                <div className="border rounded-lg p-4">
                  <h3 className="font-medium mb-2">이미지 미리보기</h3>
                  <div className="relative inline-block">
                    <img
                      id="test-image-preview"
                      src={testImage}
                      alt="Test input"
                      className="max-w-full h-auto max-h-96"
                      onLoad={(e) => {
                        const img = e.target as HTMLImageElement
                        // Store natural (original) image dimensions
                        setImageNaturalSize({
                          width: img.naturalWidth,
                          height: img.naturalHeight
                        })
                      }}
                    />
                    {testResults && testResults.detections && testResults.detections.length > 0 && imageNaturalSize && (
                      <svg
                        id="detection-overlay"
                        className="absolute top-0 left-0 pointer-events-none"
                        style={{ width: '100%', height: '100%' }}
                      >
                        {testResults.detections.map((det: any, idx: number) => {
                          const img = document.getElementById('test-image-preview') as HTMLImageElement
                          if (!img || !imageNaturalSize) return null

                          // Get displayed image size
                          const displayWidth = img.width
                          const displayHeight = img.height

                          // Get natural (original) image size
                          const naturalWidth = imageNaturalSize.width || 1
                          const naturalHeight = imageNaturalSize.height || 1

                          // Calculate aspect ratios
                          const imageAspect = naturalWidth / naturalHeight
                          const displayAspect = displayWidth / displayHeight

                          // Simulate object-contain behavior: calculate letterboxing
                          let drawWidth = displayWidth
                          let drawHeight = displayHeight
                          let offsetX = 0
                          let offsetY = 0

                          if (displayAspect > imageAspect) {
                            // Image is taller - letterbox on sides
                            drawHeight = displayHeight
                            drawWidth = drawHeight * imageAspect
                            offsetX = (displayWidth - drawWidth) / 2
                          } else {
                            // Image is wider - letterbox on top/bottom
                            drawWidth = displayWidth
                            drawHeight = drawWidth / imageAspect
                            offsetY = (displayHeight - drawHeight) / 2
                          }

                          // API returns YOLO normalized format (x_center, y_center, width, height)
                          // All values are 0-1, relative to original image dimensions
                          const xCenter = det.bbox.x_center
                          const yCenter = det.bbox.y_center
                          const width = det.bbox.width
                          const height = det.bbox.height

                          // Calculate coordinates with letterboxing offset
                          const x1 = offsetX + (xCenter - width / 2) * drawWidth
                          const y1 = offsetY + (yCenter - height / 2) * drawHeight
                          const width_px = width * drawWidth
                          const height_px = height * drawHeight

                          // Generate a color based on class_id for variety
                          const colors = ['#00ff00', '#ff0000', '#0000ff', '#ffff00', '#ff00ff', '#00ffff']
                          const color = colors[det.class_id % colors.length]

                          return (
                            <g key={idx}>
                              {/* Bounding box rectangle */}
                              <rect
                                x={x1}
                                y={y1}
                                width={width_px}
                                height={height_px}
                                fill="none"
                                stroke={color}
                                strokeWidth="3"
                                opacity="0.8"
                              />
                              {/* Label background */}
                              <rect
                                x={x1}
                                y={Math.max(0, y1 - 25)}
                                width={(det.class_name || 'Unknown').length * 8 + 60}
                                height="24"
                                fill={color}
                                opacity="0.8"
                              />
                              {/* Label text */}
                              <text
                                x={x1 + 4}
                                y={Math.max(16, y1 - 8)}
                                fill="#000000"
                                fontSize="14"
                                fontWeight="bold"
                                fontFamily="monospace"
                              >
                                {det.class_name || 'Unknown'} {(det.confidence * 100).toFixed(1)}%
                              </text>
                            </g>
                          )
                        })}
                      </svg>
                    )}
                  </div>
                </div>
              )}

              {/* Run Inference Button */}
              <Button
                onClick={runTestInference}
                disabled={!testImage || isTesting}
                className="w-full"
              >
                {isTesting ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    추론 중...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 mr-2" />
                    추론 실행
                  </>
                )}
              </Button>
            </div>

            {/* Results */}
            {testResults && (
              <div className="space-y-4 border-t pt-4">
                <h3 className="font-medium">탐지 결과</h3>

                {testResults.detections && testResults.detections.length > 0 ? (
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      {testResults.detections.length}개의 객체가 탐지되었습니다
                      {testResults.inference_time_ms && (
                        <> (추론 시간: {testResults.inference_time_ms.toFixed(1)}ms)</>
                      )}
                    </p>

                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>클래스</TableHead>
                          <TableHead>신뢰도</TableHead>
                          <TableHead>위치 (x_center, y_center, width, height)</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {testResults.detections.map((det: any, idx: number) => (
                          <TableRow key={idx}>
                            <TableCell className="font-medium">{det.class_name}</TableCell>
                            <TableCell>
                              <Badge variant={det.confidence > 0.7 ? 'default' : 'secondary'}>
                                {(det.confidence * 100).toFixed(1)}%
                              </Badge>
                            </TableCell>
                            <TableCell className="text-sm text-muted-foreground font-mono">
                              ({det.bbox.x_center.toFixed(4)}, {det.bbox.y_center.toFixed(4)}, {det.bbox.width.toFixed(4)}, {det.bbox.height.toFixed(4)})
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">탐지된 객체가 없습니다</p>
                )}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!modelToDelete} onOpenChange={(open) => !open && setModelToDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>모델 삭제 확인</AlertDialogTitle>
            <AlertDialogDescription>
              정말로 <strong>{modelToDelete?.name}</strong> 모델을 삭제하시겠습니까?
              <br /><br />
              이 작업은 다음을 수행합니다:
              <ul className="list-disc list-inside mt-2 space-y-1">
                <li>메모리에서 모델 언로드</li>
                <li>저장된 모델 파일 삭제</li>
                <li>데이터베이스에서 모델 정보 삭제</li>
              </ul>
              <br />
              <span className="text-red-600 font-semibold">이 작업은 되돌릴 수 없습니다.</span>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>취소</AlertDialogCancel>
            <AlertDialogAction
              onClick={deleteModel}
              disabled={isDeleting}
              className="bg-red-600 hover:bg-red-700"
            >
              {isDeleting ? '삭제 중...' : '삭제'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
