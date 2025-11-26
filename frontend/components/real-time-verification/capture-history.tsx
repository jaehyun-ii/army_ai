"use client"

import { useState, useEffect, useRef, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert"
import {
  FileText,
  Trash2,
  History,
  Image as ImageIcon,
  Calculator,
  AlertCircle,
  CheckCircle2,
  XCircle,
  ArrowRight
} from "lucide-react"

// No longer needed - using Next.js API routes

// Helper function for API endpoints
const getApiEndpoint = (path: string) => path

// Generate consistent color from class name
function getColorFromClassName(className: string): number {
  let hash = 0
  for (let i = 0; i < className.length; i++) {
    hash = className.charCodeAt(i) + ((hash << 5) - hash)
    hash = hash & hash
  }
  return Math.abs(hash % 360)
}

// Component to render image with bounding boxes
function ImageWithBoundingBoxes({
  imageUrl,
  detections,
  frameNumber
}: {
  imageUrl: string
  detections: any[]
  frameNumber: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    const image = imageRef.current
    const canvas = canvasRef.current

    if (!image || !canvas || !detections || detections.length === 0) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const drawBoundingBoxes = () => {
      const displayWidth = image.clientWidth
      const displayHeight = image.clientHeight

      if (!displayWidth || !displayHeight) return

      canvas.width = displayWidth
      canvas.height = displayHeight
      ctx.clearRect(0, 0, displayWidth, displayHeight)

      const naturalWidth = image.naturalWidth || 1
      const naturalHeight = image.naturalHeight || 1

      // Calculate aspect ratio and letterboxing
      const imageAspect = naturalWidth / naturalHeight
      const displayAspect = displayWidth / displayHeight

      let drawWidth = displayWidth
      let drawHeight = displayHeight
      let offsetX = 0
      let offsetY = 0

      if (displayAspect > imageAspect) {
        drawHeight = displayHeight
        drawWidth = drawHeight * imageAspect
        offsetX = (displayWidth - drawWidth) / 2
      } else {
        drawWidth = displayWidth
        drawHeight = drawWidth / imageAspect
        offsetY = (displayHeight - drawHeight) / 2
      }

      const scaleX = drawWidth / naturalWidth
      const scaleY = drawHeight / naturalHeight

      // Draw each bounding box
      detections.forEach((det: any) => {
        if (!det.bbox) return

        // bbox format: {x, y, width, height} in pixels
        const x = offsetX + det.bbox.x * scaleX
        const y = offsetY + det.bbox.y * scaleY
        const w = det.bbox.width * scaleX
        const h = det.bbox.height * scaleY

        // Generate color based on class name
        const hue = getColorFromClassName(det.class_name)
        const color = `hsl(${hue}, 70%, 50%)`

        // Draw bounding box
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, w, h)

        // Draw label
        const label = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`
        const fontSize = 12
        ctx.font = `${fontSize}px Arial`
        const textWidth = ctx.measureText(label).width
        const padding = 3

        // Background
        ctx.fillStyle = color
        ctx.fillRect(x, Math.max(0, y - fontSize - padding * 2), textWidth + padding * 2, fontSize + padding * 2)

        // Text
        ctx.fillStyle = 'white'
        ctx.fillText(label, x + padding, Math.max(fontSize, y - padding))
      })
    }

    const handleLoad = () => drawBoundingBoxes()

    if (image.complete && image.naturalWidth > 0) {
      drawBoundingBoxes()
    } else {
      image.addEventListener('load', handleLoad)
    }

    // Handle resize
    let resizeObserver: ResizeObserver | undefined
    if (typeof window !== 'undefined' && 'ResizeObserver' in window) {
      resizeObserver = new ResizeObserver(() => drawBoundingBoxes())
      resizeObserver.observe(image)
    }

    return () => {
      image.removeEventListener('load', handleLoad)
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
    }
  }, [detections, imageUrl])

  return (
    <div className="relative aspect-square bg-slate-900 rounded-lg overflow-hidden">
      <img
        ref={imageRef}
        src={imageUrl}
        alt={`Frame ${frameNumber}`}
        className="w-full h-full object-contain"
      />
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 w-full h-full pointer-events-none"
      />
    </div>
  )
}

// Reliability Evaluation Component
function ReliabilityEvaluator({
  captures,
  onEvaluate,
  evaluationResult,
  isEvaluating
}: {
  captures: any[]
  onEvaluate: (cleanId: string, attackedId: string, label: string) => void
  evaluationResult: any
  isEvaluating: boolean
}) {
  const [cleanSessionId, setCleanSessionId] = useState<string>("")
  const [attackedSessionId, setAttackedSessionId] = useState<string>("")
  const [targetLabel, setTargetLabel] = useState<string>("")
  const [commonLabels, setCommonLabels] = useState<string[]>([])
  const [isLoadingLabels, setIsLoadingLabels] = useState(false)

  // Fetch common labels when both sessions are selected
  useEffect(() => {
    if (cleanSessionId && attackedSessionId) {
      fetchCommonLabels()
    } else {
      setCommonLabels([])
      setTargetLabel("")
    }
  }, [cleanSessionId, attackedSessionId])

  const fetchCommonLabels = async () => {
    setIsLoadingLabels(true)
    try {
      // Fetch details for both sessions to extract labels
      const [cleanRes, attackedRes] = await Promise.all([
        fetch(getApiEndpoint(`/api/camera/captures/${cleanSessionId}`)),
        fetch(getApiEndpoint(`/api/camera/captures/${attackedSessionId}`))
      ])

      if (!cleanRes.ok || !attackedRes.ok) throw new Error("Failed to fetch session details")

      const cleanData = await cleanRes.json()
      const attackedData = await attackedRes.json()

      // Extract unique labels from both
      const getLabels = (data: any) => {
        const labels = new Set<string>()
        data.images?.forEach((img: any) => {
          img.detections?.forEach((det: any) => labels.add(det.class_name))
        })
        return labels
      }

      const cleanLabels = getLabels(cleanData)
      const attackedLabels = getLabels(attackedData)

      // Find intersection
      const common = Array.from(cleanLabels).filter(label => attackedLabels.has(label))
      setCommonLabels(common)
    } catch (error) {
      console.error("Error fetching labels:", error)
    } finally {
      setIsLoadingLabels(false)
    }
  }

  const handleEvaluate = () => {
    if (cleanSessionId && attackedSessionId && targetLabel) {
      onEvaluate(cleanSessionId, attackedSessionId, targetLabel)
    }
  }

  return (
    <Card className="bg-slate-800/50 border-white/10 mb-6">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2 text-white">
          <Calculator className="w-5 h-5 text-blue-400" />
          신뢰성 평가 (Reliability Evaluation)
        </CardTitle>
        <CardDescription className="text-slate-400">
          적대적 공격 전후의 데이터셋을 비교하여 모델의 신뢰성을 평가합니다.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div className="space-y-2">
            <label className="text-xs text-slate-400">적대적 공격 미적용 (Clean)</label>
            <Select value={cleanSessionId} onValueChange={setCleanSessionId}>
              <SelectTrigger className="bg-slate-900/50 border-white/10 text-white">
                <SelectValue placeholder="세션 선택" />
              </SelectTrigger>
              <SelectContent>
                {captures.map(c => (
                  <SelectItem key={c.id} value={c.id.toString()}>
                    #{c.id} ({new Date(c.created_at).toLocaleString()})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-slate-400">적대적 공격 적용 (Attacked)</label>
            <Select value={attackedSessionId} onValueChange={setAttackedSessionId}>
              <SelectTrigger className="bg-slate-900/50 border-white/10 text-white">
                <SelectValue placeholder="세션 선택" />
              </SelectTrigger>
              <SelectContent>
                {captures.map(c => (
                  <SelectItem key={c.id} value={c.id.toString()} disabled={c.id.toString() === cleanSessionId}>
                    #{c.id} ({new Date(c.created_at).toLocaleString()})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-xs text-slate-400">평가대상 라벨</label>
            <Select value={targetLabel} onValueChange={setTargetLabel} disabled={!cleanSessionId || !attackedSessionId || isLoadingLabels}>
              <SelectTrigger className="bg-slate-900/50 border-white/10 text-white">
                <SelectValue placeholder={isLoadingLabels ? "로딩 중..." : "라벨 선택"} />
              </SelectTrigger>
              <SelectContent>
                {commonLabels.length === 0 ? (
                  <SelectItem value="none" disabled>공통 라벨 없음</SelectItem>
                ) : (
                  commonLabels.map(label => (
                    <SelectItem key={label} value={label}>{label}</SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>

          <Button
            onClick={handleEvaluate}
            disabled={!targetLabel || isEvaluating}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            {isEvaluating ? "평가 중..." : "신뢰도 평가"}
          </Button>
        </div>

        {evaluationResult && (
          <div className="mt-6 p-4 bg-slate-900/50 rounded-lg border border-white/10 animate-in fade-in slide-in-from-top-2">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-white font-medium flex items-center gap-2">
                <FileText className="w-4 h-4" /> 평가 결과: {evaluationResult.label}
              </h4>
              <Badge
                variant="outline"
                className={`${
                  evaluationResult.status === 'High' ? 'bg-green-500/20 text-green-400 border-green-500/50' :
                  evaluationResult.status === 'Low' ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50' :
                  'bg-red-500/20 text-red-400 border-red-500/50'
                }`}
              >
                신뢰성 {evaluationResult.status === 'High' ? '높음' : evaluationResult.status === 'Low' ? '낮음' : '매우 낮음'}
              </Badge>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-slate-800/50 p-3 rounded border border-white/5">
                <span className="text-slate-400 block mb-1">Clean 평균 신뢰도</span>
                <span className="text-white font-mono text-lg">{(evaluationResult.cleanConf * 100).toFixed(2)}%</span>
              </div>
              <div className="flex items-center justify-center text-slate-500">
                <ArrowRight className="w-5 h-5" />
              </div>
              <div className="bg-slate-800/50 p-3 rounded border border-white/5">
                <span className="text-slate-400 block mb-1">Attacked 평균 신뢰도</span>
                <span className="text-white font-mono text-lg">{(evaluationResult.attackedConf * 100).toFixed(2)}%</span>
              </div>
            </div>

            <div className="mt-4 flex items-center gap-2 text-sm">
              <span className="text-slate-400">성능 하락율:</span>
              <span className={`font-mono font-bold ${
                evaluationResult.dropRate < 0.05 ? 'text-green-400' :
                evaluationResult.dropRate < 0.15 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {(evaluationResult.dropRate * 100).toFixed(2)}%
              </span>
              <span className="text-slate-500 text-xs ml-2">
                (기준: 5% 미만 높음, 15% 미만 낮음, 15% 이상 매우 낮음)
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export function CaptureHistory() {
  const { toast } = useToast()
  const [captureHistory, setCaptureHistory] = useState<any[]>([])
  const [selectedCapture, setSelectedCapture] = useState<any>(null)
  const [captureImages, setCaptureImages] = useState<any[]>([])
  
  // Reliability Evaluation State
  const [evalResult, setEvalResult] = useState<any>(null)
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [selectedConfidenceClass, setSelectedConfidenceClass] = useState<string>("")

  const classAverages = useMemo(() => selectedCapture?.avg_confidence_by_class || [], [selectedCapture])
  const selectedClassStat = useMemo(() => {
    if (!selectedConfidenceClass) return null
    return (
      classAverages.find((cls: any) => {
        const key = cls.class_name || String(cls.class_id || "")
        return key === selectedConfidenceClass
      }) || null
    )
  }, [classAverages, selectedConfidenceClass])

  // Load capture history on mount
  useEffect(() => {
    loadCaptureHistory()
  }, [])

  // Load capture history
  const loadCaptureHistory = async () => {
    try {
      const response = await fetch('/api/camera/captures')
      const data = await response.json()
      setCaptureHistory(data.captures || [])
    } catch (error) {
      console.error('Failed to load capture history:', error)
    }
  }

  // Load capture details
  const loadCaptureDetails = async (captureId: number) => {
    try {
      const endpoint = getApiEndpoint(`/api/camera/captures/${captureId}`)
      const response = await fetch(endpoint)
      const data = await response.json()
      setSelectedCapture(data.capture)
      const firstClass = data.capture?.avg_confidence_by_class?.[0]
      setSelectedConfidenceClass(firstClass ? (firstClass.class_name || String(firstClass.class_id || "")) : "")
      setCaptureImages(data.images || [])
    } catch (error) {
      console.error('Failed to load capture details:', error)
    }
  }

  // Perform Reliability Evaluation
  const performEvaluation = async (cleanId: string, attackedId: string, label: string) => {
    setIsEvaluating(true)
    setEvalResult(null)
    try {
      // 1. Fetch full details for both sessions to get confidence scores
      const [cleanRes, attackedRes] = await Promise.all([
        fetch(getApiEndpoint(`/api/camera/captures/${cleanId}`)),
        fetch(getApiEndpoint(`/api/camera/captures/${attackedId}`))
      ])

      if (!cleanRes.ok || !attackedRes.ok) throw new Error("Failed to fetch session data")

      const cleanData = await cleanRes.json()
      const attackedData = await attackedRes.json()

      // 2. Calculate average confidence for the target label
      const calculateAvgConf = (data: any, target: string) => {
        let totalConf = 0
        let count = 0
        data.images?.forEach((img: any) => {
          img.detections?.forEach((det: any) => {
            if (det.class_name === target) {
              totalConf += det.confidence
              count++
            }
          })
        })
        return count > 0 ? totalConf / count : 0
      }

      const cleanAvg = calculateAvgConf(cleanData, label)
      const attackedAvg = calculateAvgConf(attackedData, label)

      if (cleanAvg === 0) {
        toast({
          title: "평가 불가",
          description: "Clean 데이터셋에서 해당 라벨의 탐지 기록을 찾을 수 없습니다.",
          variant: "destructive"
        })
        setIsEvaluating(false)
        return
      }

      // 3. Calculate Drop Rate
      // Formula: (Clean - Attacked) / Clean
      const dropRate = (cleanAvg - attackedAvg) / cleanAvg

      // 4. Determine Reliability Status
      let status = "Very Low"
      if (dropRate < 0.05) status = "High"      // < 5%
      else if (dropRate < 0.15) status = "Low"  // < 15%
      // else >= 15% is Very Low

      setEvalResult({
        label,
        cleanConf: cleanAvg,
        attackedConf: attackedAvg,
        dropRate,
        status
      })

    } catch (error) {
      console.error("Evaluation failed:", error)
      toast({
        title: "평가 실패",
        description: "신뢰성 평가 중 오류가 발생했습니다.",
        variant: "destructive"
      })
    } finally {
      setIsEvaluating(false)
    }
  }

  // Download entire dataset
  const downloadDataset = (captureId: number) => {
    const url = getApiEndpoint(`/api/camera/captures/${captureId}/download`)
    const link = document.createElement('a')
    link.href = url
    link.download = `capture_${captureId}_dataset.zip`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    toast({
      title: "다운로드 시작",
      description: "전체 데이터셋 다운로드가 시작되었습니다."
    })
  }

  // Delete capture session
  const deleteCaptureSession = async (captureId: number) => {
    if (!confirm('이 캡처 세션을 삭제하시겠습니까? 모든 이미지와 데이터가 영구적으로 삭제됩니다.')) {
      return
    }

    try {
      const endpoint = getApiEndpoint(`/api/camera/captures/${captureId}`)
      const response = await fetch(endpoint, {
        method: 'DELETE'
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || 'Failed to delete capture session')
      }

      await loadCaptureHistory()

      if (selectedCapture && selectedCapture.id === captureId) {
        setSelectedCapture(null)
        setCaptureImages([])
        setSelectedConfidenceClass("")
      }

      toast({
        title: "삭제 완료",
        description: "캡처 세션이 성공적으로 삭제되었습니다."
      })
    } catch (error) {
      console.error('Failed to delete capture session:', error)
      toast({
        variant: "destructive",
        title: "삭제 실패",
        description: `캡처 세션 삭제 실패: ${error}`
      })
    }
  }

  return (
    <AdversarialToolLayout
      title="캡처 기록"
      description="저장된 캡처 세션 및 이미지 확인"
      icon={History}
      headerStats={
        <div className="flex gap-2">
          <Button
            onClick={loadCaptureHistory}
            variant="outline"
            size="sm"
            className="bg-slate-700/50 border-white/20 text-white hover:bg-slate-600"
          >
            새로고침
          </Button>
          {selectedCapture && (
            <>
              <Button
                onClick={() => {
                  setSelectedCapture(null)
                  setCaptureImages([])
                  setSelectedConfidenceClass("")
                }}
                variant="outline"
                size="sm"
                className="bg-slate-700/50 border-white/20 text-white hover:bg-slate-600"
              >
                목록으로
              </Button>
              <Button
                onClick={() => deleteCaptureSession(selectedCapture.id)}
                variant="outline"
                size="sm"
                className="bg-red-900/50 border-red-500/20 text-red-400 hover:bg-red-800/50 hover:text-red-300"
              >
                <Trash2 className="w-4 h-4 mr-1" />
                삭제
              </Button>
            </>
          )}
        </div>
      }
      rightPanel={{
        title: "캡처 세션",
        icon: FileText,
        description: "저장된 캡처 세션 및 이미지",
        children: (
          <div className="h-full overflow-y-auto">

          {!selectedCapture && (
            <ReliabilityEvaluator
              captures={captureHistory}
              onEvaluate={performEvaluation}
              evaluationResult={evalResult}
              isEvaluating={isEvaluating}
            />
          )}

          {!selectedCapture ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {captureHistory.length === 0 ? (
                <div className="col-span-3 text-center py-8">
                  <FileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">캡처 기록이 없습니다</p>
                </div>
              ) : (
                captureHistory.map((capture: any) => (
                  <Card
                    key={capture.id}
                    className="bg-slate-800/50 border-white/10 hover:bg-slate-800/70 transition-colors w-full"
                  >
                    <CardHeader className="p-3">
                      <div className="flex justify-between items-start">
                        <div
                          className="flex-1 cursor-pointer"
                          onClick={() => loadCaptureDetails(capture.id)}
                        >
                          <CardTitle className="text-white text-sm">Session #{capture.id}</CardTitle>
                          <CardDescription className="text-slate-400 text-xs mt-1">
                            {new Date(capture.created_at).toLocaleString()}
                          </CardDescription>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={capture.status === 'completed' ? 'default' : 'secondary'} className="text-xs">
                            {capture.status}
                          </Badge>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 w-6 p-0 text-red-400 hover:text-red-300 hover:bg-red-900/20"
                            onClick={(e) => {
                              e.stopPropagation()
                              deleteCaptureSession(capture.id)
                            }}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent
                      className="p-3 pt-0 cursor-pointer"
                      onClick={() => loadCaptureDetails(capture.id)}
                    >
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">이미지</span>
                          <span className="text-white font-medium">{capture.total_images}장</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">탐지 객체</span>
                          <span className="text-white font-medium">{capture.total_detections || 0}개</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">평균 Confidence</span>
                          <span className="text-white font-medium">
                            {capture.avg_confidence_overall
                              ? (capture.avg_confidence_overall * 100).toFixed(2) + '%'
                              : 'N/A'}
                          </span>
                        </div>
                        {capture.avg_confidence_by_class && capture.avg_confidence_by_class.length > 0 && (
                          <div className="pt-2 border-t border-white/10">
                            <div className="text-[10px] text-slate-400 mb-1">클래스별 평균</div>
                            <div className="flex flex-wrap gap-1">
                              {capture.avg_confidence_by_class.slice(0, 3).map((cls: any) => (
                                <Badge key={`${capture.id}-${cls.class_id}-${cls.class_name}`} variant="secondary" className="text-[10px] px-2 py-1 bg-slate-900/70 border-slate-700">
                                  <span className="mr-1">{cls.class_name || cls.class_id}</span>
                                  <span className="text-green-300 font-semibold">
                                    {cls.avg_confidence ? (cls.avg_confidence * 100).toFixed(1) : 'N/A'}%
                                  </span>
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="bg-slate-800/50 rounded-lg p-4 border border-white/10">
                <div className="flex justify-between items-start mb-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm flex-1">
                    <div>
                      <span className="text-slate-400 block mb-1">세션 ID</span>
                      <span className="text-white font-medium">#{selectedCapture.id}</span>
                    </div>
                    <div>
                      <span className="text-slate-400 block mb-1">생성 시간</span>
                      <span className="text-white font-medium">
                        {new Date(selectedCapture.created_at).toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block mb-1">탐지 객체</span>
                      <span className="text-white font-medium">{selectedCapture.total_detections || 0}개</span>
                    </div>
                    <div>
                      <span className="text-slate-400 block mb-1">평균 Confidence</span>
                      <span className="text-white font-medium">
                        {selectedCapture.avg_confidence_overall
                          ? (selectedCapture.avg_confidence_overall * 100).toFixed(2) + '%'
                          : 'N/A'}
                      </span>
                    </div>
                  </div>
                  <Button
                    onClick={() => downloadDataset(selectedCapture.id)}
                    className="bg-green-600 hover:bg-green-700 text-white ml-4"
                  >
                    전체 다운로드
                  </Button>
                </div>
              </div>

              {classAverages.length > 0 && (
                <div className="bg-slate-800/50 rounded-lg p-4 border border-white/10">
                  <div className="flex items-center justify-between mb-3">
                    <div className="text-sm text-slate-200 font-medium">클래스별 평균 Confidence</div>
                    <Select
                      value={selectedConfidenceClass || (classAverages[0]?.class_name || String(classAverages[0]?.class_id || ""))}
                      onValueChange={(value) => setSelectedConfidenceClass(value)}
                    >
                      <SelectTrigger className="w-44 bg-slate-900/60 border-slate-700 text-white">
                        <SelectValue placeholder="라벨 선택" />
                      </SelectTrigger>
                      <SelectContent>
                        {classAverages.map((cls: any) => {
                          const key = cls.class_name || String(cls.class_id || "")
                          return (
                            <SelectItem key={key} value={key}>
                              {cls.class_name || cls.class_id || 'unknown'}
                            </SelectItem>
                          )
                        })}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {classAverages.map((cls: any) => {
                      const label = cls.class_name || cls.class_id || 'unknown'
                      return (
                        <Badge
                          key={`${selectedCapture.id}-${label}`}
                          variant="secondary"
                          className="text-[11px] px-2 py-1 bg-slate-900/70 border-slate-700"
                        >
                          <span className="mr-2">{label}</span>
                          <span className="text-green-300 font-semibold">
                            {cls.avg_confidence ? (cls.avg_confidence * 100).toFixed(1) : 'N/A'}%
                          </span>
                        </Badge>
                      )
                    })}
                  </div>
                  {selectedClassStat && (
                    <div className="mt-3 text-sm text-slate-300">
                      선택 라벨 평균:{" "}
                      <span className="text-white font-semibold">
                        {selectedClassStat.avg_confidence ? (selectedClassStat.avg_confidence * 100).toFixed(2) : 'N/A'}%
                      </span>
                      <span className="text-slate-400 ml-2">(탐지 {selectedClassStat.count || 0}개)</span>
                    </div>
                  )}
                </div>
              )}

              <div className="grid grid-cols-5 gap-3">
                {captureImages.map((image: any) => (
                  <Card key={image.id} className="bg-slate-800/50 border-white/10">
                    <CardContent className="p-3">
                      <div className="mb-2">
                        <ImageWithBoundingBoxes
                          imageUrl={getApiEndpoint(`/api/camera/captures/${selectedCapture.id}/image/${image.frame_number}`)}
                          detections={image.detections || []}
                          frameNumber={image.frame_number}
                        />
                      </div>
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-400">프레임</span>
                          <span className="text-white font-medium">#{image.frame_number}</span>
                        </div>

                        {image.detections && image.detections.length > 0 ? (
                          <div className="border-t border-white/10 pt-2 mt-2">
                            <div className="text-slate-400 mb-1">탐지된 객체:</div>
                            <div className="space-y-1 max-h-32 overflow-y-auto">
                              {image.detections.map((det: any, idx: number) => (
                                <div key={idx} className="flex justify-between items-center bg-slate-900/50 rounded px-2 py-1">
                                  <span className="text-white text-[10px] truncate flex-1">
                                    {det.class_name}
                                  </span>
                                  <span className="text-green-400 font-medium text-[10px] ml-2">
                                    {(det.confidence * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <div className="flex justify-between text-slate-500">
                            <span>탐지된 객체</span>
                            <span>없음</span>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
          </div>
        )
      }}
    />
  )
}
