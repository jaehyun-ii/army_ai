"use client"

import { useState, useRef, useEffect, useCallback, memo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useToast } from "@/hooks/use-toast"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import { validateName, getNameValidationMessage } from '@/lib/validation'
import { useOperation } from '@/contexts/OperationContext'
import {
  Camera,
  CameraOff,
  Play,
  Square,
  Monitor,
  Activity,
  Brain,
  Zap,
  AlertCircle,
  Video,
  RotateCcw,
} from "lucide-react"

// Interface definitions
interface ModelInfo {
  id: string
  name: string
  type: string
  size: string
}

interface CameraDevice {
  device_id: string
  device_path: string
  name: string
  width: number
  height: number
  fps: number
  is_available: boolean
}

// Bounding box color generation utility
const classColors = new Map<number, string>()
const goldenRatio = 0.61803398875
const get_class_color = (classId: number): string => {
  if (!classColors.has(classId)) {
    const hue = (classId * goldenRatio) % 1.0
    const saturation = 0.7
    const lightness = 0.6
    const rgb = hslToRgb(hue, saturation, lightness)
    classColors.set(classId, `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`)
  }
  return classColors.get(classId)!
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  let r, g, b
  if (s === 0) {
    r = g = b = l
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1
      if (t > 1) t -= 1
      if (t < 1 / 6) return p + (q - p) * 6 * t
      if (t < 1 / 2) return q
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
      return p
    }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s
    const p = 2 * l - q
    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
}


function RealTimeCameraComponent() {
  const { toast } = useToast()
  const { setOperationInProgress } = useOperation()
  const [cameraStatus, setCameraStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected')
  const [modelStatus, setModelStatus] = useState<'idle' | 'loading' | 'ready' | 'running' | 'error'>('idle')
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [selectedCamera, setSelectedCamera] = useState<string>("")
  const [availableCameras, setAvailableCameras] = useState<CameraDevice[]>([])
  const [isLoadingCameras, setIsLoadingCameras] = useState<boolean>(false)
  const [isInferenceActive, setIsInferenceActive] = useState(false)

  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])

  const [isCapturing, setIsCapturing] = useState(false)
  const [showCaptureDialog, setShowCaptureDialog] = useState(false)
  const [captureName, setCaptureName] = useState("")
  const [captureNameError, setCaptureNameError] = useState<string | null>(null)

  const sseRef = useRef<EventSource | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imageRef = useRef<HTMLImageElement>(new Image())
  const notifyCameraStatus = useCallback((isConnected: boolean) => {
    if (typeof window === 'undefined') return
    window.dispatchEvent(new CustomEvent('camera-connection-change', { detail: { isConnected } }))
  }, [])

  // --- Data Loading Effects ---
  useEffect(() => {
    // Load available models
    const loadModels = async () => {
      try {
        const res = await fetch('/api/models'); const data = await res.json()
        setAvailableModels(Array.isArray(data) ? data.map((m: any) => ({ id: m.id, name: m.name, type: 'Object Detection', size: m.framework || 'Unknown' })) : [])
      } catch (e) { console.error('Failed to load models:', e); toast({ variant: "destructive", title: "모델 로딩 실패" }) }
    }; loadModels()
    // Load available cameras
    const loadCameras = async () => {
      try {
        setIsLoadingCameras(true)
        const res = await fetch('/api/camera/list'); const data = await res.json()
        if (data.cameras && data.cameras.length > 0) {
          setAvailableCameras(data.cameras)
          if (!selectedCamera) setSelectedCamera(data.cameras[0].device_id)
        } else {
          toast({ variant: "destructive", title: "카메라가 없습니다", description: "장치 연결 및 권한을 확인하세요." })
        }
      } catch (e) { console.error('Failed to load cameras:', e); toast({ variant: "destructive", title: "카메라 로딩 실패" }) }
      finally { setIsLoadingCameras(false) }
    }; loadCameras()
  }, [toast])


  // --- Main SSE Handler for Video, Stats, and Capture Events ---
  useEffect(() => {
    if (cameraStatus !== 'connected') {
      if (sseRef.current) { sseRef.current.close(); sseRef.current = null }
      return
    }

    const sseUrl = '/api/camera/stream'
    const eventSource = new EventSource(sseUrl)
    sseRef.current = eventSource

    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    const img = imageRef.current

    // 최신 프레임만 유지하면서 requestAnimationFrame으로 그리기 → 메인 스레드 점유 최소화
    const latestFrameRef = { frameBase64: '', detections: [] as any[] }
    let rafId: number | null = null
    let lastDraw = 0
    const targetIntervalMs = 1000 / 15 // 15fps 제한

    const drawFrame = () => {
      if (!ctx || !canvas || !latestFrameRef.frameBase64) {
        rafId = null
        return
      }
      const now = performance.now()
      if (now - lastDraw < targetIntervalMs) {
        rafId = requestAnimationFrame(drawFrame)
        return
      }
      lastDraw = now
      img.src = `data:image/jpeg;base64,${latestFrameRef.frameBase64}`
      img.onload = () => {
        const hRatio = canvas.width / img.width
        const vRatio = canvas.height / img.height
        const ratio = Math.min(hRatio, vRatio)
        const centerShift_x = (canvas.width - img.width * ratio) / 2
        const centerShift_y = (canvas.height - img.height * ratio) / 2

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(img, 0, 0, img.width, img.height, centerShift_x, centerShift_y, img.width * ratio, img.height * ratio)

        latestFrameRef.detections.forEach((det: any) => {
          const x1 = det.bbox.x1 * (img.width * ratio) + centerShift_x
          const y1 = det.bbox.y1 * (img.height * ratio) + centerShift_y
          const width = (det.bbox.x2 - det.bbox.x1) * (img.width * ratio)
          const height = (det.bbox.y2 - det.bbox.y1) * (img.height * ratio)

          const color = get_class_color(det.class_id)
          ctx.strokeStyle = color
          ctx.lineWidth = 2
          ctx.strokeRect(x1, y1, width, height)

          const label = `${det.class_name} ${det.confidence.toFixed(2)}`
          ctx.fillStyle = color
          ctx.font = '14px sans-serif'
          const textWidth = ctx.measureText(label).width
          ctx.fillRect(x1, y1 - 20, textWidth + 8, 20)
          ctx.fillStyle = 'white'
          ctx.fillText(label, x1 + 4, y1 - 5)
        })

        rafId = requestAnimationFrame(drawFrame)
      }
    }

    // Listener for video frames
    eventSource.addEventListener('video_frame', (event) => {
      if (!ctx || !canvas) return
      try {
        const data = JSON.parse(event.data)
        latestFrameRef.frameBase64 = data.frame
        latestFrameRef.detections = data.detections || []

        if (rafId === null) {
          rafId = requestAnimationFrame(drawFrame)
        }
      } catch (err) {
        console.error('Failed to parse video frame', err)
      }
    })

    eventSource.onerror = () => {
      toast({ variant: "destructive", title: "스트림 연결 오류", description: "서버와의 연결이 끊겼습니다." })
      eventSource.close()
      setCameraStatus('error')
    }

    return () => {
      if (rafId) cancelAnimationFrame(rafId)
      if (sseRef.current) { sseRef.current.close(); sseRef.current = null }
    }
  }, [cameraStatus, toast])


  // --- Control Handlers ---

  const toggleCamera = useCallback(async () => {
    if (!selectedCamera) { toast({ variant: "destructive", title: "카메라를 선택하세요" }); return }

    if (cameraStatus === 'connected') {
      try { await fetch('/api/camera/stop', { method: 'POST' }) } catch (e) { console.error(e) }

      // Clear canvas to black
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (ctx && canvas) {
        ctx.fillStyle = 'black'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
      }

      setCameraStatus('disconnected'); setIsInferenceActive(false); setModelStatus('idle'); notifyCameraStatus(false)
      setOperationInProgress(false)
    } else {
      setCameraStatus('connecting')
      try {
        const res = await fetch('/api/camera/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device: selectedCamera }) })
        if (!res.ok) throw new Error((await res.json()).detail || 'Failed to start camera')
        setCameraStatus('connected'); notifyCameraStatus(true)
        setOperationInProgress(true, 'camera')
      } catch (e: any) {
        setCameraStatus('error'); toast({ variant: "destructive", title: "카메라 연결 실패", description: e.message })
        setOperationInProgress(false)
      }
    }
  }, [selectedCamera, cameraStatus, toast, notifyCameraStatus, setOperationInProgress])

  const toggleInference = useCallback(async () => {
    const apiEndpoint = isInferenceActive ? '/api/camera/detection/stop' : '/api/camera/detection/start'
    const successState = isInferenceActive ? 'ready' : 'running'
    const failState = 'error'

    try {
      const body = isInferenceActive ? undefined : JSON.stringify({ model_path: `${selectedModel}.pt` })
      const res = await fetch(apiEndpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body })
      if (!res.ok) throw new Error((await res.json()).detail || 'Failed to toggle detection')

      setModelStatus(successState)
      setIsInferenceActive(!isInferenceActive)

      // Update operation state
      if (!isInferenceActive) {
        setOperationInProgress(true, 'inference')
      } else if (cameraStatus !== 'connected') {
        // Only clear operation if camera is also disconnected
        setOperationInProgress(false)
      }
    } catch (e: any) {
      setModelStatus(failState); toast({ variant: "destructive", title: "추론 제어 실패", description: e.message })
    }
  }, [isInferenceActive, selectedModel, toast, cameraStatus, setOperationInProgress])

  const startCapture = useCallback(async () => {
    // Validate capture name is required
    if (!captureName.trim()) {
      toast({ variant: "destructive", title: "캡처 이름 필수", description: "캡처 이름을 입력해주세요" })
      return
    }

    if (captureNameError) {
      toast({ variant: "destructive", title: "잘못된 캡처 이름", description: captureNameError })
      return
    }

    if (!validateName(captureName)) {
      setCaptureNameError(getNameValidationMessage('캡처 이름'))
      return
    }

    setShowCaptureDialog(false)
    setIsCapturing(true)

    try {
        toast({ title: "캡처 시작", description: "10장의 프레임 캡처를 시작합니다." })

        const res = await fetch('/api/camera/capture/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: captureName }) })

        if (!res.ok) {
          const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }))
          throw new Error(errorData.detail || 'Failed to start capture')
        }

        await res.json()
        setIsCapturing(false)
        toast({
          title: "캡처 완료",
          description: `프레임 캡처를 성공적으로 완료했습니다.`
        })

        // Clear capture name for next time
        setCaptureName("")
    } catch(e: any) {
        setIsCapturing(false)
        toast({ variant: "destructive", title: "캡처 실패", description: e.message })
    }
  }, [captureName, toast])

  const handleCaptureNameChange = useCallback((value: string) => {
    // Only allow valid characters (same as model name validation)
    if (value === '' || validateName(value)) {
      setCaptureName(value)
      setCaptureNameError(null)
    } else {
      // Show error message but don't update the value (input is blocked)
      setCaptureNameError(getNameValidationMessage('캡처 이름'))
    }
  }, [])

  const handleReset = useCallback(async () => {
    try {
      // Stop inference if active
      if (isInferenceActive) {
        await fetch('/api/camera/detection/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' } })
        setIsInferenceActive(false)
        setModelStatus('idle')
      }

      // Stop camera if connected
      if (cameraStatus === 'connected') {
        await fetch('/api/camera/stop', { method: 'POST' })

        // Clear canvas to black
        const canvas = canvasRef.current
        const ctx = canvas?.getContext('2d')
        if (ctx && canvas) {
          ctx.fillStyle = 'black'
          ctx.fillRect(0, 0, canvas.width, canvas.height)
        }

        setCameraStatus('disconnected')
        notifyCameraStatus(false)
      }

      // Clear operation state
      setOperationInProgress(false)

      toast({ title: "초기화 완료", description: "카메라와 추론이 중지되었습니다." })
    } catch (e: any) {
      toast({ variant: "destructive", title: "초기화 실패", description: e.message })
    }
  }, [isInferenceActive, cameraStatus, toast, notifyCameraStatus, setOperationInProgress])

  // Check if reset button should be shown
  const showResetButton = cameraStatus === 'connected' || isInferenceActive

  // --- UI Rendering ---
  return (
    <AdversarialToolLayout
      title="실시간 카메라"
      description="실물 객체에 대한 실시간 AI 모델 성능 검증"
      icon={Video}
      leftPanelWidth="lg"
      leftPanel={{
        title: "제어 패널",
        icon: Activity,
        description: "카메라 및 AI 모델 제어",
        children: (
          <div className="space-y-6">
            {/* Camera Control Card */}
            <Card variant="elevated" className="bg-surface-container border-outline-variant">
              <CardHeader className="px-4">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Camera className="w-4 h-4 text-primary" />
                  카메라 제어
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 px-4 pt-0">
                <div className="flex gap-2">
                  <Select value={selectedCamera} onValueChange={setSelectedCamera} disabled={cameraStatus === 'connected'}>
                    <SelectTrigger className="bg-surface-container-high border-outline text-foreground flex-1"><SelectValue placeholder={isLoadingCameras ? "불러오는 중..." : "카메라 선택"} /></SelectTrigger>
                    <SelectContent className="bg-surface-container border-outline">
                      {availableCameras.map(c => <SelectItem key={c.device_id} value={c.device_id} className="text-foreground">{c.name}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <Button className="ds-btn-outline flex-shrink-0" disabled={isLoadingCameras || cameraStatus === 'connected'} onClick={async () => {
                    setIsLoadingCameras(true)
                    try {
                      const data = await fetch('/api/camera/list').then(r => r.json())
                      if (data.cameras && data.cameras.length > 0) {
                        setAvailableCameras(data.cameras)
                        setSelectedCamera(data.cameras[0].device_id)
                      } else {
                        toast({ variant: "destructive", title: "카메라가 없습니다", description: "장치 연결 및 권한을 확인하세요." })
                      }
                    } catch (err) {
                      console.error(err); toast({ variant: "destructive", title: "카메라 새로고침 실패" })
                    } finally {
                      setIsLoadingCameras(false)
                    }
                  }}>새로고침</Button>
                </div>
                <Button onClick={toggleCamera} disabled={cameraStatus === 'connecting' || isCapturing} className="w-full ds-btn-primary">{cameraStatus === 'connected' ? '카메라 해제' : '카메라 연결'}</Button>
              </CardContent>
            </Card>

            {/* AI Model Control Card */}
            <Card variant="elevated" className="bg-surface-container border-outline-variant">
              <CardHeader className="px-4">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Brain className="w-4 h-4 text-tertiary" />
                  AI 모델 제어
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 px-4 pt-0">
                <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isInferenceActive}>
                    <SelectTrigger className="bg-surface-container-high border-outline text-foreground"><SelectValue placeholder="모델 선택" /></SelectTrigger>
                    <SelectContent className="bg-surface-container border-outline">
                        {availableModels.map(m => <SelectItem key={m.id} value={m.id} className="text-foreground">{m.name}</SelectItem>)}
                    </SelectContent>
                </Select>
                <Button onClick={toggleInference} disabled={cameraStatus !== 'connected' || !selectedModel || isCapturing} className="w-full ds-btn-primary">{isInferenceActive ? '추론 중지' : '실시간 추론 시작'}</Button>
              </CardContent>
            </Card>

            {/* Reset Button */}
            {showResetButton && (
              <Button
                onClick={handleReset}
                disabled={isCapturing}
                className="w-full ds-btn-danger"
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                초기화 (카메라 & 추론 중지)
              </Button>
            )}
          </div>
        )
      }}
      rightPanel={{
        title: "실시간 카메라 화면",
        icon: Monitor,
        description: "연결된 카메라 영상 및 실시간 추론 결과",
        children: (
          <div className="flex flex-col gap-4 h-full">
            <div className="relative bg-scrim rounded-lg overflow-hidden aspect-video">
              <canvas ref={canvasRef} width="640" height="480" className="w-full h-full object-contain" />
              {cameraStatus !== 'connected' && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center"><CameraOff className="w-16 h-16 text-muted mx-auto mb-4" /><p className="text-muted">{cameraStatus === 'connecting' ? '카메라 연결 중...' : '카메라 연결 안됨'}</p></div>
                </div>
              )}
            </div>

            {/* Capture UI */}
            {isInferenceActive && (
              <div className="space-y-4">
                <Button onClick={() => setShowCaptureDialog(true)} disabled={isCapturing} className="w-full ds-btn-primary">
                  {isCapturing ? '캡처 중...' : '5초간 캡처 시작 (10장)'}
                </Button>
              </div>
            )}
            
            {/* Dialog for capture name */}
            <Dialog open={showCaptureDialog} onOpenChange={setShowCaptureDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>캡처 세션 이름</DialogTitle>
                  <DialogDescription className="text-muted">
                    캡처할 프레임 세트의 이름을 입력하세요
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="capture-name" className="text-foreground">
                      세션 이름 <span className="text-error">*</span>
                    </Label>
                    <Input
                      id="capture-name"
                      placeholder="예: object_test_1"
                      value={captureName}
                      onChange={(e) => handleCaptureNameChange(e.target.value)}
                      className={`bg-surface-container-high border-border text-foreground ${captureNameError ? 'border-error' : ''}`}
                    />
                    {captureNameError ? (
                      <p className="text-xs text-error">{captureNameError}</p>
                    ) : (
                      <p className="text-xs text-muted">영문자, 숫자, 대시(-), 언더스코어(_)만 사용 가능</p>
                    )}
                  </div>
                </div>
                <DialogFooter>
                  <Button
                    onClick={() => setShowCaptureDialog(false)}
                    className="ds-btn-outline"
                  >
                    취소
                  </Button>
                  <Button
                    onClick={startCapture}
                    disabled={!captureName.trim() || !!captureNameError}
                    className="ds-btn-primary"
                  >
                    캡처 시작
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        )
      }}
    />
  )
}

// Memoize component to prevent re-renders from parent state changes
export const RealTimeCamera = memo(RealTimeCameraComponent)
