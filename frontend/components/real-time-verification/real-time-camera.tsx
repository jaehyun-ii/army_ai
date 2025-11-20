"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useToast } from "@/hooks/use-toast"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
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
} from "lucide-react"

// No longer needed - using Next.js API routes

interface ModelInfo {
  id: string
  name: string
  type: string
  size: string
}

interface PerformanceMetrics {
  fps: number
  detectedObjects: number
  processingTime: number
  totalFrames: number
  successfulDetections: number
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

export function RealTimeCamera() {
  const { toast } = useToast()
  const [cameraStatus, setCameraStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected')
  const [modelStatus, setModelStatus] = useState<'idle' | 'loading' | 'ready' | 'running' | 'error'>('idle')
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [selectedCamera, setSelectedCamera] = useState<string>("0")
  const [availableCameras, setAvailableCameras] = useState<CameraDevice[]>([])
  const [isInferenceActive, setIsInferenceActive] = useState(false)
  const [streamKey, setStreamKey] = useState(Date.now())
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    fps: 0,
    detectedObjects: 0,
    processingTime: 0,
    totalFrames: 0,
    successfulDetections: 0
  })
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([])
  const [isCapturing, setIsCapturing] = useState(false)

  const sseRef = useRef<EventSource | null>(null)

  // Load available models
  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await fetch('/api/models')
        const data = await response.json()
        const models = Array.isArray(data) ? data.map((model: any) => ({
          id: model.id,
          name: model.name,
          type: 'Object Detection',
          size: model.framework || 'Unknown'
        })) : []
        setAvailableModels(models)
      } catch (error) {
        console.error('Failed to load models:', error)
        toast({
          variant: "destructive",
          title: "모델 로딩 실패",
          description: "모델 목록을 불러오는데 실패했습니다."
        })
      }
    }
    loadModels()
  }, [toast])

  // Load available cameras
  useEffect(() => {
    const loadCameras = async () => {
      try {
        const response = await fetch('/api/camera/list')
        const data = await response.json()
        if (data.cameras && Array.isArray(data.cameras)) {
          setAvailableCameras(data.cameras)
          if (data.cameras.length > 0 && !selectedCamera) {
            setSelectedCamera(data.cameras[0].device_id)
          }
        }
      } catch (error) {
        console.error('Failed to load cameras:', error)
        toast({
          variant: "destructive",
          title: "카메라 로딩 실패",
          description: "카메라 목록을 불러오는데 실패했습니다."
        })
      }
    }
    loadCameras()
  }, [toast])

  // Sync state with backend
  useEffect(() => {
    const syncState = async () => {
      try {
        const response = await fetch('/api/camera/status')
        const data = await response.json()
        if (data.is_active) {
          setCameraStatus('connected')
          setModelStatus('running')
          setIsInferenceActive(true)
          if (data.model_id) {
            setSelectedModel(data.model_id.replace('.pt', ''))
          }
        }
      } catch (error) {
        console.error('Failed to sync state:', error)
      }
    }
    syncState()
  }, [toast])

  // Start capture
  const startCapture = async () => {
    if (cameraStatus !== 'connected') {
      toast({
        variant: "destructive",
        title: "카메라 미연결",
        description: "카메라를 먼저 연결해주세요"
      })
      return
    }

    if (!isInferenceActive || modelStatus !== 'running') {
      toast({
        variant: "destructive",
        title: "객체 감지 미실행",
        description: "객체 감지를 먼저 시작해주세요"
      })
      return
    }

    setIsCapturing(true)
    try {
      const response = await fetch('/api/camera/capture/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      })

      if (!response.ok) {
        let errorMessage = 'Failed to start capture'
        try {
          const data = await response.json()
          errorMessage = data.error || data.details || errorMessage
        } catch (e) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`
        }
        throw new Error(errorMessage)
      }

      await new Promise(resolve => setTimeout(resolve, 6000))

      toast({
        title: "캡처 완료",
        description: "5초간의 캡처가 성공적으로 완료되었습니다."
      })
    } catch (error) {
      console.error('Capture failed:', error)
      toast({
        variant: "destructive",
        title: "캡처 실패",
        description: `캡처 실패: ${error}`
      })
    } finally {
      setIsCapturing(false)
    }
  }

  // Toggle camera
  const toggleCamera = async () => {
    if (cameraStatus === 'connected') {
      try {
        await fetch('/api/camera/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        })
      } catch (error) {
        console.error('카메라 해제 실패:', error)
      }
      setCameraStatus('disconnected')
      setIsInferenceActive(false)
    } else {
      setCameraStatus('connecting')
      try {
        const response = await fetch('/api/camera/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device: selectedCamera })
        })

        if (!response.ok) {
          const data = await response.json()
          throw new Error(data.error || 'Failed to start camera')
        }

        const data = await response.json()
        if (data.status === 'started' || data.status === 'already_active') {
          setCameraStatus('connected')
          setStreamKey(Date.now())
        } else {
          throw new Error('Unexpected response from camera API')
        }
      } catch (error) {
        console.error('카메라 연결 실패:', error)
        setCameraStatus('error')
        toast({
          variant: "destructive",
          title: "카메라 연결 실패",
          description: "카메라 연결에 실패했습니다."
        })
      }
    }
  }

  // Model auto-ready when selected
  useEffect(() => {
    if (selectedModel && !isInferenceActive) {
      setModelStatus('ready')
    }
  }, [selectedModel, isInferenceActive])

  // Toggle inference
  const toggleInference = async () => {
    if (isInferenceActive) {
      try {
        const response = await fetch('/api/camera/detection/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        })

        if (!response.ok) {
          const errorData = await response.json()
          toast({
            variant: "destructive",
            title: "추론 중지 실패",
            description: errorData.error || errorData.details || 'Unknown error'
          })
          return
        }

        setIsInferenceActive(false)
        setModelStatus('ready')
        setStreamKey(Date.now())
      } catch (error) {
        console.error('Failed to stop detection:', error)
        toast({
          variant: "destructive",
          title: "추론 중지 오류",
          description: `추론 중지 중 오류 발생: ${error}`
        })
      }
    } else {
      if (modelStatus === 'ready' && cameraStatus === 'connected' && selectedModel) {
        try {
          const modelPath = `${selectedModel}.pt`
          const response = await fetch('/api/camera/detection/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model_path: modelPath,
              confidence_threshold: 0.25
            })
          })

          if (!response.ok) {
            const data = await response.json()
            const errorMessage = data.error || data.details || 'Failed to start detection'
            toast({
              variant: "destructive",
              title: "추론 시작 실패",
              description: errorMessage
            })
            setModelStatus('error')
            return
          }

          setIsInferenceActive(true)
          setModelStatus('running')
          setStreamKey(Date.now())
        } catch (error) {
          console.error('Failed to start detection:', error)
          toast({
            variant: "destructive",
            title: "추론 시작 오류",
            description: `추론 시작 중 오류 발생: ${error}`
          })
          setModelStatus('error')
        }
      } else {
        if (!selectedModel) {
          toast({
            variant: "destructive",
            title: "모델 미선택",
            description: "모델을 먼저 선택해주세요"
          })
        } else if (cameraStatus !== 'connected') {
          toast({
            variant: "destructive",
            title: "카메라 미연결",
            description: "카메라를 먼저 연결해주세요"
          })
        }
      }
    }
  }

  // Real-time metrics via SSE
  useEffect(() => {
    if (isInferenceActive && modelStatus === 'running') {
      const sseUrl = '/api/camera/stats/stream'
      const eventSource = new EventSource(sseUrl)
      sseRef.current = eventSource

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.is_active && data.stats) {
            setMetrics({
              fps: data.stats.fps || 0,
              detectedObjects: data.stats.detections || 0,
              processingTime: data.stats.inference_time_ms || 0,
              totalFrames: data.stats.frame_count || 0,
              successfulDetections: data.stats.frame_count || 0
            })
          }
        } catch (error) {
          console.error('Failed to parse SSE data:', error)
        }
      }

      eventSource.onerror = () => {
        eventSource.close()
        sseRef.current = null
      }

      return () => {
        eventSource.close()
        sseRef.current = null
      }
    } else {
      if (sseRef.current) {
        sseRef.current.close()
        sseRef.current = null
      }
    }
  }, [isInferenceActive, modelStatus])

  const selectedModelInfo = availableModels.find(m => m.id === selectedModel)

  return (
    <AdversarialToolLayout
      title="실시간 카메라"
      description="실물 객체에 대한 실시간 AI 모델 성능 검증"
      icon={Video}
      headerStats={
        <Badge
          variant={cameraStatus === 'connected' && modelStatus === 'running' ? "default" : "secondary"}
          className="px-3 py-1"
        >
          {cameraStatus === 'connected' && modelStatus === 'running' ? '실행 중' : '대기 중'}
        </Badge>
      }
      leftPanelWidth="lg"
      leftPanel={{
        title: "제어 패널",
        icon: Activity,
        description: "카메라 및 AI 모델 제어",
        children: (
          <div className="space-y-3">
            {/* 카메라 제어 */}
            <Card className="bg-slate-800/50 border-white/10">
              <CardHeader className="px-4">
                <CardTitle className="text-white flex items-center gap-2 text-sm">
                  <Camera className="w-4 h-4" />
                  카메라 제어
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 px-4 pt-0">
                <div>
                  <Select
                    value={selectedCamera}
                    onValueChange={setSelectedCamera}
                    disabled={cameraStatus === 'connected'}
                  >
                    <SelectTrigger className="bg-slate-700/50 border-white/20 text-white ">
                      <SelectValue placeholder="카메라를 선택하세요" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-white/20">
                      {availableCameras.length > 0 ? (
                        availableCameras.map((camera) => (
                          <SelectItem key={camera.device_id} value={camera.device_id} className="text-white">
                            <div>
                              <div className="font-medium">{camera.name}</div>
                              <div className="text-xs text-slate-400">
                                {camera.width}x{camera.height} @ {camera.fps}fps
                              </div>
                            </div>
                          </SelectItem>
                        ))
                      ) : (
                        <SelectItem value="0" className="text-white">
                          Camera 0 (Default)
                        </SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-slate-300">상태:</span>
                  <Badge variant={cameraStatus === 'connected' ? "default" : "secondary"}>
                    {cameraStatus === 'connected' ? '연결됨' :
                     cameraStatus === 'connecting' ? '연결 중' :
                     cameraStatus === 'error' ? '오류' : '연결 안됨'}
                  </Badge>
                </div>

                <Button
                  onClick={toggleCamera}
                  disabled={cameraStatus === 'connecting'}
                  className={`w-full ${cameraStatus === 'connected' ? 'bg-red-600 hover:bg-red-700 text-white' : ''}`}
                  variant={cameraStatus === 'connected' ? undefined : "default"}
                >
                  {cameraStatus === 'connected' ? (
                    <>
                      <CameraOff className="w-4 h-4 mr-2" />
                      카메라 해제
                    </>
                  ) : (
                    <>
                      <Camera className="w-4 h-4 mr-2" />
                      카메라 연결
                    </>
                  )}
                </Button>
                {cameraStatus === 'error' && (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      카메라 연결에 실패했습니다.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {/* AI 모델 제어 */}
            <Card className="bg-slate-800/50 border-white/10">
              <CardHeader className="px-4">
                <CardTitle className="text-white flex items-center gap-2 text-sm">
                  <Brain className="w-4 h-4" />
                  AI 모델 제어
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 p-4 pt-0">
                <div>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger className="bg-slate-700/50 border-white/20 text-white">
                      <SelectValue placeholder="모델을 선택하세요" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-white/20">
                      {availableModels.map((model) => (
                        <SelectItem key={model.id} value={model.id} className="text-white">
                          <div>
                            <div className="font-medium">{model.name}</div>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {selectedModelInfo && (
                  <div className="bg-slate-700/30 rounded-lg p-3 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">타입:</span>
                      <span className="text-white">{selectedModelInfo.type}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">모델 크기:</span>
                      <span className="text-white">{selectedModelInfo.size}</span>
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between">
                  <span className="text-slate-300">모델 상태:</span>
                  <Badge variant={
                    modelStatus === 'ready' ? "default" :
                    modelStatus === 'running' ? "default" :
                    modelStatus === 'error' ? "destructive" : "outline"
                  }>
                    {modelStatus === 'ready' ? '준비됨' :
                     modelStatus === 'running' ? '실행 중' :
                     modelStatus === 'error' ? '오류' : '대기'}
                  </Badge>
                </div>

                <Button
                  onClick={toggleInference}
                  disabled={
                    cameraStatus !== 'connected' ||
                    (!isInferenceActive && modelStatus !== 'ready')
                  }
                  className={`w-full ${isInferenceActive ? 'bg-red-600 hover:bg-red-700 text-white' : ''}`}
                  variant={isInferenceActive ? undefined : "default"}
                >
                  {isInferenceActive ? (
                    <>
                      <Square className="w-4 h-4 mr-2" />
                      추론 중지
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      실시간 추론 시작
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
        )
      }}
      rightPanel={{
        title: "실시간 카메라 화면",
        icon: Monitor,
        description: "연결된 카메라 영상 및 실시간 추론 결과",
        children: (
          <div className="flex flex-col gap-4 h-full">
                <div className="relative bg-black rounded-lg overflow-hidden" style={{ width: '100%', height: '480px' }}>
                  {cameraStatus === 'connected' ? (
                    <>
                      <img
                        key={streamKey}
                        src={`/api/camera/stream?draw_boxes=${isInferenceActive}&t=${streamKey}`}
                        alt="Camera stream"
                        className="w-full h-full object-contain"
                        style={{ display: 'block', maxWidth: '640px', maxHeight: '480px', margin: '0 auto' }}
                        onError={() => {
                          toast({
                            variant: "destructive",
                            title: "스트림 연결 실패",
                            description: "카메라 스트림을 불러올 수 없습니다."
                          })
                          setTimeout(() => setStreamKey(Date.now()), 2000)
                        }}
                      />
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-full">
                      <div className="text-center">
                        <CameraOff className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                        <p className="text-slate-400">
                          {cameraStatus === 'connecting' ? '카메라 연결 중...' :
                           cameraStatus === 'error' ? '카메라 연결 실패' :
                           '카메라를 연결하세요'}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* 상태 정보 오버레이 */}
                  {cameraStatus === 'connected' && (
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                      <div className="flex items-center justify-between text-white">
                        <div className="flex items-center gap-4">
                          <div className="flex items-center gap-2">
                            <Zap className="w-4 h-4" />
                            <span className="text-sm">{metrics.fps.toFixed(1)} FPS</span>
                          </div>
                          {selectedModelInfo && (
                            <div className="flex items-center gap-2">
                              <Brain className="w-4 h-4" />
                              <span className="text-sm">{selectedModelInfo.name}</span>
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${isInferenceActive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`} />
                          <span className="text-sm">{isInferenceActive ? '추론 중' : '대기'}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Capture Button */}
                {isInferenceActive && (
                  <div className="mt-4">
                    <Button
                      onClick={startCapture}
                      disabled={isCapturing}
                      className="w-full bg-green-600 hover:bg-green-700 text-white"
                    >
                      {isCapturing ? (
                        <>
                          <Activity className="w-4 h-4 mr-2 animate-spin" />
                          캡처 중... (5초)
                        </>
                      ) : (
                        <>
                          <Camera className="w-4 h-4 mr-2" />
                          5초간 캡처 시작 (10장)
                        </>
                      )}
                    </Button>
                  </div>
                )}
          </div>
        )
      }}
    />
  )
}
