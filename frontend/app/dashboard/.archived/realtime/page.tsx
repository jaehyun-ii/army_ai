"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { toast } from "sonner"
import {
  Camera,
  Play,
  Square,
  Video,
  AlertCircle,
  Activity,
  Zap,
  Eye
} from "lucide-react"
import {
  fetchAvailableCameras,
  fetchYoloModels,
  startRealtimeSession,
  stopRealtimeSession,
  fetchRealtimeStats,
  getRealtimeMJPEGUrl,
  type Camera as CameraType,
  type YoloModel,
  type RealtimeStats
} from "@/lib/realtime-api"

export default function RealtimeInferencePage() {
  // State
  const [cameras, setCameras] = useState<CameraType[]>([])
  const [models, setModels] = useState<YoloModel[]>([])
  const [selectedCamera, setSelectedCamera] = useState<string>("")
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [isStreaming, setIsStreaming] = useState(false)
  const [currentSession, setCurrentSession] = useState<any>(null)
  const [stats, setStats] = useState<RealtimeStats>({
    run_id: '',
    frame_count: 0,
    fps: 0,
    detections: 0,
    inference_time_ms: 0,
    last_update: 0
  })

  const statsIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const videoRef = useRef<HTMLImageElement>(null)

  // Load cameras and models on mount
  useEffect(() => {
    loadCameras()
    loadModels()

    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current)
      }
    }
  }, [])

  const loadCameras = async () => {
    try {
      const cameraList = await fetchAvailableCameras()
      setCameras(cameraList)
      if (cameraList.length > 0) {
        setSelectedCamera(cameraList[0].device_id)
      }
    } catch (error) {
      console.error('Failed to load cameras:', error)
      toast.error('카메라 목록을 불러오지 못했습니다')
    }
  }

  const loadModels = async () => {
    try {
      const modelList = await fetchYoloModels()
      setModels(modelList)
      if (modelList.length > 0) {
        setSelectedModel(modelList[0].id)
      }
    } catch (error) {
      console.error('Failed to load models:', error)
      toast.error('모델 목록을 불러오지 못했습니다')
    }
  }

  const handleStartInference = async () => {
    if (!selectedCamera || !selectedModel) {
      toast.error('카메라와 모델을 선택해주세요')
      return
    }

    try {
      // Start realtime session
      const session = await startRealtimeSession({
        run_name: `Realtime_${new Date().toISOString()}`,
        device: selectedCamera,
        model_id: selectedModel,
        fps_target: 30,
        conf_threshold: 0.25
      })

      setCurrentSession(session)
      setIsStreaming(true)
      toast.success('실시간 추론이 시작되었습니다')

      // Set MJPEG video source
      if (videoRef.current && session.run_id) {
        const mjpegUrl = getRealtimeMJPEGUrl(session.run_id, selectedModel, {
          conf_threshold: 0.25,
          draw_boxes: true
        })
        videoRef.current.src = mjpegUrl
      }

      // Start polling stats
      const pollStats = async () => {
        try {
          const statsData = await fetchRealtimeStats(session.run_id)
          setStats(statsData)
        } catch (error) {
          console.error('Failed to fetch stats:', error)
        }
      }

      // Poll every 500ms
      statsIntervalRef.current = setInterval(pollStats, 500)

    } catch (error) {
      console.error('Failed to start inference:', error)
      toast.error('추론 시작에 실패했습니다')
      setIsStreaming(false)
    }
  }

  const handleStopInference = async () => {
    try {
      // Stop stats polling
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current)
        statsIntervalRef.current = null
      }

      // Stop video
      if (videoRef.current) {
        videoRef.current.src = ''
      }

      // Stop backend session
      if (currentSession) {
        await stopRealtimeSession(currentSession.run_id)
      }

      setIsStreaming(false)
      setCurrentSession(null)
      setStats({
        run_id: '',
        frame_count: 0,
        fps: 0,
        detections: 0,
        inference_time_ms: 0,
        last_update: 0
      })
      toast.success('추론이 종료되었습니다')

    } catch (error) {
      console.error('Failed to stop inference:', error)
      toast.error('추론 종료에 실패했습니다')
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">실시간 객체 인식</h1>
          <p className="text-muted-foreground mt-2">
            카메라를 연결하여 실시간으로 객체를 인식합니다
          </p>
        </div>
        <Badge variant={isStreaming ? "default" : "secondary"} className="text-lg px-4 py-2">
          <Activity className={`w-4 h-4 mr-2 ${isStreaming ? 'animate-pulse' : ''}`} />
          {isStreaming ? '스트리밍 중' : '대기 중'}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Control Panel */}
        <div className="lg:col-span-1 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="w-5 h-5" />
                설정
              </CardTitle>
              <CardDescription>카메라와 모델을 선택하세요</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Camera Selection */}
              <div className="space-y-2">
                <Label>카메라 선택</Label>
                <Select
                  value={selectedCamera}
                  onValueChange={setSelectedCamera}
                  disabled={isStreaming}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="카메라를 선택하세요" />
                  </SelectTrigger>
                  <SelectContent>
                    {cameras.map((camera) => (
                      <SelectItem key={camera.device_id} value={camera.device_id}>
                        {camera.name} ({camera.width}x{camera.height} @ {camera.fps}fps)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {cameras.length === 0 && (
                  <p className="text-sm text-muted-foreground flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    사용 가능한 카메라가 없습니다
                  </p>
                )}
              </div>

              {/* Model Selection */}
              <div className="space-y-2">
                <Label>AI 모델 선택</Label>
                <Select
                  value={selectedModel}
                  onValueChange={setSelectedModel}
                  disabled={isStreaming}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="모델을 선택하세요" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        {model.name} ({model.framework})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Control Buttons */}
              <div className="flex gap-2 pt-4">
                {!isStreaming ? (
                  <Button
                    onClick={handleStartInference}
                    disabled={!selectedCamera || !selectedModel}
                    className="flex-1"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    추론 시작
                  </Button>
                ) : (
                  <Button
                    onClick={handleStopInference}
                    variant="destructive"
                    className="flex-1"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    추론 중지
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Stats Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                실시간 통계
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">FPS</span>
                <span className="text-xl font-bold">{stats.fps.toFixed(1)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">검출 객체</span>
                <span className="text-xl font-bold">{stats.detections}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">추론 시간</span>
                <span className="text-xl font-bold">{stats.inference_time_ms.toFixed(1)}ms</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">총 프레임</span>
                <span className="text-xl font-bold">{stats.frame_count}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Video Display */}
        <div className="lg:col-span-2">
          <Card className="h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Video className="w-5 h-5" />
                실시간 영상
              </CardTitle>
              <CardDescription>
                {isStreaming ? '검출된 객체가 바운딩 박스로 표시됩니다' : '추론을 시작하면 영상이 표시됩니다'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative bg-black rounded-lg overflow-hidden aspect-video flex items-center justify-center">
                {isStreaming ? (
                  <img
                    ref={videoRef}
                    alt="Realtime video stream"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-muted-foreground flex flex-col items-center gap-2">
                    <Eye className="w-12 h-12 opacity-20" />
                    <p>영상 대기 중</p>
                  </div>
                )}
              </div>

            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
