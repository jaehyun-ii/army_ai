"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Play,
  Pause,
  Square,
  Settings,
  Download,
  Eye,
  Map,
  Cloud,
  Car,
  AlertCircle,
  CheckCircle,
  Loader2,
  FolderOpen,
  Box,
  Sun,
  CloudRain,
  CloudFog,
  Moon,
  Sunrise,
  Monitor
} from "lucide-react"

interface GenerationConfig {
  datasetName: string
  map: string
  weather: string
  modelType: string
  imageResolution: string
  outputFormat: string
}

interface GenerationStatus {
  isGenerating: boolean
  progress: number
  totalImages: number
  generatedImages: number
  currentStatus: string
  errors: string[]
  logs: string[]
}

export function DataGeneration3DUpdated() {
  const [config, setConfig] = useState<GenerationConfig>({
    datasetName: "",
    map: "Town01",
    weather: "ClearNoon",
    modelType: "sedan",
    imageResolution: "640x640",
    outputFormat: "png"
  })

  const [status, setStatus] = useState<GenerationStatus>({
    isGenerating: false,
    progress: 0,
    totalImages: 1000, // Fixed default value
    generatedImages: 0,
    currentStatus: "대기 중",
    errors: [],
    logs: []
  })

  const simulationCanvasRef = useRef<HTMLCanvasElement>(null)
  const [rightCardHeight, setRightCardHeight] = useState<number>(600)

  // CARLA 맵 옵션
  const carlaMapOptions = [
    { value: "Town01", label: "Town01 - 작은 도시" },
    { value: "Town02", label: "Town02 - 상업 지구" },
    { value: "Town03", label: "Town03 - 대도시" },
    { value: "Town04", label: "Town04 - 산악 도시" },
    { value: "Town05", label: "Town05 - 도심 격자" },
    { value: "Town06", label: "Town06 - 고속도로" },
    { value: "Town07", label: "Town07 - 시골 마을" },
    { value: "Town10HD", label: "Town10HD - 도심 HD" }
  ]

  // 날씨 옵션
  const weatherOptions = [
    { value: "ClearNoon", label: "맑음 (낮)", icon: Sun },
    { value: "CloudyNoon", label: "흐림 (낮)", icon: Cloud },
    { value: "WetNoon", label: "비 (낮)", icon: CloudRain },
    { value: "WetCloudyNoon", label: "흐린 비 (낮)", icon: CloudRain },
    { value: "HardRainNoon", label: "폭우 (낮)", icon: CloudRain },
    { value: "ClearSunset", label: "맑음 (일몰)", icon: Sunrise },
    { value: "CloudySunset", label: "흐림 (일몰)", icon: Cloud },
    { value: "ClearNight", label: "맑음 (밤)", icon: Moon },
    { value: "CloudyNight", label: "흐림 (밤)", icon: Cloud },
    { value: "Foggy", label: "안개", icon: CloudFog }
  ]

  // 객체 탐지 모델용 일반적인 해상도
  const resolutionOptions = [
    { value: "416x416", label: "416×416"},
    { value: "640x640", label: "640×640"},
    { value: "512x512", label: "512×512"},
    { value: "608x608", label: "608×608"},
    { value: "832x832", label: "832×832"},
    { value: "1280x1280", label: "1280×1280"}
  ]

  // 시뮬레이션 시작
  const startGeneration = () => {
    if (!config.datasetName) {
      alert("데이터셋 이름을 입력해주세요.")
      return
    }

    setStatus({
      ...status,
      isGenerating: true,
      progress: 0,
      totalImages: 1000,
      generatedImages: 0,
      currentStatus: "CARLA 시뮬레이터 초기화 중...",
      errors: [],
      logs: [`[${new Date().toLocaleTimeString()}] 데이터 생성 시작: ${config.datasetName}`]
    })

    simulateGeneration(1000)
  }

  // 생성 시뮬레이션
  const simulateGeneration = (totalImages: number) => {
    let generated = 0
    const interval = setInterval(() => {
      generated += Math.floor(Math.random() * 10) + 5

      if (generated >= totalImages) {
        generated = totalImages
        clearInterval(interval)

        setStatus(prev => ({
          ...prev,
          isGenerating: false,
          progress: 100,
          generatedImages: generated,
          currentStatus: "생성 완료",
          logs: [...prev.logs, `[${new Date().toLocaleTimeString()}] 데이터 생성 완료: ${generated}개 이미지`]
        }))
      } else {
        const progress = (generated / totalImages) * 100

        setStatus(prev => ({
          ...prev,
          progress: progress,
          generatedImages: generated,
          currentStatus: `생성 중... (${generated}/${totalImages})`,
          logs: prev.logs.length < 10 ? [...prev.logs, `[${new Date().toLocaleTimeString()}] 이미지 생성: ${generated}/${totalImages}`] : prev.logs
        }))
      }
    }, 500)
  }

  // 생성 중지
  const stopGeneration = () => {
    setStatus(prev => ({
      ...prev,
      isGenerating: false,
      currentStatus: "중지됨",
      logs: [...prev.logs, `[${new Date().toLocaleTimeString()}] 사용자에 의해 중지됨`]
    }))
  }

  // 실시간 생성 화면 높이 측정
  useEffect(() => {
    const measureHeight = () => {
      const rightCard = document.getElementById('right-card');
      if (rightCard) {
        setRightCardHeight(rightCard.offsetHeight);
      }
    };

    measureHeight();
    window.addEventListener('resize', measureHeight);
    return () => window.removeEventListener('resize', measureHeight);
  }, []);

  // Canvas 시뮬레이션 렌더링
  useEffect(() => {
    if (simulationCanvasRef.current && status.isGenerating) {
      const canvas = simulationCanvasRef.current
      const ctx = canvas.getContext('2d')

      if (ctx) {
        const animate = () => {
          ctx.fillStyle = '#1a1a2e'
          ctx.fillRect(0, 0, canvas.width, canvas.height)

          // 도로 그리기
          ctx.fillStyle = '#444'
          ctx.fillRect(0, canvas.height * 0.6, canvas.width, canvas.height * 0.4)

          // 차선 그리기
          ctx.strokeStyle = '#fff'
          ctx.setLineDash([20, 10])
          ctx.beginPath()
          ctx.moveTo(0, canvas.height * 0.8)
          ctx.lineTo(canvas.width, canvas.height * 0.8)
          ctx.stroke()

          // 차량 시뮬레이션
          const carX = (Date.now() / 10) % (canvas.width + 100) - 50
          ctx.fillStyle = '#ff6b6b'
          ctx.fillRect(carX, canvas.height * 0.7, 60, 30)

          // 카메라 뷰 표시
          ctx.strokeStyle = '#00ff00'
          ctx.setLineDash([])
          ctx.lineWidth = 2
          ctx.strokeRect(10, 10, canvas.width - 20, canvas.height - 20)

          // 정보 표시
          ctx.fillStyle = '#00ff00'
          ctx.font = '14px monospace'
          ctx.fillText(`Map: ${config.map}`, 20, 30)
          ctx.fillText(`Weather: ${config.weather}`, 20, 50)
          ctx.fillText(`Model: ${config.modelType}`, 20, 70)
          ctx.fillText(`Progress: ${status.progress.toFixed(1)}%`, 20, 90)

          if (status.isGenerating) {
            requestAnimationFrame(animate)
          }
        }

        animate()
      }
    }
  }, [status.isGenerating, config, status.progress])

  return (
    <div className="h-full flex flex-col gap-2">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800/80 to-slate-900/80 rounded-xl p-3 border border-white/10 shadow-xl flex-shrink-0">
        <div className="flex-shrink-0">
          <h1 className="text-lg lg:text-xl font-bold text-white flex items-center gap-2">
            <Box className="w-6 h-6 text-blue-400" />
            3D 데이터 생성
          </h1>
          <p className="text-xs text-slate-400">CARLA 시뮬레이터를 활용한 가상 환경 3D 데이터 생성</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 space-y-6 overflow-auto">
        <div className="grid grid-cols-12 gap-6">
          {/* 왼쪽: 설정 패널 */}
          <div className="col-span-4">
            <Card className="bg-slate-800/50 border-white/10" style={{ height: `${rightCardHeight}px` }}>
              <CardHeader>
                <CardTitle className="text-white text-lg">데이터셋 설정</CardTitle>
              </CardHeader>
              <CardContent className="p-0" style={{ height: 'calc(100% - 60px)', overflow: 'hidden' }}>
                <ScrollArea className="h-full w-full">
                  <div className="px-6">
                    <div className="space-y-4 py-4">
                      <div>
                        <Label htmlFor="dataset-name" className="text-white">데이터셋 이름</Label>
                        <Input
                          id="dataset-name"
                          value={config.datasetName}
                          onChange={(e) => setConfig({ ...config, datasetName: e.target.value })}
                          placeholder="예: urban_dataset_v1"
                          className="bg-slate-700/50 border-white/10 text-white"
                        />
                      </div>

                      <Separator className="bg-white/10" />

                      {/* 시뮬레이션 환경 설정 */}
                      <div className="space-y-4">
                        <h3 className="text-white font-medium flex items-center gap-2">
                          <Map className="w-4 h-4" />
                          시뮬레이션 환경
                        </h3>

                        <div>
                          <Label className="text-white">맵 선택</Label>
                          <Select value={config.map} onValueChange={(value) => setConfig({ ...config, map: value })}>
                            <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent className="bg-slate-800 border-white/10">
                              {carlaMapOptions.map(option => (
                                <SelectItem key={option.value} value={option.value} className="text-white">
                                  {option.label}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <Label className="text-white">날씨 설정</Label>
                          <Select value={config.weather} onValueChange={(value) => setConfig({ ...config, weather: value })}>
                            <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent className="bg-slate-800 border-white/10">
                              {weatherOptions.map(option => {
                                const Icon = option.icon
                                return (
                                  <SelectItem key={option.value} value={option.value} className="text-white">
                                    <div className="flex items-center gap-2">
                                      <Icon className="w-4 h-4" />
                                      {option.label}
                                    </div>
                                  </SelectItem>
                                )
                              })}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <Separator className="bg-white/10" />

                      {/* 객체 설정 */}
                      <div className="space-y-4">
                        <h3 className="text-white font-medium flex items-center gap-2">
                          <Car className="w-4 h-4" />
                          객체 설정
                        </h3>

                        <div>
                          <Label className="text-white">3D 모델 타입</Label>
                          <Select value={config.modelType} onValueChange={(value) => setConfig({ ...config, modelType: value })}>
                            <SelectTrigger className="bg-slate-700/50 border-white/20 text-white mt-2">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent className="bg-slate-800 border-white/20">
                              <SelectItem value="sedan" className="text-white">Sedan (승용차)</SelectItem>
                              <SelectItem value="suv" className="text-white">SUV</SelectItem>
                              <SelectItem value="truck" className="text-white">Truck (트럭)</SelectItem>
                              <SelectItem value="bus" className="text-white">Bus (버스)</SelectItem>
                              <SelectItem value="van" className="text-white">Van (밴)</SelectItem>
                              <SelectItem value="motorcycle" className="text-white">Motorcycle (오토바이)</SelectItem>
                              <SelectItem value="bicycle" className="text-white">Bicycle (자전거)</SelectItem>
                              <SelectItem value="pedestrian" className="text-white">Pedestrian (보행자)</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <Separator className="bg-white/10" />

                      {/* 출력 설정 */}
                      <div className="space-y-4">
                        <h3 className="text-white font-medium flex items-center gap-2">
                          <Monitor className="w-4 h-4" />
                          출력 설정
                        </h3>

                        {/* 해상도 버튼 선택 */}
                        <div>
                          <Label className="text-white mb-3 block">이미지 해상도 (객체 탐지 모델용)</Label>
                          <div className="grid grid-cols-2 gap-2">
                            {resolutionOptions.map((res) => (
                              <Button
                                key={res.value}
                                variant={config.imageResolution === res.value ? "default" : "outline"}
                                onClick={() => setConfig({ ...config, imageResolution: res.value })}
                                className={`h-auto py-3 px-2 flex-col ${
                                  config.imageResolution === res.value
                                    ? "bg-primary text-white"
                                    : "border-white/20 text-white hover:bg-slate-700/50"
                                }`}
                              >
                                <span className="font-semibold text-sm">{res.label}</span>
                              </Button>
                            ))}
                          </div>
                          <p className="text-xs text-slate-400 mt-2">
                            선택된 해상도: {config.imageResolution}
                          </p>
                        </div>
                      </div>

                      {/* 제어 버튼 */}
                      <div className="flex gap-2 pt-4">
                        {!status.isGenerating ? (
                          <Button
                            className="flex-1 bg-gradient-to-r from-blue-600 to-blue-700"
                            onClick={startGeneration}
                          >
                            <Play className="w-4 h-4 mr-2" />
                            생성 시작
                          </Button>
                        ) : (
                          <>
                            <Button
                              className="flex-1 bg-yellow-600 hover:bg-yellow-700"
                              onClick={() => setStatus({ ...status, isGenerating: false })}
                            >
                              <Pause className="w-4 h-4 mr-2" />
                              일시 정지
                            </Button>
                            <Button
                              className="flex-1 bg-red-600 hover:bg-red-700"
                              onClick={stopGeneration}
                            >
                              <Square className="w-4 h-4 mr-2" />
                              중지
                            </Button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>

          {/* 오른쪽: 실시간 시뮬레이션 화면 */}
          <div className="col-span-8">
            <Card id="right-card" className="bg-slate-800/50 border-white/10">
              <CardHeader>
                <CardTitle className="text-white text-lg flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  실시간 생성 화면
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden">
                  <canvas
                    ref={simulationCanvasRef}
                    width={640}
                    height={360}
                    className="w-full h-full"
                  />
                  {status.isGenerating && (
                    <div className="absolute top-4 right-4">
                      <Badge className="bg-red-600 animate-pulse">
                        <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse" />
                        LIVE
                      </Badge>
                    </div>
                  )}
                </div>

                {/* 생성 진행률 */}
                <div className="mt-4 space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">생성 진행률</span>
                    <span className="text-white">{status.generatedImages} / {status.totalImages} 이미지</span>
                  </div>
                  <Progress value={status.progress} className="h-2" />
                </div>

                {/* 빠른 통계 */}
                <div className="grid grid-cols-3 gap-4 mt-4">
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-slate-400 text-xs">생성된 이미지</p>
                    <p className="text-white text-lg font-semibold">{status.generatedImages}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-slate-400 text-xs">선택된 해상도</p>
                    <p className="text-white text-lg font-semibold">{config.imageResolution}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded-lg p-3">
                    <p className="text-slate-400 text-xs">모델 타입</p>
                    <p className="text-white text-lg font-semibold capitalize">{config.modelType}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* 하단: 생성 상황 패널 */}
        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader>
            <CardTitle className="text-white text-lg">생성 상황</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-12 gap-6">
              {/* 로그 및 오류 탭 */}
              <div className="col-span-8">
                <Tabs defaultValue="logs" className="h-full">
                  <TabsList className="grid w-full grid-cols-2 bg-slate-700/30">
                    <TabsTrigger value="logs" className="text-white">로그</TabsTrigger>
                    <TabsTrigger value="errors" className="text-white">오류</TabsTrigger>
                  </TabsList>

                  <TabsContent value="logs" className="mt-4">
                    <div className="max-h-48 overflow-y-auto w-full rounded-md border border-white/10 p-4 bg-slate-800/30">
                      <div className="space-y-2">
                        {status.logs.map((log, index) => (
                          <div key={index} className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-400 mt-0.5" />
                            <p className="text-slate-300 text-sm font-mono">{log}</p>
                          </div>
                        ))}
                        {status.logs.length === 0 && (
                          <p className="text-slate-500 text-sm">로그가 없습니다.</p>
                        )}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="errors" className="mt-4">
                    <div className="max-h-48 overflow-y-auto w-full rounded-md border border-white/10 p-4 bg-slate-800/30">
                      <div className="space-y-2">
                        {status.errors.map((error, index) => (
                          <Alert key={index} className="bg-red-900/20 border-red-500/30">
                            <AlertCircle className="h-4 w-4 text-red-400" />
                            <AlertDescription className="text-red-300">
                              {error}
                            </AlertDescription>
                          </Alert>
                        ))}
                        {status.errors.length === 0 && (
                          <p className="text-slate-500 text-sm">오류가 없습니다.</p>
                        )}
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </div>

              {/* 상태 및 액션 */}
              <div className="col-span-4 space-y-4">
                {/* 상태 표시 */}
                <div className="p-3 bg-slate-700/30 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400 text-sm">현재 상태</span>
                    <div className="flex items-center gap-2">
                      {status.isGenerating && (
                        <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                      )}
                      <Badge variant={status.isGenerating ? "default" : "secondary"}>
                        {status.currentStatus}
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* 액션 버튼 */}
                <div className="space-y-2">
                  <Button
                    variant="outline"
                    className="w-full border-white/10 text-white hover:bg-slate-700/50"
                    disabled={status.generatedImages === 0}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    데이터셋 다운로드
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full border-white/10 text-white hover:bg-slate-700/50"
                  >
                    <FolderOpen className="w-4 h-4 mr-2" />
                    저장 위치 열기
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}