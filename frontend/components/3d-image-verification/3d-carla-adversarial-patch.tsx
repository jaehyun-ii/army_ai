"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Play,
  Pause,
  Square,
  Settings,
  Download,
  Eye,
  Target,
  AlertCircle,
  CheckCircle,
  Loader2,
  Box,
  Sun,
  CloudRain,
  CloudFog,
  Moon,
  Sunrise,
  Wind,
  Snowflake,
  Camera,
  Palette,
  Layers,
  RotateCcw,
  Move3D,
  RefreshCw,
  MapPin,
  Zap,
  Brain,
  Activity
} from "lucide-react"

interface CarlaPatchConfig {
  patchName: string
  carlaMap: string
  weatherCondition: string
  timeOfDay: number
  targetObjects: string[]
  spawnLocations: string[]
  attackMethod: string
  targetModel: string
  patchSize: number
  patchPosition: string
  optimizationSteps: number
  learningRate: number
  confidenceThreshold: number
  renderQuality: string
}

interface GenerationStatus {
  isGenerating: boolean
  currentPhase: string
  progress: number
  currentStep: number
  totalSteps: number
  renderingFPS: number
  patchEffectiveness: number
  confidenceReduction: number
  estimatedTime: string
  errorMessage?: string
}

interface GeneratedPatch {
  id: string
  name: string
  carlaMap: string
  targetObject: string
  attackMethod: string
  timestamp: string
  effectiveness: number
  confidenceReduction: number
  renderQuality: string
  thumbnailPath: string
  outputPath: string
  size: string
}

export function CarlaAdversarialPatchGenerator() {
  const [config, setConfig] = useState<CarlaPatchConfig>({
    patchName: "",
    carlaMap: "",
    weatherCondition: "clear",
    timeOfDay: 12,
    targetObjects: [],
    spawnLocations: [],
    attackMethod: "",
    targetModel: "",
    patchSize: 64,
    patchPosition: "vehicle-surface",
    optimizationSteps: 1000,
    learningRate: 0.01,
    confidenceThreshold: 0.5,
    renderQuality: "high"
  })

  const [status, setStatus] = useState<GenerationStatus>({
    isGenerating: false,
    currentPhase: "",
    progress: 0,
    currentStep: 0,
    totalSteps: 0,
    renderingFPS: 0,
    patchEffectiveness: 0,
    confidenceReduction: 0,
    estimatedTime: ""
  })

  const [generatedPatches] = useState<GeneratedPatch[]>([
    {
      id: "carla_patch_001",
      name: "Town01 차량 탐지 방해 패치",
      carlaMap: "Town01",
      targetObject: "vehicle.tesla.model3",
      attackMethod: "ACTIVE Algorithm",
      timestamp: "2024-01-15 14:30",
      effectiveness: 89.2,
      confidenceReduction: 0.73,
      renderQuality: "Ultra",
      thumbnailPath: "/thumbnails/carla_vehicle_patch.jpg",
      outputPath: "/patches/carla_town01_vehicle",
      size: "156MB"
    },
    {
      id: "carla_patch_002",
      name: "Town03 보행자 회피 패치",
      carlaMap: "Town03",
      targetObject: "walker.pedestrian.0001",
      attackMethod: "DTA Algorithm",
      timestamp: "2024-01-14 11:20",
      effectiveness: 76.8,
      confidenceReduction: 0.65,
      renderQuality: "High",
      thumbnailPath: "/thumbnails/carla_pedestrian_patch.jpg",
      outputPath: "/patches/carla_town03_pedestrian",
      size: "98MB"
    },
    {
      id: "carla_patch_003",
      name: "Town07 다중 객체 공격 패치",
      carlaMap: "Town07",
      targetObject: "multiple_objects",
      attackMethod: "Hybrid ACTIVE-DTA",
      timestamp: "2024-01-13 16:45",
      effectiveness: 82.5,
      confidenceReduction: 0.69,
      renderQuality: "High",
      thumbnailPath: "/thumbnails/carla_multi_patch.jpg",
      outputPath: "/patches/carla_town07_multi",
      size: "234MB"
    }
  ])

  const [simulationView, setSimulationView] = useState<string>("initialization")

  // CARLA 시뮬레이션 및 패치 생성 시뮬레이션
  useEffect(() => {
    if (status.isGenerating) {
      const phases = [
        "CARLA 시뮬레이터 초기화 중...",
        "가상 환경 로딩 중...",
        "대상 객체 스폰 중...",
        "초기 패치 생성 중...",
        "적대적 최적화 진행 중...",
        "패치 렌더링 및 테스트 중...",
        "결과 저장 및 검증 중..."
      ]

      let currentPhaseIndex = 0
      let stepCount = 0
      let currentFPS = 60
      let effectiveness = 0
      let confidenceReduction = 0

      const interval = setInterval(() => {
        setStatus(prev => {
          const newProgress = Math.min(prev.progress + Math.random() * 1.5, 100)
          stepCount = Math.min(stepCount + Math.floor(Math.random() * 5) + 1, config.optimizationSteps)
          currentFPS = 55 + Math.random() * 10
          effectiveness = Math.min(newProgress * 0.9 + Math.random() * 5, 95)
          confidenceReduction = Math.min(newProgress * 0.008 + Math.random() * 0.05, 0.8)

          if (newProgress > (currentPhaseIndex + 1) * 14.28) {
            currentPhaseIndex = Math.min(currentPhaseIndex + 1, phases.length - 1)
          }

          const estimatedTime = newProgress > 90 ? "완료 직전" :
                              newProgress > 70 ? "약 4분 남음" :
                              newProgress > 50 ? "약 8분 남음" :
                              newProgress > 30 ? "약 15분 남음" : "약 25분 남음"

          return {
            ...prev,
            progress: newProgress,
            currentPhase: phases[currentPhaseIndex],
            currentStep: stepCount,
            totalSteps: config.optimizationSteps,
            renderingFPS: currentFPS,
            patchEffectiveness: effectiveness,
            confidenceReduction,
            estimatedTime
          }
        })

        // 시뮬레이션 뷰 업데이트
        if (Math.random() > 0.6) {
          const views = ["environment", "patch-rendering", "optimization", "testing"]
          setSimulationView(views[Math.floor(Math.random() * views.length)])
        }
      }, 800)

      return () => clearInterval(interval)
    }
  }, [status.isGenerating, config.optimizationSteps])

  const startGeneration = () => {
    if (!config.patchName || !config.carlaMap || !config.attackMethod || !config.targetModel) {
      return
    }

    setStatus({
      isGenerating: true,
      currentPhase: "CARLA 시뮬레이터 초기화 중...",
      progress: 0,
      currentStep: 0,
      totalSteps: config.optimizationSteps,
      renderingFPS: 0,
      patchEffectiveness: 0,
      confidenceReduction: 0,
      estimatedTime: "계산 중..."
    })
  }

  const stopGeneration = () => {
    setStatus(prev => ({
      ...prev,
      isGenerating: false,
      currentPhase: "중단됨",
      estimatedTime: "중단됨"
    }))
  }

  const resetGeneration = () => {
    setStatus({
      isGenerating: false,
      currentPhase: "",
      progress: 0,
      currentStep: 0,
      totalSteps: 0,
      renderingFPS: 0,
      patchEffectiveness: 0,
      confidenceReduction: 0,
      estimatedTime: ""
    })
    setSimulationView("initialization")
  }

  const getAttackMethodBadge = (method: string) => {
    switch (method) {
      case 'ACTIVE Algorithm': return 'bg-red-900/40 text-red-300 border-red-500/40'
      case 'DTA Algorithm': return 'bg-blue-900/40 text-blue-300 border-blue-500/40'
      case 'Hybrid ACTIVE-DTA': return 'bg-purple-900/40 text-purple-300 border-purple-500/40'
      case 'Custom Algorithm': return 'bg-green-900/40 text-green-300 border-green-500/40'
      default: return 'bg-slate-900/40 text-slate-300 border-slate-500/40'
    }
  }

  const getSimulationViewContent = () => {
    switch (simulationView) {
      case "environment":
        return {
          title: "CARLA 환경 렌더링",
          description: `${config.carlaMap} 맵에서 ${config.weatherCondition} 날씨 조건으로 렌더링 중`,
          icon: <Sun className="w-8 h-8 text-yellow-400" />
        }
      case "patch-rendering":
        return {
          title: "패치 렌더링 진행",
          description: "대상 객체에 적대적 패치를 적용하여 렌더링 중",
          icon: <Palette className="w-8 h-8 text-purple-400" />
        }
      case "optimization":
        return {
          title: "패치 최적화 중",
          description: `${config.attackMethod}을 사용하여 패치 효과 최적화 진행`,
          icon: <Zap className="w-8 h-8 text-blue-400" />
        }
      case "testing":
        return {
          title: "공격 효과 테스트",
          description: `${config.targetModel} 모델에 대한 패치 효과 검증 중`,
          icon: <Target className="w-8 h-8 text-red-400" />
        }
      default:
        return {
          title: "시뮬레이션 대기",
          description: "CARLA 패치 생성 시작을 대기 중입니다",
          icon: <Box className="w-8 h-8 text-slate-400" />
        }
    }
  }

  return (
    <div className="h-full overflow-hidden flex flex-col space-y-6">
      {/* 헤더 */}
      <Card className="bg-slate-900/50 border-white/10">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Box className="w-6 h-6 text-blue-400" />
            3D CARLA 적대적 공격 패치 생성
          </CardTitle>
          <CardDescription className="text-slate-400">
            CARLA 시뮬레이터 가상 환경에서 객체식별 AI 모델의 취약점을 노리는 적대적 패치를 실시간 렌더링으로 생성
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-12 gap-6 flex-1 min-h-0">
        {/* 왼쪽: 설정 패널 */}
        <div className="col-span-5 min-h-0">
          <Card className="bg-slate-900/70 border-white/20">
            <CardHeader>
              <CardTitle className="text-white text-lg">CARLA 패치 생성 설정</CardTitle>
            </CardHeader>
            <CardContent className="p-0 flex-1 min-h-0">
              <div className="flex-1 min-h-0 w-full overflow-y-auto">
                <div className="px-6">
                  <div className="space-y-6 py-4">
                    {/* 기본 설정 */}
                    <div className="space-y-4">
                      <h3 className="text-white font-medium flex items-center gap-2">
                        <Settings className="w-4 h-4" />
                        기본 설정
                      </h3>

                      <div>
                        <Label htmlFor="patch-name" className="text-white">생성 공격 패치 이름</Label>
                        <Input
                          id="patch-name"
                          value={config.patchName}
                          onChange={(e) => setConfig({ ...config, patchName: e.target.value })}
                          placeholder="예: carla_town01_vehicle_attack_v1"
                          className="bg-slate-700/50 border-white/10 text-white"
                        />
                      </div>

                      <div>
                        <Label className="text-white">공격 기법</Label>
                        <Select
                          value={config.attackMethod}
                          onValueChange={(value) => setConfig({ ...config, attackMethod: value })}
                        >
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                            <SelectValue placeholder="적대적 공격 알고리즘 선택" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="active">ACTIVE Algorithm</SelectItem>
                            <SelectItem value="dta">DTA Algorithm</SelectItem>
                            <SelectItem value="hybrid">Hybrid ACTIVE-DTA</SelectItem>
                            <SelectItem value="custom">Custom Algorithm</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label className="text-white">객체식별 AI 모델</Label>
                        <Select
                          value={config.targetModel}
                          onValueChange={(value) => setConfig({ ...config, targetModel: value })}
                        >
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                            <SelectValue placeholder="공격 대상 AI 모델 선택" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="yolov8">YOLOv8 (3D 객체 탐지)</SelectItem>
                            <SelectItem value="pointnet">PointNet++ (3D 포인트 클라우드)</SelectItem>
                            <SelectItem value="voxelnet">VoxelNet (3D 복셀)</SelectItem>
                            <SelectItem value="centerpoint">CenterPoint (3D 탐지)</SelectItem>
                            <SelectItem value="second">SECOND (3D 탐지)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <Separator className="bg-white/10" />

                    {/* 시뮬레이션 환경 설정 */}
                    <div className="space-y-4">
                      <h3 className="text-white font-medium flex items-center gap-2">
                        <MapPin className="w-4 h-4" />
                        시뮬레이션 환경 설정
                      </h3>

                      <div>
                        <Label className="text-white">CARLA 맵</Label>
                        <Select
                          value={config.carlaMap}
                          onValueChange={(value) => setConfig({ ...config, carlaMap: value })}
                        >
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                            <SelectValue placeholder="CARLA 가상 환경 맵 선택" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Town01">Town01 (도시 환경)</SelectItem>
                            <SelectItem value="Town02">Town02 (주거 지역)</SelectItem>
                            <SelectItem value="Town03">Town03 (대도시)</SelectItem>
                            <SelectItem value="Town04">Town04 (고속도로)</SelectItem>
                            <SelectItem value="Town05">Town05 (시골 도로)</SelectItem>
                            <SelectItem value="Town06">Town06 (미시간 좌회전)</SelectItem>
                            <SelectItem value="Town07">Town07 (시골 환경)</SelectItem>
                            <SelectItem value="Town10">Town10 (도심 고층빌딩)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <Label className="text-white">날씨 조건</Label>
                          <Select
                            value={config.weatherCondition}
                            onValueChange={(value) => setConfig({ ...config, weatherCondition: value })}
                          >
                            <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="clear">맑음</SelectItem>
                              <SelectItem value="cloudy">흐림</SelectItem>
                              <SelectItem value="wet">비</SelectItem>
                              <SelectItem value="wetcloudy">비와 흐림</SelectItem>
                              <SelectItem value="softrain">가벼운 비</SelectItem>
                              <SelectItem value="midrain">보통 비</SelectItem>
                              <SelectItem value="hardrain">강한 비</SelectItem>
                              <SelectItem value="fog">안개</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <Label className="text-white text-sm">시간대: {config.timeOfDay}시</Label>
                          <Slider
                            value={[config.timeOfDay]}
                            onValueChange={(value) => setConfig({ ...config, timeOfDay: value[0] })}
                            min={0}
                            max={23}
                            step={1}
                            className="mt-2"
                          />
                        </div>
                      </div>

                      <div>
                        <Label className="text-white text-sm">대상 객체 종류</Label>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          {[
                            { id: "vehicle.tesla.model3", label: "Tesla Model 3" },
                            { id: "vehicle.audi.a2", label: "Audi A2" },
                            { id: "vehicle.bmw.grandtourer", label: "BMW GT" },
                            { id: "walker.pedestrian.0001", label: "성인 보행자" },
                            { id: "walker.pedestrian.0002", label: "어린이" },
                            { id: "static.prop.trafficcone01", label: "교통 콘" }
                          ].map(({ id, label }) => (
                            <div key={id} className="flex items-center space-x-2">
                              <Checkbox
                                id={id}
                                checked={config.targetObjects.includes(id)}
                                onCheckedChange={(checked) => {
                                  if (checked) {
                                    setConfig({
                                      ...config,
                                      targetObjects: [...config.targetObjects, id]
                                    })
                                  } else {
                                    setConfig({
                                      ...config,
                                      targetObjects: config.targetObjects.filter(obj => obj !== id)
                                    })
                                  }
                                }}
                              />
                              <label htmlFor={id} className="text-white text-sm">{label}</label>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div>
                        <Label className="text-white text-sm">스폰 위치</Label>
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          {[
                            { id: "spawn_point_1", label: "교차로 중앙" },
                            { id: "spawn_point_2", label: "도로 직선" },
                            { id: "spawn_point_3", label: "주차장" },
                            { id: "spawn_point_4", label: "보행자 도로" },
                            { id: "spawn_point_5", label: "고속도로" },
                            { id: "spawn_point_6", label: "무작위 위치" }
                          ].map(({ id, label }) => (
                            <div key={id} className="flex items-center space-x-2">
                              <Checkbox
                                id={id}
                                checked={config.spawnLocations.includes(id)}
                                onCheckedChange={(checked) => {
                                  if (checked) {
                                    setConfig({
                                      ...config,
                                      spawnLocations: [...config.spawnLocations, id]
                                    })
                                  } else {
                                    setConfig({
                                      ...config,
                                      spawnLocations: config.spawnLocations.filter(loc => loc !== id)
                                    })
                                  }
                                }}
                              />
                              <label htmlFor={id} className="text-white text-sm">{label}</label>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <Separator className="bg-white/10" />

                    {/* 패치 설정 */}
                    <div className="space-y-4">
                      <h3 className="text-white font-medium flex items-center gap-2">
                        <Palette className="w-4 h-4" />
                        패치 설정
                      </h3>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <Label className="text-white text-sm">패치 크기</Label>
                          <Select
                            value={config.patchSize.toString()}
                            onValueChange={(value) => setConfig({ ...config, patchSize: parseInt(value) })}
                          >
                            <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="32">32x32cm</SelectItem>
                              <SelectItem value="64">64x64cm</SelectItem>
                              <SelectItem value="128">128x128cm</SelectItem>
                              <SelectItem value="256">256x256cm</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <Label className="text-white text-sm">패치 위치</Label>
                          <Select
                            value={config.patchPosition}
                            onValueChange={(value) => setConfig({ ...config, patchPosition: value })}
                          >
                            <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="vehicle-surface">차량 표면</SelectItem>
                              <SelectItem value="vehicle-roof">차량 지붕</SelectItem>
                              <SelectItem value="vehicle-hood">차량 후드</SelectItem>
                              <SelectItem value="pedestrian-clothing">보행자 의복</SelectItem>
                              <SelectItem value="ground-plane">지면</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <div>
                        <Label className="text-white text-sm">최적화 단계: {config.optimizationSteps}</Label>
                        <Slider
                          value={[config.optimizationSteps]}
                          onValueChange={(value) => setConfig({ ...config, optimizationSteps: value[0] })}
                          min={100}
                          max={5000}
                          step={100}
                          className="mt-2"
                        />
                      </div>

                      <div>
                        <Label className="text-white text-sm">생성률: {config.learningRate}</Label>
                        <Slider
                          value={[config.learningRate * 100]}
                          onValueChange={(value) => setConfig({ ...config, learningRate: value[0] / 100 })}
                          min={0.1}
                          max={10}
                          step={0.1}
                          className="mt-2"
                        />
                      </div>

                      <div>
                        <Label className="text-white text-sm">렌더링 품질</Label>
                        <Select
                          value={config.renderQuality}
                          onValueChange={(value) => setConfig({ ...config, renderQuality: value })}
                        >
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="low">Low (빠른 프로토타이핑)</SelectItem>
                            <SelectItem value="medium">Medium (균형)</SelectItem>
                            <SelectItem value="high">High (고품질)</SelectItem>
                            <SelectItem value="ultra">Ultra (최고 품질)</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    {/* 제어 버튼 */}
                    <div className="flex items-center gap-3 pt-4">
                      <Button
                        onClick={startGeneration}
                        disabled={status.isGenerating || !config.patchName || !config.carlaMap || !config.attackMethod || !config.targetModel}
                        className="bg-gradient-to-r from-blue-600 to-blue-500 text-white flex-1"
                      >
                        {status.isGenerating ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            패치 생성 중...
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 mr-2" />
                            공격 패치 생성
                          </>
                        )}
                      </Button>

                      {status.isGenerating && (
                        <Button
                          onClick={stopGeneration}
                          variant="outline"
                          className="bg-red-900/20 border-red-500/30 text-red-300"
                        >
                          <Square className="w-4 h-4" />
                        </Button>
                      )}

                      <Button
                        onClick={resetGeneration}
                        variant="outline"
                        disabled={status.isGenerating}
                      >
                        <RefreshCw className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 오른쪽: 생성 화면 및 상황 */}
        <div className="col-span-7">
          <div className="space-y-6">
            {/* 공격 패치 생성 화면 */}
            <Card className="bg-slate-900/70 border-white/20 flex-1">
              <CardHeader>
                <CardTitle className="text-white text-lg flex items-center gap-2">
                  <Camera className="w-5 h-5" />
                  공격 패치 생성 화면
                </CardTitle>
              </CardHeader>
              <CardContent className="h-full">
                {status.isGenerating ? (
                  <div className="bg-slate-800/50 rounded-lg h-full flex items-center justify-center">
                    <div className="text-center">
                      {getSimulationViewContent().icon}
                      <h3 className="text-white font-medium mt-4 mb-2">
                        {getSimulationViewContent().title}
                      </h3>
                      <p className="text-slate-400 text-sm max-w-md">
                        {getSimulationViewContent().description}
                      </p>
                      <div className="mt-4 flex items-center justify-center gap-4 text-sm">
                        <div className="flex items-center gap-1">
                          <Activity className="w-4 h-4 text-green-400" />
                          <span className="text-green-400">{status.renderingFPS.toFixed(1)} FPS</span>
                        </div>
                        <div className="text-slate-400">
                          {config.carlaMap} • {config.weatherCondition}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-800/50 rounded-lg h-full flex items-center justify-center">
                    <div className="text-center">
                      <Box className="w-16 h-16 text-slate-500 mx-auto mb-4" />
                      <p className="text-slate-300 mb-2">CARLA 시뮬레이션 대기중</p>
                      <p className="text-slate-500 text-sm">
                        설정을 완료하고 '공격 패치 생성' 버튼을 클릭하세요
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 생성 상황 */}
            <Card className="bg-slate-900/70 border-white/20">
              <CardHeader>
                <CardTitle className="text-white text-lg flex items-center gap-2">
                  <Activity className="w-5 h-5" />
                  생성 상황
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {status.isGenerating ? (
                  <div className="space-y-4">
                    <div className="bg-slate-800/50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <p className="text-white font-medium">{status.currentPhase}</p>
                          <p className="text-slate-400 text-sm">단계 {status.currentStep}/{status.totalSteps}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-blue-400 text-sm">{status.estimatedTime}</p>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-slate-300">전체 진행률</span>
                          <span className="text-white">{Math.round(status.progress)}%</span>
                        </div>
                        <Progress value={status.progress} className="h-2" />
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div className="bg-slate-800/30 rounded-lg p-3">
                        <div className="flex justify-between">
                          <span className="text-slate-400">패치 효과:</span>
                          <span className="text-green-400">{status.patchEffectiveness.toFixed(1)}%</span>
                        </div>
                      </div>

                      <div className="bg-slate-800/30 rounded-lg p-3">
                        <div className="flex justify-between">
                          <span className="text-slate-400">신뢰도 감소:</span>
                          <span className="text-red-400">{status.confidenceReduction.toFixed(3)}</span>
                        </div>
                      </div>

                      <div className="bg-slate-800/30 rounded-lg p-3">
                        <div className="flex justify-between">
                          <span className="text-slate-400">렌더링 FPS:</span>
                          <span className="text-blue-400">{status.renderingFPS.toFixed(1)}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-800/50 rounded-lg p-6 text-center">
                    <Target className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                    <p className="text-slate-300">패치 생성 상황 표시 대기중</p>
                    <p className="text-slate-500 text-sm mt-1">
                      생성 프로세스가 시작되면 진행 상황과 오류/이슈가 여기에 표시됩니다
                    </p>
                  </div>
                )}

                {status.errorMessage && (
                  <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
                    <div className="flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-red-400" />
                      <span className="text-red-300 text-sm">{status.errorMessage}</span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 생성된 패치 목록 */}
            <Card className="bg-slate-900/70 border-white/20">
              <CardHeader>
                <CardTitle className="text-white text-lg flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  생성된 CARLA 적대적 패치
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="max-h-64 overflow-y-auto">
                  <div className="space-y-3">
                    {generatedPatches.map((patch) => (
                      <Card key={patch.id} className="bg-slate-800/50 border-white/10 p-4">
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="text-white font-medium">{patch.name}</h4>
                              <div className="flex items-center gap-4 text-sm text-slate-400">
                                <span>{patch.carlaMap}</span>
                                <span>{patch.targetObject}</span>
                                <span>{patch.timestamp}</span>
                                <span>{patch.size}</span>
                              </div>
                            </div>
                            <Badge className={getAttackMethodBadge(patch.attackMethod)}>
                              {patch.attackMethod}
                            </Badge>
                          </div>

                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4 text-sm">
                              <div>
                                <span className="text-slate-400">효과:</span>
                                <span className="text-green-400 ml-1">{patch.effectiveness}%</span>
                              </div>
                              <div>
                                <span className="text-slate-400">신뢰도 감소:</span>
                                <span className="text-red-400 ml-1">{patch.confidenceReduction}</span>
                              </div>
                              <div>
                                <span className="text-slate-400">품질:</span>
                                <span className="text-white ml-1">{patch.renderQuality}</span>
                              </div>
                            </div>

                            <div className="flex items-center gap-2">
                              <Button size="sm" variant="outline" className="text-xs">
                                <Eye className="w-3 h-3 mr-1" />
                                미리보기
                              </Button>
                              <Button size="sm" variant="outline" className="text-xs">
                                <Download className="w-3 h-3 mr-1" />
                                다운로드
                              </Button>
                            </div>
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}