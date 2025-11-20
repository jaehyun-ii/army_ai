"use client"

import { useState, useRef, useEffect, Fragment } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Play,
  Pause,
  Square,
  Settings,
  Download,
  Eye,
  Target,
  Map,
  Cloud,
  Car,
  AlertCircle,
  CheckCircle,
  CheckCircle2,
  Circle,
  Loader2,
  Brain,
  Sun,
  CloudRain,
  Moon,
  Sunrise,
  CloudFog,
  Upload,
  Activity,
  ChevronRight,
  ChevronLeft,
  BarChart3,
  RotateCcw,
  Database,
  Layers
} from "lucide-react"

interface PatchConfig {
  projectName: string
  selectedDataset: string // 3D 데이터 생성에서 만들어진 데이터셋 선택

  // 환경 설정
  map: string
  weather: string
  spawnLocation: { x: number, y: number, z: number }

  // 공격 대상
  targetModel: string
  targetClass: string
  vulnerabilityType: string

  // 패치 설정
  patchSize: number
  patchPattern: string
  optimizationMethod: string
  iterations: number
  epsilon: number

  // 물리 환경
  lightingCondition: string
  viewingAngle: number
  distance: number
  occlusion: boolean
}

interface GenerationStatus {
  isGenerating: boolean
  progress: number
  currentIteration: number
  totalIterations: number
  attackSuccessRate: number
  currentStatus: string
  logs: string[]
  metrics: {
    confidence: number
    loss: number
    perturbation: number
  }
}

export function AdversarialPatch3D() {
  const [activeStep, setActiveStep] = useState(0)
  const [completedSteps, setCompletedSteps] = useState<number[]>([])

  const [config, setConfig] = useState<PatchConfig>({
    projectName: "",
    selectedDataset: "",
    map: "Town03",
    weather: "ClearNoon",
    spawnLocation: { x: 0, y: 0, z: 0 },
    targetModel: "yolov8",
    targetClass: "car",
    vulnerabilityType: "misclassification",
    patchSize: 100,
    patchPattern: "noise",
    optimizationMethod: "pgd",
    iterations: 1000,
    epsilon: 0.03,
    lightingCondition: "normal",
    viewingAngle: 0,
    distance: 10,
    occlusion: false
  })

  const [status, setStatus] = useState<GenerationStatus>({
    isGenerating: false,
    progress: 0,
    currentIteration: 0,
    totalIterations: 1000,
    attackSuccessRate: 0,
    currentStatus: "대기 중",
    logs: [],
    metrics: {
      confidence: 0,
      loss: 0,
      perturbation: 0
    }
  })

  const steps = [
    { title: '환경 & 모델', description: '데이터셋 선택 및 설정' },
    { title: '생성 실행', description: '패치 최적화 진행' },
    { title: '결과 확인', description: '생성된 패치 분석' }
  ]

  const simulationCanvasRef = useRef<HTMLCanvasElement>(null)
  const patchPreviewRef = useRef<HTMLCanvasElement>(null)
  const datasetPreviewRef = useRef<HTMLCanvasElement>(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)

  // 3D 데이터셋 옵션 (3D 데이터 생성에서 만들어진 데이터)
  const generatedDatasets = [
    { value: "carla_urban_day_v1", label: "CARLA Urban Day v1", count: 1000, date: "2024-01-15" },
    { value: "carla_highway_night_v2", label: "CARLA Highway Night v2", count: 800, date: "2024-01-14" },
    { value: "carla_suburban_rain_v1", label: "CARLA Suburban Rain v1", count: 1200, date: "2024-01-13" },
    { value: "carla_mixed_weather_v3", label: "CARLA Mixed Weather v3", count: 2000, date: "2024-01-12" },
    { value: "custom_adversarial_v1", label: "Custom Adversarial v1", count: 500, date: "2024-01-11" },
    { value: "synthetic_urban_v2", label: "Synthetic Urban v2", count: 1500, date: "2024-01-10" }
  ]

  // AI 모델 옵션
  const modelOptions = [
    { value: "yolov8", label: "YOLO v8", type: "Detection" },
    { value: "yolov5", label: "YOLO v5", type: "Detection" },
    { value: "fasterrcnn", label: "Faster R-CNN", type: "Detection" },
    { value: "ssd", label: "SSD MobileNet", type: "Detection" },
    { value: "maskrcnn", label: "Mask R-CNN", type: "Segmentation" },
    { value: "detr", label: "DETR", type: "Detection" },
    { value: "efficientdet", label: "EfficientDet", type: "Detection" }
  ]

  // 대상 클래스 옵션
  const targetClasses = [
    "car", "truck", "bus", "motorcycle", "bicycle",
    "person", "traffic_light", "stop_sign", "traffic_sign"
  ]

  // 취약점 유형
  const vulnerabilityTypes = [
    { value: "misclassification", label: "오분류 유도" },
    { value: "disappearance", label: "객체 은폐" },
    { value: "false_positive", label: "거짓 양성" },
    { value: "confidence_reduction", label: "신뢰도 감소" }
  ]

  // 패치 패턴
  const patchPatterns = [
    { value: "noise", label: "랜덤 노이즈" },
    { value: "geometric", label: "기하학적 패턴" },
    { value: "natural", label: "자연 텍스처" },
    { value: "adversarial", label: "적대적 패턴" },
    { value: "qrcode", label: "QR 코드형" }
  ]

  // 최적화 방법
  const optimizationMethods = [
    { value: "dta", label: "DTA" },
    { value: "active", label: "ACTIVE" },

  ]

  // CARLA 맵 옵션
  const carlaMapOptions = [
    { value: "Town01", label: "Town01 - 작은 도시" },
    { value: "Town02", label: "Town02 - 상업 지구" },
    { value: "Town03", label: "Town03 - 대도시" },
    { value: "Town04", label: "Town04 - 산악 도시" },
    { value: "Town05", label: "Town05 - 도심 격자" },
    { value: "Town10HD", label: "Town10HD - 도심 HD" }
  ]

  // 날씨 옵션
  const weatherOptions = [
    { value: "ClearNoon", label: "맑음 (낮)", icon: Sun },
    { value: "CloudyNoon", label: "흐림 (낮)", icon: Cloud },
    { value: "WetNoon", label: "비 (낮)", icon: CloudRain },
    { value: "ClearSunset", label: "맑음 (일몰)", icon: Sunrise },
    { value: "ClearNight", label: "맑음 (밤)", icon: Moon },
    { value: "Foggy", label: "안개", icon: CloudFog }
  ]

  const handleNext = () => {
    if (activeStep < steps.length - 1) {
      setCompletedSteps([...completedSteps, activeStep])
      setActiveStep(activeStep + 1)
    }
  }

  const handleBack = () => {
    if (activeStep > 0) {
      setActiveStep(activeStep - 1)
    }
  }

  const handleReset = () => {
    setActiveStep(0)
    setCompletedSteps([])
    setConfig({
      projectName: "",
      selectedDataset: "",
      map: "Town03",
      weather: "ClearNoon",
      spawnLocation: { x: 0, y: 0, z: 0 },
      targetModel: "yolov8",
      targetClass: "car",
      vulnerabilityType: "misclassification",
      patchSize: 100,
      patchPattern: "noise",
      optimizationMethod: "pgd",
      iterations: 1000,
      epsilon: 0.03,
      lightingCondition: "normal",
      viewingAngle: 0,
      distance: 10,
      occlusion: false
    })
    setStatus({
      isGenerating: false,
      progress: 0,
      currentIteration: 0,
      totalIterations: 1000,
      attackSuccessRate: 0,
      currentStatus: "대기 중",
      logs: [],
      metrics: {
        confidence: 0,
        loss: 0,
        perturbation: 0
      }
    })
  }

  // 패치 생성 시작
  const startGeneration = () => {
    if (!config.projectName || !config.selectedDataset) {
      alert("프로젝트 이름과 데이터셋을 선택해주세요.")
      return
    }

    setStatus({
      ...status,
      isGenerating: true,
      progress: 0,
      currentIteration: 0,
      totalIterations: config.iterations,
      currentStatus: "패치 최적화 시작...",
      logs: [`[${new Date().toLocaleTimeString()}] 적대적 패치 생성 시작: ${config.projectName}`]
    })

    simulatePatchGeneration()
  }

  // 패치 생성 시뮬레이션
  const simulatePatchGeneration = () => {
    let iteration = 0
    const interval = setInterval(() => {
      iteration += Math.floor(Math.random() * 50) + 10

      if (iteration >= config.iterations) {
        iteration = config.iterations
        clearInterval(interval)

        setStatus(prev => ({
          ...prev,
          isGenerating: false,
          progress: 100,
          currentIteration: iteration,
          attackSuccessRate: 85 + Math.random() * 10,
          currentStatus: "생성 완료",
          logs: [...prev.logs, `[${new Date().toLocaleTimeString()}] 패치 생성 완료 - 성공률: ${(85 + Math.random() * 10).toFixed(1)}%`]
        }))
      } else {
        const progress = (iteration / config.iterations) * 100
        const successRate = Math.min(95, progress * 0.9 + Math.random() * 10)

        setStatus(prev => ({
          ...prev,
          progress: progress,
          currentIteration: iteration,
          attackSuccessRate: successRate,
          currentStatus: `최적화 진행 중... (${iteration}/${config.iterations})`,
          metrics: {
            confidence: Math.max(0, 1 - progress / 100),
            loss: Math.max(0.01, 10 - progress / 10),
            perturbation: Math.min(config.epsilon, progress / 100 * config.epsilon)
          },
          logs: iteration % 100 === 0 ?
            [...prev.logs, `[${new Date().toLocaleTimeString()}] Iteration ${iteration}: Loss=${(10 - progress / 10).toFixed(3)}, Success=${successRate.toFixed(1)}%`]
            : prev.logs
        }))
      }
    }, 100)
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

  // 3D 시뮬레이션 렌더링
  useEffect(() => {
    if (simulationCanvasRef.current && status.isGenerating) {
      const canvas = simulationCanvasRef.current
      const ctx = canvas.getContext('2d')

      if (ctx) {
        const animate = () => {
          // 배경
          ctx.fillStyle = '#1a1a2e'
          ctx.fillRect(0, 0, canvas.width, canvas.height)

          // 도로
          ctx.fillStyle = '#333'
          ctx.fillRect(0, canvas.height * 0.6, canvas.width, canvas.height * 0.4)

          // 차량 (타겟)
          const carX = canvas.width / 2 - 60
          const carY = canvas.height * 0.65
          ctx.fillStyle = '#4a5568'
          ctx.fillRect(carX, carY, 120, 60)

          // 패치 렌더링 (차량 위에)
          if (status.progress > 0) {
            ctx.fillStyle = `rgba(255, 0, 0, ${0.3 + status.progress / 200})`
            ctx.fillRect(carX + 30, carY + 10, 60, 40)

            // 패치 패턴
            ctx.strokeStyle = '#ff0000'
            ctx.lineWidth = 2
            for (let i = 0; i < 5; i++) {
              ctx.beginPath()
              ctx.moveTo(carX + 30 + i * 12, carY + 10)
              ctx.lineTo(carX + 30 + i * 12, carY + 50)
              ctx.stroke()
            }
          }

          // 탐지 박스
          ctx.strokeStyle = status.attackSuccessRate > 50 ? '#ff0000' : '#00ff00'
          ctx.lineWidth = 2
          ctx.setLineDash([5, 5])
          ctx.strokeRect(carX - 10, carY - 10, 140, 80)
          ctx.setLineDash([])

          // 정보 표시
          ctx.fillStyle = '#00ff00'
          ctx.font = '12px monospace'
          ctx.fillText(`Target: ${config.targetClass}`, 10, 20)
          ctx.fillText(`Model: ${config.targetModel}`, 10, 35)
          ctx.fillText(`Attack Success: ${status.attackSuccessRate.toFixed(1)}%`, 10, 50)
          ctx.fillText(`Iteration: ${status.currentIteration}/${config.iterations}`, 10, 65)

          if (status.isGenerating) {
            requestAnimationFrame(animate)
          }
        }

        animate()
      }
    }
  }, [status, config])

  // 데이터셋 비디오 프리뷰 애니메이션
  useEffect(() => {
    if (datasetPreviewRef.current && isPlaying && config.selectedDataset) {
      const canvas = datasetPreviewRef.current
      const ctx = canvas.getContext('2d')
      const dataset = generatedDatasets.find(d => d.value === config.selectedDataset)

      if (ctx && dataset) {
        let frameCount = 0
        const totalFrames = dataset.count

        const animate = () => {
          // 캔버스 클리어
          ctx.fillStyle = '#0f172a'
          ctx.fillRect(0, 0, canvas.width, canvas.height)

          // 도로 배경
          ctx.fillStyle = '#334155'
          ctx.fillRect(0, canvas.height * 0.5, canvas.width, canvas.height * 0.5)

          // 차선
          ctx.strokeStyle = '#64748b'
          ctx.lineWidth = 2
          ctx.setLineDash([20, 10])
          ctx.beginPath()
          ctx.moveTo(0, canvas.height * 0.75)
          ctx.lineTo(canvas.width, canvas.height * 0.75)
          ctx.stroke()
          ctx.setLineDash([])

          // 움직이는 차량들 시뮬레이션
          const time = frameCount * 0.05
          for (let i = 0; i < 3; i++) {
            const carX = ((time * 50 + i * 200) % (canvas.width + 100)) - 50
            const carY = canvas.height * 0.6 + i * 40
            const carWidth = 80
            const carHeight = 40

            // 차량 본체
            ctx.fillStyle = i === 0 ? '#ef4444' : '#64748b'
            ctx.fillRect(carX, carY, carWidth, carHeight)

            // 바운딩 박스
            if (i === 0) { // 메인 타겟 차량만
              ctx.strokeStyle = '#10b981'
              ctx.lineWidth = 2
              ctx.strokeRect(carX - 5, carY - 5, carWidth + 10, carHeight + 10)

              // 라벨
              ctx.fillStyle = '#10b981'
              ctx.font = '12px monospace'
              ctx.fillText('car 98.5%', carX, carY - 10)
            }
          }

          // 프레임 정보
          ctx.fillStyle = '#ffffff'
          ctx.font = '14px monospace'
          ctx.fillText(`Frame: ${frameCount + 1}/${totalFrames}`, 10, 25)

          // 타임스탬프
          const timestamp = new Date(Date.now() - (totalFrames - frameCount) * 33).toLocaleTimeString()
          ctx.fillStyle = '#94a3b8'
          ctx.font = '12px monospace'
          ctx.fillText(timestamp, canvas.width - 100, 25)

          frameCount++
          setCurrentFrame(frameCount)

          if (frameCount >= totalFrames) {
            frameCount = 0 // 루프
          }

          if (isPlaying) {
            requestAnimationFrame(animate)
          }
        }

        animate()
      }
    }
  }, [isPlaying, config.selectedDataset])

  // 데이터셋 변경 시 재생 중지 및 프레임 리셋
  useEffect(() => {
    setIsPlaying(false)
    setCurrentFrame(0)
  }, [config.selectedDataset])

  // 패치 프리뷰 렌더링
  useEffect(() => {
    if (patchPreviewRef.current) {
      const canvas = patchPreviewRef.current
      const ctx = canvas.getContext('2d')

      if (ctx) {
        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        // 패치 패턴 생성
        const imageData = ctx.createImageData(canvas.width, canvas.height)
        for (let i = 0; i < imageData.data.length; i += 4) {
          if (config.patchPattern === 'noise') {
            imageData.data[i] = Math.random() * 255     // R
            imageData.data[i + 1] = Math.random() * 255 // G
            imageData.data[i + 2] = Math.random() * 255 // B
          } else if (config.patchPattern === 'geometric') {
            const x = (i / 4) % canvas.width
            const y = Math.floor((i / 4) / canvas.width)
            const pattern = (x + y) % 20 < 10
            imageData.data[i] = pattern ? 255 : 0
            imageData.data[i + 1] = pattern ? 0 : 255
            imageData.data[i + 2] = 0
          }
          imageData.data[i + 3] = 255 // Alpha
        }
        ctx.putImageData(imageData, 0, 0)
      }
    }
  }, [config.patchPattern])

  const StepIndicator = ({ stepIndex }: { stepIndex: number }) => {
    const isActive = stepIndex === activeStep
    const isCompleted = completedSteps.includes(stepIndex)

    return (
      <div className="flex items-center">
        <div className={`
          w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
          ${isActive ? 'bg-gradient-to-r from-primary to-accent text-white' :
            isCompleted ? 'bg-primary/20 text-primary' :
            'bg-slate-700 text-slate-400'}
        `}>
          {isCompleted ? <CheckCircle2 className="w-5 h-5" /> : stepIndex + 1}
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col gap-2">
      {/* Compact Header with Stepper */}
      <div className="bg-gradient-to-r from-slate-800/80 to-slate-900/80 rounded-xl p-3 border border-white/10 shadow-xl flex-shrink-0">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-2">
          {/* Title Section */}
          <div className="flex-shrink-0">
            <h1 className="text-lg lg:text-xl font-bold text-white flex items-center gap-2">
              <Layers className="w-6 h-6 text-purple-400" />
              3D 적대적 패치 생성
            </h1>
            <p className="text-xs text-slate-400">CARLA 시뮬레이터 기반 물리 환경을 고려한 적대적 패치 생성</p>
          </div>

          {/* Horizontal Stepper */}
          <div className="flex items-center gap-1 lg:justify-end">
            {steps.map((step, index) => (
              <Fragment key={index}>
                <button
                  onClick={() => {
                    if (completedSteps.includes(index) || index < activeStep) {
                      setActiveStep(index);
                    }
                  }}
                  className={`flex items-center gap-1 px-2 py-1 rounded-lg transition-all ${
                    index === activeStep ? 'scale-105' :
                    completedSteps.includes(index) ? 'hover:bg-slate-700/50 cursor-pointer' :
                    'cursor-not-allowed opacity-60'
                  }`}
                  disabled={!completedSteps.includes(index) && index > activeStep}
                >
                  <div className={`
                    w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-all
                    ${index === activeStep ? 'bg-gradient-to-r from-primary to-accent text-white shadow-lg' :
                      completedSteps.includes(index) ? 'bg-primary/30 text-primary' :
                      'bg-slate-700 text-slate-500'}
                  `}>
                    {completedSteps.includes(index) ? '✓' : index + 1}
                  </div>
                  <span className={`text-xs font-medium hidden sm:block ${
                    index === activeStep ? 'text-white' : 'text-slate-400'
                  }`}>
                    {step.title}
                  </span>
                </button>
                {index < steps.length - 1 && (
                  <div className={`w-8 h-0.5 ${
                    completedSteps.includes(index) ? 'bg-primary/50' : 'bg-slate-700'
                  }`} />
                )}
              </Fragment>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content Area - Flexible Height */}
      <div className="flex-1 flex flex-col gap-2 min-h-0 overflow-hidden">
        <div className="flex-1 grid grid-cols-12 gap-2 min-h-0">

          {/* 왼쪽: 설정 패널 - col-3 */}
          <div className="col-span-12 lg:col-span-3 min-h-0">
            <Card className="bg-slate-800/50 border-white/10 h-full flex flex-col overflow-hidden">
            <CardHeader className="flex-shrink-0 py-2 px-3">
              <CardTitle className="text-sm font-semibold text-white flex items-center gap-1">
                <BarChart3 className="w-3 h-3 text-primary" />
                {activeStep === 0 && '환경 & 모델 설정'}
                {activeStep === 1 && '생성 진행'}
                {activeStep === 2 && '결과 분석'}
              </CardTitle>
              <CardDescription className="text-xs text-slate-400">
                Step {activeStep + 1} / {steps.length}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col overflow-y-auto p-3">
              <div className="flex-1 space-y-2">
                {/* Step 0: 환경 & 모델 설정 (통합) */}
                {activeStep === 0 && (
                  <>
                    <div className="space-y-2">
                      <div className="space-y-1">
                        <Label htmlFor="project-name" className="text-xs text-slate-300">프로젝트 이름</Label>
                        <Input
                          id="project-name"
                          value={config.projectName}
                          onChange={(e) => setConfig({ ...config, projectName: e.target.value })}
                          placeholder="예: car_patch_v1"
                          className="bg-slate-700/50 border-white/10 text-white h-8 text-sm"
                        />
                      </div>

                      <div className="space-y-1">
                        <Label className="text-xs text-slate-300 flex items-center gap-1">
                          <Database className="w-3 h-3" />
                          3D 데이터셋 선택
                        </Label>
                        <Select value={config.selectedDataset} onValueChange={(value) => setConfig({ ...config, selectedDataset: value })}>
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white h-8 text-sm">
                            <SelectValue placeholder="생성된 3D 데이터셋 선택" />
                          </SelectTrigger>
                          <SelectContent>
                            {generatedDatasets.map(dataset => (
                              <SelectItem key={dataset.value} value={dataset.value}>
                                <div className="flex flex-col">
                                  <span className="font-medium">{dataset.label}</span>
                                  <span className="text-xs text-slate-400">{dataset.count} images • {dataset.date}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>


                      <div className="space-y-1">
                        <Label className="text-xs text-slate-300">대상 모델</Label>
                        <Select value={config.targetModel} onValueChange={(value) => setConfig({ ...config, targetModel: value })}>
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white h-8 text-sm">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {modelOptions.map(option => (
                              <SelectItem key={option.value} value={option.value}>
                                {option.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-1">
                        <Label className="text-xs text-slate-300">대상 클래스</Label>
                        <Select value={config.targetClass} onValueChange={(value) => setConfig({ ...config, targetClass: value })}>
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white h-8 text-sm">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {targetClasses.map(cls => (
                              <SelectItem key={cls} value={cls}>
                                {cls}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>


                      <div className="space-y-1">
                        <Label className="text-xs text-slate-300">최적화 방법</Label>
                        <Select value={config.optimizationMethod} onValueChange={(value) => setConfig({ ...config, optimizationMethod: value })}>
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white h-8 text-sm">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {optimizationMethods.map(method => (
                              <SelectItem key={method.value} value={method.value}>
                                {method.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </>
                )}

                {/* Step 1: 생성 실행 */}
                {activeStep === 1 && (
                  <>
                    {!status.isGenerating ? (
                      <div className="space-y-2">
                        <div className="bg-blue-900/20 border border-blue-500/30 p-2 rounded">
                          <p className="text-xs text-blue-300">설정이 완료되었습니다. 패치 생성을 시작하세요.</p>
                        </div>
                        <Button
                          onClick={startGeneration}
                          className="w-full bg-gradient-to-r from-red-600 to-red-700 h-8 text-sm"
                        >
                          <Play className="w-3 h-3 mr-1" />
                          패치 생성 시작
                        </Button>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs text-slate-300">
                            <span>생성 진행률</span>
                            <span>{status.progress.toFixed(1)}%</span>
                          </div>
                          <Progress value={status.progress} className="h-1" />
                        </div>
                        <div className="text-xs text-slate-400">
                          반복: {status.currentIteration} / {status.totalIterations}
                        </div>
                        <div className="grid grid-cols-1 gap-1 text-xs">
                          <div className="bg-slate-700/30 p-1 rounded flex justify-between">
                            <span className="text-slate-400">성공률:</span>
                            <span className="text-white">{status.attackSuccessRate.toFixed(1)}%</span>
                          </div>
                          <div className="bg-slate-700/30 p-1 rounded flex justify-between">
                            <span className="text-slate-400">손실:</span>
                            <span className="text-white">{status.metrics.loss.toFixed(3)}</span>
                          </div>
                        </div>
                        <Button
                          onClick={stopGeneration}
                          className="w-full bg-red-600 hover:bg-red-700 h-8 text-sm"
                        >
                          <Square className="w-3 h-3 mr-1" />
                          생성 중지
                        </Button>
                      </div>
                    )}
                  </>
                )}

                {/* Step 2: 결과 확인 */}
                {activeStep === 2 && (
                  <>
                    <div className="space-y-2">
                      <div className="bg-green-900/20 border border-green-500/30 p-2 rounded">
                        <p className="text-xs text-green-300">패치 생성이 완료되었습니다!</p>
                      </div>

                      <div className="space-y-1 p-2 bg-slate-700/30 rounded border border-white/10">
                        <div className="flex justify-between text-xs text-slate-300">
                          <span>최종 성공률</span>
                          <span className="font-medium text-white">{status.attackSuccessRate.toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between text-xs text-slate-300">
                          <span>총 반복 수</span>
                          <span className="font-medium text-white">{status.currentIteration}</span>
                        </div>
                        <div className="flex justify-between text-xs text-slate-300">
                          <span>최종 손실</span>
                          <span className="font-medium text-white">{status.metrics.loss.toFixed(3)}</span>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <Button className="w-full h-7 text-xs" variant="outline">
                          <Download className="w-3 h-3 mr-1" />
                          패치 다운로드
                        </Button>
                        <Button className="w-full h-7 text-xs" variant="outline">
                          <Upload className="w-3 h-3 mr-1" />
                          실제 환경 적용
                        </Button>
                        <Button onClick={handleReset} className="w-full h-7 text-xs" variant="ghost">
                          <RotateCcw className="w-3 h-3 mr-1" />
                          새로 시작
                        </Button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
          </div>

          {/* 오른쪽: 정보 패널 - col-9 */}
          <div className="col-span-12 lg:col-span-9 min-h-0">
            <div className="h-full overflow-y-auto space-y-2 pr-2">
              {/* Step 0: 환경 & 모델 설정 화면 */}
              {activeStep === 0 && (
                <>
                  <Card className="bg-slate-800/50 border-white/10">
                    <CardHeader className="py-2 px-3">
                      <CardTitle className="text-sm text-white">3D 데이터셋 및 환경 설정</CardTitle>
                      <CardDescription className="text-xs text-slate-400">
                        생성된 3D 데이터 선택 및 시뮬레이션 환경 구성
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="p-3">
                      <div className="space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-2">
                            <h4 className="text-xs font-medium text-slate-300">선택된 데이터셋</h4>
                            {config.selectedDataset ? (
                              <div className="bg-slate-700/30 p-2 rounded space-y-1 text-xs">
                                <div className="flex justify-between">
                                  <span className="text-slate-400">데이터셋:</span>
                                  <span className="text-white">
                                    {generatedDatasets.find(d => d.value === config.selectedDataset)?.label}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-slate-400">이미지 수:</span>
                                  <span className="text-white">
                                    {generatedDatasets.find(d => d.value === config.selectedDataset)?.count}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-slate-400">생성일:</span>
                                  <span className="text-white">
                                    {generatedDatasets.find(d => d.value === config.selectedDataset)?.date}
                                  </span>
                                </div>
                              </div>
                            ) : (
                              <div className="bg-slate-700/30 p-4 rounded text-center">
                                <Database className="w-8 h-8 mx-auto mb-2 text-slate-500" />
                                <p className="text-xs text-slate-400">데이터셋을 선택해주세요</p>
                              </div>
                            )}
                          </div>
                          <div className="space-y-2">
                            <h4 className="text-xs font-medium text-slate-300">모델 설정</h4>
                            <div className="bg-slate-700/30 p-2 rounded space-y-1 text-xs">
                              <div className="flex justify-between">
                                <span className="text-slate-400">대상 모델:</span>
                                <span className="text-white">{modelOptions.find(m => m.value === config.targetModel)?.label}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-400">대상 클래스:</span>
                                <span className="text-white">{config.targetClass}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-slate-400">최적화 방법:</span>
                                <span className="text-white">{config.optimizationMethod.toUpperCase()}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {config.selectedDataset && (
                          <div className="space-y-2">
                            <h4 className="text-xs font-medium text-slate-300">데이터셋 영상 미리보기</h4>
                            <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden border border-white/10">
                              <canvas
                                ref={datasetPreviewRef}
                                className="w-full h-full"
                                width={640}
                                height={360}
                              />

                              {/* 비디오 컨트롤 오버레이 */}
                              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
                                <div className="flex items-center gap-3">
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    className="h-6 px-2 text-white hover:bg-white/20"
                                    onClick={() => setIsPlaying(!isPlaying)}
                                  >
                                    {isPlaying ? (
                                      <Pause className="w-3 h-3" />
                                    ) : (
                                      <Play className="w-3 h-3" />
                                    )}
                                  </Button>

                                  <div className="flex-1">
                                    <div className="flex items-center gap-2">
                                      <span className="text-xs text-white/80">Frame {currentFrame}/{generatedDatasets.find(d => d.value === config.selectedDataset)?.count || 0}</span>
                                      <div className="flex-1 h-1 bg-white/20 rounded-full overflow-hidden">
                                        <div
                                          className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-100"
                                          style={{ width: `${(currentFrame / (generatedDatasets.find(d => d.value === config.selectedDataset)?.count || 1)) * 100}%` }}
                                        />
                                      </div>
                                      <span className="text-xs text-white/80">30 FPS</span>
                                    </div>
                                  </div>
                                </div>
                              </div>

                              {/* 재생 상태 인디케이터 */}
                              {isPlaying && (
                                <div className="absolute top-3 right-3">
                                  <Badge className="bg-red-600 animate-pulse text-xs">
                                    <div className="w-2 h-2 bg-white rounded-full mr-1 animate-pulse" />
                                    PLAYING
                                  </Badge>
                                </div>
                              )}
                            </div>
                            <p className="text-xs text-slate-400 text-center">
                              {generatedDatasets.find(d => d.value === config.selectedDataset)?.label} -
                              {' '}{generatedDatasets.find(d => d.value === config.selectedDataset)?.count}개 프레임 재생
                            </p>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}

              {/* Step 1: 생성 실행 화면 */}
              {activeStep === 1 && (
                <>
                  <Card className="bg-slate-800/50 border-white/10">
                    <CardHeader className="py-2 px-3">
                      <CardTitle className="text-sm text-white">실시간 시뮬레이션</CardTitle>
                    </CardHeader>
                    <CardContent className="p-3">
                      <div className="relative">
                        <canvas
                          ref={simulationCanvasRef}
                          width={400}
                          height={300}
                          className="w-full border border-white/10 rounded"
                        />
                        {status.isGenerating && (
                          <div className="absolute top-2 right-2">
                            <Badge className="bg-red-600 animate-pulse">
                              <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse" />
                              LIVE
                            </Badge>
                          </div>
                        )}
                      </div>

                      {/* 실시간 메트릭 */}
                      <div className="grid grid-cols-3 gap-2 mt-3">
                        <div className="bg-slate-700/30 rounded p-2">
                          <p className="text-slate-400 text-xs">공격 성공률</p>
                          <p className="text-white text-lg font-semibold">
                            {status.attackSuccessRate.toFixed(1)}%
                          </p>
                        </div>
                        <div className="bg-slate-700/30 rounded p-2">
                          <p className="text-slate-400 text-xs">손실값</p>
                          <p className="text-white text-lg font-semibold">
                            {status.metrics.loss.toFixed(3)}
                          </p>
                        </div>
                        <div className="bg-slate-700/30 rounded p-2">
                          <p className="text-slate-400 text-xs">섭동</p>
                          <p className="text-white text-lg font-semibold">
                            {status.metrics.perturbation.toFixed(3)}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}

              {/* Step 2: 결과 확인 화면 */}
              {activeStep === 2 && (
                <>
                  <Card className="bg-slate-800/50 border-white/10">
                    <CardHeader className="py-2 px-3">
                      <CardTitle className="text-sm text-white">생성된 패치 결과</CardTitle>
                    </CardHeader>
                    <CardContent className="p-3">
                      <div className="grid grid-cols-2 gap-3">
                        <div className="space-y-2">
                          <h4 className="text-xs font-medium text-slate-300">최종 패치</h4>
                          <canvas
                            ref={patchPreviewRef}
                            width={120}
                            height={120}
                            className="w-full border border-white/10 rounded bg-slate-700/30"
                          />
                        </div>
                        <div className="space-y-2">
                          <h4 className="text-xs font-medium text-slate-300">성능 분석</h4>
                          <div className="bg-slate-700/30 p-2 rounded space-y-1 text-xs">
                            <div className="flex justify-between">
                              <span className="text-slate-400">공격 성공률:</span>
                              <span className="text-red-400">{status.attackSuccessRate.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-slate-400">평균 손실:</span>
                              <span className="text-white">{status.metrics.loss.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-slate-400">총 반복:</span>
                              <span className="text-white">{status.currentIteration}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-slate-400">평균 섭동:</span>
                              <span className="text-white">{status.metrics.perturbation.toFixed(3)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Buttons - col-12 (full width at bottom) */}
        <div className="flex gap-2 flex-shrink-0">
          {activeStep > 0 && (
            <Button
              onClick={handleBack}
              variant="outline"
              className="flex-1"
              size="default"
            >
              <ChevronLeft className="w-5 h-5 mr-2" />
              이전 단계
            </Button>
          )}
          {activeStep < steps.length - 1 && (
            <Button
              onClick={() => {
                if (activeStep === 1 && !status.isGenerating && status.progress === 0) {
                  startGeneration()
                } else if (activeStep === 1 && status.progress === 100) {
                  handleNext()
                } else {
                  handleNext()
                }
              }}
              disabled={
                (activeStep === 0 && (!config.projectName || !config.selectedDataset)) ||
                (activeStep === 1 && status.isGenerating && status.progress < 100)
              }
              className="flex-1 bg-gradient-to-r from-primary to-accent hover:from-primary/80 hover:to-accent/80 shadow-lg"
              size="default"
            >
              {activeStep === 1 && !status.isGenerating && status.progress === 0 ? '생성 시작' : '다음 단계'}
              <ChevronRight className="w-5 h-5 ml-2" />
            </Button>
          )}
          {activeStep === steps.length - 1 && (
            <Button
              onClick={handleReset}
              className="w-full"
              variant="outline"
              size="default"
            >
              <RotateCcw className="w-5 h-5 mr-2" />
              처음부터 다시 시작
            </Button>
          )}
        </div>
      </div>

    </div>
  )
}