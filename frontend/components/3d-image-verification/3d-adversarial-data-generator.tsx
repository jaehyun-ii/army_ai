"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Play,
  Square,
  Download,
  AlertCircle,
  CheckCircle,
  Database,
  Shield,
  Brain,
  Activity,
  Eye,
  Target,
  Zap,
  Settings,
  Crosshair
} from "lucide-react"

interface AttackConfig {
  attackName: string
  selectedEnvironment: string // 3D 데이터 생성에서 생성된 환경
  selectedPatch: string // 3D 적대적 패치 생성에서 생성된 패치
  targetModel: string // 공격할 AI 모델
}

interface GenerationStatus {
  isRunning: boolean
  currentPhase: string
  progress: number
  processedFrames: number
  totalFrames: number
  attackSuccessRate: number
  averageConfidence: number
  logs: string[]
}

export function AdversarialDataGenerator3D() {
  const [config, setConfig] = useState<AttackConfig>({
    attackName: "",
    selectedEnvironment: "",
    selectedPatch: "",
    targetModel: ""
  })

  const [status, setStatus] = useState<GenerationStatus>({
    isRunning: false,
    currentPhase: "대기 중",
    progress: 0,
    processedFrames: 0,
    totalFrames: 0,
    attackSuccessRate: 0,
    averageConfidence: 0,
    logs: []
  })

  const simulationCanvasRef = useRef<HTMLCanvasElement>(null)

  // 생성된 3D 환경 목록 (3D 데이터 생성에서 생성된 것들)
  const generatedEnvironments = [
    { value: "env_urban_day_1", label: "Urban Day Environment v1", frames: 1000, date: "2024-01-15" },
    { value: "env_highway_night_2", label: "Highway Night Environment v2", frames: 800, date: "2024-01-14" },
    { value: "env_suburban_rain_1", label: "Suburban Rain Environment v1", frames: 1200, date: "2024-01-13" },
    { value: "env_city_fog_3", label: "City Fog Environment v3", frames: 1500, date: "2024-01-12" },
    { value: "env_rural_sunset_1", label: "Rural Sunset Environment v1", frames: 900, date: "2024-01-11" }
  ]

  // 생성된 패치 목록 (3D 적대적 패치 생성에서 생성된 것들)
  const generatedPatches = [
    { value: "patch_car_v1", label: "Car Attack Patch v1", successRate: 92.5, date: "2024-01-15" },
    { value: "patch_truck_v2", label: "Truck Attack Patch v2", successRate: 87.3, date: "2024-01-14" },
    { value: "patch_bus_v1", label: "Bus Attack Patch v1", successRate: 89.7, date: "2024-01-13" },
    { value: "patch_pedestrian_v3", label: "Pedestrian Attack Patch v3", successRate: 85.2, date: "2024-01-12" },
    { value: "patch_universal_v1", label: "Universal Attack Patch v1", successRate: 88.9, date: "2024-01-11" }
  ]

  // AI 모델 목록
  const targetModels = [
    { value: "yolov8", label: "YOLO v8", type: "Object Detection" },
    { value: "yolov5", label: "YOLO v5", type: "Object Detection" },
    { value: "fasterrcnn", label: "Faster R-CNN", type: "Object Detection" },
    { value: "ssd", label: "SSD MobileNet", type: "Object Detection" },
    { value: "maskrcnn", label: "Mask R-CNN", type: "Instance Segmentation" },
    { value: "detr", label: "DETR", type: "Object Detection" }
  ]

  // 적대적 공격 실행
  const startAttack = () => {
    if (!config.attackName || !config.selectedEnvironment || !config.selectedPatch || !config.targetModel) {
      alert("모든 필수 항목을 선택해주세요.")
      return
    }

    const environment = generatedEnvironments.find(env => env.value === config.selectedEnvironment)
    const totalFrames = environment?.frames || 1000

    setStatus({
      isRunning: true,
      currentPhase: "적대적 공격 초기화 중...",
      progress: 0,
      processedFrames: 0,
      totalFrames: totalFrames,
      attackSuccessRate: 0,
      averageConfidence: 95,
      logs: [`[${new Date().toLocaleTimeString()}] 적대적 공격 시작: ${config.attackName}`]
    })

    simulateAttack(totalFrames)
  }

  // 공격 시뮬레이션
  const simulateAttack = (totalFrames: number) => {
    let processed = 0
    const interval = setInterval(() => {
      processed += Math.floor(Math.random() * 30) + 10

      if (processed >= totalFrames) {
        processed = totalFrames
        clearInterval(interval)

        setStatus(prev => ({
          ...prev,
          isRunning: false,
          progress: 100,
          processedFrames: processed,
          currentPhase: "공격 완료",
          attackSuccessRate: 85 + Math.random() * 10,
          averageConfidence: 30 + Math.random() * 20,
          logs: [...prev.logs, `[${new Date().toLocaleTimeString()}] 적대적 공격 완료 - 성공률: ${(85 + Math.random() * 10).toFixed(1)}%`]
        }))
      } else {
        const progress = (processed / totalFrames) * 100
        const successRate = Math.min(95, progress * 0.9 + Math.random() * 10)
        const confidence = Math.max(20, 95 - progress * 0.7)

        setStatus(prev => ({
          ...prev,
          progress: progress,
          processedFrames: processed,
          currentPhase: `프레임 처리 중... (${processed}/${totalFrames})`,
          attackSuccessRate: successRate,
          averageConfidence: confidence,
          logs: processed % 100 === 0 ?
            [...prev.logs, `[${new Date().toLocaleTimeString()}] 처리 완료: ${processed}/${totalFrames} 프레임`]
            : prev.logs
        }))
      }
    }, 100)
  }

  // 공격 중지
  const stopAttack = () => {
    setStatus(prev => ({
      ...prev,
      isRunning: false,
      currentPhase: "중지됨",
      logs: [...prev.logs, `[${new Date().toLocaleTimeString()}] 사용자에 의해 중지됨`]
    }))
  }

  // 시뮬레이션 애니메이션
  useEffect(() => {
    if (simulationCanvasRef.current && status.isRunning) {
      const canvas = simulationCanvasRef.current
      const ctx = canvas.getContext('2d')

      if (ctx) {
        const animate = () => {
          // 배경
          ctx.fillStyle = '#0f172a'
          ctx.fillRect(0, 0, canvas.width, canvas.height)

          // 도로
          ctx.fillStyle = '#334155'
          ctx.fillRect(0, canvas.height * 0.6, canvas.width, canvas.height * 0.4)

          // 차량
          const carX = canvas.width / 2 - 60
          const carY = canvas.height * 0.65
          ctx.fillStyle = '#64748b'
          ctx.fillRect(carX, carY, 120, 60)

          // 적대적 패치 (차량 위에)
          if (status.progress > 0) {
            ctx.fillStyle = `rgba(239, 68, 68, ${0.3 + status.progress / 200})`
            ctx.fillRect(carX + 30, carY + 10, 60, 40)

            // 패치 패턴
            ctx.strokeStyle = '#ef4444'
            ctx.lineWidth = 2
            for (let i = 0; i < 5; i++) {
              ctx.beginPath()
              ctx.moveTo(carX + 30 + i * 12, carY + 10)
              ctx.lineTo(carX + 30 + i * 12, carY + 50)
              ctx.stroke()
            }
          }

          // 탐지 박스 (공격 성공 시 빨간색, 실패 시 초록색)
          ctx.strokeStyle = status.attackSuccessRate > 50 ? '#ef4444' : '#10b981'
          ctx.lineWidth = 2
          ctx.setLineDash([5, 5])
          ctx.strokeRect(carX - 10, carY - 10, 140, 80)
          ctx.setLineDash([])

          // 정보 표시
          ctx.fillStyle = '#ffffff'
          ctx.font = '14px monospace'
          ctx.fillText(`Model: ${config.targetModel || 'None'}`, 10, 25)
          ctx.fillText(`Attack Success: ${status.attackSuccessRate.toFixed(1)}%`, 10, 45)
          ctx.fillText(`Confidence: ${status.averageConfidence.toFixed(1)}%`, 10, 65)
          ctx.fillText(`Frame: ${status.processedFrames}/${status.totalFrames}`, 10, 85)

          if (status.isRunning) {
            requestAnimationFrame(animate)
          }
        }

        animate()
      }
    }
  }, [status, config])

  return (
    <div className="h-full flex flex-col gap-2">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800/80 to-slate-900/80 rounded-xl p-3 border border-white/10 shadow-xl flex-shrink-0">
        <div className="flex-shrink-0">
          <h1 className="text-lg lg:text-xl font-bold text-white flex items-center gap-2">
            <Crosshair className="w-6 h-6 text-red-400" />
            3D 적대적 공격 데이터 생성
          </h1>
          <p className="text-xs text-slate-400">생성된 환경과 패치를 사용하여 적대적 공격 실행</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 space-y-6 overflow-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 왼쪽: 설정 패널 */}
        <div className="lg:col-span-1 space-y-4">
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Settings className="w-5 h-5" />
                공격 설정
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-white">공격 이름</Label>
                <Input
                  value={config.attackName}
                  onChange={(e) => setConfig({ ...config, attackName: e.target.value })}
                  placeholder="예: car_attack_test_v1"
                  className="bg-slate-700/50 border-white/20 text-white mt-2"
                />
              </div>

              <div>
                <Label className="text-white flex items-center gap-1">
                  <Database className="w-4 h-4" />
                  3D 환경 선택
                </Label>
                <Select value={config.selectedEnvironment} onValueChange={(value) => setConfig({ ...config, selectedEnvironment: value })}>
                  <SelectTrigger className="bg-slate-700/50 border-white/20 text-white mt-2">
                    <SelectValue placeholder="생성된 3D 환경 선택" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-white/20">
                    {generatedEnvironments.map((env) => (
                      <SelectItem key={env.value} value={env.value} className="text-white">
                        <div className="flex flex-col">
                          <span>{env.label}</span>
                          <span className="text-xs text-slate-400">{env.frames} frames • {env.date}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label className="text-white flex items-center gap-1">
                  <Shield className="w-4 h-4" />
                  적대적 패치 선택
                </Label>
                <Select value={config.selectedPatch} onValueChange={(value) => setConfig({ ...config, selectedPatch: value })}>
                  <SelectTrigger className="bg-slate-700/50 border-white/20 text-white mt-2">
                    <SelectValue placeholder="생성된 패치 선택" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-white/20">
                    {generatedPatches.map((patch) => (
                      <SelectItem key={patch.value} value={patch.value} className="text-white">
                        <div className="flex flex-col">
                          <span>{patch.label}</span>
                          <span className="text-xs text-slate-400">성공률: {patch.successRate}% • {patch.date}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label className="text-white flex items-center gap-1">
                  <Brain className="w-4 h-4" />
                  대상 AI 모델
                </Label>
                <Select value={config.targetModel} onValueChange={(value) => setConfig({ ...config, targetModel: value })}>
                  <SelectTrigger className="bg-slate-700/50 border-white/20 text-white mt-2">
                    <SelectValue placeholder="공격할 모델 선택" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-white/20">
                    {targetModels.map((model) => (
                      <SelectItem key={model.value} value={model.value} className="text-white">
                        <div className="flex flex-col">
                          <span>{model.label}</span>
                          <span className="text-xs text-slate-400">{model.type}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* 선택된 항목 요약 */}
              {config.selectedEnvironment && config.selectedPatch && config.targetModel && (
                <div className="bg-slate-700/30 rounded-lg p-3 space-y-2">
                  <h4 className="text-white text-sm font-medium">공격 구성 요약</h4>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">환경:</span>
                      <span className="text-white">
                        {generatedEnvironments.find(e => e.value === config.selectedEnvironment)?.label}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">패치:</span>
                      <span className="text-white">
                        {generatedPatches.find(p => p.value === config.selectedPatch)?.label}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">모델:</span>
                      <span className="text-white">
                        {targetModels.find(m => m.value === config.targetModel)?.label}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* 제어 버튼 */}
              <div className="pt-4">
                {!status.isRunning ? (
                  <Button
                    onClick={startAttack}
                    className="w-full bg-gradient-to-r from-red-600 to-red-700"
                    disabled={!config.attackName || !config.selectedEnvironment || !config.selectedPatch || !config.targetModel}
                  >
                    <Play className="w-4 h-4 mr-2" />
                    적대적 공격 실행
                  </Button>
                ) : (
                  <Button
                    onClick={stopAttack}
                    variant="destructive"
                    className="w-full"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    공격 중지
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* 실행 상태 */}
          {status.isRunning && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardHeader>
                <CardTitle className="text-white text-sm flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  실행 상태
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>진행률</span>
                    <span>{status.progress.toFixed(1)}%</span>
                  </div>
                  <Progress value={status.progress} className="h-2" />
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-slate-700/30 rounded p-2">
                    <p className="text-slate-400">처리 프레임</p>
                    <p className="text-white font-medium">{status.processedFrames}/{status.totalFrames}</p>
                  </div>
                  <div className="bg-slate-700/30 rounded p-2">
                    <p className="text-slate-400">공격 성공률</p>
                    <p className="text-red-400 font-medium">{status.attackSuccessRate.toFixed(1)}%</p>
                  </div>
                </div>

                <div className="text-xs text-slate-400">
                  상태: {status.currentPhase}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* 오른쪽: 시뮬레이션 뷰 */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Eye className="w-5 h-5" />
                실시간 시뮬레이션
              </CardTitle>
              <CardDescription className="text-slate-400">
                적대적 패치가 적용된 3D 환경에서의 공격 시뮬레이션
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative bg-slate-900 rounded-lg overflow-hidden">
                <canvas
                  ref={simulationCanvasRef}
                  width={800}
                  height={450}
                  className="w-full"
                />
                {status.isRunning && (
                  <div className="absolute top-4 right-4">
                    <Badge className="bg-red-600 animate-pulse">
                      <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse" />
                      ATTACKING
                    </Badge>
                  </div>
                )}
              </div>

              {/* 실시간 메트릭 */}
              <div className="grid grid-cols-3 gap-4 mt-4">
                <div className="bg-slate-700/30 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Target className="w-4 h-4 text-red-400" />
                    <span className="text-xs text-slate-400">공격 성공률</span>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {status.attackSuccessRate.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Brain className="w-4 h-4 text-blue-400" />
                    <span className="text-xs text-slate-400">모델 신뢰도</span>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {status.averageConfidence.toFixed(1)}%
                  </p>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Zap className="w-4 h-4 text-yellow-400" />
                    <span className="text-xs text-slate-400">처리 속도</span>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {status.isRunning ? "30 FPS" : "0 FPS"}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 로그 */}
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="text-white text-sm">실행 로그</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-900 rounded-lg p-3 h-32 overflow-y-auto">
                {status.logs.length === 0 ? (
                  <p className="text-slate-500 text-xs">대기 중...</p>
                ) : (
                  <div className="space-y-1">
                    {status.logs.map((log, index) => (
                      <p key={index} className="text-xs text-slate-300 font-mono">{log}</p>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* 완료 시 결과 */}
          {!status.isRunning && status.progress === 100 && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  공격 완료
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>
                    적대적 공격이 성공적으로 완료되었습니다.
                    총 {status.processedFrames}개 프레임에서 {status.attackSuccessRate.toFixed(1)}%의 공격 성공률을 달성했습니다.
                  </AlertDescription>
                </Alert>

                <div className="mt-4 flex gap-2">
                  <Button variant="outline" className="flex-1">
                    <Download className="w-4 h-4 mr-2" />
                    결과 다운로드
                  </Button>
                  <Button
                    onClick={() => {
                      setStatus({
                        isRunning: false,
                        currentPhase: "대기 중",
                        progress: 0,
                        processedFrames: 0,
                        totalFrames: 0,
                        attackSuccessRate: 0,
                        averageConfidence: 0,
                        logs: []
                      })
                      setConfig({
                        attackName: "",
                        selectedEnvironment: "",
                        selectedPatch: "",
                        targetModel: ""
                      })
                    }}
                    variant="ghost"
                  >
                    새로운 공격
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
        </div>
      </div>
    </div>
  )
}