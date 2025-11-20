"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { toast } from "sonner"
import {
  Play,
  Pause,
  Shield,
  Target,
  Settings,
  Database,
  AlertCircle,
  CheckCircle2,
  Clock,
  FileStack,
  TrendingDown,
  BarChart3,
  Activity,
  ArrowRight,
  Brain,
  Gauge,
  ShieldAlert,
  Award
} from "lucide-react"

interface Dataset {
  id: string
  name: string
  description: string
  imageCount: number
  type: 'normal' | 'adversarial'
  attackMethod?: string
}

interface EvaluationProgress {
  total: number
  processed: number
  currentDataset: string
  estimatedTime: string
  metrics: {
    ap50: number
    f1Score: number
    precision: number
    recall: number
  }
}

interface EvaluationResult {
  id: string
  name: string
  model: string
  timestamp: string
  normalDataset: {
    name: string
    ap50: number
    f1Score: number
    precision: number
    recall: number
  }
  adversarialDatasets: Array<{
    name: string
    attackMethod: string
    ap50: number
    f1Score: number
    precision: number
    recall: number
    performanceDrop: number
  }>
  reliabilityLevel: 'high' | 'medium' | 'low'
  reliabilityScore: number
}

export function PerformanceEvaluation() {
  const [evaluationName, setEvaluationName] = useState("")
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedNormalDatasets, setSelectedNormalDatasets] = useState<string[]>([])
  const [selectedAdversarialDatasets, setSelectedAdversarialDatasets] = useState<string[]>([])
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [showProgressModal, setShowProgressModal] = useState(false)
  const [showResultPage, setShowResultPage] = useState(false)
  const [evaluationProgress, setEvaluationProgress] = useState<EvaluationProgress | null>(null)
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResult | null>(null)

  // 데이터셋 목록 (정상 + 적대적)
  const [normalDatasets] = useState<Dataset[]>([
    {
      id: "cars_normal",
      name: "자동차 분류 데이터셋",
      description: "원본 자동차 이미지 데이터셋",
      imageCount: 15847,
      type: 'normal'
    },
    {
      id: "military_normal",
      name: "군사 장비 데이터셋",
      description: "원본 군사 장비 데이터셋",
      imageCount: 8932,
      type: 'normal'
    },
    {
      id: "drone_normal",
      name: "드론 탐지 데이터셋",
      description: "원본 드론 이미지 데이터셋",
      imageCount: 5621,
      type: 'normal'
    }
  ])

  const [adversarialDatasets] = useState<Dataset[]>([
    {
      id: "cars_fgsm",
      name: "자동차 FGSM 공격 데이터셋",
      description: "FGSM 공격이 적용된 자동차 데이터셋",
      imageCount: 15847,
      type: 'adversarial',
      attackMethod: 'FGSM'
    },
    {
      id: "cars_pgd",
      name: "자동차 PGD 공격 데이터셋",
      description: "PGD 공격이 적용된 자동차 데이터셋",
      imageCount: 15847,
      type: 'adversarial',
      attackMethod: 'PGD'
    },
    {
      id: "military_patch",
      name: "군사 장비 패치 공격 데이터셋",
      description: "적대적 패치가 적용된 군사 장비 데이터셋",
      imageCount: 8932,
      type: 'adversarial',
      attackMethod: 'Patch'
    },
    {
      id: "drone_noise",
      name: "드론 노이즈 공격 데이터셋",
      description: "노이즈 공격이 적용된 드론 데이터셋",
      imageCount: 5621,
      type: 'adversarial',
      attackMethod: 'Noise'
    }
  ])

  const aiModels = [
    { id: "yolov8", name: "YOLO v8", type: "Object Detection" },
    { id: "resnet50", name: "ResNet-50", type: "Classification" },
    { id: "efficientnet", name: "EfficientNet", type: "Classification" },
    { id: "detectron2", name: "Detectron2", type: "Object Detection" },
    { id: "maskrcnn", name: "Mask R-CNN", type: "Instance Segmentation" }
  ]

  const handleStartEvaluation = () => {
    if (!evaluationName.trim()) {
      toast.error("평가 이름을 입력해주세요")
      return
    }

    if (!selectedModel) {
      toast.error("평가할 AI 모델을 선택해주세요")
      return
    }

    if (selectedNormalDatasets.length === 0) {
      toast.error("최소 1개의 기본 데이터셋을 선택해주세요")
      return
    }

    setIsEvaluating(true)
    setShowProgressModal(true)

    const totalDatasets = selectedNormalDatasets.length + selectedAdversarialDatasets.length
    let processed = 0

    setEvaluationProgress({
      total: totalDatasets,
      processed: 0,
      currentDataset: "초기화 중...",
      estimatedTime: "계산 중...",
      metrics: {
        ap50: 0,
        f1Score: 0,
        precision: 0,
        recall: 0
      }
    })

    // 평가 시뮬레이션
    const interval = setInterval(() => {
      processed++

      if (processed > totalDatasets) {
        clearInterval(interval)
        setIsEvaluating(false)

        // 평가 결과 생성
        const result: EvaluationResult = {
          id: `eval_${Date.now()}`,
          name: evaluationName,
          model: selectedModel,
          timestamp: new Date().toISOString(),
          normalDataset: {
            name: normalDatasets.find(d => selectedNormalDatasets.includes(d.id))?.name || "",
            ap50: 0.924,
            f1Score: 0.891,
            precision: 0.903,
            recall: 0.879
          },
          adversarialDatasets: selectedAdversarialDatasets.map(datasetId => {
            const dataset = adversarialDatasets.find(d => d.id === datasetId)
            const performanceDrop = 25 + Math.random() * 30 // 25-55% 성능 하락
            return {
              name: dataset?.name || "",
              attackMethod: dataset?.attackMethod || "",
              ap50: 0.924 * (1 - performanceDrop / 100),
              f1Score: 0.891 * (1 - performanceDrop / 100),
              precision: 0.903 * (1 - performanceDrop / 100),
              recall: 0.879 * (1 - performanceDrop / 100),
              performanceDrop
            }
          }),
          reliabilityLevel: 'low',
          reliabilityScore: 45.2
        }

        // 신뢰성 수준 계산
        if (result.adversarialDatasets.length > 0) {
          const avgDrop = result.adversarialDatasets.reduce((acc, d) => acc + d.performanceDrop, 0) / result.adversarialDatasets.length
          if (avgDrop < 20) {
            result.reliabilityLevel = 'high'
            result.reliabilityScore = 85 + Math.random() * 15
          } else if (avgDrop < 40) {
            result.reliabilityLevel = 'medium'
            result.reliabilityScore = 50 + Math.random() * 35
          } else {
            result.reliabilityLevel = 'low'
            result.reliabilityScore = 20 + Math.random() * 30
          }
        }

        setEvaluationResult(result)
        toast.success("성능 평가가 완료되었습니다")
        return
      }

      const currentDatasetId = processed <= selectedNormalDatasets.length
        ? selectedNormalDatasets[processed - 1]
        : selectedAdversarialDatasets[processed - selectedNormalDatasets.length - 1]

      const currentDataset = [...normalDatasets, ...adversarialDatasets].find(d => d.id === currentDatasetId)

      setEvaluationProgress({
        total: totalDatasets,
        processed: processed,
        currentDataset: currentDataset?.name || "",
        estimatedTime: `${Math.max(0, (totalDatasets - processed) * 5)}초`,
        metrics: {
          ap50: 0.8 + Math.random() * 0.15,
          f1Score: 0.75 + Math.random() * 0.15,
          precision: 0.8 + Math.random() * 0.1,
          recall: 0.7 + Math.random() * 0.15
        }
      })
    }, 2000)
  }

  const handleViewResults = () => {
    setShowProgressModal(false)
    setShowResultPage(true)
  }

  const getReliabilityColor = (level: string) => {
    switch(level) {
      case 'high': return 'text-green-400'
      case 'medium': return 'text-white'
      case 'low': return 'text-red-400'
      default: return 'text-slate-400'
    }
  }

  const getReliabilityBadge = (level: string) => {
    switch(level) {
      case 'high': return <Badge className="bg-green-500/20 text-green-400 border-green-500/30">높음</Badge>
      case 'medium': return <Badge className="bg-amber-500/20 text-white border-amber-500/30">보통</Badge>
      case 'low': return <Badge className="bg-red-500/20 text-red-400 border-red-500/30">낮음</Badge>
      default: return null
    }
  }

  // 결과 페이지 표시
  if (showResultPage && evaluationResult) {
    return (
      <div className="space-y-6">
        {/* 헤더 */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white">{evaluationResult.name}</h2>
            <p className="text-slate-400">성능 및 신뢰성 평가 결과</p>
          </div>
          <Button variant="outline" onClick={() => setShowResultPage(false)}>
            돌아가기
          </Button>
        </div>

        {/* 요약 카드 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="w-5 h-5 text-blue-400" />
                AI 모델
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xl font-bold text-white">{evaluationResult.model}</p>
              <p className="text-sm text-slate-400 mt-1">평가 대상 모델</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <Gauge className="w-5 h-5 text-green-400" />
                기본 성능
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xl font-bold text-white">
                AP50: {(evaluationResult.normalDataset.ap50 * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-slate-400 mt-1">원본 데이터셋</p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <ShieldAlert className="w-5 h-5 text-red-400" />
                신뢰성 수준
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                {getReliabilityBadge(evaluationResult.reliabilityLevel)}
              </div>
              <p className="text-sm text-slate-400 mt-1">
                점수: {evaluationResult.reliabilityScore.toFixed(1)}/100
              </p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg flex items-center gap-2">
                <TrendingDown className="w-5 h-5 text-white" />
                평균 성능 하락
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xl font-bold text-red-400">
                {evaluationResult.adversarialDatasets.length > 0
                  ? `${(evaluationResult.adversarialDatasets.reduce((acc, d) => acc + d.performanceDrop, 0) / evaluationResult.adversarialDatasets.length).toFixed(1)}%`
                  : 'N/A'}
              </p>
              <p className="text-sm text-slate-400 mt-1">적대적 공격 영향</p>
            </CardContent>
          </Card>
        </div>

        {/* 상세 결과 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 기본 데이터셋 성능 */}
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5" />
                기본 데이터셋 성능
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="bg-slate-900/50 p-4 rounded-lg">
                  <h4 className="text-white font-medium mb-3">{evaluationResult.normalDataset.name}</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-slate-400">AP50</p>
                      <p className="text-lg font-bold text-green-400">
                        {(evaluationResult.normalDataset.ap50 * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">F1-Score</p>
                      <p className="text-lg font-bold text-blue-400">
                        {(evaluationResult.normalDataset.f1Score * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">Precision</p>
                      <p className="text-lg font-bold text-purple-400">
                        {(evaluationResult.normalDataset.precision * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400">Recall</p>
                      <p className="text-lg font-bold text-cyan-400">
                        {(evaluationResult.normalDataset.recall * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 적대적 데이터셋 성능 */}
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5" />
                적대적 데이터셋 성능
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="max-h-72 overflow-y-auto">
                <div className="space-y-3">
                  {evaluationResult.adversarialDatasets.map((dataset, index) => (
                    <div key={index} className="bg-slate-900/50 p-4 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-white font-medium text-sm">{dataset.name}</h4>
                        <Badge variant="outline" className="text-xs">
                          {dataset.attackMethod}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div className="flex justify-between">
                          <span className="text-slate-400">AP50:</span>
                          <span className="text-white">{(dataset.ap50 * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">F1:</span>
                          <span className="text-white">{(dataset.f1Score * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="mt-2 pt-2 border-t border-white/10">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-slate-400">성능 하락</span>
                          <span className="text-sm font-bold text-red-400">
                            -{dataset.performanceDrop.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 신뢰성 평가 결과 */}
        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Award className="w-5 h-5" />
              신뢰성 평가 결과
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Alert className={`border-${evaluationResult.reliabilityLevel === 'low' ? 'red' : evaluationResult.reliabilityLevel === 'medium' ? 'amber' : 'green'}-500/30`}>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-white">
                {evaluationResult.reliabilityLevel === 'low'
                  ? "이 모델은 적대적 공격에 매우 취약합니다. 보안 강화가 필요합니다."
                  : evaluationResult.reliabilityLevel === 'medium'
                  ? "이 모델은 일부 적대적 공격에 취약합니다. 추가적인 방어 메커니즘을 고려하세요."
                  : "이 모델은 적대적 공격에 대해 상대적으로 강건합니다."}
              </AlertDescription>
            </Alert>

            <div className="mt-4 p-4 bg-slate-900/50 rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <span className="text-white font-medium">신뢰성 점수</span>
                <span className={`text-2xl font-bold ${getReliabilityColor(evaluationResult.reliabilityLevel)}`}>
                  {evaluationResult.reliabilityScore.toFixed(1)}/100
                </span>
              </div>
              <Progress
                value={evaluationResult.reliabilityScore}
                className="h-3"
              />
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // 메인 평가 설정 페이지
  return (
    <div className="space-y-6">
      {/* 상단 정보 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-blue-400" />
              평가 이름
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-lg font-bold text-white truncate">
              {evaluationName || "미입력"}
            </p>
            <p className="text-sm text-slate-400 mt-1">성능 평가 식별자</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Brain className="w-5 h-5 text-green-400" />
              대상 모델
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-lg font-bold text-white">
              {selectedModel ? selectedModel.toUpperCase() : "미선택"}
            </p>
            <p className="text-sm text-slate-400 mt-1">평가할 AI 모델</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Database className="w-5 h-5 text-white" />
              기본 데이터셋
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-white">{selectedNormalDatasets.length}</p>
            <p className="text-sm text-slate-400 mt-1">선택된 데이터셋</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Shield className="w-5 h-5 text-red-400" />
              공격 데이터셋
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-white">{selectedAdversarialDatasets.length}</p>
            <p className="text-sm text-slate-400 mt-1">적대적 데이터셋</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 왼쪽: 평가 설정 */}
        <div className="lg:col-span-1">
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                평가 설정
              </CardTitle>
              <CardDescription>성능 평가 파라미터 설정</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* 평가 이름 */}
              <div className="space-y-2">
                <Label>평가 이름</Label>
                <Input
                  value={evaluationName}
                  onChange={(e) => setEvaluationName(e.target.value)}
                  placeholder="예: YOLO_신뢰성_평가_v1"
                  className="bg-slate-900/50"
                />
              </div>

              {/* 모델 선택 */}
              <div className="space-y-2">
                <Label>평가 대상 AI 모델</Label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="모델 선택" />
                  </SelectTrigger>
                  <SelectContent>
                    {aiModels.map(model => (
                      <SelectItem key={model.id} value={model.id}>
                        <div className="flex items-center justify-between w-full">
                          <span>{model.name}</span>
                          <Badge variant="outline" className="ml-2 text-xs">
                            {model.type}
                          </Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* 평가 지표 정보 */}
              <div className="bg-slate-900/50 p-4 rounded-lg space-y-2">
                <h4 className="text-sm font-medium text-white mb-2">평가 지표</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full" />
                    <span className="text-slate-300">AP50 (Average Precision)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-400 rounded-full" />
                    <span className="text-slate-300">F1-Score</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-purple-400 rounded-full" />
                    <span className="text-slate-300">Precision (정밀도)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full" />
                    <span className="text-slate-300">Recall (재현율)</span>
                  </div>
                </div>
              </div>

              {/* 신뢰성 판정 기준 */}
              <div className="bg-slate-900/50 p-4 rounded-lg space-y-2">
                <h4 className="text-sm font-medium text-white mb-2">신뢰성 판정 기준</h4>
                <div className="space-y-1 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">높음</span>
                    <span className="text-green-400">성능 하락 &lt; 20%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">보통</span>
                    <span className="text-white">성능 하락 20-40%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">낮음</span>
                    <span className="text-red-400">성능 하락 &gt; 40%</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 오른쪽: 데이터셋 선택 */}
        <div className="lg:col-span-2 space-y-4">
          {/* 기본 데이터셋 선택 */}
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5" />
                기본 데이터셋 선택
              </CardTitle>
              <CardDescription>성능 평가에 사용할 원본 데이터셋</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="max-h-48 overflow-y-auto">
                <div className="space-y-2 pr-4">
                  {normalDatasets.map(dataset => (
                    <div
                      key={dataset.id}
                      className={`p-3 rounded-lg border transition-all cursor-pointer ${
                        selectedNormalDatasets.includes(dataset.id)
                          ? 'bg-primary/10 border-primary'
                          : 'bg-slate-900/50 border-white/10 hover:border-white/20'
                      }`}
                      onClick={() => {
                        setSelectedNormalDatasets(prev =>
                          prev.includes(dataset.id)
                            ? prev.filter(id => id !== dataset.id)
                            : [...prev, dataset.id]
                        )
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <Checkbox
                          checked={selectedNormalDatasets.includes(dataset.id)}
                          onCheckedChange={() => {}}
                        />
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium text-white text-sm">{dataset.name}</h4>
                            <Badge  className="text-xs">
                              {dataset.imageCount.toLocaleString()}개
                            </Badge>
                          </div>
                          <p className="text-xs text-slate-400 mt-1">{dataset.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 적대적 공격 데이터셋 선택 */}
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5" />
                적대적 공격 데이터셋 선택
              </CardTitle>
              <CardDescription>신뢰성 평가에 사용할 공격 데이터셋</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="max-h-60 overflow-y-auto">
                <div className="space-y-2 pr-4">
                  {adversarialDatasets.map(dataset => (
                    <div
                      key={dataset.id}
                      className={`p-3 rounded-lg border transition-all cursor-pointer ${
                        selectedAdversarialDatasets.includes(dataset.id)
                          ? 'bg-primary/10 border-primary'
                          : 'bg-slate-900/50 border-white/10 hover:border-white/20'
                      }`}
                      onClick={() => {
                        setSelectedAdversarialDatasets(prev =>
                          prev.includes(dataset.id)
                            ? prev.filter(id => id !== dataset.id)
                            : [...prev, dataset.id]
                        )
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <Checkbox
                          checked={selectedAdversarialDatasets.includes(dataset.id)}
                          onCheckedChange={() => {}}
                        />
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <h4 className="font-medium text-white text-sm">{dataset.name}</h4>
                            <div className="flex gap-2">
                              <Badge variant="outline" className="text-xs">
                                {dataset.attackMethod}
                              </Badge>
                              <Badge  className="text-xs">
                                {dataset.imageCount.toLocaleString()}개
                              </Badge>
                            </div>
                          </div>
                          <p className="text-xs text-slate-400 mt-1">{dataset.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 액션 버튼 */}
          <div className="flex gap-3">
            <Button
              onClick={handleStartEvaluation}
              disabled={isEvaluating}
              className="flex-1"
            >
              <Play className="w-4 h-4 mr-2" />
              평가 시작
            </Button>
          </div>
        </div>
      </div>

      {/* 진행 상황 모달 */}
      <Dialog open={showProgressModal} onOpenChange={setShowProgressModal}>
        <DialogContent className="sm:max-w-[600px] bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-white">
              <Activity className="w-5 h-5" />
              성능 평가 진행 중
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              선택한 데이터셋에 대해 AI 모델 성능을 평가하고 있습니다
            </DialogDescription>
          </DialogHeader>

          {evaluationProgress && (
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-400">전체 진행률</span>
                  <span className="text-white">
                    {evaluationProgress.processed} / {evaluationProgress.total} 데이터셋
                  </span>
                </div>
                <Progress
                  value={(evaluationProgress.processed / evaluationProgress.total) * 100}
                  className="h-3"
                />
              </div>

              <div className="bg-slate-800/50 p-4 rounded-lg">
                <p className="text-sm text-slate-400 mb-2">현재 평가 중</p>
                <p className="text-white font-medium">{evaluationProgress.currentDataset}</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-800/50 p-3 rounded-lg">
                  <p className="text-xs text-slate-400 mb-1">AP50</p>
                  <p className="text-xl font-bold text-green-400">
                    {(evaluationProgress.metrics.ap50 * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-slate-800/50 p-3 rounded-lg">
                  <p className="text-xs text-slate-400 mb-1">F1-Score</p>
                  <p className="text-xl font-bold text-blue-400">
                    {(evaluationProgress.metrics.f1Score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">예상 남은 시간</span>
                <span className="text-white">{evaluationProgress.estimatedTime}</span>
              </div>
            </div>
          )}

          <DialogFooter>
            {isEvaluating ? (
              <Button disabled variant="outline">
                <Pause className="w-4 h-4 mr-2" />
                평가 중...
              </Button>
            ) : (
              <Button onClick={handleViewResults} className="bg-primary">
                <ArrowRight className="w-4 h-4 mr-2" />
                결과 보기
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}