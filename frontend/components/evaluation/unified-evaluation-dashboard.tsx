"use client"

import { useState, useEffect } from "react"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import {
  Plus,
  BarChart3,
  Shield,
  Activity,
  TrendingUp,
  Database,
  Zap,
  Eye,
  FileText,
  Image as ImageIcon,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Info,
  Brain
} from "lucide-react"
import { apiClient } from "@/lib/api-client"
import { getImageUrlByStorageKey } from "@/lib/adversarial-api"
import { toast } from "sonner"
import { RobustnessComparison } from "@/components/evaluation/RobustnessComparison"

interface Model {
  id: string
  name: string
  model_type: string
}

interface Dataset {
  id: string
  name: string
  image_count: number
  is_attack_dataset: boolean
  created_at?: string
  description?: string
  // Attack dataset specific fields
  attack_type?: string
  target_class?: string
  parameters?: {
    output_dataset_id?: string
    [key: string]: any
  }
  base_dataset_id?: string
}

export function UnifiedEvaluationDashboard() {
  // Form states
  const [evaluationName, setEvaluationName] = useState("")
  const [description, setDescription] = useState("")
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedBaseDataset, setSelectedBaseDataset] = useState("")
  const [selectedAttackDataset, setSelectedAttackDataset] = useState("")
  const [datasetTab, setDatasetTab] = useState<"base" | "attack">("base")

  // Data states
  const [models, setModels] = useState<Model[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [baseDatasetImages, setBaseDatasetImages] = useState<any[]>([])
  const [attackDatasetImages, setAttackDatasetImages] = useState<any[]>([])

  // UI states
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [evaluationLogs, setEvaluationLogs] = useState<string[]>([])
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [evaluationCompleted, setEvaluationCompleted] = useState(false)
  const [loadingBaseImages, setLoadingBaseImages] = useState(false)
  const [loadingAttackImages, setLoadingAttackImages] = useState(false)
  const [currentImagePage, setCurrentImagePage] = useState(0)

  // Evaluation results states
  const [completedEvaluationIds, setCompletedEvaluationIds] = useState<string[]>([])
  const [showResults, setShowResults] = useState(false)
  const [evaluationResults, setEvaluationResults] = useState<any[]>([])
  const [loadingResults, setLoadingResults] = useState(false)
  const [comparisonData, setComparisonData] = useState<any>(null)

  // Load data
  useEffect(() => {
    loadModels()
    loadDatasets()
  }, [])

  // Load base dataset images when selected or page changes
  useEffect(() => {
    if (selectedBaseDataset) {
      loadBaseDatasetImages(selectedBaseDataset, currentImagePage)
    } else {
      setBaseDatasetImages([])
    }
  }, [selectedBaseDataset, currentImagePage])

  // Load attack dataset images when selected or page changes
  useEffect(() => {
    if (selectedAttackDataset) {
      loadAttackDatasetImages(selectedAttackDataset, currentImagePage)
    } else {
      setAttackDatasetImages([])
    }
  }, [selectedAttackDataset, currentImagePage])

  // Reset page when dataset selection changes
  useEffect(() => {
    setCurrentImagePage(0)
  }, [selectedBaseDataset, selectedAttackDataset])

  const loadModels = async () => {
    try {
      const response: any = await apiClient.getModels()
      setModels(response || [])
    } catch (error) {
      console.error("Failed to load models:", error)
      toast.error("모델 목록을 불러오는데 실패했습니다")
    }
  }

  const loadDatasets = async () => {
    try {
      // Load both regular datasets and attack datasets
      const [regularDatasetsResponse, attackDatasetsResponse]: any[] = await Promise.all([
        apiClient.getDatasets(),
        apiClient.listAttackDatasets()
      ])

      // Add is_attack_dataset flag to regular datasets
      const regularDatasets = (regularDatasetsResponse || []).map((ds: any) => ({
        ...ds,
        is_attack_dataset: false
      }))

      // Add is_attack_dataset flag to attack datasets
      const attackDatasets = (attackDatasetsResponse || []).map((ds: any) => ({
        ...ds,
        is_attack_dataset: true
      }))

      // Merge both datasets
      const allDatasets = [...regularDatasets, ...attackDatasets]
      setDatasets(allDatasets)
    } catch (error) {
      console.error("Failed to load datasets:", error)
      toast.error("데이터셋 목록을 불러오는데 실패했습니다")
    }
  }

  const loadBaseDatasetImages = async (datasetId: string, page: number = 0) => {
    setLoadingBaseImages(true)
    try {
      const offset = page * 6
      console.log("📸 Loading base dataset images for:", datasetId, "page:", page, "offset:", offset)
      const response: any = await apiClient.getDatasetImages(datasetId, offset, 6)
      console.log("📸 Base dataset images response:", response)
      if (response && response.length > 0) {
        console.log("📸 First image storage_key:", response[0].storage_key)
      }
      setBaseDatasetImages(response || [])
    } catch (error) {
      console.error("❌ Failed to load base dataset images:", error)
      setBaseDatasetImages([])
    } finally {
      setLoadingBaseImages(false)
    }
  }

  const loadAttackDatasetImages = async (datasetId: string, page: number = 0) => {
    setLoadingAttackImages(true)
    try {
      const offset = page * 6
      console.log("🔴 Loading attack dataset images for:", datasetId, "page:", page, "offset:", offset)
      // For attack datasets, check if there's an output dataset
      const attackDataset = datasets.find(d => d.id === datasetId && d.is_attack_dataset)
      console.log("🔴 Attack dataset found:", attackDataset)

      if (attackDataset) {
        // Check if there's an output_dataset_id in parameters
        const outputDatasetId = attackDataset.parameters?.output_dataset_id
        console.log("🔴 Output dataset ID:", outputDatasetId)

        if (outputDatasetId) {
          // Load images from the output dataset
          const response: any = await apiClient.getDatasetImages(outputDatasetId, offset, 6)
          console.log("🔴 Attack images response (from output):", response)
          if (response && response.length > 0) {
            console.log("🔴 First image storage_key:", response[0].storage_key)
          }
          setAttackDatasetImages(response || [])
        } else {
          // Try to load from base_dataset_id with attack modifications
          const baseDatasetId = attackDataset.base_dataset_id
          console.log("🔴 Fallback to base dataset ID:", baseDatasetId)
          if (baseDatasetId) {
            const response: any = await apiClient.getDatasetImages(baseDatasetId, offset, 6)
            console.log("🔴 Attack images response (from base):", response)
            setAttackDatasetImages(response || [])
          }
        }
      }
    } catch (error) {
      console.error("❌ Failed to load attack dataset images:", error)
      setAttackDatasetImages([])
    } finally {
      setLoadingAttackImages(false)
    }
  }

  const handleSubmit = async () => {
    if (!evaluationName || !selectedModel) {
      toast.error("평가 이름과 모델을 입력해주세요")
      return
    }

    if (!selectedBaseDataset && !selectedAttackDataset) {
      toast.error("최소 하나의 데이터셋을 선택해주세요 (기준 데이터셋 또는 공격 데이터셋)")
      return
    }

    setIsSubmitting(true)
    setIsEvaluating(true)
    setEvaluationLogs([])
    setEvaluationCompleted(false)

    // SSE connection tracking
    let eventSources: EventSource[] = []

    try {
      // Start logging
      const addLog = (message: string) => {
        setEvaluationLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`])
      }

      addLog("✓ 평가 요청 생성 중...")

      // Determine phase based on selected datasets
      let phase = "pre_attack"
      let evalBaseDatasetId = selectedBaseDataset
      let evalAttackDatasetId = selectedAttackDataset

      if (selectedBaseDataset && selectedAttackDataset) {
        // Both selected: We'll create two runs (clean and adversarial)
        addLog("✓ 비교 평가 모드: 기준 데이터셋과 공격 데이터셋 평가")
      } else if (selectedAttackDataset) {
        // Only attack dataset: treat as post_attack phase
        phase = "post_attack"
        addLog("✓ 공격 데이터셋 평가 모드")
      } else {
        // Only base dataset: treat as pre_attack phase
        addLog("✓ 기준 데이터셋 평가 모드")
      }

      // Create evaluation run(s)
      const createdRuns: any[] = []

      if (selectedBaseDataset && selectedAttackDataset) {
        // Create two runs: one for clean, one for adversarial
        addLog("→ 기준 데이터셋 평가 생성 중...")
        const cleanRun: any = await apiClient.createEvaluationRun({
          name: `${evaluationName} (Clean)`,
          description: description ? `${description}\n기준 데이터셋 평가` : "기준 데이터셋 평가",
          phase: "pre_attack",
          model_id: selectedModel,
          base_dataset_id: selectedBaseDataset,
        })
        createdRuns.push(cleanRun)
        addLog(`✓ 기준 평가 생성됨 (ID: ${cleanRun.id})`)

        addLog("→ 공격 데이터셋 평가 생성 중...")
        const advRun: any = await apiClient.createEvaluationRun({
          name: `${evaluationName} (Adversarial)`,
          description: description ? `${description}\n공격 데이터셋 평가` : "공격 데이터셋 평가",
          phase: "post_attack",
          model_id: selectedModel,
          base_dataset_id: selectedBaseDataset,
          attack_dataset_id: selectedAttackDataset,
        })
        createdRuns.push(advRun)
        addLog(`✓ 공격 평가 생성됨 (ID: ${advRun.id})`)
      } else {
        // Create single run
        const runData: any = {
          name: evaluationName,
          phase: phase,
          model_id: selectedModel,
        }

        // Add optional fields only if they exist
        if (description) runData.description = description
        if (evalBaseDatasetId) runData.base_dataset_id = evalBaseDatasetId
        if (evalAttackDatasetId) runData.attack_dataset_id = evalAttackDatasetId

        const run: any = await apiClient.createEvaluationRun(runData)
        createdRuns.push(run)
        addLog(`✓ 평가 생성됨 (ID: ${run.id})`)
      }

      // Execute evaluation runs with SSE logging
      addLog("→ 평가 실행 중...")

      // Save evaluation IDs
      setCompletedEvaluationIds(createdRuns.map(r => r.id))

      for (const run of createdRuns) {
        addLog(`  실행 중: ${run.name}`)

        // Create unique session ID for SSE
        const sessionId = `eval-${run.id}-${Date.now()}`

        try {
          // Setup SSE connection for real-time logs
          const eventSource = new EventSource(
            `http://localhost:8000/api/v1/evaluation/runs/events/${sessionId}`
          )

          eventSources.push(eventSource)

          eventSource.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data)

              // Format message based on type
              let logMessage = ""
              switch (data.type) {
                case "status":
                  logMessage = `🔄 ${data.message}`
                  break
                case "info":
                  logMessage = `ℹ️  ${data.message}`
                  break
                case "success":
                  logMessage = `✅ ${data.message}`
                  break
                case "error":
                  logMessage = `❌ ${data.message}`
                  break
                case "complete":
                  logMessage = `✨ ${data.message}`
                  setEvaluationCompleted(true)
                  break
                default:
                  logMessage = data.message
              }

              addLog(logMessage)
            } catch (e) {
              console.error("Failed to parse SSE message:", e)
            }
          }

          eventSource.onerror = (error) => {
            console.error("SSE connection error:", error)
            eventSource.close()
          }

          // Execute evaluation with session ID
          await apiClient.executeEvaluationRun(run.id, {
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            session_id: sessionId,
          })

          addLog(`  ✓ ${run.name} 실행 시작됨 (실시간 로그 연결됨)`)

        } catch (execError) {
          console.error(`Failed to execute evaluation ${run.id}:`, execError)
          addLog(`  ❌ ${run.name} 실행 실패`)
        }
      }

      addLog("✓ 모든 평가가 백그라운드에서 실행 중입니다")
      addLog("📊 실시간 로그를 확인하세요")

      toast.success("평가가 성공적으로 시작되었습니다")

      // Close SSE connections after 5 minutes (cleanup)
      setTimeout(() => {
        eventSources.forEach(es => es.close())
      }, 5 * 60 * 1000)

    } catch (error) {
      console.error("Failed to create evaluation:", error)
      setEvaluationLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ❌ 오류: 평가 생성 실패`])
      toast.error("평가 생성에 실패했습니다")
      setIsEvaluating(false)

      // Close all SSE connections on error
      eventSources.forEach(es => es.close())
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleReset = () => {
    setEvaluationName("")
    setDescription("")
    setSelectedModel("")
    setSelectedBaseDataset("")
    setSelectedAttackDataset("")
    setEvaluationLogs([])
    setIsEvaluating(false)
    setEvaluationCompleted(false)
    setShowResults(false)
    setEvaluationResults([])
    setCompletedEvaluationIds([])
  }

  const handleViewResults = async () => {
    if (completedEvaluationIds.length === 0) {
      toast.error("평가 결과를 찾을 수 없습니다")
      return
    }

    setLoadingResults(true)
    setShowResults(true)

    try {
      const results = []

      for (const evalId of completedEvaluationIds) {
        // Fetch evaluation run details
        const evalRun: any = await apiClient.getEvaluationRun(evalId)

        // Class metrics removed - not used
        // let classMetrics: any[] = []
        // try {
        //   const metricsResponse: any = await apiClient.getEvaluationClassMetrics(evalId)
        //   classMetrics = metricsResponse.items || []
        // } catch (e) {
        //   console.error("Failed to load class metrics:", e)
        // }

        results.push({
          ...evalRun,
          // classMetrics,  // Removed
        })
      }

      setEvaluationResults(results)

      // If we have both pre-attack and post-attack results, fetch comparison data
      const preRun = results.find((r: any) => r.phase === 'pre_attack')
      const postRun = results.find((r: any) => r.phase === 'post_attack')

      if (preRun && postRun) {
        try {
          const comparisonData: any = await apiClient.compareRobustness({
            clean_run_id: preRun.id,
            adv_run_id: postRun.id
          })
          // Store comparison data for rendering
          setComparisonData(comparisonData)
        } catch (e) {
          console.error("Failed to load comparison data:", e)
          // Continue without comparison data
        }
      }
    } catch (error) {
      console.error("Failed to load evaluation results:", error)
      toast.error("평가 결과를 불러오는데 실패했습니다")
    } finally {
      setLoadingResults(false)
    }
  }

  // Get base datasets (non-attack datasets)
  const baseDatasets = datasets.filter(d => !d.is_attack_dataset)
  const attackDatasets = datasets.filter(d => d.is_attack_dataset)

  // Get selected items for preview
  const selectedModelData = models.find(m => m.id === selectedModel)
  const selectedBaseDatasetData = datasets.find(d => d.id === selectedBaseDataset)
  const selectedAttackDatasetData = datasets.find(d => d.id === selectedAttackDataset)

  // Check if anything is selected
  const hasSelection = selectedModel || selectedBaseDataset || selectedAttackDataset

  return (
    <AdversarialToolLayout
      title="통합 평가 대시보드"
      description="AI 모델 신뢰성 평가 생성 및 결과 분석"
      icon={BarChart3}
      leftPanelWidth="lg"
      leftPanel={{
        title: "새 평가 생성",
        icon: Plus,
        children: (
          <div className="space-y-4">
            {/* Evaluation Name */}
            <div className="space-y-2">
              <Label htmlFor="eval-name" className="text-white">평가 이름 *</Label>
              <Input
                id="eval-name"
                value={evaluationName}
                onChange={(e) => setEvaluationName(e.target.value)}
                placeholder="예: YOLOv8 종합 신뢰성 평가"
                className="bg-slate-700/50 border-white/10 text-white"
              />
            </div>

            {/* Description */}
            <div className="space-y-2">
              <Label htmlFor="eval-description" className="text-white">설명</Label>
              <Textarea
                id="eval-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="평가에 대한 설명을 입력하세요"
                className="bg-slate-700/50 border-white/10 text-white min-h-[80px]"
              />
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label className="text-white">평가 모델 *</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                  <SelectValue placeholder="모델을 선택하세요" />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Dataset Selection with Tabs */}
            <div className="space-y-2">
              <Label className="text-white">평가 데이터셋 (선택)</Label>
              <Tabs value={datasetTab} onValueChange={(value) => setDatasetTab(value as "base" | "attack")} className="w-full">
                <TabsList className="grid w-full grid-cols-2 bg-slate-700/50">
                  <TabsTrigger value="base" className="data-[state=active]:bg-green-600">
                    기준 데이터셋
                  </TabsTrigger>
                  <TabsTrigger value="attack" className="data-[state=active]:bg-red-600">
                    공격 데이터셋
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="base" className="mt-3">
                  <Select value={selectedBaseDataset || undefined} onValueChange={setSelectedBaseDataset}>
                    <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                      <SelectValue placeholder="기준 데이터셋을 선택하세요 (선택 사항)" />
                    </SelectTrigger>
                    <SelectContent>
                      {baseDatasets.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name} ({dataset.image_count} 이미지)
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selectedBaseDataset && (
                    <div className="flex items-center justify-between mt-2">
                      <p className="text-xs text-green-400">
                        ✓ {baseDatasets.find(d => d.id === selectedBaseDataset)?.name} 선택됨
                      </p>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedBaseDataset("")}
                        className="text-xs text-slate-400 hover:text-white h-6 px-2"
                      >
                        선택 해제
                      </Button>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="attack" className="mt-3">
                  <Select value={selectedAttackDataset || undefined} onValueChange={setSelectedAttackDataset}>
                    <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                      <SelectValue placeholder="공격 데이터셋을 선택하세요 (선택 사항)" />
                    </SelectTrigger>
                    <SelectContent>
                      {attackDatasets.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name} {dataset.attack_type && `(${dataset.attack_type})`}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selectedAttackDataset && (
                    <div className="flex items-center justify-between mt-2">
                      <p className="text-xs text-red-400">
                        ✓ {attackDatasets.find(d => d.id === selectedAttackDataset)?.name} 선택됨
                      </p>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedAttackDataset("")}
                        className="text-xs text-slate-400 hover:text-white h-6 px-2"
                      >
                        선택 해제
                      </Button>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </div>

            {/* Evaluation Summary */}
            {(selectedBaseDataset || selectedAttackDataset) && (
              <div className="bg-green-900/20 border border-green-500/30 rounded-md p-3">
                <p className="text-xs text-green-300 font-semibold mb-2">평가 요약</p>
                <div className="text-xs text-slate-300 space-y-1">
                  {selectedBaseDataset && !selectedAttackDataset && (
                    <div>✓ 단순 성능 평가 (기준 데이터셋)</div>
                  )}
                  {!selectedBaseDataset && selectedAttackDataset && (
                    <div>✓ 단순 성능 평가 (공격 데이터셋)</div>
                  )}
                  {selectedBaseDataset && selectedAttackDataset && (
                    <>
                      <div>✓ 비교 평가 모드</div>
                      <div className="ml-3">- 기준 데이터셋: 1개</div>
                      <div className="ml-3">- 공격 데이터셋: 1개</div>
                      <div className="ml-3 text-green-400">→ 신뢰성 분석 포함</div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Info Box */}
            <div className="bg-blue-900/20 border border-blue-500/30 rounded-md p-3">
              <div className="text-xs text-blue-300 space-y-1">
                <div>• <strong>기준 데이터셋만</strong> 선택: 단순 성능(객체식별) 평가</div>
                <div>• <strong>공격 데이터셋만</strong> 선택: 단순 성능(객체식별) 평가</div>
                <div>• <strong>둘 다</strong> 선택: 비교 평가 및 신뢰성 분석 제공</div>
                <div className="mt-2 pt-2 border-t border-blue-500/30">평가 시작 후 진행 상황은 우측 패널에서 확인할 수 있습니다.</div>
              </div>
            </div>
          </div>
        )
      }}
      rightPanel={{
        title: isEvaluating ? (evaluationCompleted ? "평가 생성 완료" : "평가 진행 상황") : (hasSelection ? "선택 항목 미리보기" : "평가 안내"),
        icon: isEvaluating ? (evaluationCompleted ? CheckCircle2 : Loader2) : (hasSelection ? Info : Activity),
        children: isEvaluating ? (
          // State 3: Evaluation Running - Show Logs
          <div className="h-full flex flex-col p-6 space-y-4">
            <div className="flex items-center gap-3 pb-4 border-b border-white/10">
              {!evaluationCompleted ? (
                <div className="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <CheckCircle2 className="w-8 h-8 text-green-400" />
              )}
              <div>
                <h3 className="text-white font-semibold">
                  {evaluationCompleted ? "평가 생성 완료" : "평가 진행 중"}
                </h3>
                <p className="text-slate-400 text-sm">
                  {evaluationCompleted
                    ? "평가가 성공적으로 생성되었습니다"
                    : "평가 프로세스가 실행되고 있습니다..."}
                </p>
              </div>
            </div>

            {/* Logs */}
            <div className="flex-1 overflow-hidden">
              <div className="bg-slate-900/50 rounded-lg p-4 h-full overflow-y-auto font-mono text-xs">
                {evaluationLogs.length === 0 ? (
                  <p className="text-slate-500">로그 대기 중...</p>
                ) : (
                  evaluationLogs.map((log, index) => (
                    <div key={index} className="text-slate-300 py-1">
                      {log}
                    </div>
                  ))
                )}
              </div>
            </div>

            {evaluationCompleted ? (
              // Action Buttons after completion
              <div className="space-y-3">
                <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4">
                  <p className="text-green-300 text-sm flex items-center gap-2 mb-3">
                    <Info className="w-4 h-4" />
                    평가가 백그라운드에서 실행됩니다. 다음 작업을 선택하세요:
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-2">
                  <Button
                    onClick={handleViewResults}
                    disabled={loadingResults}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    {loadingResults ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        결과 로딩 중...
                      </>
                    ) : (
                      <>
                        <Eye className="w-4 h-4 mr-2" />
                        결과 보기
                      </>
                    )}
                  </Button>
                </div>

                {/* Robustness Comparison - Show if both pre and post attack results exist */}
                {showResults && comparisonData && (
                  <div className="mt-4">
                    <RobustnessComparison data={comparisonData} />
                  </div>
                )}

                {/* Evaluation Results */}
                {showResults && evaluationResults.length > 0 && !comparisonData && (
                  <div className="mt-4 space-y-4">
                    <div className="flex items-center gap-2 text-white font-semibold">
                      <BarChart3 className="w-5 h-5" />
                      <h3>평가 결과</h3>
                    </div>

                    {evaluationResults.map((result, idx) => (
                      <Card key={idx} className="bg-slate-800/50 border-white/10">
                        <CardContent className="p-4">
                          {/* Evaluation Info */}
                          <div className="flex items-center justify-between mb-4">
                            <div>
                              <h4 className="text-white font-semibold">{result.name}</h4>
                              <p className="text-sm text-slate-400">
                                {result.phase === 'pre_attack' ? '정상 데이터' : '공격 데이터'}
                              </p>
                            </div>
                            <div className={`px-3 py-1 rounded text-sm ${
                              result.status === 'completed'
                                ? 'bg-green-900/30 text-green-300'
                                : result.status === 'running'
                                ? 'bg-blue-900/30 text-blue-300'
                                : 'bg-yellow-900/30 text-yellow-300'
                            }`}>
                              {result.status === 'completed' ? '완료' :
                               result.status === 'running' ? '실행 중' :
                               result.status === 'failed' ? '실패' : result.status}
                            </div>
                          </div>

                          {/* Overall Metrics */}
                          {result.metrics_summary && (
                            <div className="mb-4">
                              <h5 className="text-slate-300 text-sm font-medium mb-2">전체 메트릭</h5>
                              <div className="grid grid-cols-3 gap-3">
                                <div className="bg-slate-900/50 rounded p-3">
                                  <p className="text-xs text-slate-400 mb-1">mAP@50</p>
                                  <p className="text-lg font-bold text-blue-400">
                                    {(result.metrics_summary.map50 * 100).toFixed(1)}%
                                  </p>
                                </div>
                                <div className="bg-slate-900/50 rounded p-3">
                                  <p className="text-xs text-slate-400 mb-1">Precision</p>
                                  <p className="text-lg font-bold text-green-400">
                                    {(result.metrics_summary.precision * 100).toFixed(1)}%
                                  </p>
                                </div>
                                <div className="bg-slate-900/50 rounded p-3">
                                  <p className="text-xs text-slate-400 mb-1">Recall</p>
                                  <p className="text-lg font-bold text-purple-400">
                                    {(result.metrics_summary.recall * 100).toFixed(1)}%
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Class Metrics */}
                          {result.classMetrics && result.classMetrics.length > 0 && (
                            <div>
                              <h5 className="text-slate-300 text-sm font-medium mb-2">
                                클래스별 메트릭 ({result.classMetrics.length}개 클래스)
                              </h5>
                              <div className="space-y-2 max-h-60 overflow-y-auto">
                                {result.classMetrics.map((cm: any, cmIdx: number) => (
                                  <div key={cmIdx} className="bg-slate-900/50 rounded p-3">
                                    <div className="flex items-center justify-between mb-2">
                                      <span className="text-white font-medium">{cm.class_name}</span>
                                    </div>
                                    <div className="grid grid-cols-3 gap-2 text-xs">
                                      <div>
                                        <span className="text-slate-400">AP@50:</span>
                                        <span className="text-blue-400 ml-1 font-semibold">
                                          {((cm.metrics?.ap || 0) * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                      <div>
                                        <span className="text-slate-400">Precision:</span>
                                        <span className="text-green-400 ml-1 font-semibold">
                                          {((cm.metrics?.precision || 0) * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                      <div>
                                        <span className="text-slate-400">Recall:</span>
                                        <span className="text-purple-400 ml-1 font-semibold">
                                          {((cm.metrics?.recall || 0) * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
                <p className="text-blue-300 text-sm flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  평가는 백그라운드에서 계속 실행됩니다. 결과는 "평가 기록 관리"에서 확인하세요.
                </p>
              </div>
            )}
          </div>
        ) : hasSelection ? (
          // State 2: Selection Mode - Show Preview
          <div className="h-full flex flex-col p-6 space-y-4 overflow-y-auto">
            {/* Selected Model */}
            {selectedModelData && (
              <Card className="bg-slate-800/50 border-white/10">
                <CardContent>
                  <div className="flex items-start gap-3">
                    <Brain className="w-8 h-8 text-blue-400 flex-shrink-0" />
                    <div className="flex-1">
                      <h4 className="text-white font-semibold mb-1">평가 모델</h4>
                      <p className="text-slate-300 text-sm mb-2">{selectedModelData.name}</p>
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <span className="px-2 py-1 bg-blue-900/30 rounded">
                          {selectedModelData.model_type}
                        </span>
                      </div>
                    </div>
                    <CheckCircle2 className="w-5 h-5 text-green-400" />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Selected Base Dataset */}
            {selectedBaseDatasetData && (
              <Card className="bg-slate-800/50 border-white/10">
                <CardContent>
                  <div className="flex items-start gap-3">
                    <ImageIcon className="w-8 h-8 text-green-400 flex-shrink-0" />
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-white font-semibold">기준 데이터셋</h4>
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                      </div>
                      <div className="text-slate-300 text-sm font-medium">{selectedBaseDatasetData.name}</div>

                      {/* Dataset Info */}
                      <div className="flex flex-wrap gap-2 text-xs">
                        {selectedBaseDatasetData.image_count !== undefined && (
                          <span className="px-2 py-1 bg-green-900/30 rounded text-green-300">
                            {selectedBaseDatasetData.image_count.toLocaleString()} 이미지
                          </span>
                        )}
                        {selectedBaseDatasetData.created_at && (
                          <span className="px-2 py-1 bg-slate-700/50 rounded text-slate-400">
                            {new Date(selectedBaseDatasetData.created_at).toLocaleDateString('ko-KR')}
                          </span>
                        )}
                      </div>

                      {/* Description */}
                      {selectedBaseDatasetData.description && (
                        <div className="text-xs text-slate-400 line-clamp-2">
                          {selectedBaseDatasetData.description}
                        </div>
                      )}

                      {/* Image Preview */}
                      {loadingBaseImages ? (
                        <div className="flex items-center justify-center gap-2 py-4">
                          <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                          <span className="text-xs text-slate-400">이미지 로딩 중...</span>
                        </div>
                      ) : baseDatasetImages.length > 0 ? (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <div className="text-xs text-slate-400">샘플 이미지</div>
                            <div className="text-xs text-slate-500">페이지 {currentImagePage + 1}</div>
                          </div>
                          <div className="grid grid-cols-3 gap-2">
                            {baseDatasetImages.slice(0, 6).map((image, idx) => {
                              const imageUrl = getImageUrlByStorageKey(image.storage_key);
                              console.log(`🖼️ Base image ${idx + 1} storage_key:`, image.storage_key);
                              console.log(`🖼️ Base image ${idx + 1} URL:`, imageUrl);

                              return (
                                <div key={idx} className="relative aspect-square rounded overflow-hidden bg-slate-900/50 group">
                                  <img
                                    src={imageUrl}
                                    alt={`Sample ${idx + 1}`}
                                    className="w-full h-full object-cover transition-transform group-hover:scale-110"
                                    onLoad={() => console.log(`✅ Image ${idx + 1} loaded successfully`)}
                                    onError={(e) => {
                                      console.error(`❌ Failed to load image ${idx + 1}:`, imageUrl);
                                      const target = e.target as HTMLImageElement;
                                      target.style.display = 'none';
                                      target.parentElement!.innerHTML = '<div class="w-full h-full flex items-center justify-center bg-slate-800"><svg class="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg></div>';
                                    }}
                                  />
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Selected Attack Dataset */}
            {selectedAttackDatasetData && (
              <Card className="bg-slate-800/50 border-white/10">
                <CardContent>
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-8 h-8 text-red-400 flex-shrink-0" />
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-white font-semibold">공격 데이터셋</h4>
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                      </div>
                      <div className="text-slate-300 text-sm font-medium">{selectedAttackDatasetData.name}</div>

                      {/* Dataset Info */}
                      <div className="flex flex-wrap gap-2 text-xs">
                        {selectedAttackDatasetData.attack_type && (
                          <span className="px-2 py-1 bg-red-900/30 rounded text-red-300 uppercase">
                            {selectedAttackDatasetData.attack_type}
                          </span>
                        )}
                        {selectedAttackDatasetData.target_class && (
                          <span className="px-2 py-1 bg-purple-900/30 rounded text-purple-300">
                            타겟: {selectedAttackDatasetData.target_class}
                          </span>
                        )}
                        {selectedAttackDatasetData.created_at && (
                          <span className="px-2 py-1 bg-slate-700/50 rounded text-slate-400">
                            {new Date(selectedAttackDatasetData.created_at).toLocaleDateString('ko-KR')}
                          </span>
                        )}
                      </div>

                      {/* Description */}
                      {selectedAttackDatasetData.description && (
                        <div className="text-xs text-slate-400 line-clamp-2">
                          {selectedAttackDatasetData.description}
                        </div>
                      )}

                      {/* Image Preview */}
                      {loadingAttackImages ? (
                        <div className="flex items-center justify-center gap-2 py-4">
                          <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                          <span className="text-xs text-slate-400">이미지 로딩 중...</span>
                        </div>
                      ) : attackDatasetImages.length > 0 ? (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <div className="text-xs text-slate-400">샘플 이미지</div>
                            <div className="text-xs text-slate-500">페이지 {currentImagePage + 1}</div>
                          </div>
                          <div className="grid grid-cols-3 gap-2">
                            {attackDatasetImages.slice(0, 6).map((image, idx) => {
                              const imageUrl = getImageUrlByStorageKey(image.storage_key);
                              console.log(`🔴 Attack image ${idx + 1} storage_key:`, image.storage_key);
                              console.log(`🔴 Attack image ${idx + 1} URL:`, imageUrl);

                              return (
                                <div key={idx} className="relative aspect-square rounded overflow-hidden bg-slate-900/50 border border-red-900/30 group">
                                  <img
                                    src={imageUrl}
                                    alt={`Attack Sample ${idx + 1}`}
                                    className="w-full h-full object-cover transition-transform group-hover:scale-110"
                                    onLoad={() => console.log(`✅ Attack image ${idx + 1} loaded successfully`)}
                                    onError={(e) => {
                                      console.error(`❌ Failed to load attack image ${idx + 1}:`, imageUrl);
                                      const target = e.target as HTMLImageElement;
                                      target.style.display = 'none';
                                      target.parentElement!.innerHTML = '<div class="w-full h-full flex items-center justify-center bg-slate-800"><svg class="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg></div>';
                                    }}
                                  />
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Image Navigation Buttons */}
            {(baseDatasetImages.length > 0 || attackDatasetImages.length > 0) && (
              <div className="flex items-center justify-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentImagePage(Math.max(0, currentImagePage - 1))}
                  disabled={currentImagePage === 0 || loadingBaseImages || loadingAttackImages}
                  className="flex items-center gap-2 bg-slate-800/50 border-white/10 text-white hover:bg-slate-700/50"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  이전 6개
                </Button>
                <span className="text-xs text-slate-400 min-w-[80px] text-center">
                  페이지 {currentImagePage + 1}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentImagePage(currentImagePage + 1)}
                  disabled={
                    (baseDatasetImages.length < 6 && attackDatasetImages.length < 6) ||
                    loadingBaseImages ||
                    loadingAttackImages
                  }
                  className="flex items-center gap-2 bg-slate-800/50 border-white/10 text-white hover:bg-slate-700/50"
                >
                  다음 6개
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Button>
              </div>
            )}

            {/* Evaluation Type Info */}
            <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
              <h4 className="text-blue-300 font-semibold mb-2 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                평가 유형
              </h4>
              <p className="text-slate-300 text-sm">
                {selectedBaseDatasetData && selectedAttackDatasetData ? (
                  <>
                    <span className="text-blue-400 font-semibold">비교 평가</span>
                    <br />
                    기준 데이터셋과 공격 데이터셋의 성능을 비교하여 모델의 적대적 공격 내성을 분석합니다.
                    신뢰성 점수와 등급이 자동으로 계산됩니다.
                  </>
                ) : (
                  <>
                    <span className="text-green-400 font-semibold">단순 성능 평가</span>
                    <br />
                    선택된 데이터셋에 대한 모델의 객체 탐지 성능(mAP, Precision, Recall 등)을 측정합니다.
                  </>
                )}
              </p>
            </div>

            {/* Ready to start message */}
            <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4">
              <p className="text-green-300 text-sm flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                모든 구성이 완료되었습니다. "평가 시작" 버튼을 클릭하여 평가를 시작하세요.
              </p>
            </div>
          </div>
        ) : (
          // State 1: Initial - Show Guide
          <div className="h-full flex flex-col justify-center items-center space-y-6 p-8">
            {/* Welcome Message */}
            <div className="text-center space-y-4 max-w-2xl">
              <Shield className="w-20 h-20 mx-auto text-blue-400 opacity-50" />
              <h3 className="text-2xl font-bold text-white">
                AI 모델 성능 및 신뢰성 평가
              </h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                왼쪽 패널에서 평가할 모델과 데이터셋을 선택하여 평가를 시작하세요.
                평가가 완료되면 결과는 "평가 기록 관리" 페이지에서 확인할 수 있습니다.
              </p>
            </div>

            {/* Info Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-4xl">
              <Card className="bg-slate-800/50 border-white/10">
                <CardContent className="pt-6 text-center">
                  <Database className="w-10 h-10 mx-auto mb-3 text-green-400" />
                  <h4 className="text-white font-semibold mb-2">단순 성능 평가</h4>
                  <p className="text-slate-400 text-xs">
                    기준 데이터셋 또는 공격 데이터셋 하나만 선택하여 객체 탐지 성능을 측정합니다.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-white/10">
                <CardContent className="pt-6 text-center">
                  <TrendingUp className="w-10 h-10 mx-auto mb-3 text-blue-400" />
                  <h4 className="text-white font-semibold mb-2">비교 평가</h4>
                  <p className="text-slate-400 text-xs">
                    기준 데이터셋과 공격 데이터셋을 함께 선택하여 적대적 공격에 대한 내성을 분석합니다.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-white/10">
                <CardContent className="pt-6 text-center">
                  <BarChart3 className="w-10 h-10 mx-auto mb-3 text-purple-400" />
                  <h4 className="text-white font-semibold mb-2">신뢰성 분석</h4>
                  <p className="text-slate-400 text-xs">
                    비교 평가 시 모델의 신뢰성 점수와 등급이 자동으로 계산됩니다.
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Quick Guide */}
            <div className="bg-slate-800/30 border border-white/10 rounded-lg p-6 w-full max-w-2xl">
              <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                <Eye className="w-5 h-5 text-blue-400" />
                평가 프로세스
              </h4>
              <ol className="space-y-2 text-sm text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-blue-400 font-semibold">1.</span>
                  <span><strong>모델 선택:</strong> 평가할 AI 모델을 선택합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-400 font-semibold">2.</span>
                  <span><strong>데이터셋 선택:</strong> 기준/공격 데이터셋을 선택합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-400 font-semibold">3.</span>
                  <span><strong>미리보기:</strong> 선택된 항목을 확인합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-400 font-semibold">4.</span>
                  <span><strong>평가 실행:</strong> 평가 시작 버튼을 클릭합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-400 font-semibold">5.</span>
                  <span><strong>결과 확인:</strong> "평가 기록 관리"에서 결과를 확인합니다</span>
                </li>
              </ol>
            </div>
          </div>
        )
      }}
      actionButtons={
        <Button
          className="w-full bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600"
          onClick={handleSubmit}
          disabled={isSubmitting || !evaluationName || !selectedModel || (!selectedBaseDataset && !selectedAttackDataset)}
        >
          {isSubmitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              평가 생성 중...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 mr-2" />
              평가 시작
            </>
          )}
        </Button>
      }
    />
  )
}
