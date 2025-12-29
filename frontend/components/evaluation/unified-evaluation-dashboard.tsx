"use client"

import { useState, useEffect, useMemo } from "react"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import { useOperation } from "@/contexts/OperationContext"
import { validateName, getNameValidationMessage } from "@/lib/validation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
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
  Brain,
  Target
} from "lucide-react"
import { useModels, useCreateEvaluation, useExecuteEvaluation } from "@/hooks/api"
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
  dataset_type?: "2d" | "3d" // Added to distinguish 2D vs 3D datasets
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
  // Use custom hooks
  const { data: modelsData } = useModels(0, 100)
  const createEvaluationMutation = useCreateEvaluation()
  const executeEvaluationMutation = useExecuteEvaluation()
  const { isOperationInProgress, setOperationInProgress } = useOperation()

  // Form states
  const [datasetDimension, setDatasetDimension] = useState<"2d" | "3d">("2d")
  const [evaluationName, setEvaluationName] = useState("")
  const [evaluationNameError, setEvaluationNameError] = useState<string>("")
  const [description, setDescription] = useState("")
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedBaseDataset, setSelectedBaseDataset] = useState("")
  const [selectedAttackDataset, setSelectedAttackDataset] = useState("")
  const [targetClass, setTargetClass] = useState<string>("")
  const [availableClasses, setAvailableClasses] = useState<Array<{value: string, label: string, count?: number}>>([])

  // Evaluation parameters
  const confThreshold = 0.25  // Fixed confidence threshold
  const iouThreshold = 0.5  // Fixed IOU threshold (all IoU thresholds 0.5~0.95 are calculated automatically)

  // Data states
  const models: Model[] = useMemo(() => modelsData || [], [modelsData])
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

  // Load datasets on mount
  useEffect(() => {
    loadDatasets()
  }, [])

  // Load base dataset images when selected or page changes
  useEffect(() => {
    if (selectedBaseDataset) {
      loadBaseDatasetImages(selectedBaseDataset, currentImagePage)
    } else {
      setBaseDatasetImages([])
    }
  }, [selectedBaseDataset, currentImagePage, datasetDimension])

  // Load attack dataset images when selected or page changes
  useEffect(() => {
    if (selectedAttackDataset) {
      loadAttackDatasetImages(selectedAttackDataset, currentImagePage)
    } else {
      setAttackDatasetImages([])
    }
  }, [selectedAttackDataset, currentImagePage, datasetDimension])

  // Auto-set target class from attack dataset when selected
  useEffect(() => {
    if (selectedAttackDataset && datasets.length > 0) {
      const attackDataset = datasets.find(d => d.id === selectedAttackDataset && d.is_attack_dataset)
      if (attackDataset && attackDataset.target_class) {
        console.log("Auto-setting target class from attack dataset:", attackDataset.target_class)
        setTargetClass(attackDataset.target_class)
      }
    } else if (!selectedAttackDataset && selectedBaseDataset) {
      // 공격 데이터셋을 해제했을 때, 기준 데이터셋이 있으면 클래스 목록 다시 로드
      loadDatasetClasses(selectedBaseDataset)
    }
  }, [selectedAttackDataset, datasets])

  // Reset page when dataset selection changes
  useEffect(() => {
    setCurrentImagePage(0)
  }, [selectedBaseDataset, selectedAttackDataset])

  // Reset selections when dataset dimension changes (2D/3D tab)
  useEffect(() => {
    setSelectedBaseDataset("")
    setSelectedAttackDataset("")
    setTargetClass("")
    setAvailableClasses([])
    setBaseDatasetImages([])
    setAttackDatasetImages([])
    setCurrentImagePage(0)
  }, [datasetDimension])

  // Load available classes when base dataset is selected
  useEffect(() => {
    if (selectedBaseDataset) {
      // 공격 데이터셋이 선택되어 있지 않은 경우에만 클래스 목록 로드
      if (!selectedAttackDataset) {
        loadDatasetClasses(selectedBaseDataset)
      } else {
        // 공격 데이터셋이 선택되어 있으면, 참고용으로만 클래스 목록 로드 (타겟 클래스는 변경하지 않음)
        loadDatasetClassesWithoutAutoSelect(selectedBaseDataset)
      }
    } else {
      // 기준 데이터셋을 해제할 때, 공격 데이터셋도 없으면 타겟 클래스 초기화
      if (!selectedAttackDataset) {
        setAvailableClasses([])
        setTargetClass("")
      }
    }
  }, [selectedBaseDataset, selectedAttackDataset, datasetDimension])

  // 한국어 클래스명 매핑 함수
  const getKoreanClassName = (englishName: string): string => {
    const classNameMap: Record<string, string> = {
      'person': '사람', 'truck': '트럭', 'bus': '버스', 'motorcycle': '오토바이',
      'bicycle': '자전거', 'car': '자동차', 'traffic light': '신호등', 'stop sign': '정지표지판',
      'horse': '말', 'cat': '고양이', 'dog': '개', 'umbrella': '우산',
      'bottle': '병', 'cup': '컵', 'chair': '의자', 'couch': '소파', 'tv': 'TV',
      'laptop': '노트북', 'cell phone': '휴대폰', 'book': '책', 'clock': '시계',
    }
    return classNameMap[englishName] || englishName
  }


  const loadDatasetClasses = async (datasetId: string) => {
    try {
      let summary: any

      if (datasetDimension === "3d") {
        // Use 3D annotation API
        const response = await fetch(`/api/annotations-3d/dataset/${datasetId}`)
        if (!response.ok) {
          console.warn("Failed to load 3D dataset annotations:", await response.text())
          setAvailableClasses([])
          setTargetClass('')
          return
        }
        summary = await response.json()
      } else {
        // Use 2D annotation API
        summary = await apiClient.getDatasetAnnotationsSummary(datasetId)
      }

      // class_distribution에서 클래스 목록 추출
      if (summary.class_distribution && Object.keys(summary.class_distribution).length > 0) {
        const classes = Object.entries(summary.class_distribution).map(([className, count]) => ({
          value: className,
          label: getKoreanClassName(className),
          count: count as number
        }))

        // 개수 기준 내림차순 정렬 (많이 탐지된 클래스부터)
        classes.sort((a, b) => (b.count || 0) - (a.count || 0))
        setAvailableClasses(classes)

        // 첫 번째 클래스를 기본 선택
        if (classes.length > 0 && !targetClass) {
          setTargetClass(classes[0].value)
        }
      } else {
        // 어노테이션이 없으면 빈 클래스 목록
        setAvailableClasses([])
        setTargetClass('')
      }
    } catch (error) {
      console.error("Failed to load dataset classes:", error)
      setAvailableClasses([])
      setTargetClass('')
    }
  }

  // 공격 데이터셋이 선택되어 있을 때 사용: 클래스 목록만 로드하고 자동 선택 안 함
  const loadDatasetClassesWithoutAutoSelect = async (datasetId: string) => {
    try {
      let summary: any

      if (datasetDimension === "3d") {
        // Use 3D annotation API
        const response = await fetch(`/api/annotations-3d/dataset/${datasetId}`)
        if (!response.ok) {
          console.warn("Failed to load 3D dataset annotations (without auto select):", await response.text())
          setAvailableClasses([])
          return
        }
        summary = await response.json()
      } else {
        // Use 2D annotation API
        summary = await apiClient.getDatasetAnnotationsSummary(datasetId)
      }

      if (summary.class_distribution && Object.keys(summary.class_distribution).length > 0) {
        const classes = Object.entries(summary.class_distribution).map(([className, count]) => ({
          value: className,
          label: getKoreanClassName(className),
          count: count as number
        }))

        // 개수 기준 내림차순 정렬
        classes.sort((a, b) => (b.count || 0) - (a.count || 0))
        setAvailableClasses(classes)
        // 타겟 클래스는 변경하지 않음 (공격 데이터셋의 타겟 클래스 유지)
      } else {
        setAvailableClasses([])
      }
    } catch (error) {
      console.error("Failed to load dataset classes:", error)
      setAvailableClasses([])
    }
  }

  const loadDatasets = async () => {
    try {
      // Load 2D datasets, 2D attack datasets, 3D datasets, and 3D attack datasets
      const [
        regularDatasetsResponse,
        attackDatasetsResponse,
        datasets3DResponse,
        attackDatasets3DResponse
      ]: any[] = await Promise.all([
        apiClient.getDatasets(),
        apiClient.listAttackDatasets(),
        fetch('/api/carla/datasets_3d?limit=1000&exclude_attack_output=true').then(r => r.json()).catch(() => []),
        fetch('/api/carla/attack_datasets_3d?limit=1000').then(r => r.json()).catch(() => [])
      ])

      // Add is_attack_dataset and dataset_type flags to 2D regular datasets
      const regularDatasets = (regularDatasetsResponse || []).map((ds: any) => ({
        ...ds,
        is_attack_dataset: false,
        dataset_type: "2d" as const
      }))

      // Add is_attack_dataset and dataset_type flags to 2D attack datasets
      const attackDatasets = (attackDatasetsResponse || []).map((ds: any) => ({
        ...ds,
        is_attack_dataset: true,
        dataset_type: "2d" as const
      }))

      // Add is_attack_dataset and dataset_type flags to 3D regular datasets
      const datasets3D = (datasets3DResponse || []).map((ds: any) => {
        // Remove any existing dataset_type to ensure our type is used
        const { dataset_type, ...rest } = ds
        return {
          ...rest,
          is_attack_dataset: false,
          dataset_type: "3d" as const
        }
      })

      // Add is_attack_dataset and dataset_type flags to 3D attack datasets
      const attackDatasets3D = (attackDatasets3DResponse || []).map((ds: any) => {
        // Remove any existing dataset_type to ensure our type is used
        const { dataset_type, ...rest } = ds
        return {
          ...rest,
          is_attack_dataset: true,
          dataset_type: "3d" as const
        }
      })

      // Merge all datasets
      const allDatasets = [...regularDatasets, ...attackDatasets, ...datasets3D, ...attackDatasets3D]
      console.log("📦 All datasets loaded:", {
        total: allDatasets.length,
        "2d_regular": regularDatasets.length,
        "2d_attack": attackDatasets.length,
        "3d_regular": datasets3D.length,
        "3d_attack": attackDatasets3D.length,
        sample_3d: datasets3D[0]
      })
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
      console.log("📸 Loading base dataset images for:", datasetId, "page:", page, "offset:", offset, "dimension:", datasetDimension)

      let response: any
      if (datasetDimension === "3d") {
        // Use 3D dataset API
        const res = await fetch(`/api/carla/datasets_3d/${datasetId}/images?offset=${offset}&limit=6`)
        response = await res.json()
      } else {
        // Use 2D dataset API
        response = await apiClient.getDatasetImages(datasetId, offset, 6)
      }

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
      console.log("🔴 Loading attack dataset images for:", datasetId, "page:", page, "offset:", offset, "dimension:", datasetDimension)
      // For attack datasets, check if there's an output dataset
      const attackDataset = datasets.find(d => d.id === datasetId && d.is_attack_dataset)
      console.log("🔴 Attack dataset found:", attackDataset)

      if (attackDataset) {
        // Check if there's an output_dataset_id in parameters
        const outputDatasetId = attackDataset.parameters?.output_dataset_id
        console.log("🔴 Output dataset ID:", outputDatasetId)

        let response: any
        if (outputDatasetId) {
          // Load images from the output dataset
          if (datasetDimension === "3d") {
            const res = await fetch(`/api/carla/datasets_3d/${outputDatasetId}/images?offset=${offset}&limit=6`)
            response = await res.json()
          } else {
            response = await apiClient.getDatasetImages(outputDatasetId, offset, 6)
          }
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
            if (datasetDimension === "3d") {
              const res = await fetch(`/api/carla/datasets_3d/${baseDatasetId}/images?offset=${offset}&limit=6`)
              response = await res.json()
            } else {
              response = await apiClient.getDatasetImages(baseDatasetId, offset, 6)
            }
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

    if (!validateName(evaluationName)) {
      toast.error(getNameValidationMessage("평가 이름"))
      return
    }

    if (!selectedBaseDataset && !selectedAttackDataset) {
      toast.error("최소 하나의 데이터셋을 선택해주세요 (기준 데이터셋 또는 공격 데이터셋)")
      return
    }

    // Check for duplicate evaluation names
    try {
      const existingEvaluations: any = await apiClient.listEvaluationRuns({ page: 1, page_size: 1000 })
      const evaluations = existingEvaluations?.items || existingEvaluations || []
      const duplicateNames = evaluations.filter((run: any) => {
        const runName = run.name.replace(/ \((Clean|Adversarial)\)$/, '')
        return evaluationName.trim() === runName.trim()
      })

      if (duplicateNames.length > 0) {
        toast.error(`평가 이름 "${evaluationName}"은(는) 이미 사용 중입니다. 다른 이름을 입력해주세요.`)
        return
      }
    } catch (error) {
      console.error("Failed to check duplicate evaluation names:", error)
      // Continue anyway if check fails
    }

    setIsSubmitting(true)
    setIsEvaluating(true)
    setEvaluationLogs([])
    setEvaluationCompleted(false)
    setShowResults(false)
    setEvaluationResults([])
    setCompletedEvaluationIds([])
    setComparisonData(null)

    // Set operation in progress to disable sidebar and control panel
    setOperationInProgress(true, '평가 수행')

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
        // Both selected: We'll create two runs (pre_attack and post_attack)
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

      // Note: When both base and attack datasets are selected,
      // create a single post_attack evaluation run.
      // The backend will automatically evaluate both datasets and create
      // two eval_dataset_results (BASE and ATTACK) within the same eval_run.
      if (selectedBaseDataset && selectedAttackDataset) {
        // Create single run that evaluates both base and attack datasets
        addLog("→ 비교 평가 생성 중 (기준 + 공격 데이터셋)...")

        // Prepare run data based on dimension
        const runData: any = {
          name: evaluationName,
          description: description,
          phase: "post_attack",  // This phase evaluates both base and attack
          dimension: datasetDimension,
          model_id: selectedModel,
          params: targetClass ? { target_class: targetClass } : undefined,
        }

        // Use correct fields based on dimension
        if (datasetDimension === "3d") {
          runData.base_dataset_3d_id = selectedBaseDataset
          runData.attack_dataset_3d_id = selectedAttackDataset
        } else {
          runData.base_dataset_id = selectedBaseDataset
          runData.attack_dataset_id = selectedAttackDataset
        }

        console.log("📤 Creating comparison evaluation run:", runData)
        const run: any = await apiClient.createEvaluationRun(runData)
        createdRuns.push(run)
        addLog(`✓ 비교 평가 생성됨 (ID: ${run.id})`)
        addLog(`  → 백엔드가 자동으로 기준 및 공격 데이터셋 모두 평가합니다`)
      } else {
        // Create single run
        const runData: any = {
          name: evaluationName,
          phase: phase,
          dimension: datasetDimension,
          model_id: selectedModel,
        }

        // Add optional fields only if they exist
        if (description) runData.description = description

        // Use correct dataset fields based on dimension
        if (datasetDimension === "3d") {
          if (evalBaseDatasetId) runData.base_dataset_3d_id = evalBaseDatasetId
          if (evalAttackDatasetId) runData.attack_dataset_3d_id = evalAttackDatasetId
        } else {
          if (evalBaseDatasetId) runData.base_dataset_id = evalBaseDatasetId
          if (evalAttackDatasetId) runData.attack_dataset_id = evalAttackDatasetId
        }

        if (targetClass) runData.params = { target_class: targetClass }

        console.log("📤 Creating single evaluation run:", runData)
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
          // Setup SSE connection for real-time logs via Next.js API proxy
          const eventSource = new EventSource(
            `/api/evaluation/runs/events/${sessionId}`
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
          console.log('[Execute Evaluation] Parameters:', {
            conf_threshold: confThreshold,
            iou_threshold: iouThreshold,
            target_class: targetClass || undefined,
            session_id: sessionId,
          })
          await apiClient.executeEvaluationRun(run.id, {
            conf_threshold: confThreshold,
            iou_threshold: iouThreshold,
            target_class: targetClass || undefined,
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

      // Reset evaluation name for next evaluation
      setEvaluationName("")

      // Re-enable sidebar and control panel after evaluation is submitted
      setOperationInProgress(false)

      // Close SSE connections after 5 minutes (cleanup)
      setTimeout(() => {
        eventSources.forEach(es => es.close())
      }, 5 * 60 * 1000)

    } catch (error) {
      console.error("Failed to create evaluation:", error)
      setEvaluationLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ❌ 오류: 평가 생성 실패`])
      toast.error("평가 생성에 실패했습니다")

      // Re-enable sidebar and control panel on error
      setOperationInProgress(false)

      // Keep isEvaluating true to show error logs, don't call setIsEvaluating(false)
      // This allows users to see the error logs before closing the modal

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
        let datasetResults = evalRun.dataset_results
        if (!datasetResults || datasetResults.length === 0) {
          try {
            const res = await fetch(`/api/evaluations/${evalId}/dataset-results`)
            if (res.ok) {
              datasetResults = await res.json()
              console.log("[UnifiedEvaluationDashboard] Loaded dataset_results via fallback endpoint:", datasetResults?.length)
            }
          } catch (error) {
            console.error("[UnifiedEvaluationDashboard] Failed to load dataset_results fallback:", error)
          }
        }

        // Load class metrics for target class display
        let classMetrics: any[] = []
        try {
          const metricsResponse: any = await apiClient.getEvaluationClassMetrics(evalId)
          classMetrics = metricsResponse || []
        } catch (e) {
          console.error("Failed to load class metrics:", e)
        }

        // Extract target class from params or description
        let targetClassName = ""
        if (evalRun.params?.target_class) {
          targetClassName = evalRun.params.target_class
        } else if (evalRun.description) {
          const match = evalRun.description.match(/타겟 클래스:\s*(.+?)(?:\n|$)/i)
          if (match) {
            targetClassName = match[1].trim()
          }
        }

        results.push({
          ...evalRun,
          dataset_results: datasetResults,
          classMetrics,
          targetClassName,
        })
      }

      setEvaluationResults(results)

      // If we have both pre_attack and post_attack results, fetch comparison data
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
  const baseDatasets = datasets.filter(d => {
    const match = !d.is_attack_dataset && d.dataset_type === datasetDimension
    return match
  })
  const attackDatasets = datasets.filter(d => {
    const match = d.is_attack_dataset && d.dataset_type === datasetDimension
    return match
  })

  // Debug log for dataset filtering
  useEffect(() => {
    console.log("🔍 Dataset filtering:", {
      currentDimension: datasetDimension,
      totalDatasets: datasets.length,
      baseDatasets: baseDatasets.length,
      attackDatasets: attackDatasets.length,
      sampleBase: baseDatasets[0],
      sampleAttack: attackDatasets[0]
    })
  }, [datasetDimension, datasets, baseDatasets, attackDatasets])

  // Get selected items for preview
  const selectedModelData = models.find(m => m.id === selectedModel)
  const selectedBaseDatasetData = datasets.find(d => d.id === selectedBaseDataset)
  const selectedAttackDatasetData = datasets.find(d => d.id === selectedAttackDataset)

  // Check if anything is selected
  const hasSelection = selectedModel || selectedBaseDataset || selectedAttackDataset

  // Calculate reliability based on performance drop
  const calculateReliability = (baseAP50: number, attackAP50: number) => {
    if (baseAP50 === 0) return { level: '측정 불가', dropRate: 0, color: 'text-muted' }

    const dropRate = ((baseAP50 - attackAP50) / baseAP50) * 100

    if (dropRate < 5) {
      return { level: '신뢰성 높음', dropRate, color: 'text-tertiary' }
    } else if (dropRate < 15) {
      return { level: '신뢰성 낮음', dropRate, color: 'text-tertiary' }
    } else {
      return { level: '신뢰성 매우 낮음', dropRate, color: 'text-error' }
    }
  }

  const sanitizeNumber = (value: any) => {
    const parsed = typeof value === "string" ? parseFloat(value) : value
    return Number.isFinite(parsed) ? parsed : 0
  }

  const hasCoreMetrics = (metrics: any) => {
    if (!metrics) return false
    return (
      metrics.map !== undefined ||
      metrics.map50 !== undefined ||
      metrics.ap !== undefined ||
      metrics.ap50 !== undefined ||
      metrics.precision !== undefined ||
      metrics.recall !== undefined ||
      metrics.f1 !== undefined
    )
  }

  const normalizeMetrics = (metrics: any) => {
    if (!metrics) return null
    return {
      ...metrics,
      map: sanitizeNumber(metrics.map),
      map50: sanitizeNumber(metrics.map50),
      map75: sanitizeNumber(metrics.map75),
      ap: sanitizeNumber(metrics.ap ?? metrics.map),
      ap50: sanitizeNumber(metrics.ap50 ?? metrics.map50),
      ap75: sanitizeNumber(metrics.ap75 ?? metrics.map75),
      precision: sanitizeNumber(metrics.precision),
      recall: sanitizeNumber(metrics.recall),
      f1: sanitizeNumber(metrics.f1),
    }
  }

  const getDatasetMetrics = (run: any, datasetType: "base" | "attack") => {
    if (!run) return null
    const datasetResult = run.dataset_results?.find((dr: any) => dr.dataset_type === datasetType)
    if (datasetResult?.metrics_summary) {
      return normalizeMetrics(datasetResult.metrics_summary)
    }
    if (datasetType === "base" && run.metrics_summary?.original_metrics) {
      return normalizeMetrics(run.metrics_summary.original_metrics)
    }
    if (hasCoreMetrics(run.metrics_summary)) {
      return normalizeMetrics(run.metrics_summary)
    }
    return null
  }

  return (
    <AdversarialToolLayout
      title="통합 평가 대시보드"
      description="AI 모델 신뢰성 평가 생성 및 결과 분석"
      icon={BarChart3}
      leftPanelWidth="lg"
      disabled={isOperationInProgress}
      leftPanel={{
        title: "새 평가 생성",
        icon: Plus,
        children: (
          <div className="space-y-4">
            {/* 2D/3D Selection */}
            <div className="space-y-2">
              <Label className="text-foreground">데이터셋 타입</Label>
              <Tabs value={datasetDimension} onValueChange={(value) => setDatasetDimension(value as "2d" | "3d")} className="w-full">
                <TabsList className="grid w-full grid-cols-2 bg-surface-container-low">
                  <TabsTrigger value="2d">
                    2D 이미지
                  </TabsTrigger>
                  <TabsTrigger value="3d">
                    3D 이미지
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            {/* Base Dataset Selection */}
            <div className="space-y-2">
              <Label className="text-foreground">기준 데이터셋 (선택 사항)</Label>
              <Select key={`base-${datasetDimension}`} value={selectedBaseDataset || undefined} onValueChange={setSelectedBaseDataset}>
                <SelectTrigger className="bg-surface-container border-outline-variant text-foreground">
                  <SelectValue placeholder="기준 데이터셋을 선택하세요" />
                </SelectTrigger>
                <SelectContent>
                  {baseDatasets.map((dataset) => (
                    <SelectItem key={dataset.id} value={dataset.id}>
                      {dataset.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Attack Dataset Selection */}
            <div className="space-y-2">
              <Label className="text-foreground">공격 데이터셋 (선택 사항)</Label>
              <Select key={`attack-${datasetDimension}`} value={selectedAttackDataset || undefined} onValueChange={setSelectedAttackDataset}>
                <SelectTrigger className="bg-surface-container border-outline-variant text-foreground">
                  <SelectValue placeholder="공격 데이터셋을 선택하세요" />
                </SelectTrigger>
                <SelectContent>
                  {attackDatasets.map((dataset) => (
                    <SelectItem key={dataset.id} value={dataset.id}>
                      {dataset.name} {dataset.attack_type && `(${dataset.attack_type})`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Evaluation Name */}
            <div className="space-y-2">
              <Label htmlFor="eval-name" className="text-foreground">평가 이름 *</Label>
              <Input
                id="eval-name"
                value={evaluationName}
                onChange={(e) => {
                  const value = e.target.value
                  // Only allow valid characters
                  if (value === '' || validateName(value)) {
                    setEvaluationName(value)
                    setEvaluationNameError("")
                  } else {
                    // Show error message but don't update the value
                    setEvaluationNameError(getNameValidationMessage("평가 이름"))
                  }
                }}
                placeholder="예: YOLOv8_reliability_test"
                className={`bg-surface-container border-outline-variant text-foreground ${evaluationNameError ? "border-error" : ""}`}
              />
              {evaluationNameError ? (
                <p className="text-xs text-error">{evaluationNameError}</p>
              ) : (
                <p className="text-xs text-muted">영문자, 숫자, - (대시), _ (언더스코어)만 사용 가능</p>
              )}
            </div>

            {/* Description */}
            <div className="space-y-2">
              <Label htmlFor="eval-description" className="text-foreground">설명</Label>
              <Textarea
                id="eval-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="평가에 대한 설명을 입력하세요"
                className="bg-surface-container border-outline-variant text-foreground min-h-[80px]"
              />
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label className="text-foreground">평가 모델 *</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger className="bg-surface-container border-outline-variant text-foreground">
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

            {/* Target Class Selection - Only show when attack dataset is NOT selected */}
            {!selectedAttackDataset ? (
              <div className="space-y-2">
                <Label className="text-foreground">
                  타겟 클래스
                </Label>
                <Select
                  key={`target-${datasetDimension}`}
                  value={targetClass || undefined}
                  onValueChange={(value) => setTargetClass(value || "")}
                  disabled={!selectedBaseDataset || availableClasses.length === 0}
                >
                  <SelectTrigger className="bg-surface-container border-outline-variant text-foreground">
                    <SelectValue placeholder={
                      !selectedBaseDataset
                        ? "데이터셋 선택시 활성화"
                        : availableClasses.length === 0
                          ? "탐지된 클래스가 없습니다"
                          : "전체 클래스 (타겟 클래스 미선택)"
                    } />
                  </SelectTrigger>
                  <SelectContent>
                    {availableClasses.map((classItem) => (
                      <SelectItem key={classItem.value} value={classItem.value}>
                        <div className="flex items-center justify-between gap-4 w-full">
                          <span>{classItem.label}</span>
                          {classItem.count && (
                            <span className="text-xs text-muted">({classItem.count}개)</span>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            ) : (
              /* Target Class Display for Attack Dataset */
              <div className="bg-primary-container border border-primary rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <Target className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-foreground mb-1">타겟 클래스</h4>
                    <p className="text-lg font-bold text-primary mb-2">
                      {getKoreanClassName(targetClass)}
                    </p>
                    <p className="text-xs text-muted">
                      공격 데이터셋의 타겟 클래스가 자동으로 적용되었습니다
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Evaluation Summary */}
            {(selectedBaseDataset || selectedAttackDataset) && (
              <div className="bg-tertiary-container border border-tertiary rounded-md p-3">
                <p className="text-xs text-tertiary font-semibold mb-2">평가 요약</p>
                <div className="text-xs text-muted space-y-1">
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
                      <div className="ml-3 text-tertiary">→ 신뢰성 분석 포함</div>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Info Box */}
            <div className="bg-primary-container border border-primary rounded-md p-3">
              <div className="text-xs text-primary space-y-1">
                <div>• <strong>기준 데이터셋</strong> 선택: 단순 성능(객체 식별) 평가</div>
                <div>• <strong>공격 데이터셋</strong> 선택: 비교 평가 및 신뢰성 분석 제공</div>
                <div className="mt-2 pt-2 border-t border-primary">평가 시작 후 진행 상황은 우측 패널에서 확인할 수 있습니다.</div>
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
            <div className="flex items-center gap-3 pb-4 border-b border-border">
              {!evaluationCompleted ? (
                <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              ) : (
                <CheckCircle2 className="w-8 h-8 text-tertiary" />
              )}
              <div>
                <h3 className="text-foreground font-semibold">
                  {evaluationCompleted ? "평가 생성 완료" : "평가 진행 중"}
                </h3>
                <p className="text-muted text-sm">
                  {evaluationCompleted
                    ? "평가가 성공적으로 생성되었습니다"
                    : "평가 프로세스가 실행되고 있습니다..."}
                </p>
              </div>
            </div>

            {/* Logs */}
            <div className="flex-1 overflow-hidden">
              <div className="bg-surface-container rounded-lg p-4 h-full overflow-y-auto font-mono text-xs">
                {evaluationLogs.length === 0 ? (
                  <p className="text-muted">로그 대기 중...</p>
                ) : (
                  evaluationLogs.map((log, index) => (
                    <div key={index} className="text-muted py-1">
                      {log}
                    </div>
                  ))
                )}
              </div>
            </div>

            {evaluationCompleted ? (
              // Action Buttons after completion
              <div className="space-y-3">
                <div className="bg-tertiary-container border border-tertiary rounded-lg p-4">
                  <p className="text-tertiary text-sm flex items-center gap-2 mb-3">
                    <Info className="w-4 h-4" />
                    평가가 백그라운드에서 실행됩니다. 다음 작업을 선택하세요:
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-2">
                  <Button
                    onClick={handleViewResults}
                    disabled={loadingResults}
                    className="w-full ds-btn-primary"
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

                {/* Evaluation Results - New Unified Display */}
                {showResults && evaluationResults.length > 0 && !comparisonData && (
                  <div className="mt-4 space-y-4">
                    <div className="flex items-center gap-2 text-foreground font-semibold">
                      <BarChart3 className="w-5 h-5" />
                      <h3>평가 결과</h3>
                    </div>

                    {(() => {
                      const preRun = evaluationResults.find((r: any) => r.phase === 'pre_attack')
                      const postRun = evaluationResults.find((r: any) => r.phase === 'post_attack')

                      const baseMetrics = preRun
                        ? getDatasetMetrics(preRun, "base")
                        : getDatasetMetrics(postRun, "base")
                      const attackMetrics = postRun ? getDatasetMetrics(postRun, "attack") : null

                      const baseAP50 = ((baseMetrics?.ap50 || baseMetrics?.map50 || 0) * 100)
                      const attackAP50 = ((attackMetrics?.ap50 || attackMetrics?.map50 || 0) * 100)

                      // Calculate reliability
                      const reliability = postRun ? calculateReliability(baseAP50, attackAP50) : null

                      // Use attack run if exists, otherwise use pre run
                      const displayRun = postRun || preRun
                      const displayMetrics = postRun ? attackMetrics : baseMetrics
                      const targetClassName = displayMetrics?.target_class || displayRun?.params?.target_class || null

                      if (!displayRun || !displayMetrics) return null

                      return (
                        <Card key="unified-result" className="bg-surface-container/50 border-border">
                          <CardContent className="p-6 space-y-6">
                            {/* Performance Metrics Cards */}
                            <div>
                              <h4 className="text-foreground font-semibold mb-4 flex items-center gap-2">
                                <Activity className="w-5 h-5" />
                                성능 지표
                              </h4>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <Card className="bg-surface-container border-border">
                                  <CardContent className="p-4">
                                    <p className="text-xs text-muted mb-1">F1 Score</p>
                                    <p className="text-2xl font-bold text-tertiary">
                                      {((displayMetrics.f1 || 0) * 100).toFixed(1)}%
                                    </p>
                                  </CardContent>
                                </Card>
                                <Card className="bg-surface-container border-border">
                                  <CardContent className="p-4">
                                    <p className="text-xs text-muted mb-1">
                                      {targetClassName
                                        ? `${getKoreanClassName(targetClassName)} AP@50`
                                        : 'mAP@50'}
                                    </p>
                                    <p className="text-2xl font-bold text-secondary">
                                      {((displayMetrics.ap50 || displayMetrics.map50 || 0) * 100).toFixed(1)}%
                                    </p>
                                  </CardContent>
                                </Card>
                                <Card className="bg-surface-container border-border">
                                  <CardContent className="p-4">
                                    <p className="text-xs text-muted mb-1">Precision</p>
                                    <p className="text-2xl font-bold text-tertiary">
                                      {((displayMetrics.precision || 0) * 100).toFixed(1)}%
                                    </p>
                                  </CardContent>
                                </Card>
                                <Card className="bg-surface-container border-border">
                                  <CardContent className="p-4">
                                    <p className="text-xs text-muted mb-1">Recall</p>
                                    <p className="text-2xl font-bold text-info">
                                      {((displayMetrics.recall || 0) * 100).toFixed(1)}%
                                    </p>
                                  </CardContent>
                                </Card>
                              </div>

                              {/* Target Class Info */}
                              {targetClassName && (
                                <div className="mt-4 bg-primary-container border border-primary rounded-lg p-3">
                                  <div className="flex items-start gap-2">
                                    <Info className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
                                    <div className="text-xs text-primary">
                                      <strong>AP (Average Precision)</strong>: IoU threshold 0.5~0.95 평균
                                      <br />
                                      타겟 클래스 "<strong>{getKoreanClassName(targetClassName)}</strong>"만 평가되었습니다
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* Reliability Display */}
                            {reliability && (
                              <div className="bg-background/30 border border-border rounded-lg p-6">
                                <h4 className="text-foreground font-semibold mb-4 flex items-center gap-2">
                                  <Shield className="w-5 h-5" />
                                  신뢰성 분석
                                </h4>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                  <div className="text-center">
                                    <p className="text-sm text-muted mb-2">정상 AP@50</p>
                                    <p className="text-3xl font-bold text-tertiary">
                                      {baseAP50.toFixed(1)}%
                                    </p>
                                  </div>
                                  <div className="text-center">
                                    <p className="text-sm text-muted mb-2">공격 AP@50</p>
                                    <p className="text-3xl font-bold text-error">
                                      {attackAP50.toFixed(1)}%
                                    </p>
                                  </div>
                                  <div className="text-center">
                                    <p className="text-sm text-muted mb-2">성능 하락율</p>
                                    <p className="text-3xl font-bold text-tertiary">
                                      {reliability.dropRate.toFixed(1)}%
                                    </p>
                                  </div>
                                </div>
                                <div className="mt-6 text-center">
                                  <div className={`inline-flex items-center gap-2 px-6 py-3 rounded-lg ${
                                    reliability.level === '신뢰성 높음'
                                      ? 'bg-tertiary-container border border-tertiary text-on-tertiary-container'
                                      : reliability.level === '신뢰성 낮음'
                                        ? 'bg-warning-container border border-warning text-on-warning-container'
                                        : 'bg-error-container border border-error text-on-error-container'
                                  }`}>
                                    <Shield className="w-6 h-6" />
                                    <span className={`text-2xl font-bold ${reliability.color}`}>
                                      {reliability.level}
                                    </span>
                                  </div>
                                  <p className="text-xs text-muted mt-3">
                                    {reliability.level === '신뢰성 높음' && '성능 하락율이 5% 미만으로 매우 안정적입니다'}
                                    {reliability.level === '신뢰성 낮음' && '성능 하락율이 5-15%로 추가 방어 기법이 필요합니다'}
                                    {reliability.level === '신뢰성 매우 낮음' && '성능 하락율이 15% 이상으로 강화된 방어 기법이 필요합니다'}
                                  </p>
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      )
                    })()}
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-primary-container border border-primary rounded-lg p-4">
                <p className="text-primary text-sm flex items-center gap-2">
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
              <Card className="bg-surface-container/50 border-border">
                <CardContent>
                  <div className="flex items-start gap-3">
                    <Brain className="w-8 h-8 text-primary flex-shrink-0" />
                    <div className="flex-1">
                      <h4 className="text-foreground font-semibold mb-1">평가 모델</h4>
                      <p className="text-muted text-sm mb-2">{selectedModelData.name}</p>
                      <div className="flex items-center gap-2 text-xs text-muted">
                        <span className="px-2 py-1 bg-primary-container rounded">
                          {selectedModelData.model_type}
                        </span>
                      </div>
                    </div>
                    <CheckCircle2 className="w-5 h-5 text-tertiary" />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Selected Base Dataset */}
            {selectedBaseDatasetData && (
              <Card className="bg-surface-container/50 border-border">
                <CardContent>
                  <div className="flex items-start gap-3">
                    <ImageIcon className="w-8 h-8 text-tertiary flex-shrink-0" />
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-foreground font-semibold">기준 데이터셋</h4>
                        <CheckCircle2 className="w-5 h-5 text-tertiary" />
                      </div>
                      <div className="text-muted text-sm font-medium">{selectedBaseDatasetData.name}</div>

                      {/* Dataset Info */}
                      <div className="flex flex-wrap gap-2 text-xs">
                        {selectedBaseDatasetData.image_count !== undefined && (
                          <span className="px-2 py-1 bg-tertiary-container rounded text-tertiary">
                            {selectedBaseDatasetData.image_count.toLocaleString()} 이미지
                          </span>
                        )}
                        {selectedBaseDatasetData.created_at && (
                          <span className="px-2 py-1 bg-surface-container-high rounded text-muted">
                            {new Date(selectedBaseDatasetData.created_at).toLocaleDateString('ko-KR')}
                          </span>
                        )}
                      </div>

                      {/* Description */}
                      {selectedBaseDatasetData.description && (
                        <div className="text-xs text-muted line-clamp-2">
                          {selectedBaseDatasetData.description}
                        </div>
                      )}

                      {/* Image Preview */}
                      {loadingBaseImages ? (
                        <div className="flex items-center justify-center gap-2 py-4">
                          <Loader2 className="w-4 h-4 animate-spin text-muted" />
                          <span className="text-xs text-muted">이미지 로딩 중...</span>
                        </div>
                      ) : baseDatasetImages.length > 0 ? (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <div className="text-xs text-muted">샘플 이미지</div>
                            <div className="text-xs text-muted">페이지 {currentImagePage + 1}</div>
                          </div>
                          <div className="grid grid-cols-3 gap-2">
                            {baseDatasetImages.slice(0, 6).map((image, idx) => {
                              const imageUrl = getImageUrlByStorageKey(image.storage_key);
                              console.log(`🖼️ Base image ${idx + 1} storage_key:`, image.storage_key);
                              console.log(`🖼️ Base image ${idx + 1} URL:`, imageUrl);

                              return (
                                <div key={idx} className="relative aspect-square rounded overflow-hidden bg-surface-container group">
                                  <img
                                    src={imageUrl}
                                    alt={`Sample ${idx + 1}`}
                                    className="w-full h-full object-cover transition-transform group-hover:scale-110"
                                    onLoad={() => console.log(`✅ Image ${idx + 1} loaded successfully`)}
                                    onError={(e) => {
                                      console.error(`❌ Failed to load image ${idx + 1}:`, imageUrl);
                                      const target = e.target as HTMLImageElement;
                                      target.style.display = 'none';
                                      target.parentElement!.innerHTML = '<div class="w-full h-full flex items-center justify-center bg-surface-container"><svg class="w-6 h-6 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg></div>';
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
              <Card className="bg-surface-container/50 border-border">
                <CardContent>
                  <div className="flex items-start gap-3">
                    <AlertCircle className="w-8 h-8 text-error flex-shrink-0" />
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="text-foreground font-semibold">공격 데이터셋</h4>
                        <CheckCircle2 className="w-5 h-5 text-tertiary" />
                      </div>
                      <div className="text-muted text-sm font-medium">{selectedAttackDatasetData.name}</div>

                      {/* Dataset Info */}
                      <div className="flex flex-wrap gap-2 text-xs">
                        {selectedAttackDatasetData.attack_type && (
                          <span className="px-2 py-1 bg-error-container rounded text-error uppercase">
                            {selectedAttackDatasetData.attack_type}
                          </span>
                        )}
                        {selectedAttackDatasetData.target_class && (
                          <span className="px-2 py-1 bg-secondary-container rounded text-secondary">
                            타겟: {selectedAttackDatasetData.target_class}
                          </span>
                        )}
                        {selectedAttackDatasetData.created_at && (
                          <span className="px-2 py-1 bg-surface-container-high rounded text-muted">
                            {new Date(selectedAttackDatasetData.created_at).toLocaleDateString('ko-KR')}
                          </span>
                        )}
                      </div>

                      {/* Description */}
                      {selectedAttackDatasetData.description && (
                        <div className="text-xs text-muted line-clamp-2">
                          {selectedAttackDatasetData.description}
                        </div>
                      )}

                      {/* Image Preview */}
                      {loadingAttackImages ? (
                        <div className="flex items-center justify-center gap-2 py-4">
                          <Loader2 className="w-4 h-4 animate-spin text-muted" />
                          <span className="text-xs text-muted">이미지 로딩 중...</span>
                        </div>
                      ) : attackDatasetImages.length > 0 ? (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <div className="text-xs text-muted">샘플 이미지</div>
                            <div className="text-xs text-muted">페이지 {currentImagePage + 1}</div>
                          </div>
                          <div className="grid grid-cols-3 gap-2">
                            {attackDatasetImages.slice(0, 6).map((image, idx) => {
                              const imageUrl = getImageUrlByStorageKey(image.storage_key);
                              console.log(`🔴 Attack image ${idx + 1} storage_key:`, image.storage_key);
                              console.log(`🔴 Attack image ${idx + 1} URL:`, imageUrl);

                              return (
                                <div key={idx} className="relative aspect-square rounded overflow-hidden bg-surface-container border border-red-900/30 group">
                                  <img
                                    src={imageUrl}
                                    alt={`Attack Sample ${idx + 1}`}
                                    className="w-full h-full object-cover transition-transform group-hover:scale-110"
                                    onLoad={() => console.log(`✅ Attack image ${idx + 1} loaded successfully`)}
                                    onError={(e) => {
                                      console.error(`❌ Failed to load attack image ${idx + 1}:`, imageUrl);
                                      const target = e.target as HTMLImageElement;
                                      target.style.display = 'none';
                                      target.parentElement!.innerHTML = '<div class="w-full h-full flex items-center justify-center bg-surface-container"><svg class="w-6 h-6 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg></div>';
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
                  className="flex items-center gap-2 ds-btn-outline"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  이전 6개
                </Button>
                <span className="text-xs text-muted min-w-[80px] text-center">
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
                  className="flex items-center gap-2 ds-btn-outline"
                >
                  다음 6개
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Button>
              </div>
            )}

            {/* Evaluation Type Info */}
            <div className="bg-primary-container border border-primary rounded-lg p-4">
              <h4 className="text-primary font-semibold mb-2 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                평가 유형
              </h4>
              <p className="text-muted text-sm">
                {selectedBaseDatasetData && selectedAttackDatasetData ? (
                  <>
                    <span className="text-primary font-semibold">비교 평가</span>
                    <br />
                    기준 데이터셋과 공격 데이터셋의 성능을 비교하여 모델의 적대적 공격 내성을 분석합니다.
                    신뢰성 점수와 등급이 자동으로 계산됩니다.
                  </>
                ) : (
                  <>
                    <span className="text-tertiary font-semibold">단순 성능 평가</span>
                    <br />
                    선택된 데이터셋에 대한 모델의 객체 식별 성능(mAP, Precision, Recall 등)을 측정합니다.
                  </>
                )}
              </p>
            </div>

            {/* Ready to start message */}
            <div className="bg-tertiary-container border border-tertiary rounded-lg p-4">
              <p className="text-tertiary text-sm flex items-center gap-2">
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
              <Shield className="w-20 h-20 mx-auto text-primary opacity-50" />
              <h3 className="text-2xl font-bold text-secondary">
                AI 모델 성능 및 신뢰성 평가
              </h3>
              <p className="text-muted text-sm leading-relaxed">
                왼쪽 패널에서 평가할 모델과 데이터셋을 선택하여 평가를 시작하세요.
                평가가 완료되면 결과는 "평가 기록 관리" 페이지에서 확인할 수 있습니다.
              </p>
            </div>

            {/* Info Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-4xl">
              <Card className="bg-surface-container/50 border-border">
                <CardContent className="pt-6 text-center">
                  <Database className="w-10 h-10 mx-auto mb-3 text-tertiary" />
                  <h4 className="text-foreground font-semibold mb-2">단순 성능 평가</h4>
                  <p className="text-muted text-xs">
                    기준 데이터셋 또는 공격 데이터셋 하나만 선택하여 객체 식별 성능을 측정합니다.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-surface-container/50 border-border">
                <CardContent className="pt-6 text-center">
                  <TrendingUp className="w-10 h-10 mx-auto mb-3 text-primary" />
                  <h4 className="text-foreground font-semibold mb-2">비교 평가</h4>
                  <p className="text-muted text-xs">
                    기준 데이터셋과 공격 데이터셋을 함께 선택하여 적대적 공격에 대한 내성을 분석합니다.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-surface-container/50 border-border">
                <CardContent className="pt-6 text-center">
                  <BarChart3 className="w-10 h-10 mx-auto mb-3 text-secondary" />
                  <h4 className="text-foreground font-semibold mb-2">신뢰성 분석</h4>
                  <p className="text-muted text-xs">
                    비교 평가 시 모델의 신뢰성 점수와 등급이 자동으로 계산됩니다.
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Quick Guide */}
            <div className="bg-surface-container/30 border border-border rounded-lg p-6 w-full max-w-2xl">
              <h4 className="text-foreground font-semibold mb-3 flex items-center gap-2">
                <Eye className="w-5 h-5 text-primary" />
                평가 프로세스
              </h4>
              <ol className="space-y-2 text-sm text-muted">
                <li className="flex items-start gap-2">
                  <span className="text-primary font-semibold">1.</span>
                  <span><strong>모델 선택:</strong> 평가할 AI 모델을 선택합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary font-semibold">2.</span>
                  <span><strong>데이터셋 선택:</strong> 기준/공격 데이터셋을 선택합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary font-semibold">3.</span>
                  <span><strong>미리보기:</strong> 선택된 항목을 확인합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary font-semibold">4.</span>
                  <span><strong>평가 실행:</strong> 평가 시작 버튼을 클릭합니다</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary font-semibold">5.</span>
                  <span><strong>결과 확인:</strong> "평가 기록 관리"에서 결과를 확인합니다</span>
                </li>
              </ol>
            </div>
          </div>
        )
      }}
      actionButtons={
        <Button
          className="w-full ds-btn-primary"
          onClick={handleSubmit}
          disabled={
            isSubmitting ||
            !evaluationName.trim() ||
            !selectedModel ||
            (!selectedBaseDataset && !selectedAttackDataset)
          }
        >
          {isSubmitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              평가 생성 중...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 mr-2" />
              신뢰도 평가
            </>
          )}
        </Button>
      }
    />
  )
}
