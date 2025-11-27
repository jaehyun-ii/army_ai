"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"
import { toast } from "sonner"
import {
  Play,
  Shield,
  Zap,
  Target,
  Settings,
  Database,
  Download,
  CheckCircle2,
  Clock,
  FileStack,
  FileText,
  Image as ImageIcon,
  Volume2,
  Grid3x3,
  Eye,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Info,
  Brain,
  AlertCircle,
  X
} from "lucide-react"
import { AdversarialToolLayout, StatCard } from "@/components/layouts/adversarial-tool-layout"
import {
  startAdversarialDataGeneration,
  connectAdversarialDataSSE,
  downloadAdversarialDataset,
  fetchBackendDatasets,
  // Removed fetchYoloModels import
  fetchAdversarialDatasetImages,
  previewPatchOnImage,
  type AdversarialDataLog,
  type BackendDataset,
  // Removed YoloModel type
} from "@/lib/adversarial-api"
import { ImageWithBBox } from "@/components/annotations/ImageWithBBox"

interface ModelInfo {
  id: string
  name: string
}

// FastAPI backend URL - Use NEXT_PUBLIC_BACKEND_API_URL environment variable
const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
const API_V1_BASE = `${BACKEND_API_URL}/api/v1`

// Validate name: only alphanumeric, dash, underscore
const validateName = (name: string): boolean => {
  const validPattern = /^[a-zA-Z0-9_-]+$/
  return validPattern.test(name)
}

interface Dataset {
  id: string
  name: string
  imageCount: number
  metadata?: {
    model?: string
    timestamp?: string
    detectedClasses?: Array<{
      class: string
      count: number
      avgConfidence: number
    }>
    totalDetections?: number
    imagesWithDetections?: number
  }
}

interface SavedPatch {
  id: string
  name: string
  method?: string
  targetClass: string
  createdAt: string
  effectiveness: number
  datasetIds: string[]
  previewUrl?: string
}

interface AttackProgress {
  total: number
  processed: number
  successful: number
  failed: number
  currentImage: string
  estimatedTime: string
}

export function AdversarialDataGeneratorUpdated() {
  const [attackType, setAttackType] = useState<"patch" | "noise">("patch")
  const [datasetName, setDatasetName] = useState("")
  const [selectedDataset, setSelectedDataset] = useState<string>("")
  const [availableDatasets, setAvailableDatasets] = useState<Dataset[]>([])
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [attackProgress, setAttackProgress] = useState<AttackProgress | null>(null)
  const [showResults, setShowResults] = useState(false)
  const [currentView, setCurrentView] = useState<'main' | 'results'>('main')
  const [generationResults, setGenerationResults] = useState<any[]>([])
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const [generationLogs, setGenerationLogs] = useState<AdversarialDataLog[]>([])
  const [generatedDatasetId, setGeneratedDatasetId] = useState<string | null>(null)
  const [attackDatasetId, setAttackDatasetId] = useState<string | null>(null)
  const [backendDatasets, setBackendDatasets] = useState<BackendDataset[]>([])
  const [yoloModels, setYoloModels] = useState<ModelInfo[]>([])
  const [noiseConfig, setNoiseConfig] = useState({
    targetModel: "",
    method: "pgd", // Default to PGD
    intensity: 8.0, // Match test file default (epsilon)
    epsilon: 8.0, // Match test file default
    alpha: 2.0, // Match test file default
    iterations: 10 // Match test file default
  })

  const [savedPatches, setSavedPatches] = useState<SavedPatch[]>([])
  const [selectedPatch, setSelectedPatch] = useState<string>("")
  const [targetClass, setTargetClass] = useState<string>("")
  const [patchScale, setPatchScale] = useState<number>(30)
  const [datasetImages, setDatasetImages] = useState<any[]>([])
  const [loadingImages, setLoadingImages] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalImages, setTotalImages] = useState(0)
  const [previewImageIndex, setPreviewImageIndex] = useState(0)
  const [previewFromServer, setPreviewFromServer] = useState<Map<string, string>>(new Map()) // imageId -> base64 data
  const [loadingServerPreview, setLoadingServerPreview] = useState(false)
  const [availableClasses, setAvailableClasses] = useState<Array<{value: string, label: string, count?: number}>>([])
  const [resultPreviewImages, setResultPreviewImages] = useState<any[]>([])
  const [loadingResultImages, setLoadingResultImages] = useState(false)
  const [resultCurrentPage, setResultCurrentPage] = useState(1)
  const [resultTotalImages, setResultTotalImages] = useState(0)
  const imagesPerPage = 20
  const resultImagesPerPage = 5

  useEffect(() => {
    loadDatasets()
    loadSavedPatches()
    loadYoloModels()
  }, [])

  // Reset selections when attack type changes
  useEffect(() => {
    if (attackType === 'patch') {
      setSelectedModel('') // Clear model selection for patch attacks
    } else if (attackType === 'noise') {
      setSelectedPatch('') // Clear patch selection for noise attacks
    }

    // Reset preview state
    setPreviewImageIndex(0)
    setPreviewFromServer(new Map())
  }, [attackType])

  // Auto-detect completion from generation logs
  useEffect(() => {
    if (!isGenerating || generationResults.length > 0) {
      return
    }

    // Check if any log indicates successful completion
    const hasCompletionLog = generationLogs.some(log =>
      log.type === 'complete' ||
      log.type === 'success' ||
      (log.message && (
        log.message.toLowerCase().includes('successfully generated') ||
        log.message.toLowerCase().includes('generation completed') ||
        log.message.toLowerCase().includes('completed successfully')
      ))
    )

    if (hasCompletionLog) {
      console.log('[useEffect] Detected completion from logs')
      console.log('[useEffect] Generation logs:', generationLogs)

      // Find the completion log to extract dataset info
      const completionLog = generationLogs.find(log =>
        log.type === 'complete' || log.type === 'success'
      )

      if (completionLog) {
        // Use output_dataset_id for images, not generation_id/attack_dataset_id
        const outputDatasetId = completionLog.output_dataset_id
        const attackDatasetId = completionLog.generation_id || completionLog.attack_dataset_id
        const totalProcessed = completionLog.processed || completionLog.processed_images || completionLog.total || 0
        const successful = completionLog.successful || completionLog.processed_images || totalProcessed
        const failed = completionLog.failed || completionLog.failed_images || 0

        // Use output_dataset_id for loading images, attack_dataset_id for download
        setGeneratedDatasetId(outputDatasetId || generatedDatasetId)
        setAttackDatasetId(attackDatasetId || null)
        setGenerationResults([{
          id: outputDatasetId || attackDatasetId || `attack_${Date.now()}`,
          name: datasetName,
          attackType: attackType,
          totalProcessed: totalProcessed,
          successful: successful,
          failed: failed,
          outputDir: completionLog.output_dir || completionLog.storage_path,
          createdAt: new Date().toISOString()
        }])

        toast.success("공격 데이터셋 생성이 완료되었습니다")
        setAttackProgress(prev => prev ? {
          ...prev,
          total: totalProcessed || prev.total,
          processed: totalProcessed || prev.total,
          successful: successful || prev.successful,
          failed: failed || prev.failed,
          currentImage: "완료",
          estimatedTime: "완료"
        } : null)
        setIsGenerating(false)
        setShowResults(true)
        setDatasetName("")
        setSelectedDataset("")
        setTargetClass("")
        setSelectedPatch("")
        setSelectedModel("")
        setAvailableClasses([])
        setDatasetImages([])
        setCurrentPage(1)
        setPreviewImageIndex(0)
        if (eventSource) {
          eventSource.close()
          setEventSource(null)
        }
      }
    }
  }, [generationLogs, generatedDatasetId, generationResults, isGenerating, datasetName, attackType, eventSource])

  // Load result images when showResults becomes true
  useEffect(() => {
    if (showResults && generatedDatasetId && resultPreviewImages.length === 0) {
      console.log('[useEffect] Loading result images for dataset:', generatedDatasetId)
      setResultCurrentPage(1)
      loadResultImages(1)
    }
  }, [showResults, generatedDatasetId])

  // Load result images when page changes
  useEffect(() => {
    if (showResults && generatedDatasetId) {
      console.log('[useEffect] Loading result images for page:', resultCurrentPage)
      loadResultImages(resultCurrentPage)
    }
  }, [resultCurrentPage])

  const getKoreanClassName = (englishName: string): string => {
    const classNameMap: Record<string, string> = {
      'person': '사람',
      'truck': '트럭',
      'bus': '버스',
      'motorcycle': '오토바이',
      'bicycle': '자전거',
      'car': '자동차',
      'traffic light': '신호등',
      'stop sign': '정지표지판',
      'horse': '말',
      'cat': '고양이',
      'dog': '개',
      'umbrella': '우산',
      'surfboard': '서핑보드',
      'bottle': '병',
      'cup': '컵',
      'pizza': '피자',
      'cake': '케이크',
      'chair': '의자',
      'couch': '소파',
      'dining table': '식탁',
      'tv': 'TV',
      'laptop': '노트북',
      'cell phone': '휴대폰',
      'microwave': '전자레인지',
      'oven': '오븐',
      'refrigerator': '냉장고',
      'sink': '싱크대',
      'clock': '시계',
      'tennis racket': '테니스 라켓',
      'sports ball': '스포츠볼',
      'baseball bat': '야구방망이',
      'baseball glove': '야구글러브',
      'skateboard': '스케이트보드',
      'wine glass': '와인잔',
      'fork': '포크',
      'knife': '나이프',
      'spoon': '숟가락',
      'bowl': '그릇',
      'banana': '바나나',
      'apple': '사과',
      'sandwich': '샌드위치',
      'orange': '오렌지',
      'broccoli': '브로콜리',
      'carrot': '당근',
      'hot dog': '핫도그',
      'donut': '도넛',
      'potted plant': '화분',
      'bed': '침대',
      'toilet': '변기',
      'remote': '리모컨',
      'keyboard': '키보드',
      'mouse': '마우스',
      'book': '책',
      'vase': '꽃병',
      'scissors': '가위',
      'teddy bear': '테디베어',
      'hair drier': '헤어드라이어',
      'toothbrush': '칫솔',
      'tie': '넥타이',
      'suitcase': '여행가방',
      'frisbee': '프리스비',
      'skis': '스키',
      'snowboard': '스노우보드',
      'kite': '연',
      'fire hydrant': '소화전',
      'bench': '벤치',
      'bird': '새',
      'sheep': '양',
      'cow': '소',
      'elephant': '코끼리',
      'bear': '곰',
      'zebra': '얼룩말',
      'giraffe': '기린'
    }
    return classNameMap[englishName] || englishName
  }

  useEffect(() => {
    if (selectedDataset) {
      setCurrentPage(1)
      setDatasetImages([])
      setTotalImages(0)

      // 백엔드에서 데이터셋의 어노테이션 요약 정보를 가져와 클래스 목록 추출
      const fetchAnnotationSummary = async () => {
        try {
          const response = await fetch(`/api/annotations/dataset/${selectedDataset}`)
          if (response.ok) {
            const summary = await response.json()

            // class_distribution에서 클래스 목록 추출
            if (summary.class_distribution && Object.keys(summary.class_distribution).length > 0) {
              const classes = Object.entries(summary.class_distribution).map(([className, count]) => ({
                value: className,
                label: getKoreanClassName(className),
                count: count as number
              }))

              // 개수 기준 내림차순 정렬 (많이 탐지된 클래스부터) 후 상위 5개만 선택
              classes.sort((a, b) => (b.count || 0) - (a.count || 0))
              setAvailableClasses(classes.slice(0, 5))

              // 첫 번째 클래스를 기본 선택
              if (classes.length > 0) {
                setTargetClass(classes[0].value)
              }
            } else {
              // 어노테이션이 없으면 빈 클래스 목록
              setAvailableClasses([])
              setTargetClass('')
            }
          } else {
            // API 호출 실패 시 빈 클래스 목록
            setAvailableClasses([])
            setTargetClass('')
          }
        } catch (error) {
          console.error('Failed to fetch annotation summary:', error)
          setAvailableClasses([])
          setTargetClass('')
        }
      }

      fetchAnnotationSummary()
    } else {
      setDatasetImages([])
      setTotalImages(0)
      setCurrentPage(1)
      setAvailableClasses([])
      setTargetClass('')
    }
  }, [selectedDataset])

  const loadDatasets = async () => {
    try {
      const response = await fetch('/api/datasets?type=2D_IMAGE')
      const data = await response.json()

      console.log('Raw API response:', data)

      const formattedDatasets = data.map((dataset: any) => {
        console.log('Processing dataset:', dataset.name)
        console.log('dataset.metadata:', dataset.metadata)

        return {
          id: dataset.id,
          name: dataset.name,
          imageCount: dataset.size || 0,
          metadata: dataset.metadata
        }
      })

      console.log('Formatted datasets:', formattedDatasets)
      setAvailableDatasets(formattedDatasets)
    } catch (error) {
      console.error('Failed to load datasets:', error)
      toast.error('데이터셋 목록을 불러오는데 실패했습니다')
    }
  }

  const loadSavedPatches = async () => {
    try {
      // Use Next.js proxy instead of direct backend call
      const response = await fetch(`/api/patches?limit=100`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()

      if (!Array.isArray(data)) {
        console.error('Invalid patches data:', data)
        throw new Error('Invalid response format')
      }

      const formattedPatches = data.map((patch: any) => {
        const bestScoreRaw = patch.patch_metadata?.best_score
        const bestScore = typeof bestScoreRaw === 'number'
          ? bestScoreRaw
          : typeof bestScoreRaw === 'string'
            ? Number(bestScoreRaw)
            : NaN

        // Use storage_key to generate preview URL via Next.js API proxy
        const previewUrl = patch.storage_key
          ? `/api/storage/${patch.storage_key}`
          : undefined

        const createdAtIso = patch.created_at ? new Date(patch.created_at).toISOString() : new Date().toISOString()

        console.log(`[Patch ${patch.id}] storage_key: ${patch.storage_key}, Preview URL:`, previewUrl)

        return {
          id: patch.id as string,
          name: patch.name ?? `Patch ${patch.id}`,
          method: patch.method as string | undefined,
          targetClass: patch.target_class ?? 'unknown',
          createdAt: createdAtIso,
          effectiveness: Number.isFinite(bestScore) ? bestScore * 100 : 0,
          datasetIds: patch.source_dataset_id ? [patch.source_dataset_id] : [],
          previewUrl: previewUrl
        } satisfies SavedPatch
      }).sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())

      setSavedPatches(formattedPatches)

      if (formattedPatches.length > 0) {
        toast.success(`${formattedPatches.length}개의 패치를 불러왔습니다`)
      }
    } catch (error) {
      console.error('Failed to load saved patches:', error)
      toast.error('저장된 패치 목록을 불러오는데 실패했습니다')
    }
  }

  const loadYoloModels = async () => {
    try {
      const response = await fetch('/api/models') // Use the consistent API endpoint
      const data = await response.json()

      const models = Array.isArray(data) ? data.map((model: any) => ({
        id: model.id,
        name: model.name || 'Unknown Model'
      })) : []
      setYoloModels(models)
      console.log('Loaded models:', models)
    } catch (error) {
      console.error('Failed to load models:', error)
      toast.error('모델 목록을 불러오는데 실패했습니다')
    }
  }

  const noiseMethods = [
    { value: "pgd", label: "PGD (Projected Gradient Descent)", desc: "반복적 그래디언트 기반 적대적 공격" },
    { value: "fgsm", label: "FGSM (Fast Gradient Sign Method)", desc: "빠른 그래디언트 기반 적대적 공격" },
    { value: "universal_noise", label: "Universal Noise (범용 노이즈)", desc: "모든 이미지에 적용 가능한 범용 적대적 노이즈 (Pseudo-GT, Object Masking)" },
    { value: "noise_osfd", label: "Noise OSFD (특징 왜곡)", desc: "특징 레벨 왜곡을 통한 적대적 노이즈 (RRB Augmentation)" }
  ]

  const targetClasses = [
    { value: "none", label: "타겟 없음 (무작위)" },
    { value: "person", label: "사람" },
    { value: "car", label: "자동차" },
    { value: "truck", label: "트럭" },
    { value: "bus", label: "버스" },
    { value: "motorcycle", label: "오토바이" }
  ]


  useEffect(() => {
    if (selectedDataset) {
      loadDatasetImages(selectedDataset, currentPage)
    }
  }, [currentPage])

  useEffect(() => {
    if (selectedDataset) {
      setCurrentPage(1)
      loadDatasetImages(selectedDataset, 1)
    }
  }, [selectedDataset])


  const loadDatasetImages = async (datasetId: string, page: number = 1) => {
    setLoadingImages(true)

    try {
      const offset = (page - 1) * imagesPerPage
      const response = await fetch(`/api/datasets/${datasetId}/images?limit=${imagesPerPage}&offset=${offset}`, {
        cache: "no-store"
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      if (data.images && Array.isArray(data.images)) {
        setDatasetImages(data.images)
        if (data.total !== undefined) {
          setTotalImages(data.total)
        } else {
          const dataset = availableDatasets.find(d => d.id === datasetId)
          setTotalImages(dataset?.imageCount || 0)
        }
      } else {
        console.warn('No images found in response')
        setDatasetImages([])
        setTotalImages(0)
      }
    } catch (error) {
      console.error('Failed to load dataset images:', error)
      toast.error('이미지를 불러오는데 실패했습니다')
      setDatasetImages([])
      setTotalImages(0)
    } finally {
      setLoadingImages(false)
    }
  }

  const handleGenerateAttack = async () => {
    console.log('handleGenerateAttack called')
    console.log('datasetName:', datasetName)
    console.log('selectedDataset:', selectedDataset)
    console.log('selectedModel:', selectedModel)
    console.log('attackType:', attackType)
    console.log('selectedPatch:', selectedPatch)

    if (!datasetName.trim()) {
      toast.error("데이터셋 이름을 입력해주세요")
      return
    }

    if (!validateName(datasetName)) {
      toast.error("데이터셋 이름은 영문자, 숫자, - (대시), _ (언더스코어)만 사용할 수 있습니다")
      return
    }

    if (!selectedDataset) {
      toast.error("데이터셋을 선택해주세요")
      return
    }

    if (attackType === "patch" && !selectedPatch) {
      toast.error("적용할 패치를 선택해주세요")
      return
    }

    if (attackType === "noise" && !selectedModel) {
      toast.error("AI 모델을 선택해주세요")
      return
    }

    if (attackType === "patch" && !targetClass) {
      toast.error("타겟 클래스를 선택해주세요")
      return
    }

    try {
      console.log('Starting generation...')
      setIsGenerating(true)
      setShowResults(false)
      setGenerationLogs([])

      const dataset = availableDatasets.find(d => d.id === selectedDataset)
      const totalImages = dataset?.imageCount || 0

      setAttackProgress({
        total: totalImages,
        processed: 0,
        successful: 0,
        failed: 0,
        currentImage: "초기화 중...",
        estimatedTime: "계산 중..."
      })

      // STEP 1: Generate session ID FIRST
      const sessionId = `attack_${Date.now()}`
      console.log('[AttackGen] Generated session_id:', sessionId)

      // STEP 2: Connect to SSE endpoint FIRST (before starting generation)
      console.log('[AttackGen] Connecting to SSE endpoint BEFORE starting generation')
      const sse = connectAdversarialDataSSE(sessionId, attackType, {
        onOpen: () => {
          console.log('[SSE] Connection opened, ready to receive events')
        },
        onMessage: (log) => {
          console.log('[SSE] Received log:', log)
          setGenerationLogs(prev => [...prev, log])

          if (log.type === 'status' || log.type === 'info') {
            setAttackProgress(prev => prev ? {
              ...prev,
              currentImage: log.message || "처리 중..."
            } : null)
          }

          if (log.type === 'progress' || (log.processed !== undefined && log.total !== undefined)) {
            setAttackProgress(prev => prev ? {
              ...prev,
              total: log.total || prev.total,
              processed: log.processed || 0,
              successful: log.successful || 0,
              failed: log.failed || log.failed_images || 0,
              currentImage: log.message || `${log.processed}/${log.total} 처리 중...`,
              estimatedTime: log.total && log.processed
                ? `${Math.max(0, Math.floor((log.total - log.processed) / 2))}초 남음`
                : "계산 중..."
            } : null)
          }

          if (log.type === 'success') {
            // Don't change isGenerating - keep showing logs
            setAttackProgress(prev => prev ? {
              ...prev,
              processed: log.processed || log.processed_images || prev.total,
              successful: log.successful || prev.successful,
              failed: log.failed || log.failed_images || prev.failed,
              currentImage: "완료",
              estimatedTime: "완료"
            } : null)

            console.log('[SSE] Success message received - useEffect will handle result generation')
            // Note: useEffect will handle result generation to avoid closure issues
          }

          if (log.type === 'complete') {
            // Don't change isGenerating - keep showing logs until user clicks button
            setAttackProgress(prev => prev ? {
              ...prev,
              processed: log.processed_images || prev.total,
              successful: log.processed_images || prev.successful,
              currentImage: "완료",
              estimatedTime: "완료"
            } : null)

            console.log('[SSE] Complete message received - useEffect will handle result generation')
            // Note: useEffect will handle result generation to avoid closure issues
          }

          if (log.type === 'error') {
            // Don't change isGenerating - keep showing logs
            toast.error(log.message || "오류가 발생했습니다")
            // 모달을 닫지 않고 유지하여 사용자가 오류 로그를 확인할 수 있도록 함
          }

          if (log.type === 'warning') {
            toast.warning(log.message)
          }
        },
        onError: (error) => {
          console.error('[SSE] Error:', error)
          // Don't automatically stop generating - let the user see the logs and decide
          console.log('[SSE] 연결이 종료되었습니다. 로그를 확인하세요.')
        },
        onClose: () => {
          console.log('[SSE] 연결 종료 - 사용자가 로그를 확인한 후 결과 보기 버튼을 클릭할 수 있습니다')
        }
      })

      setEventSource(sse)

      // STEP 3: Wait a bit to ensure SSE connection is established
      await new Promise(resolve => setTimeout(resolve, 500))
      console.log('[AttackGen] SSE connection established, now starting generation')

      // STEP 4: Now start generation with the same session_id
      const config = {
        dataset_name: datasetName,
        source_dataset_id: selectedDataset,
        training_id: attackType === "patch" ? selectedPatch : undefined,
        attack_type: attackType,
        patch_scale: attackType === "patch" ? (patchScale / 100) : undefined,  // Convert percentage to ratio
        noise_method: attackType === "noise" ? noiseConfig.method as ("pgd" | "fgsm" | "universal_noise" | "noise_osfd") : undefined,
        noise_epsilon: attackType === "noise" ? noiseConfig.epsilon : undefined,
        noise_alpha: attackType === "noise" ? noiseConfig.alpha : undefined,
        noise_iterations: attackType === "noise" ? noiseConfig.iterations : undefined,
        target_class: targetClass || undefined,  // ✅ Send target_class for both patch and noise attacks
        model_id: attackType === "noise" ? selectedModel : undefined,  // Only send model_id for noise attacks
        session_id: sessionId  // Pass the same session_id
      }

      console.log('[AttackGen] Starting generation with session_id:', sessionId)
      console.log('[AttackGen] Generation config:', config)

      const response = await startAdversarialDataGeneration(config)
      console.log('[AttackGen] Generation started:', response)
      const newDatasetId = response.output_dataset_id || response.generation_id || null
      if (newDatasetId) {
        setGeneratedDatasetId(newDatasetId)
      }
      toast.success("공격 데이터셋 생성이 시작되었습니다")

    } catch (error) {
      console.error('Failed to start adversarial data generation:', error)

      // Extract user-friendly error message
      let errorMessage = "적대적 데이터 생성 시작 실패"
      if (error instanceof Error) {
        // Check if error message contains user-friendly Korean message from backend
        if (error.message.includes("이미 존재하는 데이터셋")) {
          errorMessage = error.message
        } else if (error.message.includes("리소스를 찾을 수 없습니다")) {
          errorMessage = error.message
        } else if (error.message.includes("잘못된 입력")) {
          errorMessage = error.message
        } else if (error.message.includes("데이터베이스")) {
          errorMessage = error.message
        } else {
          errorMessage = `데이터 생성 시작 실패: ${error.message}`
        }
      }

      toast.error(errorMessage)

      // Keep isGenerating true to show error in logs
      setGenerationLogs(prev => [...prev, {
        type: 'error',
        message: errorMessage
      }])
    }
  }

  // Left Panel: Attack Configuration
  const leftPanelContent = (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>공격 데이터셋 이름</Label>
        <Input
          value={datasetName}
          onChange={(e) => {
            const value = e.target.value
            if (value === '' || validateName(value)) {
              setDatasetName(value)
            }
          }}
          placeholder="공격 데이터셋 이름 입력"
          disabled={isGenerating}
        />
        <p className="text-xs text-slate-400">영문자, 숫자, - (대시), _ (언더스코어)만 사용 가능</p>
      </div>

      <div className="space-y-2">
        <Label>데이터셋 선택</Label>
        <Select value={selectedDataset} onValueChange={setSelectedDataset} disabled={isGenerating}>
          <SelectTrigger>
            <SelectValue placeholder="데이터셋 선택" />
          </SelectTrigger>
          <SelectContent>
            {availableDatasets.map(dataset => (
              <SelectItem key={dataset.id} value={dataset.id}>
                <div className="flex items-center justify-between w-full">
                  <span>{dataset.name}</span>
                  <Badge className="ml-2 text-xs">{dataset.imageCount}개</Badge>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {selectedDataset && (
          <div className="text-xs text-slate-400">
            선택됨: {availableDatasets.find(d => d.id === selectedDataset)?.imageCount || 0}개 이미지
          </div>
        )}
      </div>

      <div className="space-y-2">
        <Label>타겟 클래스</Label>
        <Select
          value={targetClass}
          onValueChange={setTargetClass}
          disabled={isGenerating || !selectedDataset || availableClasses.length === 0}
        >
          <SelectTrigger>
            <SelectValue placeholder={
              !selectedDataset
                ? "데이터셋 선택시 활성화"
                : availableClasses.length === 0
                  ? "탐지된 클래스가 없습니다"
                  : "클래스 선택"
            } />
          </SelectTrigger>
          <SelectContent>
            {availableClasses.map(cls => (
              <SelectItem key={cls.value} value={cls.value}>
                <div className="flex items-center justify-between w-full">
                  <span>{cls.label}</span>
                  {cls.count && (
                    <Badge variant="outline" className="ml-2 text-xs">
                      {cls.count}개 탐지
                    </Badge>
                  )}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Tabs value={attackType} onValueChange={(v) => { if (isGenerating) return; setAttackType(v as "patch" | "noise") }}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="patch">
            <Grid3x3 className="w-4 h-4 mr-2" />
            2D 패치
          </TabsTrigger>
          <TabsTrigger value="noise">
            <Volume2 className="w-4 h-4 mr-2" />
            노이즈
          </TabsTrigger>
        </TabsList>

        <TabsContent value="patch" className="space-y-4 mt-4">



          <div className="space-y-2">
            <Label>생성된 패치 선택</Label>
            <div className="flex-1 min-h-0 w-full rounded-md border border-slate-700 p-2 overflow-y-auto max-h-[300px]">
              <div className="space-y-2">
                {savedPatches.filter(patch => !targetClass || targetClass === "" || patch.targetClass === targetClass).map(patch => (
                  <div
                    key={patch.id}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      selectedPatch === patch.id
                        ? 'bg-primary/10 border-primary'
                        : 'bg-slate-900/50 border-white/10 hover:border-white/20'
                    }`}
                    onClick={() => {
                      if (isGenerating) return
                      setSelectedPatch(patch.id)
                    }}
                  >
                    <div className="flex items-start gap-2">
                      {patch.previewUrl && (
                        <div className="w-12 h-12 rounded overflow-hidden bg-slate-800 flex-shrink-0">
                          <img
                            src={patch.previewUrl}
                            alt={patch.name}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium text-white text-sm truncate">{patch.name}</h4>
                        <div className="flex gap-1 mt-1">
                          <Badge variant="outline" className="text-xs">
                            {getKoreanClassName(patch.targetClass)}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <Label>패치 크기 비율</Label>
            <Select value={patchScale.toString()} onValueChange={(value) => setPatchScale(Number(value))} disabled={isGenerating}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="30">30%</SelectItem>
                <SelectItem value="40">40%</SelectItem>
                <SelectItem value="50">50%</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </TabsContent>

        <TabsContent value="noise" className="space-y-4 mt-4">
          <div className="space-y-2">
            <Label>대상 AI 모델</Label>
            <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isGenerating}>
              <SelectTrigger>
                <SelectValue placeholder="모델 선택" />
              </SelectTrigger>
              <SelectContent>
                {yoloModels.length > 0 ? (
                  yoloModels.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))
                ) : (
                  <SelectItem value="no-model" disabled>등록된 모델이 없습니다</SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>노이즈 방법</Label>
            <Select
              value={noiseConfig.method}
              onValueChange={(v) => setNoiseConfig(prev => ({ ...prev, method: v }))}
              disabled={isGenerating}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {noiseMethods.map(method => (
                  <SelectItem key={method.value} value={method.value}>
                    <div className="flex flex-col items-start">
                      <span className="font-medium">{method.label}</span>
                      <span className="text-xs text-slate-400">{method.desc}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Epsilon (노이즈 강도)</Label>
            <div className="flex items-center gap-4">
              <Slider
                value={[noiseConfig.epsilon]}
                onValueChange={(values) => setNoiseConfig(prev => ({ ...prev, epsilon: values[0], intensity: values[0] }))}
                min={1}
                max={20}
                step={0.5}
                className="flex-1"
                disabled={isGenerating}
              />
              <span className="text-sm text-white min-w-[60px] text-right">
                {noiseConfig.epsilon.toFixed(1)}
              </span>
            </div>
          </div>

          {(noiseConfig.method === "pgd" || noiseConfig.method === "universal_noise" || noiseConfig.method === "noise_osfd") && (
            <>
              <div className="space-y-2">
                <Label>Alpha (스텝 크기)</Label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[noiseConfig.alpha]}
                    onValueChange={(values) => setNoiseConfig(prev => ({ ...prev, alpha: values[0] }))}
                    min={0.1}
                    max={5}
                    step={0.1}
                    className="flex-1"
                    disabled={isGenerating}
                  />
                  <span className="text-sm text-white min-w-[60px] text-right">
                    {noiseConfig.alpha.toFixed(1)}
                  </span>
                </div>
              </div>

              <div className="space-y-2">
                <Label>반복 횟수 (Iterations)</Label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[noiseConfig.iterations]}
                    onValueChange={(values) => setNoiseConfig(prev => ({ ...prev, iterations: values[0] }))}
                    min={1}
                    max={noiseConfig.method === "universal_noise" ? 100 : 50}
                    step={1}
                    className="flex-1"
                    disabled={isGenerating}
                  />
                  <span className="text-sm text-white min-w-[60px] text-right">
                    {noiseConfig.iterations}
                  </span>
                </div>
                {noiseConfig.method === "universal_noise" && (
                  <p className="text-xs text-slate-400">범용 노이즈는 50-100회 반복 권장</p>
                )}
                {noiseConfig.method === "noise_osfd" && (
                  <p className="text-xs text-slate-400">OSFD는 30회 이상 반복 권장</p>
                )}
              </div>
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )

  // Check if anything is selected
  const hasSelection = datasetName || selectedDataset || targetClass || selectedPatch || selectedModel
  const selectedDatasetData = availableDatasets.find(d => d.id === selectedDataset)
  const selectedPatchData = savedPatches.find(p => p.id === selectedPatch)
  const selectedModelData = yoloModels.find(m => m.id === selectedModel)

  // Right Panel: Four States - Initial Guide, Selection Preview, Generation Progress, Results
  const rightPanelContent = showResults && generationResults.length > 0 ? (
    // State 4: Results - Show Generated Dataset Info (takes priority)
    <div className="h-full flex flex-col p-6 space-y-4 overflow-hidden">


      {generationResults[0] && (
        <Card className="bg-slate-800/50 border-white/10 flex-shrink-0">
          <CardContent>
            
            <div className="space-y-3">
              <h4 className="text-white font-semibold mb-3">처리 정보</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">데이터셋 이름</span>
                  <span className="text-white">{generationResults[0].name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">공격 유형</span>
                  <span className="text-white">
                    {generationResults[0].attackType === 'patch' ? '적대적 패치' : '노이즈 공격'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">처리된 이미지</span>
                  <span className="text-white">{generationResults[0].totalProcessed}개</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">성공</span>
                  <span className="text-green-400">{generationResults[0].successful}개</span>
                </div>
                {generationResults[0].failed > 0 && (
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-400">실패</span>
                    <span className="text-red-400">{generationResults[0].failed}개</span>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Dataset Images Preview */}
      {generatedDatasetId && (
        <div className="flex-1 flex flex-col space-y-3 overflow-hidden">
          <h4 className="text-white font-semibold text-sm flex-shrink-0">생성된 이미지 미리보기</h4>

          {loadingResultImages ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-2" />
                <p className="text-slate-400 text-sm">이미지 로딩 중...</p>
              </div>
            </div>
          ) : resultPreviewImages.length > 0 ? (
            <>
              <div className="grid grid-cols-5 gap-2 flex-shrink-0">
                {resultPreviewImages.slice(0, 5).map((img, idx) => (
                  <div key={idx} className="aspect-square bg-slate-900/50 rounded-lg overflow-hidden relative group border border-slate-700">
                    {img.data ? (
                      <img
                        src={`data:${img.mimeType || 'image/jpeg'};base64,${img.data}`}
                        alt={img.filename || `Result ${idx + 1}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <ImageIcon className="w-6 h-6 text-slate-500" />
                      </div>
                    )}
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/60 transition-all duration-200 flex items-center justify-center opacity-0 group-hover:opacity-100">
                      <span className="text-white text-xs font-medium px-2 text-center break-all">
                        {img.filename || `이미지 ${idx + 1}`}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Pagination */}
              {resultTotalImages > 5 && (
                <div className="flex items-center justify-between bg-slate-800/50 rounded-lg p-3 flex-shrink-0">
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setResultCurrentPage(prev => Math.max(prev - 1, 1))}
                      disabled={resultCurrentPage === 1 || loadingResultImages}
                      className="h-8 px-3 border-slate-700 bg-slate-900/70 text-slate-300 hover:bg-slate-800"
                    >
                      <ChevronLeft className="h-4 w-4 mr-1" />
                      이전
                    </Button>
                    <span className="text-sm text-slate-300 min-w-[100px] text-center">
                      {resultCurrentPage} / {Math.ceil(resultTotalImages / 5)} 페이지
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setResultCurrentPage(prev => Math.min(prev + 1, Math.ceil(resultTotalImages / 5)))}
                      disabled={resultCurrentPage >= Math.ceil(resultTotalImages / 5) || loadingResultImages}
                      className="h-8 px-3 border-slate-700 bg-slate-900/70 text-slate-300 hover:bg-slate-800"
                    >
                      다음
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </Button>
                  </div>
                  <span className="text-xs text-slate-400">
                    전체 {resultTotalImages.toLocaleString()}개 중 {((resultCurrentPage - 1) * 5 + 1)}-{Math.min(resultCurrentPage * 5, resultTotalImages)}개 표시
                  </span>
                </div>
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-slate-400 text-sm">이미지를 불러올 수 없습니다</p>
            </div>
          )}
        </div>
      )}

      {attackDatasetId && (
        <Button
          onClick={async () => {
            try {
              await downloadAdversarialDataset(attackDatasetId, datasetName)
              toast.success('데이터셋 다운로드가 시작되었습니다')
            } catch (error) {
              toast.error('데이터셋 다운로드에 실패했습니다')
            }
          }}
          className="w-full bg-gradient-to-r from-green-600 to-green-700 flex-shrink-0"
        >
          <Download className="w-4 h-4 mr-2" />
          데이터셋 다운로드
        </Button>
      )}
    </div>
  ) : isGenerating ? (
    // State 3: Generation in Progress - Show Logs
    <div className="h-full flex flex-col p-6 space-y-4">
      <div className="flex items-center gap-3 pb-4 border-b border-white/10">
        <div className="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        <div>
          <h3 className="text-white font-semibold">데이터셋 생성 진행 중</h3>
          <p className="text-slate-400 text-sm">적대적 공격 데이터를 생성하고 있습니다...</p>
        </div>
      </div>

      {/* Progress Stats */}
      {attackProgress && (
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <div className="text-xs text-slate-400 mb-1">진행률</div>
            <div className="text-2xl font-bold text-blue-400">
              {attackProgress.total > 0 ? Math.round((attackProgress.processed / attackProgress.total) * 100) : 0}%
            </div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <div className="text-xs text-slate-400 mb-1">처리됨</div>
            <div className="text-2xl font-bold text-green-400">
              {attackProgress.processed}/{attackProgress.total}
            </div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <div className="text-xs text-slate-400 mb-1">성공</div>
            <div className="text-2xl font-bold text-green-400">
              {attackProgress.successful}
            </div>
          </div>
        </div>
      )}

      {/* Progress Bar */}
      {attackProgress && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">전체 진행률</span>
            <span className="text-white">
              {attackProgress.processed} / {attackProgress.total}
            </span>
          </div>
          <Progress
            value={attackProgress.total > 0 ? (attackProgress.processed / attackProgress.total) * 100 : 0}
            className="h-2"
          />
          <div className="text-xs text-slate-400">
            {attackProgress.currentImage}
          </div>
        </div>
      )}

      {/* Logs */}
      <div className="flex-1 overflow-hidden">
        <div className="bg-slate-900/50 rounded-lg p-4 h-full overflow-y-auto font-mono text-xs">
          {generationLogs.length === 0 ? (
            <p className="text-slate-500">로그 대기 중...</p>
          ) : (
            generationLogs.map((log, index) => (
              <div
                key={index}
                className={`py-1 ${
                  log.type === 'error' ? 'text-red-400' :
                  log.type === 'warning' ? 'text-yellow-400' :
                  log.type === 'success' || log.type === 'complete' ? 'text-green-400' :
                  'text-slate-300'
                }`}
              >
                [{new Date().toLocaleTimeString()}] [{log.type?.toUpperCase() || 'INFO'}] {log.message}
              </div>
            ))
          )}
        </div>
      </div>

      {generationResults.length > 0 ? (
        <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4">
          <p className="text-green-300 text-sm flex items-center gap-2 mb-3">
            <CheckCircle2 className="w-4 h-4" />
            데이터셋 생성이 완료되었습니다!
          </p>
          <Button
            onClick={() => {
              setIsGenerating(false)
              setShowResults(true)
            }}
            className="w-full bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800"
          >
            <Eye className="w-4 h-4 mr-2" />
            결과 보기
          </Button>
        </div>
      ) : generationLogs.some(log => log.type === 'error') ? (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
          <p className="text-red-300 text-sm flex items-center gap-2 mb-3">
            <AlertCircle className="w-4 h-4" />
            오류가 발생했습니다. 로그를 확인하세요.
          </p>
          <Button
            onClick={() => {
              setIsGenerating(false)
              setShowResults(false)
            }}
            variant="outline"
            className="w-full border-red-500/30 hover:bg-red-900/20"
          >
            <X className="w-4 h-4 mr-2" />
            닫기
          </Button>
        </div>
      ) : (
        <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
          <p className="text-blue-300 text-sm flex items-center gap-2">
            <Info className="w-4 h-4" />
            데이터셋 생성이 완료되면 결과를 확인할 수 있습니다.
          </p>
        </div>
      )}
    </div>
  ) : hasSelection ? (
    // State 2: Selection Mode - Show Preview with 2 Column Layout
    <div className="h-full flex flex-col p-6 space-y-4">
      <div className="flex-1 grid grid-cols-2 gap-4 overflow-hidden">
        {/* Left Column: Image Preview */}
        <div className="flex flex-col space-y-3 overflow-hidden">
          {/* Large single image */}
          {datasetImages.length > 0 && (
            <>
              <div className="flex-1 bg-slate-900 rounded-lg overflow-hidden relative min-h-0">
                {(() => {
                  const currentImage = datasetImages[previewImageIndex]
                  const patchImageUrl = selectedPatchData?.previewUrl
                  return currentImage?.data ? (
                    <ImageWithBBox
                      imageId={currentImage.id}
                      imageData={currentImage.data}
                      imageMimeType={currentImage.mimeType || 'image/jpeg'}
                      imageWidth={currentImage.width || 640}
                      imageHeight={currentImage.height || 640}
                      targetClass={targetClass}
                      minConfidence={0.3}
                      className="w-full h-full"
                      patchImageUrl={patchImageUrl}
                      patchScale={patchScale}
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-slate-700">
                      <ImageIcon className="w-8 h-8 text-slate-500" />
                    </div>
                  )
                })()}
              </div>

              {/* Image Navigation Controls */}
              <div className="flex items-center justify-between gap-3 px-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPreviewImageIndex(Math.max(0, previewImageIndex - 1))}
                  disabled={isGenerating || previewImageIndex === 0}
                  className="flex-shrink-0"
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>

                <div className="flex-1 text-center">
                  <p className="text-xs text-slate-400">
                    이미지 {previewImageIndex + 1} / {datasetImages.length}
                  </p>
                  {datasetImages[previewImageIndex] && (
                    <p className="text-xs text-slate-500 truncate">
                      {datasetImages[previewImageIndex].filename || `Image ${previewImageIndex + 1}`}
                    </p>
                  )}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPreviewImageIndex(Math.min(datasetImages.length - 1, previewImageIndex + 1))}
                  disabled={isGenerating || previewImageIndex >= datasetImages.length - 1}
                  className="flex-shrink-0"
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </>
          )}

          {datasetImages.length === 0 && (
            <div className="flex-1 bg-slate-900 rounded-lg flex items-center justify-center">
              <div className="text-center space-y-2">
                <ImageIcon className="w-12 h-12 mx-auto text-slate-600" />
                <p className="text-slate-500 text-sm">데이터셋과 타겟 클래스를 선택하면 미리보기가 표시됩니다</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Selected Components */}
        <div className="flex flex-col space-y-3 overflow-y-auto pr-2">
          {/* Dataset Name Card */}
          {datasetName && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent>
                <div className="flex items-start gap-3">
                  <FileText className="w-8 h-8 text-cyan-400 flex-shrink-0" />
                  <div className="flex-1">
                    <h4 className="text-white font-semibold mb-1">생성될 데이터셋 이름</h4>
                    <p className="text-slate-300 text-sm">{datasetName}</p>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Selected Dataset */}
          {selectedDatasetData && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent >
                <div className="flex items-start gap-3">
                  <Database className="w-8 h-8 text-blue-400 flex-shrink-0" />
                  <div className="flex-1">
                    <h4 className="text-white font-semibold mb-1">기준 데이터셋</h4>
                    <p className="text-slate-300 text-sm mb-2">{selectedDatasetData.name}</p>
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <span className="px-2 py-1 bg-blue-900/30 rounded">
                        {selectedDatasetData.imageCount.toLocaleString()} 이미지
                      </span>
                    </div>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Selected Target Class */}
          {targetClass && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent>
                <div className="flex items-start gap-3">
                  <Target className="w-8 h-8 text-red-400 flex-shrink-0" />
                  <div className="flex-1">
                    <h4 className="text-white font-semibold mb-1">타겟 클래스</h4>
                    <p className="text-slate-300 text-sm mb-2">
                      {availableClasses.find(c => c.value === targetClass)?.label || targetClass}
                    </p>
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      {availableClasses.find(c => c.value === targetClass)?.count && (
                        <span className="px-2 py-1 bg-red-900/30 rounded">
                          {availableClasses.find(c => c.value === targetClass)?.count}개 탐지
                        </span>
                      )}
                    </div>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Selected Patch (for patch attack) */}
          {attackType === "patch" && selectedPatchData && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent>
                <div className="flex items-start gap-3">
                  <Grid3x3 className="w-8 h-8 text-purple-400 flex-shrink-0" />
                  <div className="flex-1">
                    <h4 className="text-white font-semibold mb-1">적용할 패치</h4>
                    <p className="text-slate-300 text-sm mb-2">{selectedPatchData.name}</p>
                    <div className="flex items-center gap-2 mt-2">
                      {selectedPatchData.previewUrl && (
                        <div className="w-16 h-16 rounded overflow-hidden bg-slate-900 border border-slate-700">
                          <img
                            src={selectedPatchData.previewUrl}
                            alt={selectedPatchData.name}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      )}
                      <div className="text-xs text-slate-400">
                        패치 크기: {patchScale}% | 타겟: {getKoreanClassName(selectedPatchData.targetClass)}
                      </div>
                    </div>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Selected Model (for noise attack) */}
          {attackType === "noise" && selectedModelData && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent>
                <div className="flex items-start gap-3">
                  <Brain className="w-8 h-8 text-purple-400 flex-shrink-0" />
                  <div className="flex-1">
                    <h4 className="text-white font-semibold mb-1">대상 AI 모델</h4>
                    <p className="text-slate-300 text-sm mb-2">{selectedModelData.name}</p>
                    <div className="text-xs text-slate-400 mt-2">
                      생성된 노이즈는 이 모델의 탐지 성능을 저하시키도록 최적화됩니다.
                    </div>
                    <div className="flex items-center gap-2 text-xs text-slate-400 mt-2">
                      <span className="px-2 py-1 bg-purple-900/30 rounded">
                        노이즈 강도: {(noiseConfig.intensity * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Ready to start message */}
          {datasetName && selectedDataset && targetClass && ((attackType === "patch" && selectedPatch) || (attackType === "noise" && selectedModel)) && (
            <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4 mt-auto">
              <p className="text-green-300 text-sm flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                모든 구성이 완료되었습니다. "공격 데이터셋 생성" 버튼을 클릭하여 생성을 시작하세요.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  ) : (
    // State 1: Initial - Show Guide
    <div className="h-full flex flex-col justify-center items-center space-y-6 p-8">
      {/* Welcome Message */}
      <div className="text-center space-y-4 max-w-2xl">
        <Zap className="w-20 h-20 mx-auto text-blue-400 opacity-50" />
        <h3 className="text-2xl font-bold text-white">
          2D 적대적 공격 데이터 생성
        </h3>
        <p className="text-slate-400 text-sm leading-relaxed">
          왼쪽 패널에서 공격 유형(패치 또는 노이즈)을 선택하고 필요한 구성 요소를 설정하여
          적대적 공격 데이터셋 생성을 시작하세요.
        </p>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-4xl">
        <Card className="bg-slate-800/50 border-white/10">
          <CardContent className="pt-6 text-center">
            <Grid3x3 className="w-10 h-10 mx-auto mb-3 text-purple-400" />
            <h4 className="text-white font-semibold mb-2">2D 패치 공격</h4>
            <p className="text-slate-400 text-xs">
              생성된 적대적 패치를 데이터셋 이미지에 적용하여 새로운 공격 데이터셋을 생성합니다.
              패치는 타겟 클래스의 바운딩 박스 위에 렌더링됩니다.
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardContent className="pt-6 text-center">
            <Volume2 className="w-10 h-10 mx-auto mb-3 text-blue-400" />
            <h4 className="text-white font-semibold mb-2">노이즈 공격</h4>
            <p className="text-slate-400 text-xs">
              선택한 AI 모델을 대상으로 최적화된 노이즈를 생성하여 데이터셋 전체에 적용합니다.
              노이즈는 모델의 탐지 성능을 저하시키도록 설계됩니다.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Guide */}
      <div className="bg-slate-800/30 border border-white/10 rounded-lg p-6 w-full max-w-2xl">
        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
          <Eye className="w-5 h-5 text-blue-400" />
          데이터셋 생성 프로세스
        </h4>
        <ol className="space-y-2 text-sm text-slate-300">
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">1.</span>
            <span><strong>데이터셋 이름:</strong> 생성할 데이터셋의 이름을 입력합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">2.</span>
            <span><strong>기준 데이터셋:</strong> 공격을 적용할 원본 데이터셋을 선택합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">3.</span>
            <span><strong>타겟 클래스:</strong> 공격 대상이 되는 객체 클래스를 선택합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">4.</span>
            <span><strong>공격 유형:</strong> 패치 또는 노이즈 공격을 선택합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">5.</span>
            <span><strong>패치/모델 선택:</strong> 패치 공격 시 패치 선택, 노이즈 공격 시 모델 선택</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">6.</span>
            <span><strong>미리보기:</strong> 선택된 항목을 확인합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">7.</span>
            <span><strong>데이터셋 생성:</strong> 생성 시작 버튼을 클릭합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">8.</span>
            <span><strong>결과 확인:</strong> 생성된 데이터셋을 다운로드합니다</span>
          </li>
        </ol>
      </div>
    </div>
  )

  // Action Buttons
  const actionButtons = showResults && generationResults.length > 0 ? (
    <Button
      onClick={() => {
        setShowResults(false)
        setGenerationResults([])
        setAttackProgress(null)
        setGeneratedDatasetId(null)
        setAttackDatasetId(null)
        setGenerationLogs([])
        setResultPreviewImages([])
        setResultCurrentPage(1)
        setResultTotalImages(0)
        if (eventSource) {
          eventSource.close()
          setEventSource(null)
        }
      }}
      className="w-full bg-gradient-to-r from-blue-600 to-blue-700"
    >
      <Play className="w-4 h-4 mr-2" />
      초기화
    </Button>
  ) : (
    <Button
      onClick={handleGenerateAttack}
      disabled={isGenerating}
      className="w-full"
    >
      <Play className="w-4 h-4 mr-2" />
      공격 데이터셋 생성
    </Button>
  )

  useEffect(() => {
    if (currentView === 'results' && generatedDatasetId) {
      setResultCurrentPage(1)
      loadResultImagesCount()
      loadResultImages(1)  // Always load images when view changes
    }
  }, [currentView, generatedDatasetId])

  useEffect(() => {
    if (currentView === 'results' && generatedDatasetId) {
      loadResultImages(resultCurrentPage)
    }
  }, [resultCurrentPage])

  const loadResultImagesCount = async () => {
    if (!generatedDatasetId) return

    try {
      const response = await fetch(`/api/datasets/${generatedDatasetId}/images?limit=1&offset=0`)
      if (response.ok) {
        const data = await response.json()
        setResultTotalImages(data.total || 0)
      }
    } catch (error) {
      console.error('Failed to load result images count:', error)
    }
  }

  const loadResultImages = async (page: number = 1) => {
    if (!generatedDatasetId) return

    setLoadingResultImages(true)
    try {
      const offset = (page - 1) * resultImagesPerPage
      const response = await fetch(`/api/datasets/${generatedDatasetId}/images?limit=${resultImagesPerPage}&offset=${offset}`)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      console.log('[AdversarialDataGenerator] loadResultImages response:', data)
      console.log('[AdversarialDataGenerator] data.images length:', data.images?.length)
      console.log('[AdversarialDataGenerator] First image data exists:', !!data.images?.[0]?.data)
      setResultPreviewImages(data.images || [])
      if (data.total !== undefined) {
        setResultTotalImages(data.total)
      }
    } catch (error) {
      console.error('Failed to load result images:', error)
      toast.error('결과 이미지를 불러오지 못했습니다')
    } finally {
      setLoadingResultImages(false)
    }
  }

  if (currentView === 'results') {
    return (
      <>
        <AdversarialToolLayout
          title="생성 결과"
          description="적대적 공격 데이터 생성 결과"
          icon={CheckCircle2}

          leftPanel={{
            title: "생성된 데이터셋",
            icon: FileStack,
            description: "생성 완료된 데이터셋 정보",
            children: (
              <div className="space-y-4">
                {generationResults.map((result, idx) => (
                  <div key={idx} className="bg-slate-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-white font-medium">{result.name}</h4>
                      <Badge className="bg-green-500/10 text-green-400">완료</Badge>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">총 이미지</span>
                        <span className="text-white">{result.totalProcessed}개</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">성공</span>
                        <span className="text-green-400">{result.successful}개</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">실패</span>
                        <span className="text-red-400">{result.failed}개</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">공격 유형</span>
                        <span className="text-white">{result.attackType === "patch" ? "2D 패치" : "노이즈"}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">생성 시간</span>
                        <span className="text-white">{new Date(result.createdAt).toLocaleString('ko-KR')}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )
          }}
          rightPanel={{
            title: "결과 미리보기",
            icon: Eye,
            description: "생성된 데이터셋 샘플",
            children: (
              <div className="space-y-4">
                {loadingResultImages ? (
                  <div className="text-center text-slate-400 py-8">
                    <div className="flex items-center justify-center gap-2">
                      <Clock className="w-4 h-4 animate-spin" />
                      이미지 로딩 중...
                    </div>
                  </div>
                ) : resultPreviewImages.length > 0 ? (
                  <>
                    <div className="grid grid-cols-3 gap-3">
                      {resultPreviewImages.map((img, idx) => (
                        <div key={idx} className="aspect-square bg-slate-900/50 rounded-lg overflow-hidden relative group">
                          {img.data ? (
                            <img
                              src={`data:image/jpeg;base64,${img.data}`}
                              alt={img.filename || `Result ${idx + 1}`}
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              <ImageIcon className="w-8 h-8 text-slate-500" />
                            </div>
                          )}
                          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/50 transition-all duration-200 flex items-center justify-center opacity-0 group-hover:opacity-100">
                            <span className="text-white text-xs font-medium">
                              {img.filename || `이미지 ${idx + 1}`}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>

                    {resultTotalImages > resultImagesPerPage && (
                      <div className="flex items-center justify-between bg-slate-800/50 rounded-lg p-3">
                        <div className="flex items-center gap-2">
                          <Button
                            variant="outline"
                            size="icon"
                            onClick={() => setResultCurrentPage(prev => Math.max(prev - 1, 1))}
                            disabled={resultCurrentPage === 1 || loadingResultImages}
                            className="h-8 w-8 border-slate-700 bg-slate-900/70 text-slate-300 hover:bg-slate-800"
                          >
                            <ChevronLeft className="h-4 w-4" />
                          </Button>
                          <span className="text-sm text-slate-300 min-w-[100px] text-center">
                            {resultCurrentPage} / {Math.ceil(resultTotalImages / resultImagesPerPage)} 페이지
                          </span>
                          <Button
                            variant="outline"
                            size="icon"
                            onClick={() => setResultCurrentPage(prev => Math.min(prev + 1, Math.ceil(resultTotalImages / resultImagesPerPage)))}
                            disabled={resultCurrentPage >= Math.ceil(resultTotalImages / resultImagesPerPage) || loadingResultImages}
                            className="h-8 w-8 border-slate-700 bg-slate-900/70 text-slate-300 hover:bg-slate-800"
                          >
                            <ChevronRight className="h-4 w-4" />
                          </Button>
                        </div>
                        <span className="text-xs text-slate-400">
                          전체 {resultTotalImages.toLocaleString()}개 중 {((resultCurrentPage - 1) * resultImagesPerPage + 1)}-{Math.min(resultCurrentPage * resultImagesPerPage, resultTotalImages)}개 표시
                        </span>
                      </div>
                    )}

                    {resultTotalImages <= resultImagesPerPage && resultTotalImages > 0 && (
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-slate-400">총 이미지</span>
                          <span className="text-lg font-bold text-white">
                            {resultTotalImages.toLocaleString()}개
                          </span>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center text-slate-400 py-8">
                    결과 이미지가 없습니다
                  </div>
                )}
              </div>
            )
          }}
          actionButtons={
            <div className="flex gap-2">
              <Button
                className="bg-primary"
                onClick={async () => {
                  if (generatedDatasetId) {
                    try {
                      await downloadAdversarialDataset(generatedDatasetId, datasetName)
                      toast.success("데이터셋 다운로드를 시작합니다")
                    } catch (error) {
                      toast.error("다운로드 실패")
                    }
                  }
                }}
                disabled={!generatedDatasetId}
              >
                <Download className="w-4 h-4 mr-2" />
                데이터셋 다운로드
              </Button>
              <Button
                onClick={() => setCurrentView('main')}
                variant="outline"
              >
                <ChevronLeft className="w-4 h-4 mr-2" />
                돌아가기
              </Button>
            </div>
          }
        />
      </>
    )
  }

  return (
    <>
      <AdversarialToolLayout
        title="2D 적대적 공격 데이터 생성"
        description="적대적 공격이 적용된 데이터셋을 생성합니다"
        icon={Zap}
        leftPanel={{
          title: "공격 설정",
          icon: Settings,
          description: "적대적 패치 파라미터 설정",
          children: leftPanelContent,
          className: "space-y-4"
        }}
        rightPanel={{
          title: isGenerating
            ? "데이터셋 생성 진행 상황"
            : showResults && generationResults.length > 0
              ? "생성된 데이터셋 결과"
              : hasSelection
                ? "선택 항목 미리보기"
                : "데이터셋 생성 안내",
          icon: isGenerating
            ? Loader2
            : showResults && generationResults.length > 0
              ? CheckCircle2
              : hasSelection
                ? Info
                : Zap,
          children: rightPanelContent
        }}
        actionButtons={actionButtons}
      />
    </>
  )
}
