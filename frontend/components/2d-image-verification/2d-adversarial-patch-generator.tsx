"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent } from "@/components/ui/card"
import { toast } from "sonner"
import {
  Play,
  Pause,
  Shield,
  Zap,
  Target,
  Settings,
  Sparkles,
  Database,
  Download,
  AlertCircle,
  CheckCircle2,
  Clock,
  FileStack,
  FileText,
  Eye,
  Image as ImageIcon,
  ChevronLeft,
  ChevronRight,
  Loader2,
  X
} from "lucide-react"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import {
  fetchBackendDatasets,
  fetchYoloModels,
  startTraining,
  connectPatchGenerationSSE,
  fetchTrainingResult,
  downloadPatch,
  type BackendDataset,
  type YoloModel,
  type TrainingLog
} from "@/lib/adversarial-api"
import { ImageWithBBox } from "@/components/annotations/ImageWithBBox"

// FastAPI backend URL - Use NEXT_PUBLIC_BACKEND_API_URL environment variable
const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

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

interface AttackConfig {
  method: "patch" | "dpatch" | "robust_dpatch" // Updated to match backend attack methods
  targetClass: string
  iterations: number
  patchSize: number
  learningRate: number // Added learning rate parameter
}

interface BatchProgress {
  total: number
  processed: number
  successful: number
  failed: number
  currentImage: string
  estimatedTime: string
}

export function AdversarialPatchGeneratorUpdated() {

  const [selectedDataset, setSelectedDataset] = useState<string>("")
  const [availableDatasets, setAvailableDatasets] = useState<Dataset[]>([])
  const [backendDatasets, setBackendDatasets] = useState<BackendDataset[]>([])
  const [yoloModels, setYoloModels] = useState<YoloModel[]>([])
  const [datasetImages, setDatasetImages] = useState<any[]>([])
  const [loadingImages, setLoadingImages] = useState(false)
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [currentTrainingId, setCurrentTrainingId] = useState<number | null>(null)
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const [trainingLogs, setTrainingLogs] = useState<TrainingLog[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null)
  const [generatedPatches, setGeneratedPatches] = useState<any[]>([])
  const [showPatchResult, setShowPatchResult] = useState(false) // 패치 결과 표시 여부
  const [currentPage, setCurrentPage] = useState(1)
  const [totalImages, setTotalImages] = useState(0)
  const [availableClasses, setAvailableClasses] = useState<Array<{value: string, label: string, count?: number}>>([])
  const [patchName, setPatchName] = useState<string>("")
  const [previewImageIndex, setPreviewImageIndex] = useState(0)
  const imagesPerPage = 20

  const [attackConfig, setAttackConfig] = useState<AttackConfig>({
    method: "robust_dpatch", // Default to robust_dpatch
    targetClass: "",
    iterations: 50, // Match test file default
    patchSize: 150,
    learningRate: 5.0, // Match test file default
  })

  useEffect(() => {
    loadDatasets()
    loadBackendDatasets()
    loadYoloModels()
  }, [])

  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close()
      }
    }
  }, [eventSource])

  // Auto-detect completion from training logs
  useEffect(() => {
    if (!isGenerating || generatedPatches.length > 0) {
      return
    }

    // IMPORTANT: Only look for 'complete' type which has patch_id
    // The 'success' type is sent before patch is saved and doesn't have patch_id
    const completionLog = trainingLogs.find(log => log.type === 'complete')

    if (completionLog) {
      console.log('[useEffect] Detected completion from logs')
      console.log('[useEffect] Completion log:', completionLog)

      const totalImages = availableDatasets.find(d => d.id === selectedDataset)?.imageCount || 0

      // Extract patch_id and file_path from the completion log
      const patchId = (completionLog as any).patch_id
      const filePath = (completionLog as any).file_path

      console.log('[useEffect] Extracted patch_id:', patchId)
      console.log('[useEffect] Extracted file_path:', filePath)

      if (patchId) {
        console.log('[useEffect] Using patch_id:', patchId)

        // Extract storage_key from file_path
        // file_path format: /home/.../storage/patches/filename.png
        // storage_key format: patches/filename.png
        let storageKey = ''
        if (filePath && typeof filePath === 'string') {
          const match = filePath.match(/storage\/(.+)$/)
          if (match) {
            storageKey = match[1]
          }
        }

        // Create patch result immediately with available data
        setGeneratedPatches([{
          id: `patch_${patchId}`,
          method: attackConfig.method.toUpperCase(),
          targetClass: attackConfig.targetClass,
          successRate: 0, // Will be updated if available
          averageConfidenceReduction: 0,
          totalImages: totalImages,
          createdAt: new Date().toISOString(),
          trainingId: patchId,
          storageKey: storageKey  // Add storage key for image URL
        }])
        toast.success("적대적 패치 생성이 완료되었습니다")
      } else {
        console.error('[useEffect] No patch_id found in completion log')
        console.error('[useEffect] Completion log:', completionLog)
      }
    }
  }, [trainingLogs, generatedPatches, isGenerating, attackConfig, selectedDataset, availableDatasets])

  // 한국어 클래스명 매핑 함수
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


  // 데이터셋 선택 시 페이지 초기화 및 클래스 목록 업데이트
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
                setAttackConfig(prev => ({ ...prev, targetClass: classes[0].value }))
              }
            } else {
              // 어노테이션이 없으면 빈 클래스 목록
              setAvailableClasses([])
              setAttackConfig(prev => ({ ...prev, targetClass: '' }))
            }
          } else {
            // API 호출 실패 시 빈 클래스 목록
            setAvailableClasses([])
            setAttackConfig(prev => ({ ...prev, targetClass: '' }))
          }
        } catch (error) {
          console.error('Failed to fetch annotation summary:', error)
          setAvailableClasses([])
          setAttackConfig(prev => ({ ...prev, targetClass: '' }))
        }
      }

      fetchAnnotationSummary()
    } else {
      setDatasetImages([])
      setTotalImages(0)
      setCurrentPage(1)
      setAvailableClasses([])
      setAttackConfig(prev => ({ ...prev, targetClass: '' }))
    }
  }, [selectedDataset])

  // 페이지 변경 시 이미지 로드
  useEffect(() => {
    if (selectedDataset) {
      loadDatasetImages(selectedDataset, currentPage)
      setPreviewImageIndex(0) // Reset preview index when dataset changes
    }
  }, [selectedDataset, currentPage])


  const loadDatasets = async () => {
    try {
      const response = await fetch('/api/datasets?type=2D_IMAGE')
      const data = await response.json()

      const formattedDatasets = data.map((dataset: any) => ({
        id: dataset.id,
        name: dataset.name,
        imageCount: dataset.size || 0,
        metadata: dataset.metadata
      }))

      setAvailableDatasets(formattedDatasets)
    } catch (error) {
      console.error('Failed to load datasets:', error)
      toast.error('데이터셋 목록을 불러오는데 실패했습니다')
    }
  }

  const loadBackendDatasets = async () => {
    try {
      const datasets = await fetchBackendDatasets()
      setBackendDatasets(datasets)
    } catch (error) {
      console.error('Failed to load backend datasets:', error)
      toast.error('백엔드 데이터셋 목록을 불러오는데 실패했습니다')
    }
  }

  const loadYoloModels = async () => {
    try {
      const models = await fetchYoloModels()
      setYoloModels(models)
    } catch (error) {
      console.error('Failed to load YOLO models:', error)
      toast.error('YOLO 모델 목록을 불러오는데 실패했습니다')
    }
  }

  const loadDatasetImages = async (datasetId: string, page: number = 1) => {
    setLoadingImages(true)

    try {
      // 페이지네이션 파라미터 추가
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
        // 전체 이미지 수를 설정 (API가 총 개수를 반환한다고 가정)
        if (data.total !== undefined) {
          setTotalImages(data.total)
        } else {
          // total이 없으면 데이터셋 정보에서 가져옴
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

  const handleGeneratePatch = async () => {
    if (!patchName.trim()) {
      toast.error("패치 이름을 입력해주세요")
      return
    }

    if (!selectedDataset) {
      toast.error("데이터셋을 선택해주세요")
      return
    }

    if (!attackConfig.targetClass) {
      toast.error("타겟 클래스를 선택해주세요")
      return
    }

    if (!selectedModel) {
      toast.error("AI 모델을 선택해주세요")
      return
    }

    const dataset = availableDatasets.find(d => d.id === selectedDataset)
    const backendDataset = backendDatasets.find(d => d.name === dataset?.name)

    if (!backendDataset) {
      toast.error("백엔드 데이터셋을 찾을 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.")
      console.error('Available backend datasets:', backendDatasets)
      console.error('Selected dataset name:', dataset?.name)
      setIsGenerating(false)
      return
    }

    setIsGenerating(true)
    setShowPatchResult(false)
    setTrainingLogs([])

    const totalImages = dataset?.imageCount || 0

    setBatchProgress({
      total: totalImages,
      processed: 0,
      successful: 0,
      failed: 0,
      currentImage: "초기화 중...",
      estimatedTime: "계산 중..."
    })

    // CRITICAL FIX: Generate session_id FIRST and connect to SSE BEFORE starting training
    const sessionId = `training_${Date.now()}`
    console.log('[FrontendFix] Generated session_id:', sessionId)

    try {
      // STEP 1: Connect to SSE endpoint FIRST (this creates the session on backend)
      console.log('[FrontendFix] Connecting to SSE endpoint BEFORE starting training')
      const sse = connectPatchGenerationSSE(sessionId, {
        onOpen: () => {
          console.log('[SSE] Connection opened, ready to receive events')
        },
        onMessage: (data) => {
          console.log('[SSE Message] Raw data received:', data)
          console.log('[SSE Message] Data type:', data.type)
          console.log('[SSE Message] Full data structure:', JSON.stringify(data, null, 2))

          setTrainingLogs(prev => [...prev, data])

          switch(data.type) {
            case 'progress':
              if (data.iteration && data.total_iterations) {
                setBatchProgress({
                  processed: data.iteration,
                  total: data.total_iterations,
                  successful: data.detected_count || 0,
                  failed: 0,
                  currentImage: `반복 ${data.iteration}/${data.total_iterations}`,
                  estimatedTime: `Loss: ${data.avg_loss?.toFixed(4) || 'N/A'}`
                })
              }
              break

            case 'success':
              // Don't change isGenerating - keep showing logs
              setBatchProgress(prev => prev ? {
                ...prev,
                processed: prev.total,
                currentImage: "완료",
                estimatedTime: "완료"
              } : null)

              console.log('[SSE] Success message received - useEffect will handle result generation')
              // Note: We don't process the result here because of closure issues with currentTrainingId
              // The useEffect hook will detect this 'success' log and handle result generation
              break

            case 'complete':
              // Don't change isGenerating - keep showing logs until user clicks button
              setBatchProgress(prev => prev ? {
                ...prev,
                processed: prev.total,
                currentImage: "완료",
                estimatedTime: "완료"
              } : null)

              console.log('[SSE] Complete message received - useEffect will handle result generation')
              // Note: We don't process the result here because of closure issues with currentTrainingId
              // The useEffect hook will detect this 'complete' log and handle result generation
              break

            case 'error':
              // Don't change isGenerating - keep showing logs
              toast.error(`오류: ${data.message}`)
              // 모달을 닫지 않고 유지하여 사용자가 오류 로그를 확인할 수 있도록 함
              break

            case 'warning':
              toast.warning(data.message)
              break

            default:
              console.log('[SSE Message] Unhandled message type:', data.type)
              // The useEffect hook will handle any completion messages based on log content
              break
          }
        },
        onError: (error) => {
          console.error('SSE 오류:', error)
          // useEffect will automatically detect completion from logs
        },
        onClose: () => {
          console.log('[SSE] 연결 종료')
          // useEffect will automatically detect completion from logs
        }
      })

      setEventSource(sse)

      // STEP 2: Wait a bit to ensure SSE connection is established
      await new Promise(resolve => setTimeout(resolve, 500))
      console.log('[FrontendFix] SSE connection established, now starting training')

      // STEP 3: Now start training with the same session_id
      const trainingConfig = {
        patch_name: patchName,
        dataset_id: backendDataset.id,
        target_class: attackConfig.targetClass,
        model_path: selectedModel,
        attack_method: attackConfig.method, // Added: attack method selection
        iterations: attackConfig.iterations,
        patch_size: attackConfig.patchSize,
        learning_rate: attackConfig.learningRate, // Added: learning rate parameter
        session_id: sessionId  // Pass the same session_id
      }

      console.log('[FrontendFix] Starting training with session_id:', sessionId)
      const result = await startTraining(trainingConfig)
      // result.training_id could be string or number; ensure we store number or null per expected type
      const trainingId = typeof result.training_id === 'number'
        ? result.training_id
        : (typeof result.training_id === 'string' && !isNaN(Number(result.training_id)) ? Number(result.training_id) : null)
      setCurrentTrainingId(trainingId)
      toast.success("학습이 시작되었습니다")

    } catch (error) {
      console.error('Failed to start training:', error)

      // Extract user-friendly error message
      let errorMessage = "학습 시작에 실패했습니다"
      if (error instanceof Error) {
        // Check if error message contains user-friendly Korean message from backend
        if (error.message.includes("이미 존재하는 패치")) {
          errorMessage = error.message
        } else if (error.message.includes("리소스를 찾을 수 없습니다")) {
          errorMessage = error.message
        } else if (error.message.includes("잘못된 입력")) {
          errorMessage = error.message
        } else if (error.message.includes("데이터베이스")) {
          errorMessage = error.message
        } else if (error.message.includes("GPU") || error.message.includes("메모리")) {
          errorMessage = error.message
        } else {
          errorMessage = `학습 시작 실패: ${error.message}`
        }
      }

      toast.error(errorMessage)

      // Keep isGenerating true to show error in logs, add error log
      setTrainingLogs(prev => [...prev, {
        type: 'error',
        message: errorMessage
      }])
    }
  }



  // Left Panel: Attack Configuration
  const leftPanelContent = (
    <div className="space-y-4">

      {/* 패치 이름 */}
      <div className="space-y-2">
        <Label>패치 이름</Label>
        <Input
          type="text"
          placeholder="패치 이름 입력"
          value={patchName}
          onChange={(e) => setPatchName(e.target.value)}
          disabled={showPatchResult && generatedPatches.length > 0}
        />
        {patchName && (
          <div className="text-xs text-slate-400">
            다운로드 파일명: {patchName.replace(/[^a-zA-Z0-9가-힣_-]/g, '_') || 'adversarial_patch'}.png
          </div>
        )}
      </div>

      {/* 데이터셋 선택 */}
      <div className="space-y-2">
        <Label>데이터셋 선택</Label>
        <Select value={selectedDataset} onValueChange={setSelectedDataset} disabled={showPatchResult && generatedPatches.length > 0}>
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

      {/* 타겟 클래스 */}
      <div className="space-y-2">
        <Label>타겟 클래스</Label>
        <Select
          value={attackConfig.targetClass}
          onValueChange={(v) => setAttackConfig({...attackConfig, targetClass: v})}
          disabled={!selectedDataset || availableClasses.length === 0 || (showPatchResult && generatedPatches.length > 0)}
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

      {/* 공격 방법 선택 */}
      <div className="space-y-2">
        <Label>공격 방법</Label>
        <Select
          value={attackConfig.method}
          onValueChange={(v) => setAttackConfig({...attackConfig, method: v as "patch" | "dpatch" | "robust_dpatch"})}
          disabled={showPatchResult && generatedPatches.length > 0}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="robust_dpatch">
              <div className="flex flex-col items-start">
                <span className="font-medium">RobustDPatch</span>
              </div>
            </SelectItem>
            <SelectItem value="dpatch">
              <div className="flex flex-col items-start">
                <span className="font-medium">DPatch</span>
              </div>
            </SelectItem>
            <SelectItem value="patch">
              <div className="flex flex-col items-start">
                <span className="font-medium">Adversarial Patch</span>
              </div>
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* 학습률 */}
      <div className="space-y-2">
        <Label>학습률 (Learning Rate)</Label>
        <div className="flex items-center gap-4">
          <Slider
            value={[attackConfig.learningRate]}
            onValueChange={(values) => setAttackConfig({...attackConfig, learningRate: values[0]})}
            min={0.1}
            max={20}
            step={0.1}
            disabled={showPatchResult && generatedPatches.length > 0}
            className="flex-1"
          />
          <span className="text-sm text-white min-w-[60px] text-right">
            {attackConfig.learningRate.toFixed(1)}
          </span>
        </div>
      </div>

      {/* 반복 횟수 */}
      <div className="space-y-2">
        <Label>반복 횟수 (Iterations)</Label>
        <div className="flex items-center gap-4">
          <Slider
            value={[attackConfig.iterations]}
            onValueChange={(values) => setAttackConfig({...attackConfig, iterations: values[0]})}
            min={10}
            max={500}
            step={10}
            disabled={showPatchResult && generatedPatches.length > 0}
            className="flex-1"
          />
          <span className="text-sm text-white min-w-[60px] text-right">
            {attackConfig.iterations}
          </span>
        </div>
      </div>

      {/* 패치 크기 */}
      <div className="space-y-2">
        <Label>패치 크기 (Patch Size)</Label>
        <div className="flex items-center gap-4">
          <Slider
            value={[attackConfig.patchSize]}
            onValueChange={(values) => setAttackConfig({...attackConfig, patchSize: values[0]})}
            min={50}
            max={300}
            step={10}
            disabled={showPatchResult && generatedPatches.length > 0}
            className="flex-1"
          />
          <span className="text-sm text-white min-w-[60px] text-right">
            {attackConfig.patchSize}px
          </span>
        </div>
      </div>

      {/* AI 모델 선택 */}
      <div className="space-y-2">
        <Label>대상 AI 모델</Label>
        <Select value={selectedModel} onValueChange={setSelectedModel} disabled={showPatchResult && generatedPatches.length > 0}>
          <SelectTrigger>
            <SelectValue placeholder="모델 선택" />
          </SelectTrigger>
          <SelectContent>
            {Array.isArray(yoloModels) && yoloModels.length > 0 ? (
              yoloModels.map(model => (
                <SelectItem
                  key={model?.path || model?.name || Math.random().toString()}
                  value={model?.path || ""}
                  disabled={!model?.path}
                >
                  {model?.name || "이름 없는 모델"}
                </SelectItem>
              ))
            ) : (
              <SelectItem value="no-model" disabled>등록된 모델이 없습니다</SelectItem>
            )}
          </SelectContent>
        </Select>
      </div>
    </div>
  )

  // Check if anything is selected
  const hasSelection = selectedDataset || attackConfig.targetClass || selectedModel
  const selectedDatasetData = availableDatasets.find(d => d.id === selectedDataset)
  const selectedModelData = yoloModels.find(m => m.path === selectedModel)

  // Right Panel: Four States - Initial Guide, Selection Preview, Generation Progress, Results
  const rightPanelContent = showPatchResult && generatedPatches.length > 0 ? (
    // State 4: Patch Result - Show Generated Patch (takes priority)
    <div className="h-full flex flex-col p-6 space-y-4 overflow-y-auto">
      <div className="pb-4 border-b border-white/10">
        <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-green-400" />
          패치 생성 완료
        </h3>
        <p className="text-slate-400 text-sm">생성된 적대적 패치를 확인하세요</p>
      </div>

      {/* Generated Patch Info */}
      {generatedPatches[0] && (
        <Card className="bg-slate-800/50 border-white/10">
          <CardContent className="pt-6">
            <div className="space-y-3">
              <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                처리 정보
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">타겟 클래스</span>
                  <span className="text-white">
                    {availableClasses.find(c => c.value === attackConfig.targetClass)?.label || attackConfig.targetClass}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">처리된 이미지</span>
                  <span className="text-white">{generatedPatches[0].totalImages.toLocaleString()}개</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">생성 시간</span>
                  <span className="text-white">
                    {new Date(generatedPatches[0].createdAt).toLocaleTimeString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-400">성공률</span>
                  <span className="text-green-400 font-semibold">
                    {generatedPatches[0].successRate.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Patch Image */}
      <div className="bg-slate-900/50 rounded-lg p-6 flex flex-col items-center justify-center">
        {generatedPatches[0]?.storageKey ? (
          <div className="flex flex-col items-center gap-4 w-full max-w-md">
            <div className="text-sm text-slate-400 mb-2">생성된 패치 이미지</div>
            <div className="w-full aspect-square bg-slate-800 rounded-lg overflow-hidden border-2 border-slate-700 shadow-xl">
              <img
                src={`/api/storage/${(generatedPatches[0] as any).storageKey}`}
                alt="Generated adversarial patch"
                className="w-full h-full object-contain"
                onError={(e) => {
                  console.error('Failed to load patch image from:', e.currentTarget.src)
                  e.currentTarget.style.display = 'none'
                  e.currentTarget.parentElement!.innerHTML = '<div class="w-full h-full flex items-center justify-center text-slate-400">패치 이미지를 불러올 수 없습니다</div>'
                }}
              />
            </div>
            <div className="text-xs text-slate-500 text-center">
              Patch ID: {generatedPatches[0].trainingId}
            </div>
          </div>
        ) : (
          <div className="text-slate-400">패치 이미지를 불러오는 중...</div>
        )}
      </div>
    </div>
  ) : isGenerating ? (
    // State 3: Generation in Progress - Show Logs
    <div className="h-full flex flex-col p-6 space-y-4">
      <div className="flex items-center gap-3 pb-4 border-b border-white/10">
        <div className="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        <div>
          <h3 className="text-white font-semibold">패치 생성 진행 중</h3>
          <p className="text-slate-400 text-sm">적대적 패치를 생성하고 있습니다...</p>
        </div>
      </div>

      {/* Progress Stats */}
      {batchProgress && (
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <div className="text-xs text-slate-400 mb-1">진행률</div>
            <div className="text-2xl font-bold text-blue-400">
              {batchProgress.total > 0 ? Math.round((batchProgress.processed / batchProgress.total) * 100) : 0}%
            </div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <div className="text-xs text-slate-400 mb-1">처리됨</div>
            <div className="text-2xl font-bold text-green-400">
              {batchProgress.processed}/{batchProgress.total}
            </div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
            <div className="text-xs text-slate-400 mb-1">에러</div>
            <div className="text-2xl font-bold text-red-400">
              {batchProgress.failed}
            </div>
          </div>
        </div>
      )}

      {/* Progress Bar */}
      {batchProgress && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">전체 진행률</span>
            <span className="text-white">
              {batchProgress.processed} / {batchProgress.total}
            </span>
          </div>
          <Progress
            value={batchProgress.total > 0 ? (batchProgress.processed / batchProgress.total) * 100 : 0}
            className="h-2"
          />
          <div className="text-xs text-slate-400">
            {batchProgress.currentImage}
          </div>
        </div>
      )}

      {/* Logs */}
      <div className="flex-1 overflow-hidden">
        <div className="bg-slate-900/50 rounded-lg p-4 h-full overflow-y-auto font-mono text-xs">
          {trainingLogs.length === 0 ? (
            <p className="text-slate-500">로그 대기 중...</p>
          ) : (
            trainingLogs.map((log, index) => (
              <div
                key={index}
                className={`py-1 ${
                  {
                    status: 'text-blue-400',
                    info: 'text-gray-400',
                    progress: 'text-green-400',
                    complete: 'text-green-500 font-bold',
                    success: 'text-green-500 font-semibold',
                    warning: 'text-yellow-400 font-semibold',
                    error: 'text-red-500 font-bold'
                  }[log.type] || 'text-gray-400'
                }`}
              >
                [{new Date().toLocaleTimeString()}] [{log.type?.toUpperCase() || 'INFO'}] {log.message}
              </div>
            ))
          )}
        </div>
      </div>

      {generatedPatches.length > 0 ? (
        <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4">
          <p className="text-green-300 text-sm flex items-center gap-2 mb-3">
            <CheckCircle2 className="w-4 h-4" />
            패치 생성이 완료되었습니다!
          </p>
          <Button
            onClick={() => {
              setIsGenerating(false)
              setShowPatchResult(true)
            }}
            className="w-full bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800"
          >
            <Eye className="w-4 h-4 mr-2" />
            결과 보기
          </Button>
        </div>
      ) : trainingLogs.some(log => log.type === 'error') ? (
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
          <p className="text-red-300 text-sm flex items-center gap-2 mb-3">
            <AlertCircle className="w-4 h-4" />
            오류가 발생했습니다. 로그를 확인하세요.
          </p>
          <Button
            onClick={() => {
              setIsGenerating(false)
              setShowPatchResult(false)
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
            <AlertCircle className="w-4 h-4" />
            패치 생성이 완료되면 결과를 확인할 수 있습니다.
          </p>
        </div>
      )}
    </div>
  ) : hasSelection ? (
    // State 2: Selection Mode - Show Preview with 2 Column Layout
    <div className="h-full flex flex-col p-6 space-y-4">
      {/* 2 Column Layout */}
      <div className="flex-1 grid grid-cols-2 gap-4 overflow-hidden">
        {/* Left Column: Image Preview */}
        <div className="flex flex-col space-y-3 overflow-hidden">
          <div className="flex items-center justify-between">
            <h4 className="text-white font-semibold text-sm">이미지 미리보기</h4>
            {datasetImages.length > 0 && (
              <span className="text-xs text-slate-400">
                {previewImageIndex + 1} / {datasetImages.length}
              </span>
            )}
          </div>

          {datasetImages.length > 0 ? (
            <div className="flex-1 flex flex-col space-y-3 min-h-0">
              {/* Image Display */}
              <div className="flex-1 bg-slate-900 rounded-lg overflow-hidden border border-white/10 relative min-h-0">
                <div className="w-full h-full p-2">
                  {datasetImages[previewImageIndex]?.data ? (
                    <ImageWithBBox
                      imageId={datasetImages[previewImageIndex].id}
                      imageData={datasetImages[previewImageIndex].data}
                      imageMimeType={datasetImages[previewImageIndex].mimeType || 'image/jpeg'}
                      imageWidth={datasetImages[previewImageIndex].width || 640}
                      imageHeight={datasetImages[previewImageIndex].height || 640}
                      targetClass={attackConfig.targetClass}
                      minConfidence={0.3}
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="text-center">
                        <ImageIcon className="w-12 h-12 text-slate-600 mx-auto mb-2" />
                        <p className="text-slate-500 text-sm">이미지를 불러올 수 없습니다</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Navigation Buttons */}
              <div className="flex items-center justify-between gap-2 flex-shrink-0">
                <Button
                  onClick={() => setPreviewImageIndex(Math.max(0, previewImageIndex - 1))}
                  disabled={previewImageIndex === 0}
                  variant="outline"
                  size="sm"
                  className="flex-1 border-white/20 text-white hover:bg-slate-700"
                >
                  <ChevronLeft className="w-4 h-4 mr-1" />
                  이전
                </Button>
                <Button
                  onClick={() => setPreviewImageIndex(Math.min(datasetImages.length - 1, previewImageIndex + 1))}
                  disabled={previewImageIndex === datasetImages.length - 1}
                  variant="outline"
                  size="sm"
                  className="flex-1 border-white/20 text-white hover:bg-slate-700"
                >
                  다음
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </div>

              {/* Image Info */}
              <div className="bg-slate-800/50 rounded-lg p-3 border border-white/10 flex-shrink-0">
                <div className="text-xs text-slate-400 space-y-1">
                  <div className="flex justify-between">
                    <span>파일명:</span>
                    <span className="text-white truncate ml-2">
                      {datasetImages[previewImageIndex]?.filename || 'Unknown'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>크기:</span>
                    <span className="text-white">
                      {datasetImages[previewImageIndex]?.width} × {datasetImages[previewImageIndex]?.height}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 bg-slate-900/50 rounded-lg flex items-center justify-center border border-white/10">
              <div className="text-center">
                <ImageIcon className="w-16 h-16 text-slate-600 mx-auto mb-3" />
                <p className="text-slate-400">데이터셋을 선택하면 이미지가 표시됩니다</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Selected Components */}
        <div className="flex flex-col space-y-3 overflow-y-auto pr-2">
          <h4 className="text-white font-semibold text-sm">선택된 구성 요소</h4>

          {/* Patch Name */}
          {patchName && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent className="pt-4 pb-4">
                <div className="flex items-start gap-3">
                  <FileText className="w-6 h-6 text-green-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <h5 className="text-white font-semibold text-sm mb-1">패치 이름</h5>
                    <p className="text-slate-300 text-sm truncate">{patchName}</p>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Selected Dataset */}
          {selectedDatasetData && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent>
                <div className="flex items-start gap-3">
                  <Database className="w-6 h-6 text-blue-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <h5 className="text-white font-semibold text-sm mb-1">데이터셋</h5>
                    <p className="text-slate-300 text-sm mb-2 truncate">{selectedDatasetData.name}</p>
                    <span className="px-2 py-1 bg-blue-900/30 rounded text-xs text-slate-400">
                      {selectedDatasetData.imageCount.toLocaleString()} 이미지
                    </span>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Selected Target Class */}
          {attackConfig.targetClass && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent>
                <div className="flex items-start gap-3">
                  <Target className="w-6 h-6 text-red-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <h5 className="text-white font-semibold text-sm mb-1">타겟 클래스</h5>
                    <p className="text-slate-300 text-sm mb-2">
                      {availableClasses.find(c => c.value === attackConfig.targetClass)?.label || attackConfig.targetClass}
                    </p>
                    {availableClasses.find(c => c.value === attackConfig.targetClass)?.count && (
                      <span className="px-2 py-1 bg-red-900/30 rounded text-xs text-slate-400">
                        {availableClasses.find(c => c.value === attackConfig.targetClass)?.count}개 탐지
                      </span>
                    )}
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Selected Model */}
          {selectedModelData && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent>
                <div className="flex items-start gap-3">
                  <Sparkles className="w-6 h-6 text-purple-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <h5 className="text-white font-semibold text-sm mb-1">대상 AI 모델</h5>
                    <p className="text-slate-300 text-sm truncate">{selectedModelData.name}</p>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
                </div>
              </CardContent>
            </Card>
          )}

        </div>
      </div>

      {/* Ready to start message */}
      {patchName && selectedDataset && attackConfig.targetClass && selectedModel && (
        <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4">
          <p className="text-green-300 text-sm flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            모든 구성이 완료되었습니다. "패치 생성 시작" 버튼을 클릭하여 생성을 시작하세요.
          </p>
        </div>
      )}
    </div>
  ) : (
    // State 1: Initial - Show Guide
    <div className="h-full flex flex-col justify-center items-center space-y-6 p-8">
      {/* Welcome Message */}
      <div className="text-center space-y-4 max-w-2xl">
        <Shield className="w-20 h-20 mx-auto text-blue-400 opacity-50" />
        <h3 className="text-2xl font-bold text-white">
          2D 적대적 패치 생성
        </h3>
        <p className="text-slate-400 text-sm leading-relaxed">
          왼쪽 패널에서 데이터셋, 타겟 클래스, AI 모델을 선택하여 적대적 패치 생성을 시작하세요.
          생성된 패치는 선택한 AI 모델의 객체 탐지 성능을 저하시키도록 최적화됩니다.
        </p>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-4xl">
        <Card className="bg-slate-800/50 border-white/10">
          <CardContent className="pt-6 text-center">
            <Database className="w-10 h-10 mx-auto mb-3 text-blue-400" />
            <h4 className="text-white font-semibold mb-2">데이터셋 선택</h4>
            <p className="text-slate-400 text-xs">
              패치 생성에 사용할 이미지 데이터셋을 선택합니다. 데이터셋의 이미지들을 기반으로 패치가 최적화됩니다.
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardContent className="pt-6 text-center">
            <Target className="w-10 h-10 mx-auto mb-3 text-red-400" />
            <h4 className="text-white font-semibold mb-2">타겟 클래스</h4>
            <p className="text-slate-400 text-xs">
              숨기고자 하는 객체 클래스를 선택합니다. 패치는 이 클래스의 탐지를 방해하도록 생성됩니다.
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardContent className="pt-6 text-center">
            <Sparkles className="w-10 h-10 mx-auto mb-3 text-purple-400" />
            <h4 className="text-white font-semibold mb-2">AI 모델</h4>
            <p className="text-slate-400 text-xs">
              공격 대상이 되는 AI 모델을 선택합니다. 패치는 이 모델을 속이도록 최적화됩니다.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Guide */}
      <div className="bg-slate-800/30 border border-white/10 rounded-lg p-6 w-full max-w-2xl">
        <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
          <Eye className="w-5 h-5 text-blue-400" />
          패치 생성 프로세스
        </h4>
        <ol className="space-y-2 text-sm text-slate-300">
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">1.</span>
            <span><strong>패치 이름:</strong> 생성할 패치의 이름을 입력합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">2.</span>
            <span><strong>데이터셋 선택:</strong> 패치 생성에 사용할 이미지 데이터셋을 선택합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">3.</span>
            <span><strong>타겟 클래스:</strong> 숨기고자 하는 객체 클래스를 선택합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">4.</span>
            <span><strong>AI 모델:</strong> 공격 대상 모델을 선택합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">5.</span>
            <span><strong>미리보기:</strong> 선택된 항목을 확인합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">6.</span>
            <span><strong>패치 생성:</strong> 생성 시작 버튼을 클릭합니다</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-400 font-semibold">7.</span>
            <span><strong>결과 확인:</strong> 생성된 패치를 다운로드합니다</span>
          </li>
        </ol>
      </div>
    </div>
  )

  // Action Buttons
  const actionButtons = (
    <>
      {showPatchResult && generatedPatches.length > 0 ? (
        <div className="space-y-2">
          <Button
            onClick={async () => {
              const patch = generatedPatches[0]
              const patchId = patch?.trainingId
              const storageKey = (patch as any)?.storageKey
              if (patchId) {
                try {
                  const filename = patchName
                    ? `${patchName.replace(/[^a-zA-Z0-9가-힣_-]/g, '_')}.png`
                    : `adversarial_patch_${attackConfig.targetClass}_${Date.now()}.png`
                  await downloadPatch(patchId, filename, storageKey)
                  toast.success('패치가 다운로드되었습니다')
                } catch (error) {
                  toast.error('패치 다운로드에 실패했습니다')
                }
              }
            }}
            disabled={!generatedPatches[0]?.trainingId}
            className="w-full bg-gradient-to-r from-green-600 to-green-700"
          >
            <Download className="w-4 h-4 mr-2" />
            패치 다운로드
          </Button>

          <Button
            onClick={() => {
              setShowPatchResult(false)
              setGeneratedPatches([])
              setBatchProgress(null)
              setCurrentTrainingId(null)
              setTrainingLogs([])
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
        </div>
      ) : (
        <>
          <Button
            onClick={handleGeneratePatch}
            disabled={
              isGenerating ||
              !patchName.trim() ||
              !selectedDataset ||
              !attackConfig.targetClass ||
              !selectedModel
            }
            className="w-full bg-gradient-to-r from-blue-600 to-blue-700"
          >
            <Play className="w-4 h-4 mr-2" />
            패치 생성 시작
          </Button>
        </>
      )}
    </>
  )

  return (
    <>
      <AdversarialToolLayout
        title="2D 적대적 패치 생성"
        description="객체 탐지 AI를 속이는 적대적 패치를 생성합니다"
        icon={Shield}
        leftPanel={{
          title: "공격 설정",
          icon: Settings,
          description: "적대적 패치 파라미터 설정",
          children: leftPanelContent
        }}
        rightPanel={{
          title: isGenerating
            ? "패치 생성 진행 상황"
            : showPatchResult && generatedPatches.length > 0
              ? "생성된 패치 결과"
              : hasSelection
                ? "선택 항목 미리보기"
                : "패치 생성 안내",
          icon: isGenerating
            ? Loader2
            : showPatchResult && generatedPatches.length > 0
              ? CheckCircle2
              : hasSelection
                ? AlertCircle
                : Shield,
          children: rightPanelContent
        }}
        actionButtons={actionButtons}
      />
    </>
  )
}
