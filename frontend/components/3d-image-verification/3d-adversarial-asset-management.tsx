"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { toast } from "sonner"
import {
  Shield,
  Database,
  Search,
  Download,
  Trash2,
  Eye,
  FileStack,
  Target,
  Calendar,
  Image as ImageIcon,
  Layers,
  Settings,
  X,
  ChevronLeft,
  ChevronRight,
  Box
} from "lucide-react"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import {
  downloadPatch,
  downloadAdversarialDataset,
  getImageUrlByStorageKey,
  fetchBackendDatasets,
  type BackendDataset,
} from "@/lib/adversarial-api"

interface AdversarialPatch {
  id: string
  name: string
  targetClass: string
  datasetName: string
  createdAt: string
  thumbnailUrl?: string
  trainingId?: number
  metadata?: {
    iterations?: number
    patchSize?: number
    imagesProcessed?: number
  }
}

interface AdversarialDataset {
  id: string
  name: string
  type: "patch" | "noise"
  originalDataset: string
  createdAt: string
  imageCount: number
  outputDatasetId?: string
  output_dataset_id?: string
  parameters?: {
    output_dataset_id?: string
  }
  metadata?: {
    attackMethod?: string
    targetClass?: string
    model?: string
  }
  sampleImages?: any[]
}

interface Dataset3D {
  id: string
  name: string
  type: string
  created_at: string
  image_count?: number
  metadata?: Record<string, any>
  is_attack_output?: boolean
  sampleImages?: any[]  // Add sample images for preview
}

export function AdversarialAssetManagement3D() {
  const [activeTab, setActiveTab] = useState<"datasets" | "patches" | "attack-datasets">("datasets")
  const [patches, setPatches] = useState<AdversarialPatch[]>([])
  const [datasets, setDatasets] = useState<AdversarialDataset[]>([])
  const [datasets3D, setDatasets3D] = useState<Dataset3D[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedPatch, setSelectedPatch] = useState<AdversarialPatch | null>(null)
  const [selectedDataset, setSelectedDataset] = useState<AdversarialDataset | null>(null)
  const [selected3DDataset, setSelected3DDataset] = useState<Dataset3D | null>(null)
  const [showPatchDetails, setShowPatchDetails] = useState(false)
  const [showDatasetDetails, setShowDatasetDetails] = useState(false)
  const [show3DDatasetDetails, setShow3DDatasetDetails] = useState(false)
  const [datasetSampleImages, setDatasetSampleImages] = useState<any[]>([])
  const [loadingSampleImages, setLoadingSampleImages] = useState(false)
  const [imageCarouselPage, setImageCarouselPage] = useState(0)
  const [datasetNameMap, setDatasetNameMap] = useState<Map<string, string>>(new Map())

  const getDatasetImageCount = (dataset: Dataset3D) => {
    if (dataset.image_count && dataset.image_count > 0) {
      return dataset.image_count
    }

    const meta = dataset.metadata || {}
    const direct = meta.total_images ?? meta.processed_images ?? meta.image_count
    if (typeof direct === "number" && direct > 0) {
      return direct
    }

    const locTotal = meta.loc_total
    const batchTotal = meta.batch_total
    if (typeof locTotal === "number" && typeof batchTotal === "number") {
      return locTotal * batchTotal
    }

    return 0
  }

  useEffect(() => {
    const initializeData = async () => {
      const nameMap = await loadDatasetNames()
      await load3DDatasets()
      await loadPatches(nameMap)
      await loadAttackDatasets(nameMap)
    }
    initializeData()
  }, [])

  const loadDatasetNames = async (): Promise<Map<string, string>> => {
    try {
      const backendDatasets = await fetchBackendDatasets()
      const nameMap = new Map<string, string>()
      backendDatasets.forEach((dataset: BackendDataset) => {
        nameMap.set(dataset.id, dataset.name)
      })
      setDatasetNameMap(nameMap)
      console.log("[3D Asset] Dataset name map loaded:", nameMap)
      return nameMap
    } catch (error) {
      console.error("[3D Asset] Failed to load dataset names:", error)
      return new Map()
    }
  }

  const load3DDatasets = async () => {
    try {
      // Fetch 3D datasets using dedicated 3D API
      // exclude_attack_output=true filters out attack output datasets at backend level
      const response = await fetch('/api/carla/datasets_3d?limit=1000&exclude_attack_output=true')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      console.log('[3D Asset] Loaded 3D datasets (excluding attack outputs):', data)
      console.log('[3D Asset] Number of 3D datasets:', data?.length || 0)

      // Load sample images for each dataset (non-blocking)
      const datasetsWithImages = await Promise.all(
        data.map(async (dataset: Dataset3D) => {
          let sampleImages: any[] = []
          try {
            const imgResponse = await fetch(`/api/carla/datasets_3d/${dataset.id}/images?limit=3`)
            if (imgResponse.ok) {
              const imgData = await imgResponse.json()
              sampleImages = imgData.images || []
              console.log(`[3D Asset load3DDatasets] Loaded ${sampleImages.length} sample images for dataset ${dataset.id}`)
            }
          } catch (error) {
            console.error(`[3D Asset] Failed to load sample images for dataset ${dataset.id}:`, error)
          }
          return {
            ...dataset,
            sampleImages
          }
        })
      )

      setDatasets3D(datasetsWithImages || [])
    } catch (error) {
      console.error("[3D Asset] Failed to load 3D datasets:", error)
      toast.error("3D 데이터셋 목록을 불러오는데 실패했습니다")
    }
  }

  // Load sample images when dataset details modal opens
  useEffect(() => {
    const loadSampleImages = async () => {
      if (selectedDataset && showDatasetDetails) {
        setLoadingSampleImages(true)
        setImageCarouselPage(0)
        try {
          const datasetId = selectedDataset.output_dataset_id ||
                           selectedDataset.outputDatasetId
          if (datasetId) {
            const limit = selectedDataset.imageCount || 10000
            console.log(`[3D Asset loadSampleImages] Loading ${limit} images for dataset ${datasetId}`)
            const response = await fetch(`/api/carla/datasets_3d/${datasetId}/images?limit=${limit}`)
            if (response.ok) {
              const data = await response.json()
              const images = data.images || []
              console.log(`[3D Asset loadSampleImages] Loaded ${images.length} images`)
              setDatasetSampleImages(images)
            } else {
              console.error("[3D Asset] Failed to load images:", response.status)
              setDatasetSampleImages([])
            }
          } else {
            console.warn("[3D Asset] No output dataset ID found for dataset:", selectedDataset.id)
            setDatasetSampleImages([])
          }
        } catch (error) {
          console.error("[3D Asset] Failed to load sample images:", error)
          setDatasetSampleImages([])
        } finally {
          setLoadingSampleImages(false)
        }
      } else {
        setDatasetSampleImages([])
      }
    }

    loadSampleImages()
  }, [selectedDataset, showDatasetDetails])

  // Load sample images for 3D datasets
  useEffect(() => {
    const load3DDatasetImages = async () => {
      if (selected3DDataset && show3DDatasetDetails) {
        setLoadingSampleImages(true)
        setImageCarouselPage(0)
        try {
          const limit = getDatasetImageCount(selected3DDataset) || 10000
          console.log(`[3D Asset load3DDatasetImages] Loading ${limit} images for dataset ${selected3DDataset.id}`)
          const response = await fetch(`/api/carla/datasets_3d/${selected3DDataset.id}/images?limit=${limit}`)
          if (response.ok) {
            const data = await response.json()
            const images = data.images || []
            console.log(`[3D Asset load3DDatasetImages] Loaded ${images.length} images`)
            setDatasetSampleImages(images)
          } else {
            console.error("[3D Asset] Failed to load 3D dataset images:", response.status)
            setDatasetSampleImages([])
          }
        } catch (error) {
          console.error("[3D Asset] Failed to load 3D dataset images:", error)
          setDatasetSampleImages([])
        } finally {
          setLoadingSampleImages(false)
        }
      } else if (!selected3DDataset) {
        setDatasetSampleImages([])
      }
    }

    load3DDatasetImages()
  }, [selected3DDataset, show3DDatasetDetails])

  const loadPatches = async (nameMap?: Map<string, string>) => {
    try {
      // Use dedicated 3D patches API
      const response = await fetch('/api/carla/patches_3d?limit=1000')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const patchesData = await response.json()
      const mapToUse = nameMap || datasetNameMap

      // Transform API response
      const transformedPatches: AdversarialPatch[] = patchesData.map((p: any) => ({
        id: p.id,
        name: p.name,
        targetClass: p.target_class || "Unknown",
        datasetName: mapToUse.get(p.source_dataset_id) || p.source_dataset_id || "Unknown Dataset",
        createdAt: p.created_at,
        thumbnailUrl: p.storage_key ? getImageUrlByStorageKey(p.storage_key) : undefined,
        metadata: {
          iterations: p.patch_metadata?.iterations,
          patchSize: p.patch_metadata?.patch_size,
          imagesProcessed: p.patch_metadata?.num_training_samples
        }
      }))

      setPatches(transformedPatches)
    } catch (error) {
      console.error("[3D Asset] Failed to load patches:", error)
      toast.error("패치 목록을 불러오는데 실패했습니다")
    }
  }

  const loadAttackDatasets = async (nameMap?: Map<string, string>) => {
    try {
      // Use dedicated 3D attack datasets API
      const response = await fetch('/api/carla/attack_datasets_3d?limit=1000')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const datasetsData = await response.json()

      console.log('[3D Asset] Attack datasets API response:', datasetsData)
      console.log('[3D Asset] Number of attack datasets:', datasetsData?.length || 0)

      const mapToUse = nameMap || datasetNameMap

      // Transform API response
      const transformedDatasets: AdversarialDataset[] = await Promise.all(
        datasetsData.map(async (d: any) => {
          const outputDatasetId = d.output_dataset_id

          console.log(`[3D Asset loadAttackDatasets] Processing attack dataset:`, {
            id: d.id,
            name: d.name,
            attack_type: d.attack_type,
            output_dataset_id: outputDatasetId,
            parameters: d.parameters
          })

          // Load sample images for each dataset
          let sampleImages: any[] = []
          if (outputDatasetId) {
            try {
              const imgResponse = await fetch(`/api/carla/datasets_3d/${outputDatasetId}/images?limit=3`)
              if (imgResponse.ok) {
                const imgData = await imgResponse.json()
                sampleImages = imgData.images || []
                console.log(`[3D Asset loadAttackDatasets] Loaded ${sampleImages.length} sample images for output dataset ${outputDatasetId}`)
              }
            } catch (error) {
              console.error(`[3D Asset] Failed to load sample images for dataset ${outputDatasetId}:`, error)
            }
          } else {
            console.warn(`[3D Asset loadAttackDatasets] No output_dataset_id found for attack dataset ${d.id}`)
          }

          // Get original dataset name from output_dataset_id
          // Note: For 3D attack datasets, we use output_dataset name as reference
          const originalDatasetName = outputDatasetId
            ? (mapToUse.get(outputDatasetId) || outputDatasetId)
            : "Unknown Dataset"

          return {
            id: d.id,
            name: d.name,
            type: d.attack_type,
            originalDataset: originalDatasetName,  // Fixed: Use output_dataset name for display
            outputDatasetId,
            createdAt: d.created_at,
            // Use image_count from backend (which is calculated from actual Image3D records)
            // Fallback to parameters.processed_images if not available
            imageCount: d.image_count ?? d.parameters?.processed_images ?? 0,
            metadata: {
              attackMethod: d.attack_type === "patch" ? "Adversarial Patch" : "Noise Attack",
              targetClass: d.target_class,
              model: d.target_model_id
            },
            sampleImages
          }
        })
      )

      console.log('[3D Asset] Transformed attack datasets:', transformedDatasets)
      setDatasets(transformedDatasets)
    } catch (error) {
      console.error("[3D Asset] Failed to load attack datasets:", error)
      toast.error("적대적 공격 데이터셋 목록을 불러오는데 실패했습니다")
    }
  }

  const handleDeletePatch = async (patchId: string) => {
    try {
      const response = await fetch(`/api/carla/patches_3d/${patchId}`, {
        method: 'DELETE'
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      setPatches(patches.filter(p => p.id !== patchId))
      toast.success("패치가 삭제되었습니다")
    } catch (error) {
      console.error("[3D Asset] Failed to delete patch:", error)
      toast.error("패치 삭제에 실패했습니다")
    }
  }

  const handleDeleteDataset = async (datasetId: string) => {
    try {
      const response = await fetch(`/api/carla/attack_datasets_3d/${datasetId}`, {
        method: 'DELETE'
      })
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      setDatasets(datasets.filter(d => d.id !== datasetId))
      toast.success("데이터셋이 삭제되었습니다")
    } catch (error) {
      console.error("[3D Asset] Failed to delete dataset:", error)
      toast.error("데이터셋 삭제에 실패했습니다")
    }
  }

  const handleDownloadDataset3D = async (datasetId: string) => {
    try {
      const response = await fetch(`/api/carla/datasets_3d/${datasetId}/download`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      // Download the file
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url

      // Get filename from Content-Disposition header or use default
      const contentDisposition = response.headers.get('Content-Disposition')
      const filename = contentDisposition
        ? contentDisposition.split('filename=')[1]?.replace(/"/g, '')
        : `dataset_${datasetId}.zip`

      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)

      toast.success("3D 데이터셋 다운로드가 시작되었습니다")
    } catch (error) {
      console.error("[3D Asset] Failed to download 3D dataset:", error)
      toast.error("3D 데이터셋 다운로드에 실패했습니다")
    }
  }

  const handleDelete3DDataset = async (datasetId: string) => {
    try {
      const response = await fetch(`/api/carla/datasets_3d/${datasetId}`, {
        method: 'DELETE'
      })
      if (!response.ok && response.status !== 204) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      setDatasets3D(prev => prev.filter(d => d.id !== datasetId))
      toast.success("데이터셋이 삭제되었습니다")
    } catch (error) {
      console.error("[3D Asset] Failed to delete 3D dataset:", error)
      toast.error("3D 데이터셋 삭제에 실패했습니다")
    }
  }

  const handleDownloadPatch = async (patch: AdversarialPatch) => {
    try {
      // Use 3D patch download API
      const url = `/api/carla/patches_3d/${patch.id}/download`
      const link = document.createElement('a')
      link.href = url
      link.download = `${patch.name}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      toast.success(`${patch.name} 다운로드 시작`)
    } catch (error) {
      console.error("[3D Asset] Failed to download patch:", error)
      toast.error("패치 다운로드에 실패했습니다")
    }
  }

  // NOTE: Attack output datasets are now filtered at the backend level (exclude_attack_output=true by default)
  // This frontend filtering is kept as a defensive measure for backward compatibility
  const attackOutputIds = new Set(
    datasets
      .map(d => d.outputDatasetId || d.output_dataset_id)
      .filter((id): id is string => Boolean(id))
  )

  const visible3DDatasets = datasets3D.filter(dataset =>
    !dataset.is_attack_output && !attackOutputIds.has(dataset.id)
  )

  const filtered3DDatasets = visible3DDatasets.filter(dataset => {
    const matchesName = dataset.name.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesName
  })

  const filteredPatches = patches.filter(patch => {
    const matchesName = patch.name.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesName
  })

  const filteredAttackDatasets = datasets.filter(dataset => {
    const matchesName = dataset.name.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesName
  })

  const currentItems = activeTab === "datasets"
    ? filtered3DDatasets
    : activeTab === "patches"
    ? filteredPatches
    : filteredAttackDatasets

  const totalItems = activeTab === "datasets"
    ? visible3DDatasets.length
    : activeTab === "patches"
    ? patches.length
    : datasets.length

  // Left Panel - Filters and Actions
  const leftPanelContent = (
    <div className="space-y-4">
      {/* Search */}
      <div>
        <label className="text-xs sm:text-sm font-medium text-muted mb-2 block">이름 검색</label>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 sm:w-4 sm:h-4 text-muted" />
          <Input
            type="text"
            placeholder="자산 이름..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 sm:pl-10 pr-8 text-sm bg-surface-container border-outline text-foreground placeholder:text-muted"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-surface-container-high rounded transition-colors"
            >
              <X className="w-3 h-3 text-muted" />
            </button>
          )}
        </div>
      </div>

      {/* Tab Selection */}
      <div>
        <label className="text-xs sm:text-sm font-medium text-muted mb-2 block">자산 유형</label>
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "datasets" | "patches" | "attack-datasets")}>
          <TabsList className="flex flex-col md:grid md:grid-cols-3 gap-1 bg-surface-container-low p-1 h-auto w-full">
            <TabsTrigger value="datasets" className="h-8 px-2 py-1 w-full justify-center">
              <Box className="w-3 h-3 mr-1 flex-shrink-0" />
              <span className="text-xs truncate">3D 데이터셋</span>
            </TabsTrigger>
            <TabsTrigger value="patches" className="h-8 px-2 py-1 w-full justify-center">
              <Shield className="w-3 h-3 mr-1 flex-shrink-0" />
              <span className="text-xs truncate">적대적 패치</span>
            </TabsTrigger>
            <TabsTrigger value="attack-datasets" className="h-8 px-2 py-1 w-full justify-center">
              <Database className="w-3 h-3 mr-1 flex-shrink-0" />
              <span className="text-xs truncate">적대적 공격 데이터</span>
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Filter Reset */}
      {searchQuery && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            setSearchQuery("")
          }}
          className="w-full border-outline hover:bg-surface-container-high"
        >
          <X className="w-3 h-3 mr-2" />
          필터 초기화
        </Button>
      )}

      {/* Statistics */}
      <div className="space-y-3 pt-2">
        <div className="bg-surface-container rounded-lg p-3 border border-outline">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-muted">전체</span>
            <span className="text-sm font-bold text-secondary">{totalItems}개</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted">필터 결과</span>
            <span className="text-sm font-bold text-primary">{currentItems.length}개</span>
          </div>
        </div>

        {activeTab === "datasets" && (
          <div className="bg-surface-container rounded-lg p-3 border border-outline">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted">총 이미지</span>
              <span className="text-sm font-bold text-secondary">
                {visible3DDatasets.reduce((sum, d) => sum + getDatasetImageCount(d), 0).toLocaleString()}개
              </span>
            </div>
          </div>
        )}

        {activeTab === "patches" && (
          <>
            <div className="bg-surface-container rounded-lg p-3 border border-outline">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted">총 처리 이미지</span>
                <span className="text-sm font-bold text-secondary">
                  {patches.reduce((sum, p) => sum + (p.metadata?.imagesProcessed || 0), 0).toLocaleString()}개
                </span>
              </div>
            </div>
            <div className="bg-surface-container rounded-lg p-3 border border-outline">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted">대상 클래스</span>
                <span className="text-sm font-bold text-secondary">
                  {new Set(patches.map(p => p.targetClass)).size}개
                </span>
              </div>
            </div>
          </>
        )}

        {activeTab === "attack-datasets" && (
          <>
            <div className="bg-surface-container rounded-lg p-3 border border-outline">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted">총 이미지</span>
                <span className="text-sm font-bold text-secondary">
                  {datasets.reduce((sum, d) => sum + d.imageCount, 0).toLocaleString()}개
                </span>
              </div>
            </div>
            <div className="bg-surface-container rounded-lg p-3 border border-outline">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted">평균 이미지</span>
                <span className="text-sm font-bold text-secondary">
                  {datasets.length > 0
                    ? Math.round(datasets.reduce((sum, d) => sum + d.imageCount, 0) / datasets.length)
                    : 0}개
                </span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )

  // Remove _attack_{timestamp} suffix from attack dataset names
  const getDisplayName = (name: string) => {
    // Remove _attack_YYYYMMDD_HHMMSS pattern from the end
    return name.replace(/_attack_\d{8}_\d{6}$/, '')
  }

  // Render 3D dataset cards
  const render3DDatasetCards = (datasetsList: Dataset3D[]) => (
    datasetsList.length === 0 ? (
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <div className="text-center">
          <Box className="w-16 h-16 text-muted mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-foreground mb-2">
            {searchQuery ? "검색 결과가 없습니다" : "생성된 3D 데이터셋이 없습니다"}
          </h3>
          <p className="text-muted mb-4">
            {searchQuery ? "다른 검색어를 시도해보세요" : "새로운 3D 데이터셋을 생성하세요"}
          </p>
        </div>
      </div>
    ) : (
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {datasetsList.map((dataset) => (
          <Card key={dataset.id} className="bg-surface-container border-outline-variant hover:border-primary transition-colors">
            <CardHeader className="pb-3">
              <CardTitle className="text-primary mb-2 truncate" title={dataset.name}>
                {dataset.name}
              </CardTitle>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="bg-primary-container text-on-primary-container border-primary text-xs">
                  <Box className="w-3 h-3 mr-1" />
                  3D
                </Badge>
                <Badge variant="outline" className="bg-surface-container-high/50 text-muted border-outline text-[10px] sm:text-xs">
                  <Calendar className="w-3 h-3 mr-1" />
                  {new Date(dataset.created_at).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })}
                </Badge>
              </div>
            </CardHeader>

            <CardContent className="space-y-3">
              {/* Sample Images Preview */}
              {dataset.sampleImages && dataset.sampleImages.length > 0 && (
                <div className="space-y-2">
                  <span className="text-xs text-muted">샘플 이미지</span>
                  <div className="grid grid-cols-3 gap-1.5">
                    {dataset.sampleImages.slice(0, 3).map((img, i) => (
                      <div
                        key={i}
                        className="relative aspect-square rounded-md overflow-hidden bg-surface-container border border-outline-variant"
                      >
                        <img
                          src={`/api/storage/${img.storage_key}`}
                          alt={`Sample ${i + 1}`}
                          onError={(e) => {
                            e.currentTarget.src = '/placeholder.png'
                          }}
                          className="w-full h-full object-cover hover:scale-110 transition-transform duration-200"
                          loading="lazy"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Metadata */}
              <div className="text-xs space-y-1.5 bg-surface-container rounded-lg p-2.5 border border-outline-variant">
                <div className="flex justify-between">
                  <span className="text-muted">데이터셋 타입</span>
                  <span className="text-foreground font-medium">{dataset.type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">이미지 수</span>
                  <span className="text-foreground font-medium">{getDatasetImageCount(dataset)}개</span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="grid grid-cols-2 gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setSelected3DDataset(dataset)
                    setShow3DDatasetDetails(true)
                  }}
                  className="border-outline hover:bg-surface-container-high"
                >
                  <Eye className="w-3 h-3 mr-1" />
                  보기
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleDownloadDataset3D(dataset.id)}
                  className="border-outline hover:bg-surface-container-high"
                >
                  <Download className="w-3 h-3 mr-1" />
                  다운로드
                </Button>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handleDelete3DDataset(dataset.id)}
                className="w-full text-error hover:text-error hover:bg-error-container"
              >
                <Trash2 className="w-3 h-3 mr-1" />
                삭제
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  )

  // Render attack dataset cards
  const renderAttackDatasetCards = (datasetsList: AdversarialDataset[]) => (
    datasetsList.length === 0 ? (
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <div className="text-center">
          <Database className="w-16 h-16 text-muted mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-foreground mb-2">
            {searchQuery ? "검색 결과가 없습니다" : "생성된 적대적 공격 데이터가 없습니다"}
          </h3>
          <p className="text-muted mb-4">
            {searchQuery ? "다른 검색어를 시도해보세요" : "새로운 적대적 공격 데이터를 생성하세요"}
          </p>
        </div>
      </div>
    ) : (
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {datasetsList.map((dataset) => (
          <Card key={dataset.id} className="bg-surface-container border-outline-variant hover:border-primary transition-colors">
            <CardHeader className="pb-3">
              <CardTitle className="text-primary mb-2 truncate" title={getDisplayName(dataset.name)}>
                {getDisplayName(dataset.name)}
              </CardTitle>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className={
                  dataset.type === "patch"
                    ? "bg-primary-container text-on-primary-container border-primary text-xs"
                    : "bg-tertiary-container text-on-tertiary-container border-tertiary text-xs"
                }>
                  {dataset.type === "patch" ? <Shield className="w-3 h-3 mr-1" /> : <FileStack className="w-3 h-3 mr-1" />}
                  {dataset.type === "patch" ? "패치" : "노이즈"}
                </Badge>
                <Badge variant="outline" className="bg-surface-container-high/50 text-muted border-outline text-[10px] sm:text-xs">
                  <Calendar className="w-3 h-3 mr-1" />
                  {new Date(dataset.createdAt).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })}
                </Badge>
              </div>
            </CardHeader>

            <CardContent className="space-y-3">
              {/* Sample Images Preview */}
              <div className="grid grid-cols-3 gap-2">
                {dataset.sampleImages && dataset.sampleImages.length > 0 ? (
                  dataset.sampleImages.slice(0, 3).map((img, i) => {
                    const imageUrl = img.storage_key ? getImageUrlByStorageKey(img.storage_key) : ''
                    return (
                      <div key={img.id || i} className="aspect-square bg-surface-container rounded-lg flex items-center justify-center border border-outline overflow-hidden">
                        {imageUrl ? (
                          <img
                            src={imageUrl}
                            alt={img.filename || img.file_name || `Sample ${i + 1}`}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              console.error('[3D Asset Image Error] Failed to load:', imageUrl, 'for image:', img)
                              e.currentTarget.style.display = 'none'
                              const parent = e.currentTarget.parentElement
                              if (parent) {
                                parent.innerHTML = '<svg class="w-6 h-6 text-muted" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>'
                              }
                            }}
                          />
                        ) : (
                          <ImageIcon className="w-6 h-6 text-muted" />
                        )}
                      </div>
                    )
                  })
                ) : (
                  [1, 2, 3].map((i) => (
                    <div key={i} className="aspect-square bg-surface-container rounded-lg flex items-center justify-center border border-outline">
                      <ImageIcon className="w-6 h-6 text-muted" />
                    </div>
                  ))
                )}
              </div>

              {/* Metadata */}
              <div className="text-xs space-y-1.5 bg-surface-container rounded-lg p-2.5 border border-outline-variant">
                <div className="flex justify-between">
                  <span className="text-muted">원본 데이터셋</span>
                  <span className="text-foreground font-medium truncate ml-2" title={dataset.originalDataset}>
                    {dataset.originalDataset}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">이미지 수</span>
                  <span className="text-foreground font-medium">{dataset.imageCount}개</span>
                </div>
                {dataset.metadata?.attackMethod && (
                  <div className="flex justify-between">
                    <span className="text-muted">공격 방법</span>
                    <span className="text-foreground font-medium">{dataset.metadata.attackMethod}</span>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setSelectedDataset(dataset)
                      setShowDatasetDetails(true)
                    }}
                    className="border-outline hover:bg-surface-container-high"
                  >
                    <Eye className="w-3 h-3 mr-1" />
                    보기
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={async () => {
                      try {
                        await downloadAdversarialDataset(dataset.id, dataset.name)
                        toast.success(`${dataset.name} 다운로드 시작`)
                      } catch (error) {
                        console.error("[3D Asset] Failed to download dataset:", error)
                        toast.error("데이터셋 다운로드에 실패했습니다")
                      }
                    }}
                    className="border-outline hover:bg-surface-container-high"
                  >
                    <Download className="w-3 h-3 mr-1" />
                    다운로드
                  </Button>
                </div>

                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleDeleteDataset(dataset.id)}
                  className="w-full text-error hover:text-error hover:bg-error-container"
                >
                  <Trash2 className="w-3 h-3 mr-1" />
                  삭제
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  )

  // Right Panel - Asset Cards
  const rightPanelContent = (
    <div>
      {activeTab === "datasets" ? (
        render3DDatasetCards(filtered3DDatasets)
      ) : activeTab === "patches" ? (
        filteredPatches.length === 0 ? (
          <div className="flex items-center justify-center h-full min-h-[400px]">
            <div className="text-center">
              <Shield className="w-16 h-16 text-muted mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">
                {searchQuery ? "검색 결과가 없습니다" : "생성된 패치가 없습니다"}
              </h3>
              <p className="text-muted mb-4">
                {searchQuery ? "다른 검색어를 시도해보세요" : "새로운 적대적 패치를 생성하세요"}
              </p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredPatches.map((patch) => (
              <Card key={patch.id} className="bg-surface-container border-outline-variant hover:border-primary transition-colors">
                <CardHeader className="pb-3">
                  <CardTitle className="text-primary mb-2 truncate" title={patch.name}>
                    {patch.name}
                  </CardTitle>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline" className="bg-primary-container text-on-primary-container border-primary text-[10px] sm:text-xs">
                      <Target className="w-3 h-3 mr-1" />
                      {patch.targetClass}
                    </Badge>
                    <Badge variant="outline" className="bg-surface-container-high text-muted border-outline text-[10px] sm:text-xs">
                      <Calendar className="w-3 h-3 mr-1" />
                      {new Date(patch.createdAt).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })}
                    </Badge>
                  </div>
                </CardHeader>

                <CardContent className="space-y-3">
                  {/* Patch Preview */}
                  <div className="aspect-square bg-surface-container rounded-lg flex items-center justify-center border border-outline overflow-hidden relative">
                    {patch.thumbnailUrl ? (
                      <img
                        src={patch.thumbnailUrl}
                        alt={patch.name}
                        className="w-full h-full object-contain"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none'
                          const parent = e.currentTarget.parentElement
                          if (parent) {
                            const placeholder = parent.querySelector('.placeholder')
                            if (placeholder) {
                              (placeholder as HTMLElement).style.display = 'block'
                            }
                          }
                        }}
                      />
                    ) : null}
                    <div className="placeholder text-center absolute inset-0 flex items-center justify-center" style={{ display: patch.thumbnailUrl ? 'none' : 'flex' }}>
                      <div>
                        <Shield className="w-12 h-12 text-muted mx-auto mb-2" />
                        <p className="text-xs text-muted">패치 미리보기</p>
                      </div>
                    </div>
                  </div>

                  {/* Metadata */}
                  <div className="text-xs space-y-1.5 bg-surface-container rounded-lg p-2.5 border border-outline-variant">
                    <div className="flex justify-between">
                      <span className="text-muted">원본 데이터셋</span>
                      <span className="text-foreground font-medium truncate ml-2" title={patch.datasetName}>
                        {patch.datasetName}
                      </span>
                    </div>
                    {patch.metadata?.imagesProcessed && (
                      <div className="flex justify-between">
                        <span className="text-muted">처리 이미지</span>
                        <span className="text-foreground font-medium">{patch.metadata.imagesProcessed}개</span>
                      </div>
                    )}
                    {patch.metadata?.iterations && (
                      <div className="flex justify-between">
                        <span className="text-muted">반복 횟수</span>
                        <span className="text-foreground font-medium">{patch.metadata.iterations}</span>
                      </div>
                    )}
                  </div>

                  {/* Action Buttons */}
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          setSelectedPatch(patch)
                          setShowPatchDetails(true)
                        }}
                        className="border-outline hover:bg-surface-container-high"
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        보기
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDownloadPatch(patch)}
                        className="border-outline hover:bg-surface-container-high"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        다운로드
                      </Button>
                    </div>

                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDeletePatch(patch.id)}
                      className="w-full text-error hover:text-error hover:bg-error-container"
                    >
                      <Trash2 className="w-3 h-3 mr-1" />
                      삭제
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )
      ) : (
        renderAttackDatasetCards(filteredAttackDatasets)
      )}
    </div>
  )

  const actionButtons = null

  return (
    <>
      <AdversarialToolLayout
        title="3D 적대적 자산 관리"
        description="생성된 3D 데이터셋, 적대적 패치 및 공격 데이터를 관리합니다"
        icon={Layers}
        leftPanel={{
          title: "필터 및 설정",
          icon: Settings,
          description: "자산을 검색하고 필터링합니다",
          children: leftPanelContent
        }}
        rightPanel={{
          title: activeTab === "datasets"
            ? "3D 데이터셋"
            : activeTab === "patches"
            ? "적대적 패치"
            : "적대적 공격 데이터",
          icon: activeTab === "datasets"
            ? Box
            : activeTab === "patches"
            ? Shield
            : Database,
          description: activeTab === "datasets"
            ? `총 ${filtered3DDatasets.length}개의 3D 데이터셋`
            : activeTab === "patches"
            ? `총 ${filteredPatches.length}개의 패치`
            : `총 ${filteredAttackDatasets.length}개의 적대적 공격 데이터`,
          children: rightPanelContent
        }}
        actionButtons={actionButtons}
      />

      {/* 3D Dataset Details Modal */}
      <Dialog open={show3DDatasetDetails} onOpenChange={setShow3DDatasetDetails}>
        <DialogContent className="sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-foreground">
              <Box className="w-5 h-5" />
              3D 데이터셋 상세 정보
            </DialogTitle>
            <DialogDescription className="text-muted">
              생성된 3D 데이터셋의 상세 정보입니다
            </DialogDescription>
          </DialogHeader>

          {selected3DDataset && (
            <div className="space-y-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-xs text-muted">데이터셋 이름</p>
                  <p className="text-sm text-foreground font-medium">{selected3DDataset.name}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">타입</p>
                  <Badge className="bg-primary-container text-on-primary-container border-primary">
                    <Box className="w-3 h-3 mr-1" />
                    {selected3DDataset.type}
                  </Badge>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">이미지 수</p>
                  <p className="text-sm text-foreground font-medium">{getDatasetImageCount(selected3DDataset)}개</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">생성 일시</p>
                  <p className="text-sm text-foreground font-medium">
                    {new Date(selected3DDataset.created_at).toLocaleString('ko-KR')}
                  </p>
                </div>
              </div>

              {/* Image Carousel */}
              <div className="border-t border-border pt-4">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-sm text-muted">샘플 이미지</p>
                  {datasetSampleImages.length > 8 && (
                    <p className="text-xs text-muted">
                      {imageCarouselPage + 1} / {Math.ceil(datasetSampleImages.length / 8)}
                    </p>
                  )}
                </div>

                {loadingSampleImages ? (
                  <div className="text-center py-8">
                    <p className="text-muted text-sm">이미지 로딩 중...</p>
                  </div>
                ) : (
                  <div className="relative">
                    {/* Image Grid (4 columns x 2 rows = 8 images) */}
                    <div className="grid grid-cols-4 gap-2 mb-3">
                      {datasetSampleImages.length > 0 ? (
                        datasetSampleImages.slice(imageCarouselPage * 8, (imageCarouselPage + 1) * 8).map((img, i) => {
                          const imageUrl = img.storage_key ? getImageUrlByStorageKey(img.storage_key) : ''
                          return (
                            <div key={img.id || i} className="aspect-square bg-surface-container rounded-lg flex items-center justify-center border border-outline overflow-hidden">
                              {imageUrl ? (
                                <img
                                  src={imageUrl}
                                  alt={img.filename || img.file_name || `Sample ${i + 1}`}
                                  className="w-full h-full object-cover"
                                  onError={(e) => {
                                    console.error('[3D Dataset Modal Image Error] Failed to load:', imageUrl, 'for image:', img)
                                    e.currentTarget.style.display = 'none'
                                    const parent = e.currentTarget.parentElement
                                    if (parent) {
                                      const icon = document.createElement('div')
                                      icon.className = 'w-8 h-8 text-muted'
                                      icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>'
                                      parent.appendChild(icon)
                                    }
                                  }}
                                />
                              ) : (
                                <ImageIcon className="w-8 h-8 text-muted" />
                              )}
                            </div>
                          )
                        })
                      ) : (
                        [1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                          <div key={i} className="aspect-square bg-surface-container rounded-lg flex items-center justify-center border border-outline">
                            <ImageIcon className="w-8 h-8 text-muted" />
                          </div>
                        ))
                      )}
                    </div>

                    {/* Navigation Buttons */}
                    {datasetSampleImages.length > 8 && (
                      <div className="flex justify-center gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setImageCarouselPage(Math.max(0, imageCarouselPage - 1))}
                          disabled={imageCarouselPage === 0}
                          className="border-outline hover:bg-surface-container-high"
                        >
                          <ChevronLeft className="w-4 h-4 mr-1" />
                          이전
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setImageCarouselPage(Math.min(Math.ceil(datasetSampleImages.length / 8) - 1, imageCarouselPage + 1))}
                          disabled={imageCarouselPage >= Math.ceil(datasetSampleImages.length / 8) - 1}
                          className="border-outline hover:bg-surface-container-high"
                        >
                          다음
                          <ChevronRight className="w-4 h-4 ml-1" />
                        </Button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          <DialogFooter className="gap-2">
            <Button onClick={() => setShow3DDatasetDetails(false)} variant="outline" className="border-outline">
              닫기
            </Button>
            {selected3DDataset && (
              <Button
                onClick={async () => {
                  try {
                    const url = `/api/carla/datasets_3d/${selected3DDataset.id}/download`
                    const link = document.createElement('a')
                    link.href = url
                    link.download = `${selected3DDataset.name}.zip`
                    document.body.appendChild(link)
                    link.click()
                    document.body.removeChild(link)
                    toast.success(`${selected3DDataset.name} 다운로드 시작`)
                  } catch (error) {
                    console.error("[3D Dataset] Failed to download:", error)
                    toast.error("데이터셋 다운로드에 실패했습니다")
                  }
                }}
                variant="default"
              >
                <Download className="w-4 h-4 mr-2" />
                다운로드
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Patch Details Modal */}
      <Dialog open={showPatchDetails} onOpenChange={setShowPatchDetails}>
        <DialogContent className="sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-foreground">
              <Shield className="w-5 h-5" />
              패치 상세 정보
            </DialogTitle>
            <DialogDescription className="text-muted">
              생성된 적대적 패치의 상세 정보입니다
            </DialogDescription>
          </DialogHeader>

          {selectedPatch && (
            <div className="space-y-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-xs text-muted">패치 이름</p>
                  <p className="text-sm text-foreground font-medium">{selectedPatch.name}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">대상 클래스</p>
                  <Badge className="bg-primary-container text-on-primary-container border-primary">
                    <Target className="w-3 h-3 mr-1" />
                    {selectedPatch.targetClass}
                  </Badge>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">원본 데이터셋</p>
                  <p className="text-sm text-foreground font-medium">{selectedPatch.datasetName}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">생성 일시</p>
                  <p className="text-sm text-foreground font-medium">
                    {new Date(selectedPatch.createdAt).toLocaleString('ko-KR')}
                  </p>
                </div>
                {selectedPatch.metadata?.imagesProcessed && (
                  <div className="space-y-1">
                    <p className="text-xs text-muted">처리된 이미지</p>
                    <p className="text-sm text-foreground font-medium">{selectedPatch.metadata.imagesProcessed}개</p>
                  </div>
                )}
                {selectedPatch.metadata?.iterations && (
                  <div className="space-y-1">
                    <p className="text-xs text-muted">반복 횟수</p>
                    <p className="text-sm text-foreground font-medium">{selectedPatch.metadata.iterations}</p>
                  </div>
                )}
              </div>

              {/* Patch Preview */}
              <div className="bg-surface-container rounded-lg p-4 border border-outline-variant">
                <p className="text-sm text-muted mb-2">패치 미리보기</p>
                <div className="aspect-square max-w-md mx-auto bg-surface-container rounded-lg flex items-center justify-center border border-outline overflow-hidden relative">
                  {selectedPatch.thumbnailUrl ? (
                    <img
                      src={selectedPatch.thumbnailUrl}
                      alt={selectedPatch.name}
                      className="w-full h-full object-contain"
                      onError={(e) => {
                        e.currentTarget.style.display = 'none'
                        const parent = e.currentTarget.parentElement
                        if (parent) {
                          const placeholder = parent.querySelector('.placeholder')
                          if (placeholder) {
                            (placeholder as HTMLElement).style.display = 'block'
                          }
                        }
                      }}
                    />
                  ) : null}
                  <div className="placeholder text-center absolute inset-0 flex items-center justify-center" style={{ display: selectedPatch.thumbnailUrl ? 'none' : 'flex' }}>
                    <div>
                      <Shield className="w-16 h-16 text-muted mx-auto mb-2" />
                      <p className="text-muted">미리보기를 불러올 수 없습니다</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <DialogFooter className="gap-2">
            <Button onClick={() => setShowPatchDetails(false)} variant="outline" className="border-outline">
              닫기
            </Button>
            {selectedPatch && (
              <Button
                onClick={() => handleDownloadPatch(selectedPatch)}
                variant="default"
              >
                <Download className="w-4 h-4 mr-2" />
                다운로드
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Dataset Details Modal */}
      <Dialog open={showDatasetDetails} onOpenChange={setShowDatasetDetails}>
        <DialogContent className="sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-foreground">
              <Database className="w-5 h-5" />
              적대적 공격 데이터 상세 정보
            </DialogTitle>
            <DialogDescription className="text-muted">
              생성된 적대적 공격 데이터의 상세 정보입니다
            </DialogDescription>
          </DialogHeader>

          {selectedDataset && (
            <div className="space-y-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-xs text-muted">데이터셋 이름</p>
                  <p className="text-sm text-foreground font-medium">{selectedDataset.name}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">공격 유형</p>
                  <Badge className={
                    selectedDataset.type === "patch"
                      ? "bg-primary-container text-on-primary-container border-primary"
                      : "bg-tertiary-container text-on-tertiary-container border-tertiary"
                  }>
                    {selectedDataset.type === "patch" ? <Shield className="w-3 h-3 mr-1" /> : <FileStack className="w-3 h-3 mr-1" />}
                    {selectedDataset.type === "patch" ? "패치" : "노이즈"}
                  </Badge>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">원본 데이터셋</p>
                  <p className="text-sm text-foreground font-medium">{selectedDataset.originalDataset}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">이미지 수</p>
                  <p className="text-sm text-foreground font-medium">{selectedDataset.imageCount}개</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-muted">생성 일시</p>
                  <p className="text-sm text-foreground font-medium">
                    {new Date(selectedDataset.createdAt).toLocaleString('ko-KR')}
                  </p>
                </div>
                {selectedDataset.metadata?.attackMethod && (
                  <div className="space-y-1">
                    <p className="text-xs text-muted">공격 방법</p>
                    <p className="text-sm text-foreground font-medium">{selectedDataset.metadata.attackMethod}</p>
                  </div>
                )}
              </div>

              {/* Sample Images Grid with Carousel */}
              <div className="bg-surface-container rounded-lg p-4 border border-outline-variant">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-sm text-muted">샘플 이미지</p>
                  {datasetSampleImages.length > 8 && (
                    <p className="text-xs text-muted">
                      {imageCarouselPage + 1} / {Math.ceil(datasetSampleImages.length / 8)}
                    </p>
                  )}
                </div>

                {loadingSampleImages ? (
                  <div className="text-center py-8">
                    <p className="text-muted text-sm">이미지 로딩 중...</p>
                  </div>
                ) : (
                  <div className="relative">
                    {/* Image Grid (4 columns x 2 rows = 8 images) */}
                    <div className="grid grid-cols-4 gap-2 mb-3">
                      {datasetSampleImages.length > 0 ? (
                        datasetSampleImages.slice(imageCarouselPage * 8, (imageCarouselPage + 1) * 8).map((img, i) => {
                          const imageUrl = img.storage_key ? getImageUrlByStorageKey(img.storage_key) : ''
                          return (
                            <div key={img.id || i} className="aspect-square bg-surface-container rounded-lg flex items-center justify-center border border-outline overflow-hidden">
                              {imageUrl ? (
                                <img
                                  src={imageUrl}
                                  alt={img.filename || img.file_name || `Sample ${i + 1}`}
                                  className="w-full h-full object-cover"
                                  onError={(e) => {
                                    console.error('[3D Asset Modal Image Error] Failed to load:', imageUrl, 'for image:', img)
                                    e.currentTarget.style.display = 'none'
                                    const parent = e.currentTarget.parentElement
                                    if (parent) {
                                      const icon = document.createElement('div')
                                      icon.className = 'w-8 h-8 text-muted'
                                      icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>'
                                      parent.appendChild(icon)
                                    }
                                  }}
                                />
                              ) : (
                                <ImageIcon className="w-8 h-8 text-muted" />
                              )}
                            </div>
                          )
                        })
                      ) : (
                        [1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                          <div key={i} className="aspect-square bg-surface-container rounded-lg flex items-center justify-center border border-outline">
                            <ImageIcon className="w-8 h-8 text-muted" />
                          </div>
                        ))
                      )}
                    </div>

                    {/* Navigation Buttons */}
                    {datasetSampleImages.length > 8 && (
                      <div className="flex justify-center gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setImageCarouselPage(Math.max(0, imageCarouselPage - 1))}
                          disabled={imageCarouselPage === 0}
                          className="border-outline hover:bg-surface-container-high"
                        >
                          <ChevronLeft className="w-4 h-4 mr-1" />
                          이전
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setImageCarouselPage(Math.min(Math.ceil(datasetSampleImages.length / 8) - 1, imageCarouselPage + 1))}
                          disabled={imageCarouselPage >= Math.ceil(datasetSampleImages.length / 8) - 1}
                          className="border-outline hover:bg-surface-container-high"
                        >
                          다음
                          <ChevronRight className="w-4 h-4 ml-1" />
                        </Button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          <DialogFooter className="gap-2">
            <Button onClick={() => setShowDatasetDetails(false)} variant="outline" className="border-outline">
              닫기
            </Button>
            {selectedDataset && (
              <>
                <Button
                  onClick={async () => {
                    try {
                      await downloadAdversarialDataset(selectedDataset.id, selectedDataset.name)
                      toast.success(`${selectedDataset.name} 다운로드 시작`)
                    } catch (error) {
                      console.error("[3D Asset] Failed to download dataset:", error)
                      toast.error("데이터셋 다운로드에 실패했습니다")
                    }
                  }}
                  variant="default"
                >
                  <Download className="w-4 h-4 mr-2" />
                  다운로드
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
