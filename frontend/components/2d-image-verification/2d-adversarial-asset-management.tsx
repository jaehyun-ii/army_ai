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
  PlayCircle,
  Layers,
  Settings,
  X,
  ChevronLeft,
  ChevronRight
} from "lucide-react"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import {
  fetchPatches,
  deletePatch,
  fetchAttackDatasets,
  deleteAttackDataset,
  downloadPatch,
  downloadAdversarialDataset,
  getPatchImageUrl,
  getImageUrlByStorageKey,
  fetchDatasetImages,
  fetchBackendDatasets,
  type PatchAsset,
  type AttackDatasetAsset,
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
  outputDatasetId?: string // Added for accessing actual images
  metadata?: {
    attackMethod?: string
    targetClass?: string
    model?: string
  }
  sampleImages?: any[] // Added for caching sample images
}

export function AdversarialAssetManagement() {
  const [activeTab, setActiveTab] = useState<"patches" | "patch-datasets" | "noise-datasets">("patches")
  const [patches, setPatches] = useState<AdversarialPatch[]>([])
  const [datasets, setDatasets] = useState<AdversarialDataset[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedPatch, setSelectedPatch] = useState<AdversarialPatch | null>(null)
  const [selectedDataset, setSelectedDataset] = useState<AdversarialDataset | null>(null)
  const [showPatchDetails, setShowPatchDetails] = useState(false)
  const [showDatasetDetails, setShowDatasetDetails] = useState(false)
  const [datasetSampleImages, setDatasetSampleImages] = useState<any[]>([])
  const [loadingSampleImages, setLoadingSampleImages] = useState(false)
  const [imageCarouselPage, setImageCarouselPage] = useState(0)
  const [datasetNameMap, setDatasetNameMap] = useState<Map<string, string>>(new Map())

  useEffect(() => {
    const initializeData = async () => {
      const nameMap = await loadDatasetNames()
      await loadPatches(nameMap)
      await loadDatasets(nameMap)
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
      console.log("Dataset name map loaded:", nameMap)
      return nameMap
    } catch (error) {
      console.error("Failed to load dataset names:", error)
      return new Map()
    }
  }

  // Load sample images when dataset details modal opens
  useEffect(() => {
    const loadSampleImages = async () => {
      if (selectedDataset && showDatasetDetails) {
        setLoadingSampleImages(true)
        setImageCarouselPage(0) // Reset page when opening modal
        try {
          // Use the output dataset ID where actual images are stored
          const datasetId = selectedDataset.outputDatasetId
          if (datasetId) {
            // Load more images for pagination (8 per page, load 24 total)
            const images = await fetchDatasetImages(datasetId, 24)
            setDatasetSampleImages(images)
          } else {
            console.warn("No output dataset ID found for dataset:", selectedDataset.id)
            setDatasetSampleImages([])
          }
        } catch (error) {
          console.error("Failed to load sample images:", error)
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

  const loadPatches = async (nameMap?: Map<string, string>) => {
    try {
      const patchesData = await fetchPatches({ limit: 1000 })
      const mapToUse = nameMap || datasetNameMap

      // Transform API response to component format
      const transformedPatches: AdversarialPatch[] = patchesData.map((p: PatchAsset) => ({
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
      console.error("Failed to load patches:", error)
      toast.error("패치 목록을 불러오는데 실패했습니다")
    }
  }

  const loadDatasets = async (nameMap?: Map<string, string>) => {
    try {
      const datasetsData = await fetchAttackDatasets({ limit: 1000 })
      const mapToUse = nameMap || datasetNameMap

      // Transform API response to component format
      const transformedDatasets: AdversarialDataset[] = await Promise.all(
        datasetsData.map(async (d: AttackDatasetAsset) => {
          // Get the output dataset ID where the actual images are stored
          const outputDatasetId = d.parameters?.output_dataset_id

          // Load sample images for each dataset (non-blocking)
          let sampleImages: any[] = []
          if (outputDatasetId) {
            try {
              sampleImages = await fetchDatasetImages(outputDatasetId, 3)
            } catch (error) {
              console.error(`Failed to load sample images for dataset ${outputDatasetId}:`, error)
            }
          }

          return {
            id: d.id,
            name: d.name,
            type: d.attack_type,
            originalDataset: mapToUse.get(d.base_dataset_id) || d.base_dataset_id || "Unknown Dataset",
            outputDatasetId,
            createdAt: d.created_at,
            imageCount: d.parameters?.processed_images || 0,
            metadata: {
              attackMethod: d.attack_type === "patch" ? "Adversarial Patch" : "Noise Attack",
              targetClass: d.target_class,
              model: d.target_model_id
            },
            sampleImages
          }
        })
      )

      setDatasets(transformedDatasets)
    } catch (error) {
      console.error("Failed to load datasets:", error)
      toast.error("데이터셋 목록을 불러오는데 실패했습니다")
    }
  }

  const handleDeletePatch = async (patchId: string) => {
    try {
      await deletePatch(patchId)
      setPatches(patches.filter(p => p.id !== patchId))
      toast.success("패치가 삭제되었습니다")
    } catch (error) {
      console.error("Failed to delete patch:", error)
      toast.error("패치 삭제에 실패했습니다")
    }
  }

  const handleDeleteDataset = async (datasetId: string) => {
    try {
      await deleteAttackDataset(datasetId)
      setDatasets(datasets.filter(d => d.id !== datasetId))
      toast.success("데이터셋이 삭제되었습니다")
    } catch (error) {
      console.error("Failed to delete dataset:", error)
      toast.error("데이터셋 삭제에 실패했습니다")
    }
  }

  const handleDownloadPatch = async (patch: AdversarialPatch) => {
    try {
      await downloadPatch(patch.id, patch.name)
      toast.success(`${patch.name} 다운로드 시작`)
    } catch (error) {
      console.error("Failed to download patch:", error)
      toast.error("패치 다운로드에 실패했습니다")
    }
  }

  const handleRunEvaluation = (_dataset: AdversarialDataset) => {
    toast.info("평가 실행 페이지로 이동합니다")
  }

  const filteredPatches = patches.filter(patch => {
    const matchesName = patch.name.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesName
  })

  const filteredPatchDatasets = datasets.filter(dataset => {
    const matchesType = dataset.type === "patch"
    const matchesName = dataset.name.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesType && matchesName
  })

  const filteredNoiseDatasets = datasets.filter(dataset => {
    const matchesType = dataset.type === "noise"
    const matchesName = dataset.name.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesType && matchesName
  })

  const patchDatasets = datasets.filter(d => d.type === "patch")
  const noiseDatasets = datasets.filter(d => d.type === "noise")

  const currentItems = activeTab === "patches"
    ? filteredPatches
    : activeTab === "patch-datasets"
    ? filteredPatchDatasets
    : filteredNoiseDatasets

  const totalItems = activeTab === "patches"
    ? patches.length
    : activeTab === "patch-datasets"
    ? patchDatasets.length
    : noiseDatasets.length

  // Left Panel - Filters and Actions
  const leftPanelContent = (
    <div className="space-y-4">
      {/* Search */}
      <div>
        <label className="text-sm font-medium text-slate-300 mb-2 block">이름 검색</label>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
          <Input
            type="text"
            placeholder="자산 이름..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 pr-8 bg-slate-900/50 border-slate-700 text-white placeholder:text-slate-500"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 hover:bg-slate-700 rounded transition-colors"
            >
              <X className="w-3 h-3 text-slate-400" />
            </button>
          )}
        </div>
      </div>

      {/* Tab Selection */}
      <div>
        <label className="text-sm font-medium text-slate-300 mb-2 block">자산 유형</label>
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as "patches" | "patch-datasets" | "noise-datasets")}>
          <TabsList className="flex flex-col xl:grid xl:grid-cols-3 gap-1 bg-slate-900/50 border border-slate-700 p-1 h-auto w-full">
            <TabsTrigger value="patches" className="data-[state=active]:bg-slate-700 h-8 px-2 py-1 w-full justify-center">
              <Shield className="w-3 h-3 mr-1" />
              <span className="text-xs">적대적 패치</span>
            </TabsTrigger>
            <TabsTrigger value="patch-datasets" className="data-[state=active]:bg-slate-700 h-8 px-2 py-1 w-full justify-center">
              <Database className="w-3 h-3 mr-1" />
              <span className="text-xs">패치 데이터셋</span>
            </TabsTrigger>
            <TabsTrigger value="noise-datasets" className="data-[state=active]:bg-slate-700 h-8 px-2 py-1 w-full justify-center">
              <FileStack className="w-3 h-3 mr-1" />
              <span className="text-xs">노이즈 데이터셋</span>
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
          className="w-full border-slate-600 hover:bg-slate-700"
        >
          <X className="w-3 h-3 mr-2" />
          필터 초기화
        </Button>
      )}

      {/* Statistics */}
      <div className="space-y-3 pt-2">
        <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-slate-400">전체</span>
            <span className="text-sm font-bold text-white">{totalItems}개</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">필터 결과</span>
            <span className="text-sm font-bold text-blue-400">{currentItems.length}개</span>
          </div>
        </div>

        {activeTab === "patches" && (
          <>
            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">총 처리 이미지</span>
                <span className="text-sm font-bold text-white">
                  {patches.reduce((sum, p) => sum + (p.metadata?.imagesProcessed || 0), 0).toLocaleString()}개
                </span>
              </div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">대상 클래스</span>
                <span className="text-sm font-bold text-white">
                  {new Set(patches.map(p => p.targetClass)).size}개
                </span>
              </div>
            </div>
          </>
        )}

        {activeTab === "patch-datasets" && (
          <>
            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">총 이미지</span>
                <span className="text-sm font-bold text-white">
                  {patchDatasets.reduce((sum, d) => sum + d.imageCount, 0).toLocaleString()}개
                </span>
              </div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">평균 이미지</span>
                <span className="text-sm font-bold text-white">
                  {patchDatasets.length > 0
                    ? Math.round(patchDatasets.reduce((sum, d) => sum + d.imageCount, 0) / patchDatasets.length)
                    : 0}개
                </span>
              </div>
            </div>
          </>
        )}

        {activeTab === "noise-datasets" && (
          <>
            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">총 이미지</span>
                <span className="text-sm font-bold text-white">
                  {noiseDatasets.reduce((sum, d) => sum + d.imageCount, 0).toLocaleString()}개
                </span>
              </div>
            </div>
            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">평균 이미지</span>
                <span className="text-sm font-bold text-white">
                  {noiseDatasets.length > 0
                    ? Math.round(noiseDatasets.reduce((sum, d) => sum + d.imageCount, 0) / noiseDatasets.length)
                    : 0}개
                </span>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )

  // Render dataset cards
  const renderDatasetCards = (datasetsList: AdversarialDataset[]) => (
    datasetsList.length === 0 ? (
      <div className="flex items-center justify-center h-full min-h-[400px]">
        <div className="text-center">
          <Database className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">
            {searchQuery ? "검색 결과가 없습니다" : "생성된 데이터셋이 없습니다"}
          </h3>
          <p className="text-slate-400 mb-4">
            {searchQuery ? "다른 검색어를 시도해보세요" : "새로운 적대적 데이터셋을 생성하세요"}
          </p>
        </div>
      </div>
    ) : (
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {datasetsList.map((dataset) => (
          <Card key={dataset.id} className="bg-slate-800/50 border-white/10 hover:border-blue-500/50 transition-colors">
            <CardHeader className="pb-3">
              <CardTitle className="text-base text-white mb-2 truncate" title={dataset.name}>
                {dataset.name}
              </CardTitle>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className={
                  dataset.type === "patch"
                    ? "bg-blue-500/10 text-blue-400 border-blue-500/30 text-xs"
                    : "bg-amber-500/10 text-amber-400 border-amber-500/30 text-xs"
                }>
                  {dataset.type === "patch" ? <Shield className="w-3 h-3 mr-1" /> : <FileStack className="w-3 h-3 mr-1" />}
                  {dataset.type === "patch" ? "패치" : "노이즈"}
                </Badge>
                <Badge variant="outline" className="bg-slate-700/50 text-slate-300 border-slate-600 text-xs">
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
                      <div key={img.id || i} className="aspect-square bg-slate-900/50 rounded-lg flex items-center justify-center border border-slate-700 overflow-hidden">
                        {imageUrl ? (
                          <img
                            src={imageUrl}
                            alt={img.file_name || `Sample ${i + 1}`}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              console.error('[Image Error] Failed to load:', imageUrl, 'for image:', img)
                              e.currentTarget.style.display = 'none'
                              const parent = e.currentTarget.parentElement
                              if (parent) {
                                parent.innerHTML = '<svg class="w-6 h-6 text-slate-600" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>'
                              }
                            }}
                          />
                        ) : (
                          <ImageIcon className="w-6 h-6 text-slate-600" />
                        )}
                      </div>
                    )
                  })
                ) : (
                  [1, 2, 3].map((i) => (
                    <div key={i} className="aspect-square bg-slate-900/50 rounded-lg flex items-center justify-center border border-slate-700">
                      <ImageIcon className="w-6 h-6 text-slate-600" />
                    </div>
                  ))
                )}
              </div>

              {/* Metadata */}
              <div className="text-xs space-y-1.5 bg-slate-800/50 rounded-lg p-2.5 border border-slate-700">
                <div className="flex justify-between">
                  <span className="text-slate-400">원본 데이터셋</span>
                  <span className="text-white font-medium truncate ml-2" title={dataset.originalDataset}>
                    {dataset.originalDataset}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">이미지 수</span>
                  <span className="text-white font-medium">{dataset.imageCount}개</span>
                </div>
                {dataset.metadata?.attackMethod && (
                  <div className="flex justify-between">
                    <span className="text-slate-400">공격 방법</span>
                    <span className="text-white font-medium">{dataset.metadata.attackMethod}</span>
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
                    className="border-slate-600 hover:bg-slate-700"
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
                        console.error("Failed to download dataset:", error)
                        toast.error("데이터셋 다운로드에 실패했습니다")
                      }
                    }}
                    className="border-slate-600 hover:bg-slate-700"
                  >
                    <Download className="w-3 h-3 mr-1" />
                    다운로드
                  </Button>
                </div>

                <Button
                  size="sm"
                  onClick={() => handleRunEvaluation(dataset)}
                  className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800"
                >
                  <PlayCircle className="w-3 h-3 mr-1" />
                  평가 실행
                </Button>

                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleDeleteDataset(dataset.id)}
                  className="w-full text-red-400 hover:text-red-300 hover:bg-red-900/20"
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
      {activeTab === "patches" ? (
        filteredPatches.length === 0 ? (
          <div className="flex items-center justify-center h-full min-h-[400px]">
            <div className="text-center">
              <Shield className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">
                {searchQuery ? "검색 결과가 없습니다" : "생성된 패치가 없습니다"}
              </h3>
              <p className="text-slate-400 mb-4">
                {searchQuery ? "다른 검색어를 시도해보세요" : "새로운 적대적 패치를 생성하세요"}
              </p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredPatches.map((patch) => (
              <Card key={patch.id} className="bg-slate-800/50 border-white/10 hover:border-blue-500/50 transition-colors">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base text-white mb-2 truncate" title={patch.name}>
                    {patch.name}
                  </CardTitle>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/30 text-xs">
                      <Target className="w-3 h-3 mr-1" />
                      {patch.targetClass}
                    </Badge>
                    <Badge variant="outline" className="bg-slate-700/50 text-slate-300 border-slate-600 text-xs">
                      <Calendar className="w-3 h-3 mr-1" />
                      {new Date(patch.createdAt).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })}
                    </Badge>
                  </div>
                </CardHeader>

                <CardContent className="space-y-3">
                  {/* Patch Preview */}
                  <div className="aspect-square bg-slate-900/50 rounded-lg flex items-center justify-center border border-slate-700 overflow-hidden relative">
                    {patch.thumbnailUrl ? (
                      <img
                        src={patch.thumbnailUrl}
                        alt={patch.name}
                        className="w-full h-full object-contain"
                        onError={(e) => {
                          // Fallback to placeholder if image fails to load
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
                        <Shield className="w-12 h-12 text-slate-600 mx-auto mb-2" />
                        <p className="text-xs text-slate-500">패치 미리보기</p>
                      </div>
                    </div>
                  </div>

                  {/* Metadata */}
                  <div className="text-xs space-y-1.5 bg-slate-800/50 rounded-lg p-2.5 border border-slate-700">
                    <div className="flex justify-between">
                      <span className="text-slate-400">원본 데이터셋</span>
                      <span className="text-white font-medium truncate ml-2" title={patch.datasetName}>
                        {patch.datasetName}
                      </span>
                    </div>
                    {patch.metadata?.imagesProcessed && (
                      <div className="flex justify-between">
                        <span className="text-slate-400">처리 이미지</span>
                        <span className="text-white font-medium">{patch.metadata.imagesProcessed}개</span>
                      </div>
                    )}
                    {patch.metadata?.iterations && (
                      <div className="flex justify-between">
                        <span className="text-slate-400">반복 횟수</span>
                        <span className="text-white font-medium">{patch.metadata.iterations}</span>
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
                        className="border-slate-600 hover:bg-slate-700"
                      >
                        <Eye className="w-3 h-3 mr-1" />
                        보기
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleDownloadPatch(patch)}
                        className="border-slate-600 hover:bg-slate-700"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        다운로드
                      </Button>
                    </div>

                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDeletePatch(patch.id)}
                      className="w-full text-red-400 hover:text-red-300 hover:bg-red-900/20"
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
      ) : activeTab === "patch-datasets" ? (
        renderDatasetCards(filteredPatchDatasets)
      ) : (
        renderDatasetCards(filteredNoiseDatasets)
      )}
    </div>
  )

  // Action Buttons (removed - no longer needed)
  const actionButtons = null

  return (
    <>
      <AdversarialToolLayout
        title="적대적 자산 관리"
        description="생성된 적대적 패치 및 데이터셋을 관리합니다"
        icon={Layers}
        leftPanel={{
          title: "필터 및 설정",
          icon: Settings,
          description: "자산을 검색하고 필터링합니다",
          children: leftPanelContent
        }}
        rightPanel={{
          title: activeTab === "patches"
            ? "적대적 패치"
            : activeTab === "patch-datasets"
            ? "패치 데이터셋"
            : "노이즈 데이터셋",
          icon: activeTab === "patches"
            ? Shield
            : activeTab === "patch-datasets"
            ? Database
            : FileStack,
          description: activeTab === "patches"
            ? `총 ${filteredPatches.length}개의 패치`
            : activeTab === "patch-datasets"
            ? `총 ${filteredPatchDatasets.length}개의 패치 데이터셋`
            : `총 ${filteredNoiseDatasets.length}개의 노이즈 데이터셋`,
          children: rightPanelContent
        }}
        actionButtons={actionButtons}
        leftPanelWidth="md"
      />

      {/* Patch Details Modal */}
      <Dialog open={showPatchDetails} onOpenChange={setShowPatchDetails}>
        <DialogContent className="sm:max-w-[700px] bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-white">
              <Shield className="w-5 h-5" />
              패치 상세 정보
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              생성된 적대적 패치의 상세 정보입니다
            </DialogDescription>
          </DialogHeader>

          {selectedPatch && (
            <div className="space-y-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">패치 이름</p>
                  <p className="text-sm text-white font-medium">{selectedPatch.name}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">대상 클래스</p>
                  <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/30">
                    <Target className="w-3 h-3 mr-1" />
                    {selectedPatch.targetClass}
                  </Badge>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">원본 데이터셋</p>
                  <p className="text-sm text-white font-medium">{selectedPatch.datasetName}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">생성 일시</p>
                  <p className="text-sm text-white font-medium">
                    {new Date(selectedPatch.createdAt).toLocaleString('ko-KR')}
                  </p>
                </div>
                {selectedPatch.metadata?.imagesProcessed && (
                  <div className="space-y-1">
                    <p className="text-xs text-slate-400">처리된 이미지</p>
                    <p className="text-sm text-white font-medium">{selectedPatch.metadata.imagesProcessed}개</p>
                  </div>
                )}
                {selectedPatch.metadata?.iterations && (
                  <div className="space-y-1">
                    <p className="text-xs text-slate-400">반복 횟수</p>
                    <p className="text-sm text-white font-medium">{selectedPatch.metadata.iterations}</p>
                  </div>
                )}
              </div>

              {/* Patch Preview */}
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <p className="text-sm text-slate-400 mb-2">패치 미리보기</p>
                <div className="aspect-square max-w-md mx-auto bg-slate-900/50 rounded-lg flex items-center justify-center border border-slate-700 overflow-hidden relative">
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
                      <Shield className="w-16 h-16 text-slate-600 mx-auto mb-2" />
                      <p className="text-slate-500">미리보기를 불러올 수 없습니다</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <DialogFooter className="gap-2">
            <Button onClick={() => setShowPatchDetails(false)} variant="outline" className="border-slate-600">
              닫기
            </Button>
            {selectedPatch && (
              <Button
                onClick={() => handleDownloadPatch(selectedPatch)}
                className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800"
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
        <DialogContent className="sm:max-w-[700px] bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-white">
              <Database className="w-5 h-5" />
              데이터셋 상세 정보
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              생성된 적대적 데이터셋의 상세 정보입니다
            </DialogDescription>
          </DialogHeader>

          {selectedDataset && (
            <div className="space-y-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">데이터셋 이름</p>
                  <p className="text-sm text-white font-medium">{selectedDataset.name}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">공격 유형</p>
                  <Badge className={
                    selectedDataset.type === "patch"
                      ? "bg-blue-500/10 text-blue-400 border-blue-500/30"
                      : "bg-amber-500/10 text-amber-400 border-amber-500/30"
                  }>
                    {selectedDataset.type === "patch" ? <Shield className="w-3 h-3 mr-1" /> : <FileStack className="w-3 h-3 mr-1" />}
                    {selectedDataset.type === "patch" ? "패치" : "노이즈"}
                  </Badge>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">원본 데이터셋</p>
                  <p className="text-sm text-white font-medium">{selectedDataset.originalDataset}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">이미지 수</p>
                  <p className="text-sm text-white font-medium">{selectedDataset.imageCount}개</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-slate-400">생성 일시</p>
                  <p className="text-sm text-white font-medium">
                    {new Date(selectedDataset.createdAt).toLocaleString('ko-KR')}
                  </p>
                </div>
                {selectedDataset.metadata?.attackMethod && (
                  <div className="space-y-1">
                    <p className="text-xs text-slate-400">공격 방법</p>
                    <p className="text-sm text-white font-medium">{selectedDataset.metadata.attackMethod}</p>
                  </div>
                )}
              </div>

              {/* Sample Images Grid with Carousel */}
              <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-sm text-slate-400">샘플 이미지</p>
                  {datasetSampleImages.length > 8 && (
                    <p className="text-xs text-slate-500">
                      {imageCarouselPage + 1} / {Math.ceil(datasetSampleImages.length / 8)}
                    </p>
                  )}
                </div>

                {loadingSampleImages ? (
                  <div className="text-center py-8">
                    <p className="text-slate-500 text-sm">이미지 로딩 중...</p>
                  </div>
                ) : (
                  <div className="relative">
                    {/* Image Grid (4 columns x 2 rows = 8 images) */}
                    <div className="grid grid-cols-4 gap-2 mb-3">
                      {datasetSampleImages.length > 0 ? (
                        datasetSampleImages.slice(imageCarouselPage * 8, (imageCarouselPage + 1) * 8).map((img, i) => {
                          const imageUrl = img.storage_key ? getImageUrlByStorageKey(img.storage_key) : ''
                          return (
                            <div key={img.id || i} className="aspect-square bg-slate-900/50 rounded-lg flex items-center justify-center border border-slate-700 overflow-hidden">
                              {imageUrl ? (
                                <img
                                  src={imageUrl}
                                  alt={img.file_name || `Sample ${i + 1}`}
                                  className="w-full h-full object-cover"
                                  onError={(e) => {
                                    console.error('[Modal Image Error] Failed to load:', imageUrl, 'for image:', img)
                                    e.currentTarget.style.display = 'none'
                                    const parent = e.currentTarget.parentElement
                                    if (parent) {
                                      const icon = document.createElement('div')
                                      icon.className = 'w-8 h-8 text-slate-600'
                                      icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>'
                                      parent.appendChild(icon)
                                    }
                                  }}
                                />
                              ) : (
                                <ImageIcon className="w-8 h-8 text-slate-600" />
                              )}
                            </div>
                          )
                        })
                      ) : (
                        [1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                          <div key={i} className="aspect-square bg-slate-900/50 rounded-lg flex items-center justify-center border border-slate-700">
                            <ImageIcon className="w-8 h-8 text-slate-600" />
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
                          className="border-slate-600 hover:bg-slate-700"
                        >
                          <ChevronLeft className="w-4 h-4 mr-1" />
                          이전
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setImageCarouselPage(Math.min(Math.ceil(datasetSampleImages.length / 8) - 1, imageCarouselPage + 1))}
                          disabled={imageCarouselPage >= Math.ceil(datasetSampleImages.length / 8) - 1}
                          className="border-slate-600 hover:bg-slate-700"
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
            <Button onClick={() => setShowDatasetDetails(false)} variant="outline" className="border-slate-600">
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
                      console.error("Failed to download dataset:", error)
                      toast.error("데이터셋 다운로드에 실패했습니다")
                    }
                  }}
                  className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800"
                >
                  <Download className="w-4 h-4 mr-2" />
                  다운로드
                </Button>
                <Button
                  onClick={() => {
                    handleRunEvaluation(selectedDataset)
                    setShowDatasetDetails(false)
                  }}
                  className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800"
                >
                  <PlayCircle className="w-4 h-4 mr-2" />
                  평가 실행
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
