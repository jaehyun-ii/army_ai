"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Loader2,
  Image as ImageIcon,
  TrendingDown,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Filter,
} from "lucide-react"
import { toast } from "sonner"

interface EvaluationImageComparisonProps {
  runId: string
  phase: string
  baseDatasetId?: string
  attackDatasetId?: string
}

interface ImageData {
  item_id: string
  file_name: string
  storage_key: string
  dataset_type: "base" | "attack" | null
  predicted_boxes: any[]
  ground_truth_boxes: any[]
  metrics: any
  visualization_url: string
}

export function EvaluationImageComparison({
  runId,
  phase,
  baseDatasetId,
  attackDatasetId,
}: EvaluationImageComparisonProps) {
  const [loading, setLoading] = useState(true)

  // 데이터
  const [baseImages, setBaseImages] = useState<ImageData[]>([])
  const [attackImages, setAttackImages] = useState<ImageData[]>([])

  // UI 상태
  const [showBboxes, setShowBboxes] = useState(true)
  const [selectedImageIndex, setSelectedImageIndex] = useState<number>(0)

  // Tab/Filter 상태
  const [viewMode, setViewMode] = useState<"all" | "base" | "attack">("all")

  useEffect(() => {
    detectAvailableRuns()
  }, [runId, phase])

  useEffect(() => {
    loadImages()
  }, [viewMode])  // viewMode 변경 시 리로드

  useEffect(() => {
    setSelectedImageIndex(0)
  }, [baseImages, attackImages])

  const detectAvailableRuns = async () => {
    // 간단히 현재 run만 사용
    // API에서 dataset_type을 제공하므로 별도 run 검색 불필요
    console.log('[detectAvailableRuns] Using current run:', runId, 'phase:', phase)

    // 이미지 로드
    await loadImages()
  }

  const loadImages = async () => {
    setLoading(true)
    console.log('[loadImages] Loading images from run:', runId, 'viewMode:', viewMode)
    try {
      // viewMode에 따라 dataset_type 파라미터와 page_size 조정
      // all 모드일 때는 base와 attack을 모두 가져와야 하므로 각각 별도로 요청
      if (viewMode === "all") {
        // All 모드: base와 attack을 각각 요청
        const [baseResponse, attackResponse] = await Promise.all([
          fetch(`http://localhost:8000/api/v1/evaluation/runs/${runId}/images-with-predictions?dataset_type=base`),
          fetch(`http://localhost:8000/api/v1/evaluation/runs/${runId}/images-with-predictions?dataset_type=attack`)
        ])

        if (!baseResponse.ok || !attackResponse.ok) {
          throw new Error("Failed to load images")
        }

        const baseData = await baseResponse.json()
        const attackData = await attackResponse.json()

        console.log('[loadImages] ALL mode - Base data:', {
          total: baseData.total,
          itemsCount: baseData.items?.length || 0
        })
        console.log('[loadImages] ALL mode - Attack data:', {
          total: attackData.total,
          itemsCount: attackData.items?.length || 0
        })

        // 파일명 기준으로 정렬 및 매칭
        const baseItems = (baseData.items || []).sort((a: ImageData, b: ImageData) =>
          (a.file_name || '').localeCompare(b.file_name || '')
        )

        // Attack 이미지는 "ad_" prefix를 제거하고 정렬
        const attackItems = (attackData.items || []).sort((a: ImageData, b: ImageData) => {
          const aName = (a.file_name || '').replace(/^ad_/, '')
          const bName = (b.file_name || '').replace(/^ad_/, '')
          return aName.localeCompare(bName)
        })

        console.log('[loadImages] Sorted images:', {
          firstBase: baseItems[0]?.file_name,
          firstAttack: attackItems[0]?.file_name,
          baseCount: baseItems.length,
          attackCount: attackItems.length
        })

        setBaseImages(baseItems)
        setAttackImages(attackItems)

        console.log('[loadImages] State updated - base:', baseItems.length, 'attack:', attackItems.length)
        return
      }

      // Base 또는 Attack 모드: 단일 요청
      let url = `http://localhost:8000/api/v1/evaluation/runs/${runId}/images-with-predictions`

      if (viewMode === "base") {
        url += "?dataset_type=base"
      } else if (viewMode === "attack") {
        url += "?dataset_type=attack"
      }

      const response = await fetch(url)

      if (!response.ok) {
        throw new Error("Failed to load images")
      }

      const data = await response.json()
      console.log('[loadImages] Response data:', {
        total: data.total,
        itemsCount: data.items?.length || 0,
        viewMode,
        firstItemType: data.items?.[0]?.dataset_type
      })

      // dataset_type으로 base와 attack 이미지 분리 및 정렬
      const allImages = data.items || []

      if (viewMode === "base") {
        // Base 모드: 파일명으로 정렬
        const sortedImages = allImages.sort((a: ImageData, b: ImageData) =>
          (a.file_name || '').localeCompare(b.file_name || '')
        )
        console.log('[loadImages] BASE mode - setting baseImages:', sortedImages.length)
        setBaseImages(sortedImages)
        setAttackImages([])
      } else if (viewMode === "attack") {
        // Attack 모드: ad_ prefix 제거 후 정렬
        const sortedImages = allImages.sort((a: ImageData, b: ImageData) => {
          const aName = (a.file_name || '').replace(/^ad_/, '')
          const bName = (b.file_name || '').replace(/^ad_/, '')
          return aName.localeCompare(bName)
        })
        console.log('[loadImages] ATTACK mode - setting attackImages:', sortedImages.length)
        setBaseImages([])
        setAttackImages(sortedImages)
      }

      console.log('[loadImages] State updated - will render with:', {
        baseImagesLength: viewMode === "attack" ? 0 : (viewMode === "base" ? allImages.length : allImages.filter((img: ImageData) => img.dataset_type === "base").length),
        attackImagesLength: viewMode === "base" ? 0 : (viewMode === "attack" ? allImages.length : allImages.filter((img: ImageData) => img.dataset_type === "attack").length)
      })
    } catch (error) {
      console.error("[loadImages] Error:", error)
      toast.error("이미지를 불러오는데 실패했습니다")
    } finally {
      setLoading(false)
    }
  }

  const getImageUrl = (storageKey: string) => {
    if (!storageKey) return ""
    const baseUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
    return `${baseUrl}/storage/${storageKey}`
  }

  const getVisualizationUrl = (visualizationPath: string) => {
    const baseUrl = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
    return `${baseUrl}${visualizationPath}`
  }

  if (loading) {
    return (
      <Card className="bg-slate-800/50 border-white/10">
        <CardContent className="py-12">
          <div className="flex items-center justify-center gap-3">
            <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
            <span className="text-slate-400">이미지를 불러오는 중...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  // 현재 선택된 이미지들
  const currentBaseImage = baseImages[selectedImageIndex]
  const currentAttackImage = attackImages[selectedImageIndex]
  const totalImages = Math.max(baseImages.length, attackImages.length)

  // viewMode에 따라 표시할 내용 결정
  const shouldShowBase = viewMode === "all" || viewMode === "base"
  const shouldShowAttack = viewMode === "all" || viewMode === "attack"
  const showComparison = viewMode === "all" && currentBaseImage && currentAttackImage

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card className="bg-slate-800/50 border-white/10">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-blue-400" />
              추론 결과 이미지
            </CardTitle>

            <div className="flex items-center gap-4">
              {/* Dataset Type Filter Tabs */}
              <Tabs value={viewMode} onValueChange={(value) => {
                setViewMode(value as "all" | "base" | "attack")
              }}>
                <TabsList className="bg-slate-700/50 border border-white/10">
                  <TabsTrigger value="all" className="data-[state=active]:bg-blue-600 data-[state=active]:text-white">
                    <Filter className="w-4 h-4 mr-2" />
                    All ({baseImages.length + attackImages.length})
                  </TabsTrigger>
                  <TabsTrigger value="base" className="data-[state=active]:bg-green-600 data-[state=active]:text-white">
                    Base ({baseImages.length})
                  </TabsTrigger>
                  {(phase === "post_attack" || attackImages.length > 0) && (
                    <TabsTrigger value="attack" className="data-[state=active]:bg-red-600 data-[state=active]:text-white">
                      Attack ({attackImages.length})
                    </TabsTrigger>
                  )}
                </TabsList>
              </Tabs>

              {/* 바운딩박스 토글 */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowBboxes(!showBboxes)}
                className="bg-slate-700/50 border-white/10 text-white hover:bg-slate-600/50"
              >
                {showBboxes ? (
                  <>
                    <Eye className="w-4 h-4 mr-2" />
                    바운딩 박스 표시
                  </>
                ) : (
                  <>
                    <EyeOff className="w-4 h-4 mr-2" />
                    바운딩 박스 숨김
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {totalImages === 0 ? (
            <div className="text-center py-12">
              <ImageIcon className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400">이미지가 없습니다</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* 이미지 선택기 */}
              <div className="bg-slate-700/30 border border-white/10 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <ImageIcon className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-semibold text-white">이미지 선택</span>
                  </div>
                  <span className="text-xs text-slate-400">
                    {selectedImageIndex + 1} / {totalImages}
                  </span>
                </div>

                {/* 드롭다운 */}
                <Select
                  value={selectedImageIndex.toString()}
                  onValueChange={(value) => setSelectedImageIndex(parseInt(value))}
                >
                  <SelectTrigger className="w-full bg-slate-800/50 border-white/10 text-white">
                    <SelectValue>
                      {(() => {
                        if (currentBaseImage?.file_name) return currentBaseImage.file_name
                        if (currentAttackImage?.file_name) return currentAttackImage.file_name.replace(/^ad_/, '')
                        return "이미지를 선택하세요"
                      })()}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent className="max-h-[300px]">
                    {Array.from({ length: totalImages }).map((_, idx) => {
                      const baseImg = baseImages[idx]
                      const attackImg = attackImages[idx]

                      // Base 파일명 우선, 없으면 Attack 파일명에서 ad_ 제거
                      let displayName = baseImg?.file_name
                      if (!displayName && attackImg?.file_name) {
                        displayName = attackImg.file_name.replace(/^ad_/, '')
                      }
                      if (!displayName) {
                        displayName = `Image ${idx + 1}`
                      }

                      return (
                        <SelectItem key={idx} value={idx.toString()}>
                          {displayName}
                        </SelectItem>
                      )
                    })}
                  </SelectContent>
                </Select>

                {/* 이전/다음 버튼 */}
                <div className="flex items-center gap-2 mt-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedImageIndex(Math.max(0, selectedImageIndex - 1))}
                    disabled={selectedImageIndex === 0}
                    className="flex-1 bg-slate-800/50 border-white/10 text-white hover:bg-slate-700/50"
                  >
                    <ChevronLeft className="w-4 h-4 mr-1" />
                    이전
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedImageIndex(Math.min(totalImages - 1, selectedImageIndex + 1))}
                    disabled={selectedImageIndex === totalImages - 1}
                    className="flex-1 bg-slate-800/50 border-white/10 text-white hover:bg-slate-700/50"
                  >
                    다음
                    <ChevronRight className="w-4 h-4 ml-1" />
                  </Button>
                </div>
              </div>

              {/* 이미지 비교 뷰 */}
              {showComparison ? (
                // 두 개 다 선택: 반반 렌더링
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white text-center">
                    {currentBaseImage.file_name}
                  </h3>

                  <div className="grid grid-cols-2 gap-6">
                    {/* 좌측: 기준 데이터셋 */}
                    <Card className="bg-slate-700/30 border-white/10">
                      <CardHeader>
                        <CardTitle className="text-lg font-semibold text-green-400">
                          기준 데이터셋 추론 결과
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {/* 이미지 */}
                        <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden">
                          {showBboxes ? (
                            <img
                              src={getVisualizationUrl(currentBaseImage.visualization_url)}
                              alt="Base with predictions"
                              className="w-full h-full object-contain"
                              onError={(e) => {
                                e.currentTarget.src = getImageUrl(currentBaseImage.storage_key)
                              }}
                            />
                          ) : (
                            <img
                              src={getImageUrl(currentBaseImage.storage_key)}
                              alt="Base original"
                              className="w-full h-full object-contain"
                            />
                          )}
                        </div>

                        {/* 탐지 객체 목록 */}
                        <div className="bg-slate-800/50 rounded-lg p-4 space-y-3">
                          <div className="flex items-center justify-between pb-2 border-b border-white/10">
                            <span className="text-sm font-semibold text-slate-300">탐지된 객체 목록</span>
                            <span className="text-xs text-slate-400">
                              {currentBaseImage.predicted_boxes.length}개 탐지됨
                            </span>
                          </div>
                          <ScrollArea className="h-[300px]">
                            <div className="space-y-2 pr-4">
                              {currentBaseImage.predicted_boxes.length > 0 ? (
                                currentBaseImage.predicted_boxes.map((box: any, idx: number) => (
                                  <div key={idx} className="bg-slate-900/50 rounded-lg p-3 space-y-2">
                                    <div className="flex items-center justify-between">
                                      <span className="font-semibold text-white">{box.class_name}</span>
                                      <span className="text-green-400 font-bold">{(box.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="text-xs text-slate-400 font-mono">
                                      [{box.bbox.x1.toFixed(3)}, {box.bbox.y1.toFixed(3)}, {box.bbox.x2.toFixed(3)}, {box.bbox.y2.toFixed(3)}]
                                    </div>
                                  </div>
                                ))
                              ) : (
                                <div className="text-center py-8 text-slate-500">탐지된 객체가 없습니다</div>
                              )}
                            </div>
                          </ScrollArea>
                
                        </div>
                      </CardContent>
                    </Card>

                    {/* 우측: 공격 데이터셋 */}
                    <Card className="bg-slate-700/30 border-red-900/30 border-2">
                      <CardHeader>
                        <CardTitle className="text-lg font-semibold text-red-400 flex items-center justify-between">
                          <span>공격 데이터셋 추론 결과</span>
                          {currentAttackImage.predicted_boxes.length < currentBaseImage.predicted_boxes.length && (
                            <Badge className="bg-red-900/40 text-red-300 border-red-500/40">
                              <TrendingDown className="w-3 h-3 mr-1" />
                              탐지 감소
                            </Badge>
                          )}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {/* 이미지 */}
                        <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden">
                          {showBboxes ? (
                            <img
                              src={getVisualizationUrl(currentAttackImage.visualization_url)}
                              alt="Attack with predictions"
                              className="w-full h-full object-contain"
                              onError={(e) => {
                                e.currentTarget.src = getImageUrl(currentAttackImage.storage_key)
                              }}
                            />
                          ) : (
                            <img
                              src={getImageUrl(currentAttackImage.storage_key)}
                              alt="Attack original"
                              className="w-full h-full object-contain"
                            />
                          )}
                        </div>

                        {/* 탐지 객체 목록 */}
                        <div className="bg-slate-800/50 rounded-lg p-4 space-y-3">
                          <div className="flex items-center justify-between pb-2 border-b border-white/10">
                            <span className="text-sm font-semibold text-slate-300">탐지된 객체 목록</span>
                            <span className="text-xs text-slate-400">
                              {currentAttackImage.predicted_boxes.length}개 탐지됨
                              <span className={currentAttackImage.predicted_boxes.length < currentBaseImage.predicted_boxes.length ? "text-red-400 ml-1 font-bold" : "text-green-400 ml-1"}>
                                ({currentAttackImage.predicted_boxes.length - currentBaseImage.predicted_boxes.length >= 0 ? "+" : ""}{currentAttackImage.predicted_boxes.length - currentBaseImage.predicted_boxes.length})
                              </span>
                            </span>
                          </div>
                          <ScrollArea className="h-[300px]">
                            <div className="space-y-2 pr-4">
                              {currentAttackImage.predicted_boxes.length > 0 ? (
                                currentAttackImage.predicted_boxes.map((box: any, idx: number) => (
                                  <div key={idx} className="bg-slate-900/50 rounded-lg p-3 space-y-2">
                                    <div className="flex items-center justify-between">
                                      <span className="font-semibold text-white">{box.class_name}</span>
                                      <span className="text-red-400 font-bold">{(box.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="text-xs text-slate-400 font-mono">
                                      [{box.bbox.x1.toFixed(3)}, {box.bbox.y1.toFixed(3)}, {box.bbox.x2.toFixed(3)}, {box.bbox.y2.toFixed(3)}]
                                    </div>
                                  </div>
                                ))
                              ) : (
                                <div className="text-center py-8 text-slate-500">탐지된 객체가 없습니다</div>
                              )}
                            </div>
                          </ScrollArea>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                </div>
              ) : shouldShowBase && currentBaseImage ? (
                // 기준 데이터셋만 선택
                <Card className="bg-slate-700/30 border-white/10">
                  <CardHeader>
                    <CardTitle className="text-lg font-semibold text-green-400">
                      기준 데이터셋 추론 결과 - {currentBaseImage.file_name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden">
                      {showBboxes ? (
                        <img
                          src={getVisualizationUrl(currentBaseImage.visualization_url)}
                          alt="Base with predictions"
                          className="w-full h-full object-contain"
                          onError={(e) => {
                            e.currentTarget.src = getImageUrl(currentBaseImage.storage_key)
                          }}
                        />
                      ) : (
                        <img
                          src={getImageUrl(currentBaseImage.storage_key)}
                          alt="Base original"
                          className="w-full h-full object-contain"
                        />
                      )}
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <div className="text-xs text-slate-400">탐지된 객체</div>
                        <div className="text-2xl font-bold text-white">{currentBaseImage.predicted_boxes.length}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">Ground Truth</div>
                        <div className="text-2xl font-bold text-white">{currentBaseImage.ground_truth_boxes.length}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">mAP</div>
                        <div className="text-2xl font-bold text-green-400">{((currentBaseImage.metrics.map || 0) * 100).toFixed(2)}%</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ) : shouldShowAttack && currentAttackImage ? (
                // 공격 데이터셋만 선택
                <Card className="bg-slate-700/30 border-red-900/30 border-2">
                  <CardHeader>
                    <CardTitle className="text-lg font-semibold text-red-400">
                      공격 데이터셋 추론 결과 - {currentAttackImage.file_name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden">
                      {showBboxes ? (
                        <img
                          src={getVisualizationUrl(currentAttackImage.visualization_url)}
                          alt="Attack with predictions"
                          className="w-full h-full object-contain"
                          onError={(e) => {
                            e.currentTarget.src = getImageUrl(currentAttackImage.storage_key)
                          }}
                        />
                      ) : (
                        <img
                          src={getImageUrl(currentAttackImage.storage_key)}
                          alt="Attack original"
                          className="w-full h-full object-contain"
                        />
                      )}
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <div className="text-xs text-slate-400">탐지된 객체</div>
                        <div className="text-2xl font-bold text-white">{currentAttackImage.predicted_boxes.length}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">Ground Truth</div>
                        <div className="text-2xl font-bold text-white">{currentAttackImage.ground_truth_boxes.length}</div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-400">mAP</div>
                        <div className="text-2xl font-bold text-red-400">{((currentAttackImage.metrics.map || 0) * 100).toFixed(2)}%</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <div className="text-center py-12">
                  <p className="text-slate-400">표시할 데이터셋을 선택해주세요</p>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
