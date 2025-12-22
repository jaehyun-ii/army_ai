"use client"

import { useEffect, useMemo, useState } from "react"
import { Activity, Map, Layers, Image as ImageIcon, Loader2, Info, Play, CheckCircle2, AlertCircle, X, RefreshCw, Eye, FileText, ChevronLeft, ChevronRight } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { toast } from "sonner"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import { useOperation } from "@/contexts/OperationContext"

type StatusType = "idle" | "processing" | "success" | "error"

interface DropdownState {
  maps: string[]
  weathers: string[]
  times: string[]
  objects: string[]
  attacks: string[]
  patches: string[]
}

interface StreamLog {
  message: string
  ts: string
  type?: StatusType
}

// 메인 백엔드 URL (이미지는 메인 백엔드의 /storage 경로에서 제공됨)
const mainBackendUrl = typeof window !== "undefined"
  ? (process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || `http://localhost:${process.env.BACKEND_PORT || 54321}`)
  : ""

export function AdversarialDataGenerator3D() {
  const { isOperationInProgress, setOperationInProgress } = useOperation()
  const [dropdowns, setDropdowns] = useState<DropdownState>({
    maps: [],
    weathers: [],
    times: [],
    objects: [],
    attacks: [],
    patches: []
  })
  const [loadingDropdowns, setLoadingDropdowns] = useState(false)

  const [datasetName, setDatasetName] = useState("k2_tank_adv")
  const [mapName, setMapName] = useState("")
  const [weatherName, setWeatherName] = useState("")
  const [timeName, setTimeName] = useState("")
  const [objectName, setObjectName] = useState("")
  const [attackMethod, setAttackMethod] = useState("")
  const [patchName, setPatchName] = useState("")
  const [imageWidth, setImageWidth] = useState<number>(512)
  const [imageHeight, setImageHeight] = useState<number>(512)

  const [status, setStatus] = useState<StatusType>("idle")
  const [statusMessage, setStatusMessage] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ current: number; total: number }>({ current: 0, total: 0 })
  const [logs, setLogs] = useState<StreamLog[]>([])
  const [showResult, setShowResult] = useState(false)
  const [generatedDatasets, setGeneratedDatasets] = useState<any[]>([])
  const [totalImages, setTotalImages] = useState(0)
  const [generatedDatasetId, setGeneratedDatasetId] = useState<string | null>(null)
  const [resultPreviewImages, setResultPreviewImages] = useState<any[]>([])
  const [loadingResultImages, setLoadingResultImages] = useState(false)
  const [resultCurrentPage, setResultCurrentPage] = useState(1)
  const [resultTotalImages, setResultTotalImages] = useState(0)
  const resultImagesPerPage = 5

  useEffect(() => {
    loadDropdowns()
  }, [])

  // Load result images when showResult becomes true
  useEffect(() => {
    if (showResult && generatedDatasetId && resultPreviewImages.length === 0) {
      console.log('[3D Attack Dataset] Loading result images for dataset:', generatedDatasetId)
      setResultCurrentPage(1)
      loadResultImages(1)
    }
  }, [showResult, generatedDatasetId])

  // Load result images when page changes
  useEffect(() => {
    if (showResult && generatedDatasetId) {
      console.log('[3D Attack Dataset] Loading result images for page:', resultCurrentPage)
      loadResultImages(resultCurrentPage)
    }
  }, [resultCurrentPage])

  const loadResultImages = async (page: number = 1) => {
    if (!generatedDatasetId) {
      console.warn('[3D Attack Dataset] No generatedDatasetId available')
      return
    }

    console.log('[3D Attack Dataset] Loading images for dataset:', generatedDatasetId, 'page:', page)
    setLoadingResultImages(true)
    try {
      const offset = (page - 1) * resultImagesPerPage
      const url = `/api/carla/datasets_3d/${generatedDatasetId}/images?limit=${resultImagesPerPage}&offset=${offset}`
      console.log('[3D Attack Dataset] Fetching URL:', url)
      const response = await fetch(url)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('[3D Attack Dataset] HTTP error:', response.status, errorText)
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      console.log('[3D Attack Dataset] Response data:', data)
      console.log('[3D Attack Dataset] Images count:', data.images?.length)
      console.log('[3D Attack Dataset] Total images:', data.total)

      setResultPreviewImages(data.images || [])
      if (data.total !== undefined) {
        setResultTotalImages(data.total)
      }
    } catch (error) {
      console.error('[3D Attack Dataset] Error:', error)
      toast.error('결과 이미지를 불러오지 못했습니다')
    } finally {
      setLoadingResultImages(false)
    }
  }

  const resetState = () => {
    setIsGenerating(false)
    setOperationInProgress(false)
    setStatus("idle")
    setStatusMessage("")
    setLogs([])
    setGeneratedDatasets([])
    setShowResult(false)
    setProgress({ current: 0, total: 0 })
    setCurrentImage(null)
    setTotalImages(0)
    setGeneratedDatasetId(null)
    setResultPreviewImages([])
    setResultCurrentPage(1)
    setResultTotalImages(0)
    setDatasetName("k2_tank_adv")
    setMapName(dropdowns.maps[0] || "")
    setWeatherName(dropdowns.weathers[0] || "")
    setTimeName(dropdowns.times[0] || "")
    setObjectName(dropdowns.objects[0] || "")
    setAttackMethod(dropdowns.attacks[0] || "")
    setPatchName(dropdowns.patches[0] || "")
    setImageWidth(512)
    setImageHeight(512)
  }

  useEffect(() => {
    if (objectName && attackMethod) {
      loadPatchList(objectName, attackMethod)
    } else {
      setDropdowns(prev => ({ ...prev, patches: [] }))
      setPatchName("")
    }
  }, [objectName, attackMethod])

  const logMessage = (message: string, type: StatusType = "processing") => {
    setLogs(prev => [...prev, { message, ts: new Date().toLocaleTimeString(), type }])
  }

  const loadDropdowns = async () => {
    setLoadingDropdowns(true)
    try {
      const [mapRes, weatherRes, timeRes, objectRes, attackRes] = await Promise.all([
        fetch("/api/carla/sim_map_list"),
        fetch("/api/carla/sim_weather_list"),
        fetch("/api/carla/sim_time_list"),
        fetch("/api/carla/sim_object_list"),
        fetch("/api/carla/sim_attack_list")
      ])

      const mapData = mapRes.ok ? await mapRes.json() : { result: [] }
      const weatherData = weatherRes.ok ? await weatherRes.json() : { result: [] }
      const timeData = timeRes.ok ? await timeRes.json() : { result: [] }
      const objectData = objectRes.ok ? await objectRes.json() : { result: [] }
      const attackData = attackRes.ok ? await attackRes.json() : { result: [] }

      setDropdowns(prev => ({
        ...prev,
        maps: mapData.result || [],
        weathers: weatherData.result || [],
        times: timeData.result || [],
        objects: objectData.result || [],
        attacks: attackData.result || []
      }))

      setMapName(mapData.result?.[0] || "")
      setWeatherName(weatherData.result?.[0] || "")
      setTimeName(timeData.result?.[0] || "")
      setObjectName(objectData.result?.[0] || "")
      setAttackMethod(attackData.result?.[0] || "")
    } catch (error) {
      console.error("드롭다운 로드 실패:", error)
      toast.error("환경 설정 정보를 불러오지 못했습니다.")
    } finally {
      setLoadingDropdowns(false)
    }
  }

  const loadPatchList = async (object: string, attack: string) => {
    try {
      const url = `/api/carla/sim_patch_list?object_name=${encodeURIComponent(object)}&attack_method=${encodeURIComponent(attack)}`
      const res = await fetch(url)
      if (!res.ok) throw new Error("패턴 목록 로드 실패")
      const data = await res.json()
      setDropdowns(prev => ({ ...prev, patches: data.result || [] }))
      setPatchName(data.result?.[0] || "")
    } catch (error) {
      console.error("패치 목록 로드 실패:", error)
      setDropdowns(prev => ({ ...prev, patches: [] }))
      setPatchName("")
    }
  }

  const handleGenerate = async () => {
    if (!datasetName || !mapName || !weatherName || !timeName || !objectName || !attackMethod || !patchName) {
      toast.error("모든 필수 항목을 입력해주세요.")
      return
    }

    setIsGenerating(true)
    setOperationInProgress(true, "3D 적대적 공격 데이터 생성")
    setStatus("processing")
    setStatusMessage("데이터셋 생성을 시작합니다...")
    setLogs([])
    setProgress({ current: 0, total: 0 })
    setCurrentImage(null)
    setShowResult(false)
    setGeneratedDatasets([])
    setTotalImages(0)

    try {
      const response = await fetch("/api/carla/sim_apply_texture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_name: datasetName,
          map_name: mapName,
          weather_name: weatherName,
          time_name: timeName,
          object_name: objectName,
          image_width: imageWidth,
          image_height: imageHeight,
          attack_method: attackMethod,
          patch_name: patchName
        })
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.message || "데이터셋 생성 실패")
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n")
          buffer = lines.pop() || ""

          for (const line of lines) {
            if (!line.trim()) continue

            // 파싱 전 원본 로그 출력
            console.log('[3D Adversarial Attack] Raw SSE line:', line)

            try {
              const data = JSON.parse(line)
              console.log('[3D Adversarial Attack] Parsed data:', data)

              if (data.state === 200 && data.result) {
                const result = data.result

                // Dataset 저장 완료 시 dataset_id 저장 (attack_dataset_id가 아닌 output_dataset_id 사용)
                if (data.message === "Attack dataset saved to database") {
                  // AttackDataset3D는 이미지가 없으므로 무시
                  console.log('[3D Attack Dataset] Attack dataset saved, but no images')
                } else if (result.dataset_id) {
                  console.log('[3D Attack Dataset] Received dataset_id:', result.dataset_id)
                  setGeneratedDatasetId(result.dataset_id)
                }

                const locIdx = result.loc_idx || 0
                const locTotal = result.loc_total || 0
                const batchIdx = result.batch_idx || 0
                const batchTotal = result.batch_total || 0
                setProgress({ current: locIdx, total: locTotal })

                // 총 이미지 수 계산 및 저장
                const calculatedTotalImages = locTotal * batchTotal
                if (calculatedTotalImages > 0) {
                  setTotalImages(calculatedTotalImages)
                  setResultTotalImages(calculatedTotalImages)
                }

                setStatusMessage(`생성장소: ${locIdx}/${locTotal}, Batch: ${batchIdx}/${batchTotal}`)

                // 로그: 상세하게 표시 (위치, 배치, 이미지 경로 포함)
                const currentImageNumber = (locIdx - 1) * batchTotal + batchIdx
                const totalImageCount = locTotal * batchTotal

                // 이미지 경로 정보 추출
                const imagePathInfo = result.storage_key
                  ? `storage_key: ${result.storage_key}`
                  : result.image_path
                    ? `image_path: ${result.image_path}`
                    : ''

                logMessage(
                  `[${currentImageNumber}/${totalImageCount}] loc: ${locIdx}/${locTotal}, batch: ${batchIdx}/${batchTotal}${imagePathInfo ? `, ${imagePathInfo}` : ''}`
                )

                // 2D와 동일하게 storage_key 우선 사용
                if (result.storage_key) {
                  setCurrentImage(`/api/storage/${result.storage_key}`)
                } else if (result.image_path) {
                  setCurrentImage(`${mainBackendUrl}${result.image_path}`)
                }
              }
            } catch (e) {
              console.error("스트림 파싱 실패", e)
            }
          }
        }
      }

      setStatus("success")
      setStatusMessage("적대적 공격 데이터 생성 완료")
      logMessage("데이터셋 생성 완료", "success")

      // 데이터셋 결과 저장
      setGeneratedDatasets([{
        id: `dataset_${Date.now()}`,
        datasetName: datasetName,
        mapName: mapName,
        weatherName: weatherName,
        timeName: timeName,
        objectName: objectName,
        attackMethod: attackMethod,
        patchName: patchName,
        totalImages: totalImages,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
        createdAt: new Date().toISOString()
      }])

      toast.success("적대적 공격 데이터 생성 완료")
    } catch (error) {
      console.error(error)
      const msg = error instanceof Error ? error.message : "데이터셋 생성 실패"
      setStatus("error")
      setStatusMessage(msg)
      logMessage(msg, "error")
      toast.error(msg)
      setIsGenerating(false)
      setOperationInProgress(false)
    }
  }

  const statusBadge = useMemo(() => {
    if (status === "success") return <Badge className="bg-emerald-500/10 text-emerald-300 border-emerald-500/30">완료</Badge>
    if (status === "error") return <Badge className="bg-red-500/10 text-red-300 border-red-500/30">오류</Badge>
    if (status === "processing") return <Badge className="bg-blue-500/10 text-blue-200 border-blue-500/30">진행 중</Badge>
    return <Badge variant="outline">대기</Badge>
  }, [status])

  const leftPanel = (
    <div className="space-y-4">
      <div>
        <Label className="text-sm">데이터셋 이름 *</Label>
        <Input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} placeholder="예: k2_tank_adv" disabled={isGenerating || (showResult && generatedDatasets.length > 0)} />
      </div>
      <div className="space-y-2">
        <Label className="text-sm">맵 선택 *</Label>
        <Select value={mapName} onValueChange={setMapName} disabled={isGenerating || loadingDropdowns || (showResult && generatedDatasets.length > 0)}>
          <SelectTrigger><SelectValue placeholder="맵 선택" /></SelectTrigger>
          <SelectContent>
            {dropdowns.maps.map(m => <SelectItem key={m} value={m}>{m}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="text-sm">날씨 *</Label>
        <Select value={weatherName} onValueChange={setWeatherName} disabled={isGenerating || loadingDropdowns || (showResult && generatedDatasets.length > 0)}>
          <SelectTrigger><SelectValue placeholder="날씨 선택" /></SelectTrigger>
          <SelectContent>
            {dropdowns.weathers.map(w => <SelectItem key={w} value={w}>{w}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="text-sm">시간 *</Label>
        <Select value={timeName} onValueChange={setTimeName} disabled={isGenerating || loadingDropdowns || (showResult && generatedDatasets.length > 0)}>
          <SelectTrigger><SelectValue placeholder="시간 선택" /></SelectTrigger>
          <SelectContent>
            {dropdowns.times.map(t => <SelectItem key={t} value={t}>{t}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="text-sm">객체 선택 *</Label>
        <Select value={objectName} onValueChange={setObjectName} disabled={isGenerating || loadingDropdowns || (showResult && generatedDatasets.length > 0)}>
          <SelectTrigger><SelectValue placeholder="객체 선택" /></SelectTrigger>
          <SelectContent>
            {dropdowns.objects.map(o => <SelectItem key={o} value={o}>{o}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="text-sm">공격 기법 *</Label>
        <Select value={attackMethod} onValueChange={setAttackMethod} disabled={isGenerating || loadingDropdowns || (showResult && generatedDatasets.length > 0)}>
          <SelectTrigger><SelectValue placeholder="공격 기법 선택" /></SelectTrigger>
          <SelectContent>
            {dropdowns.attacks.map(a => <SelectItem key={a} value={a}>{a}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="text-sm">패턴 선택 *</Label>
        <Select value={patchName} onValueChange={setPatchName} disabled={isGenerating || loadingDropdowns || !dropdowns.patches.length || (showResult && generatedDatasets.length > 0)}>
          <SelectTrigger><SelectValue placeholder={objectName && attackMethod ? "패턴 선택" : "객체/공격 기법을 먼저 선택"} /></SelectTrigger>
          <SelectContent>
            {dropdowns.patches.length === 0 ? (
              <SelectItem value="__no_patch__" disabled>패턴 없음</SelectItem>
            ) : dropdowns.patches.map(p => <SelectItem key={p} value={p}>{p}</SelectItem>)
            }
          </SelectContent>
        </Select>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label className="text-sm">이미지 너비</Label>
          <Input type="number" value={imageWidth} onChange={(e) => setImageWidth(parseInt(e.target.value) || 0)} disabled={isGenerating || (showResult && generatedDatasets.length > 0)} />
        </div>
        <div>
          <Label className="text-sm">이미지 높이</Label>
          <Input type="number" value={imageHeight} onChange={(e) => setImageHeight(parseInt(e.target.value) || 0)} disabled={isGenerating || (showResult && generatedDatasets.length > 0)} />
        </div>
      </div>
    </div>
  )

  const rightPanel = showResult && generatedDatasets.length > 0 ? (
    // State: Result - Show Generated Dataset
    <div className="h-full flex flex-col p-6 space-y-4 overflow-hidden">
      {/* Generated Dataset Info */}
      {generatedDatasets[0] && (
        <Card className="bg-surface-container/50 border-border flex-shrink-0">
          <CardContent className="pt-6">
            <div className="space-y-3">
              <h4 className="text-foreground font-semibold mb-3">적대적 공격 데이터 정보</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted">데이터셋 이름</span>
                  <span className="text-foreground">{generatedDatasets[0].datasetName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">맵</span>
                  <span className="text-foreground">{generatedDatasets[0].mapName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">날씨</span>
                  <span className="text-foreground">{generatedDatasets[0].weatherName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">시간</span>
                  <span className="text-foreground">{generatedDatasets[0].timeName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">객체</span>
                  <span className="text-foreground">{generatedDatasets[0].objectName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">공격 기법</span>
                  <span className="text-foreground">{generatedDatasets[0].attackMethod}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">패치</span>
                  <span className="text-foreground">{generatedDatasets[0].patchName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">생성된 이미지</span>
                  <span className="text-foreground font-semibold">{resultTotalImages}개</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">생성 시간</span>
                  <span className="text-foreground">
                    {new Date(generatedDatasets[0].createdAt).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Dataset Images Preview */}
      {generatedDatasetId && (
        <div className="flex-1 flex flex-col space-y-3 overflow-hidden">
          <h4 className="text-foreground font-semibold text-sm flex-shrink-0">생성된 이미지 미리보기</h4>

          {loadingResultImages ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-2" />
                <p className="text-muted text-sm">이미지 로딩 중...</p>
              </div>
            </div>
          ) : resultPreviewImages.length > 0 ? (
            <>
              <div className="grid grid-cols-5 gap-2 flex-shrink-0">
                {resultPreviewImages.slice(0, 5).map((img, idx) => (
                  <div key={idx} className="aspect-square bg-surface-container rounded-lg overflow-hidden relative group border border-outline">
                    {img.storage_key || img.data ? (
                      <img
                        src={
                          img.storage_key
                            ? `/api/storage/${img.storage_key}`
                            : `data:${img.mimeType || img.mime_type || 'image/jpeg'};base64,${img.data}`
                        }
                        alt={img.filename || img.file_name || `Result ${idx + 1}`}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          console.error('[ResultImage] Failed to load image:', {
                            storage_key: img.storage_key,
                            hasData: !!img.data,
                            src: e.currentTarget.src
                          })
                        }}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <ImageIcon className="w-6 h-6 text-muted" />
                      </div>
                    )}
                    <div className="absolute inset-0 bg-scrim/0 group-hover:bg-scrim/60 transition-all duration-200 flex items-center justify-center opacity-0 group-hover:opacity-100">
                      <span className="text-foreground text-xs font-medium px-2 text-center break-all">
                        {img.filename || img.file_name || `이미지 ${idx + 1}`}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Pagination */}
              {resultTotalImages > 5 && (
                <div className="flex items-center justify-between bg-surface-container/50 rounded-lg p-3 flex-shrink-0">
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setResultCurrentPage(prev => Math.max(prev - 1, 1))}
                      disabled={resultCurrentPage === 1 || loadingResultImages}
                      className="h-8 px-3"
                    >
                      <ChevronLeft className="h-4 w-4 mr-1" />
                      이전
                    </Button>
                    <span className="text-sm text-muted min-w-[100px] text-center">
                      {resultCurrentPage} / {Math.ceil(resultTotalImages / 5)} 페이지
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setResultCurrentPage(prev => Math.min(prev + 1, Math.ceil(resultTotalImages / 5)))}
                      disabled={resultCurrentPage >= Math.ceil(resultTotalImages / 5) || loadingResultImages}
                      className="h-8 px-3"
                    >
                      다음
                      <ChevronRight className="h-4 w-4 ml-1" />
                    </Button>
                  </div>
                  <span className="text-xs text-muted">
                    전체 {resultTotalImages.toLocaleString()}개 중 {((resultCurrentPage - 1) * 5 + 1)}-{Math.min(resultCurrentPage * 5, resultTotalImages)}개 표시
                  </span>
                </div>
              )}
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-muted text-sm">이미지를 불러올 수 없습니다</p>
            </div>
          )}
        </div>
      )}
    </div>
  ) : isGenerating ? (
    // State: Generation in Progress - Show Real-time Preview
    <div className="h-full flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-foreground">적대적 공격 데이터 생성 진행 중 - 실시간으로 생성된 이미지를 확인하세요</p>
        {status === "processing" && <Badge className="bg-blue-500/10 text-blue-200 border-blue-500/30">진행 중</Badge>}
      </div>

      {/* Progress Card */}
      <Card className="p-3 bg-surface-container-high/40 border-border">
        <div className="flex flex-col gap-2">
          {progress.total > 0 && (
            <>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-surface-container rounded-lg p-2 border border-outline-variant">
                  <div className="text-xs text-muted mb-1">진행률</div>
                  <div className="text-xl font-bold text-primary">
                    {Math.round((progress.current / progress.total) * 100)}%
                  </div>
                </div>
                <div className="bg-surface-container rounded-lg p-2 border border-outline-variant">
                  <div className="text-xs text-muted mb-1">생성 위치</div>
                  <div className="text-xl font-bold text-tertiary">
                    {progress.current}/{progress.total}
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted">전체 진행률</span>
                <span className="text-xs text-muted">{progress.current}/{progress.total}</span>
              </div>
              <Progress value={progress.total ? (progress.current / progress.total) * 100 : 0} className="h-2" />
            </>
          )}
        </div>
      </Card>

      {/* Real-time Preview and Logs - Same layout as initial state */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-3 overflow-hidden">
        <Card className="p-3 bg-surface-container/40 border-border flex flex-col">
          <p className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
            <ImageIcon className="w-4 h-4" />
            실시간 미리보기
          </p>
          <div className="flex-1 bg-surface rounded-lg border border-border flex items-center justify-center overflow-hidden">
            {currentImage ? (
              <img src={currentImage} alt="real-time preview" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center text-muted text-sm p-6">
                <Info className="w-6 h-6 mx-auto mb-2 text-muted" />
                생성된 공격 데이터가 표시됩니다.
              </div>
            )}
          </div>
        </Card>

        <Card className="p-3 bg-surface-container/40 border-border flex flex-col overflow-hidden">
          <p className="text-sm font-semibold text-foreground mb-2">상태 로그</p>
          <div className="flex-1 h-full overflow-y-auto text-xs font-mono space-y-1">
            {logs.length === 0 && <p className="text-muted">로그 대기 중...</p>}
            {logs.map((log, idx) => (
              <div key={`${log.ts}-${idx}`} className={log.type === "error" ? "text-error" : log.type === "success" ? "text-tertiary" : "text-muted"}>
                [{log.ts}] {log.message}
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Status Messages */}
      {generatedDatasets.length > 0 && (
        <div className="bg-tertiary-container border border-tertiary rounded-lg p-3">
          <p className="text-tertiary text-sm flex items-center gap-2 mb-2">
            <CheckCircle2 className="w-4 h-4" />
            적대적 공격 데이터 생성이 완료되었습니다!
          </p>
          <Button
            onClick={() => {
              setIsGenerating(false)
              setShowResult(true)
              setOperationInProgress(false)
            }}
            variant="tertiary"
            className="w-full"
          >
            <Eye className="w-4 h-4 mr-2" />
            결과 보기
          </Button>
        </div>
      )}
      {logs.some(log => log.type === 'error') && (
        <div className="bg-error-container border border-error rounded-lg p-3">
          <p className="text-error text-sm flex items-center gap-2 mb-2">
            <AlertCircle className="w-4 h-4" />
            오류가 발생했습니다.
          </p>
          <div className="flex gap-2">
            <Button
              onClick={() => {
                setIsGenerating(false)
                setShowResult(false)
                setOperationInProgress(false)
              }}
              variant="outline"
              size="sm"
              className="flex-1 border-error hover:bg-error-container"
            >
              <X className="w-4 h-4 mr-1" />
              닫기
            </Button>
            <Button
              onClick={resetState}
              variant="default"
              size="sm"
              className="flex-1"
            >
              <RefreshCw className="w-4 h-4 mr-1" />
              초기화
            </Button>
          </div>
        </div>
      )}
    </div>
  ) : (
    // State: Initial - Show Guide
    <div className="h-full flex flex-col gap-3">
      <div className="flex items-center gap-3">
        {status === "processing" && <Loader2 className="w-4 h-4 animate-spin text-primary" />}
        <p className="text-sm text-foreground">{statusMessage || "대기 중"}</p>
        {statusBadge}
      </div>

      <Card className="p-3 bg-surface-container-high/40 border-border">
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted">진행률</span>
            <span className="text-xs text-muted">{progress.current}/{progress.total || 0}</span>
          </div>
          <Progress value={progress.total ? (progress.current / progress.total) * 100 : 0} className="h-2" />
        </div>
      </Card>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-3 overflow-hidden">
        <Card className="p-3 bg-surface-container/40 border-border flex flex-col">
          <p className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2"><ImageIcon className="w-4 h-4" />미리보기</p>
          <div className="flex-1 bg-surface rounded-lg border border-border flex items-center justify-center overflow-hidden">
            {currentImage ? (
              <img src={currentImage} alt="generated preview" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center text-muted text-sm p-6">
                <Info className="w-6 h-6 mx-auto mb-2 text-muted" />
                생성된 공격 데이터가 표시됩니다.
              </div>
            )}
          </div>
        </Card>

        <Card className="p-3 bg-surface-container/40 border-border flex flex-col overflow-hidden">
          <p className="text-sm font-semibold text-foreground mb-2">상태 로그</p>
          <div className="flex-1 h-full overflow-y-auto text-xs font-mono space-y-1">
            {logs.length === 0 && <p className="text-muted">로그 대기 중...</p>}
            {logs.map((log, idx) => (
              <div key={`${log.ts}-${idx}`} className={log.type === "error" ? "text-error" : log.type === "success" ? "text-tertiary" : "text-muted"}>
                [{log.ts}] {log.message}
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )

  const actionButtons = showResult && generatedDatasets.length > 0 ? (
    <Button
      onClick={resetState}
      className="w-full ds-btn-outline"
    >
      <RefreshCw className="w-4 h-4 mr-2" />
      초기화
    </Button>
  ) : (
    <Button className="w-full ds-btn-primary" disabled={isGenerating || isOperationInProgress} onClick={handleGenerate}>
      <Play className="w-4 h-4 mr-2" />
      적대적 공격 데이터 생성
    </Button>
  )

  return (
    <AdversarialToolLayout
      title="적대적 공격 데이터 생성"
      description="생성된 패치를 대상 객체와 환경에 적용해 공격 데이터를 생성합니다."
      icon={Activity}
      leftPanel={{
        title: "생성 설정",
        icon: Map,
        description: "데이터셋, 환경, 공격 기법 및 패턴을 선택하세요.",
        children: leftPanel
      }}
      rightPanel={{
        title: "생성 상태",
        icon: Layers,
        children: rightPanel
      }}
      actionButtons={actionButtons}
      disabled={isOperationInProgress}
    />
  )
}
