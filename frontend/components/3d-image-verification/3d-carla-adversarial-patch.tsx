"use client"

import { useEffect, useMemo, useState } from "react"
import { Shield, Target, Zap, Loader2, Image as ImageIcon, Info, Play, CheckCircle2, AlertCircle, X, RefreshCw, Eye, Download, FileText } from "lucide-react"
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

interface ModelOption {
  id: string
  name: string
}

interface DropdownState {
  objects: string[]
  attackMethods: string[]
  detectors: ModelOption[]
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

export function CarlaAdversarialPatchGenerator() {
  const { isOperationInProgress, setOperationInProgress } = useOperation()
  const [dropdowns, setDropdowns] = useState<DropdownState>({ objects: [], attackMethods: [], detectors: [] })
  const [loadingDropdowns, setLoadingDropdowns] = useState(false)

  const [patchName, setPatchName] = useState("3d_k2_tank")
  const [objectName, setObjectName] = useState("")
  const [attackMethod, setAttackMethod] = useState("")
  const [detectorId, setDetectorId] = useState("")  // detectorName -> detectorId로 변경

  const [status, setStatus] = useState<StatusType>("idle")
  const [statusMessage, setStatusMessage] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ current: number; total: number }>({ current: 0, total: 0 })
  const [logs, setLogs] = useState<StreamLog[]>([])
  const [showPatchResult, setShowPatchResult] = useState(false)
  const [generatedPatches, setGeneratedPatches] = useState<any[]>([])
  const [lastStorageKey, setLastStorageKey] = useState<string | null>(null)

  useEffect(() => {
    loadDropdowns()
  }, [])

  const resetPatchState = () => {
    setIsGenerating(false)
    setOperationInProgress(false)
    setStatus("idle")
    setStatusMessage("")
    setLogs([])
    setGeneratedPatches([])
    setShowPatchResult(false)
    setProgress({ current: 0, total: 0 })
    setCurrentImage(null)
    setLastStorageKey(null)
    setPatchName("")
    setObjectName(dropdowns.objects[0] || "")
    setAttackMethod(dropdowns.attackMethods[0] || "")
    setDetectorId(dropdowns.detectors[0]?.id || "")
  }

  const logMessage = (message: string, type: StatusType = "processing") => {
    setLogs(prev => [...prev, { message, ts: new Date().toLocaleTimeString(), type }])
  }

  const loadDropdowns = async () => {
    setLoadingDropdowns(true)
    try {
      const [objectRes, attackRes, modelsRes] = await Promise.all([
        fetch("/api/carla/sim_object_list"),
        fetch("/api/carla/sim_attack_list"),
        fetch("/api/models")  // 2D와 동일하게 /api/models에서 모델 조회
      ])

      const objectData = objectRes.ok ? await objectRes.json() : { result: [] }
      const attackData = attackRes.ok ? await attackRes.json() : { result: [] }
      const modelsData = modelsRes.ok ? await modelsRes.json() : []

      // 모델 데이터를 {id, name} 형식으로 변환
      const models = Array.isArray(modelsData) ? modelsData.map((model: any) => ({
        id: model.id,
        name: model.name || 'Unknown Model'
      })) : []

      setDropdowns({
        objects: objectData.result || [],
        attackMethods: attackData.result || [],
        detectors: models
      })

      setObjectName(objectData.result?.[0] || "")
      setAttackMethod(attackData.result?.[0] || "")
      setDetectorId(models[0]?.id || "")  // 첫 번째 모델의 id 사용
    } catch (error) {
      console.error("드롭다운 로드 실패:", error)
      toast.error("패치 생성 정보를 불러오지 못했습니다.")
    } finally {
      setLoadingDropdowns(false)
    }
  }

  const handleGenerate = async () => {
    if (!patchName || !objectName || !attackMethod || !detectorId) {
      toast.error("모든 필수 항목을 입력해주세요.")
      return
    }

    setIsGenerating(true)
    setOperationInProgress(true, "3D 적대적 패치 생성")
    setStatus("processing")
    setStatusMessage("패치 생성을 시작합니다...")
    setLogs([])
    setProgress({ current: 0, total: 0 })
    setCurrentImage(null)
    setShowPatchResult(false)
    setGeneratedPatches([])
    setLastStorageKey(null)

    try {
      const response = await fetch("/api/carla/sim_gen_patch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patch_name: patchName,
          object_name: objectName,
          attack_method_name: attackMethod,
          model_id: detectorId,  // model_id 전달 (메인 백엔드에서 모델 정보 조회)
          // 아래 필드는 메인 백엔드에서 자동으로 채워짐
          class_name: ["car"],
          input_size: [640, 640]
        })
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.message || "패치 생성 실패")
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
            console.log('[3D Adversarial Patch] Raw SSE line:', line)

            try {
              const data = JSON.parse(line)
              console.log('[3D Adversarial Patch] Parsed data:', data)

              if (data.state === 200 && data.result) {
                const result = data.result
                const epoch = result.epoch || 0
                const epochTotal = result.epoch_total || 0
                const batchIdx = result.batch_idx || 0
                const batchTotal = result.batch_total || 0
                const loss = result.loss
                const lossAdv = result.loss_adv
                const lossSmooth = result.loss_smooth
                const detectionScore = result.detection_score
                setProgress({ current: epoch, total: epochTotal })

                // 상태 메시지: 간략하게 표시
                setStatusMessage(
                  `Epoch ${epoch}/${epochTotal}, Batch ${batchIdx}/${batchTotal}, ` +
                  `Loss ${loss !== undefined ? loss.toFixed(4) : 'N/A'}, ` +
                  `탐지점수 ${detectionScore !== undefined ? detectionScore.toFixed(4) : 'N/A'}`
                )

                // 로그: 상세하게 표시 (loss_adv, loss_smooth 포함)
                logMessage(
                  `Epoch ${epoch}/${epochTotal} - ` +
                  `Loss ${loss !== undefined ? loss.toFixed(4) : 'N/A'} ` +
                  `(Adv: ${lossAdv !== undefined ? lossAdv.toFixed(4) : 'N/A'}, ` +
                  `Smooth: ${lossSmooth !== undefined ? lossSmooth.toFixed(4) : 'N/A'}), ` +
                  `Score ${detectionScore !== undefined ? detectionScore.toFixed(4) : 'N/A'}`
                )

                // storage_key 우선 사용, 없으면 image_path 기반으로 storage_key 생성
                if (result.storage_key) {
                  console.log('Using storage_key:', result.storage_key)
                  setLastStorageKey(result.storage_key)
                  setCurrentImage(`/api/storage/${result.storage_key}`)
                } else if (result.image_path) {
                  // image_path가 /storage/3d/... 형식이면 /api/storage/3d/...로 변환
                  const imagePath = result.image_path
                  console.log('Using image_path:', imagePath)
                  if (imagePath.startsWith('/storage/')) {
                    const storageKey = imagePath.replace('/storage/', '')
                    console.log('Converted to storage_key:', storageKey)
                    setLastStorageKey(storageKey)
                    setCurrentImage(`/api/storage/${storageKey}`)
                  } else {
                    console.warn('Unexpected image_path format:', imagePath)
                  }
                }
              }
            } catch (e) {
              console.error("스트림 파싱 실패", e)
            }
          }
        }
      }

      setStatus("success")
      setStatusMessage("적대적 패치 생성 완료")
      logMessage("패치 생성 완료", "success")

      // 패치 결과 저장 (lastStorageKey 사용)
      setGeneratedPatches([{
        id: `patch_${Date.now()}`,
        patchName: patchName,
        objectName: objectName,
        attackMethod: attackMethod,
        detectorId: detectorId,
        storageKey: lastStorageKey,
        createdAt: new Date().toISOString()
      }])

      toast.success("패치 생성 완료")
    } catch (error) {
      console.error(error)
      const msg = error instanceof Error ? error.message : "패치 생성 실패"
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
        <Label className="text-sm">패턴 이름 *</Label>
        <Input value={patchName} onChange={(e) => setPatchName(e.target.value)} placeholder="예: 3d_k2_tank" disabled={isGenerating || (showPatchResult && generatedPatches.length > 0)} />
      </div>
      <div className="space-y-2">
        <Label className="text-sm">객체 선택 *</Label>
        <Select value={objectName} onValueChange={setObjectName} disabled={isGenerating || loadingDropdowns || (showPatchResult && generatedPatches.length > 0)}>
          <SelectTrigger><SelectValue placeholder="객체 선택" /></SelectTrigger>
          <SelectContent>
            {dropdowns.objects.map(obj => <SelectItem key={obj} value={obj}>{obj}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="text-sm">공격 기법 *</Label>
        <Select value={attackMethod} onValueChange={setAttackMethod} disabled={isGenerating || loadingDropdowns || (showPatchResult && generatedPatches.length > 0)}>
          <SelectTrigger><SelectValue placeholder="공격 기법 선택" /></SelectTrigger>
          <SelectContent>
            {dropdowns.attackMethods.map(a => <SelectItem key={a} value={a}>{a}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="text-sm">객체 탐지 모델 *</Label>
        <Select value={detectorId} onValueChange={setDetectorId} disabled={isGenerating || loadingDropdowns || (showPatchResult && generatedPatches.length > 0)}>
          <SelectTrigger><SelectValue placeholder="탐지 모델 선택" /></SelectTrigger>
          <SelectContent>
            {dropdowns.detectors.map(d => <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
    </div>
  )

  const rightPanel = showPatchResult && generatedPatches.length > 0 ? (
    // State: Patch Result - Show Generated Patch
    <div className="h-full flex flex-col p-6 space-y-4 overflow-y-auto">
      <div className="pb-4 border-b border-border">
        <h3 className="text-foreground font-semibold mb-2 flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-tertiary" />
          패치 생성 완료
        </h3>
        <p className="text-muted text-sm">생성된 적대적 패치를 확인하세요</p>
      </div>

      {/* Generated Patch Info */}
      {generatedPatches[0] && (
        <Card className="bg-surface-container/50 border-border">
          <CardContent className="pt-6">
            <div className="space-y-3">
              <h4 className="text-foreground font-semibold mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4" />
                처리 정보
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted">패치 이름</span>
                  <span className="text-foreground">{generatedPatches[0].patchName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">객체</span>
                  <span className="text-foreground">{generatedPatches[0].objectName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">공격 기법</span>
                  <span className="text-foreground">{generatedPatches[0].attackMethod}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted">생성 시간</span>
                  <span className="text-foreground">
                    {new Date(generatedPatches[0].createdAt).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Patch Image */}
      <div className="bg-surface-container rounded-lg p-6 flex flex-col items-center justify-center">
        {generatedPatches[0]?.storageKey ? (
          <div className="flex flex-col items-center gap-4 w-full max-w-md">
            <div className="text-sm text-muted mb-2">생성된 패치 이미지</div>
            <div className="w-full aspect-square bg-surface-container rounded-lg overflow-hidden border-2 border-outline shadow-xl">
              <img
                src={`/api/storage/${generatedPatches[0].storageKey}`}
                alt="Generated adversarial patch"
                className="w-full h-full object-contain"
                onError={(e) => {
                  console.error('Failed to load patch image')
                  e.currentTarget.style.display = 'none'
                }}
              />
            </div>
          </div>
        ) : (
          <div className="text-muted">패치 이미지를 불러오는 중...</div>
        )}
      </div>
    </div>
  ) : isGenerating ? (
    // State: Generation in Progress - Show Real-time Preview
    <div className="h-full flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-foreground">패치 생성 진행 중 - 실시간으로 생성된 패치를 확인하세요</p>
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
                  <div className="text-xs text-muted mb-1">현재 Epoch</div>
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
            실시간 패치 미리보기
          </p>
          <div className="flex-1 bg-surface rounded-lg border border-border flex items-center justify-center overflow-hidden">
            {currentImage ? (
              <img src={currentImage} alt="real-time patch preview" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center text-muted text-sm p-6">
                <Info className="w-6 h-6 mx-auto mb-2 text-muted" />
                생성된 패치가 표시됩니다.
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
      {generatedPatches.length > 0 && (
        <div className="bg-tertiary-container border border-tertiary rounded-lg p-3">
          <p className="text-tertiary text-sm flex items-center gap-2 mb-2">
            <CheckCircle2 className="w-4 h-4" />
            패치 생성이 완료되었습니다!
          </p>
          <Button
            onClick={() => {
              setIsGenerating(false)
              setShowPatchResult(true)
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
                setShowPatchResult(false)
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
              onClick={resetPatchState}
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
              <img src={currentImage} alt="patch preview" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center text-muted text-sm p-6">
                <Info className="w-6 h-6 mx-auto mb-2 text-muted" />
                생성된 패치 이미지가 표시됩니다.
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

  const actionButtons = showPatchResult && generatedPatches.length > 0 ? (
    <div className="space-y-2">
      <Button
        onClick={async () => {
          const patch = generatedPatches[0]
          const storageKey = patch?.storageKey
          if (storageKey) {
            try {
              // 다운로드 로직
              const response = await fetch(`/api/storage/${storageKey}`)
              const blob = await response.blob()
              const url = window.URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = `${patch.patchName}.png`
              document.body.appendChild(a)
              a.click()
              window.URL.revokeObjectURL(url)
              document.body.removeChild(a)
              toast.success('패치가 다운로드되었습니다')
            } catch (error) {
              toast.error('패치 다운로드에 실패했습니다')
            }
          }
        }}
        disabled={!generatedPatches[0]?.storageKey}
        variant="tertiary"
        className="w-full"
      >
        <Download className="w-4 h-4 mr-2" />
        패치 다운로드
      </Button>

      <Button
        onClick={resetPatchState}
        className="w-full ds-btn-outline"
      >
        <Play className="w-4 h-4 mr-2" />
        초기화
      </Button>
    </div>
  ) : (
    <Button className="w-full ds-btn-primary" disabled={isGenerating || isOperationInProgress} onClick={handleGenerate}>
      <Play className="w-4 h-4 mr-2" />
      3D 적대적 패치 생성
    </Button>
  )

  return (
    <AdversarialToolLayout
      title="3D 적대적 패치 생성"
      description="CARLA 객체/공격/탐지 모델을 선택해 적대적 패치를 생성합니다."
      icon={Shield}
      leftPanel={{
        title: "패치 설정",
        icon: Target,
        description: "패턴 이름과 객체, 공격 기법, 탐지 모델을 선택하세요.",
        children: leftPanel
      }}
      rightPanel={{
        title: "생성 상태",
        icon: Zap,
        children: rightPanel
      }}
      actionButtons={actionButtons}
      disabled={isOperationInProgress}
    />
  )
}
