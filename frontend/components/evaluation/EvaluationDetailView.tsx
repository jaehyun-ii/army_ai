"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"
import {
  ArrowLeft,
  BarChart3,
  CheckCircle,
  XCircle,
  Loader2,
  AlertTriangle,
  Brain,
  Database,
  Images,
  Target,
  FileText,
} from "lucide-react"
import { apiClient } from "@/lib/api-client"
import { toast } from "sonner"
import { EvaluationImageComparison } from "./EvaluationImageComparison"
import { PRCurveChart } from "./pr-curve-chart"
import { IoUDistributionChart } from "./iou-distribution-chart"
import { usePRCurveData, useIoUDistribution } from "@/hooks/api"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"

interface MetricsSummary {
  // Multi-class metrics
  map?: number
  map50?: number
  map75?: number
  // Single-class metrics (target class)
  ap?: number
  ap50?: number
  ap75?: number
  target_class?: string
  // Common metrics
  precision: number
  recall: number
  f1: number
  ar_100?: number
  original_metrics?: {
    map?: number
    map50?: number
    map75?: number
    ap?: number
    ap50?: number
    ap75?: number
    precision: number
    recall: number
    f1: number
    ar_100?: number
  }
  robustness?: {
    delta_map: number
    delta_map50: number
    drop_percentage: number
    robustness_ratio: number
    delta_recall: number
    delta_precision: number
    delta_f1: number
  }
}

interface EvaluationRun {
  id: string
  name: string
  description?: string
  phase: string
  status: string
  model_id: string
  base_dataset_id?: string
  attack_dataset_id?: string
  params?: {
    target_class?: string
    conf_threshold?: number
    iou_threshold?: number
  }
  metrics_summary?: MetricsSummary
  dataset_results?: any[]
  created_at: string
  started_at?: string
  ended_at?: string
}

interface Model {
  id: string
  name: string
  model_type: string
}

interface Dataset {
  id: string
  name: string
  description?: string
}

interface EvaluationDetailViewProps {
  runId: string
  onBack: () => void
}

export function EvaluationDetailView({ runId, onBack }: EvaluationDetailViewProps) {
  const [evaluationRun, setEvaluationRun] = useState<EvaluationRun | null>(null)
  const [model, setModel] = useState<Model | null>(null)
  const [baseDataset, setBaseDataset] = useState<Dataset | null>(null)
  const [attackDataset, setAttackDataset] = useState<Dataset | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedIoU, setSelectedIoU] = useState<"0.50" | "0.75" | "0.95">("0.50")

  // Fetch data using hooks
  const prCurveQuery = usePRCurveData(runId, !!runId)
  const iouDistQuery = useIoUDistribution(runId, !!runId)

  const sanitizeNumber = (value: any) => {
    const parsed = typeof value === "string" ? parseFloat(value) : value
    return Number.isFinite(parsed) ? parsed : 0
  }

  const normalizeMetrics = (metrics: MetricsSummary | null | undefined): MetricsSummary | null => {
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
      ar_100: sanitizeNumber(metrics.ar_100),
      original_metrics: metrics.original_metrics
        ? {
            ...metrics.original_metrics,
            map: sanitizeNumber(metrics.original_metrics.map),
            map50: sanitizeNumber(metrics.original_metrics.map50),
            map75: sanitizeNumber(metrics.original_metrics.map75),
            ap: sanitizeNumber(metrics.original_metrics.ap ?? metrics.original_metrics.map),
            ap50: sanitizeNumber(metrics.original_metrics.ap50 ?? metrics.original_metrics.map50),
            ap75: sanitizeNumber(metrics.original_metrics.ap75 ?? metrics.original_metrics.map75),
            precision: sanitizeNumber(metrics.original_metrics.precision),
            recall: sanitizeNumber(metrics.original_metrics.recall),
            f1: sanitizeNumber(metrics.original_metrics.f1),
            ar_100: sanitizeNumber(metrics.original_metrics.ar_100),
          }
        : undefined,
    }
  }

  // Helper function to get attack dataset metrics
  const getAttackDatasetMetrics = () => {
    if (!evaluationRun) return null

    // First try to find attack dataset result
    const attackDatasetResult = evaluationRun.dataset_results?.find(
      (dr: any) => dr.dataset_type === 'attack'
    )

    if (attackDatasetResult?.metrics_summary) {
      console.log('[EvaluationDetailView] ✓ Using attack dataset_results metrics:', attackDatasetResult.metrics_summary)
      return normalizeMetrics(attackDatasetResult.metrics_summary)
    }

    // Fallback 1: For post_attack evaluations, the attack metrics are in eval_run.metrics_summary.overall
    // (This is the old structure before dataset_results was introduced)
    if (evaluationRun.phase === 'post_attack' && evaluationRun.metrics_summary) {
      console.log('[EvaluationDetailView] ⚠ Using eval_run.metrics_summary (fallback for post_attack):', evaluationRun.metrics_summary)
      return normalizeMetrics(evaluationRun.metrics_summary as MetricsSummary)
    }

    // Fallback 2: General fallback
    if (evaluationRun.metrics_summary) {
      console.log('[EvaluationDetailView] ⚠ Using eval_run.metrics_summary (general fallback):', evaluationRun.metrics_summary)
      return normalizeMetrics(evaluationRun.metrics_summary as MetricsSummary)
    }

    console.error('[EvaluationDetailView] ✗ No attack dataset metrics found!')
    console.error('[EvaluationDetailView] dataset_results:', evaluationRun.dataset_results)
    console.error('[EvaluationDetailView] metrics_summary:', evaluationRun.metrics_summary)
    return null
  }

  // Helper function to get base dataset metrics
  const getBaseDatasetMetrics = () => {
    if (!evaluationRun) return null

    // For post_attack evaluations, try dataset_results first
    if (evaluationRun.phase === 'post_attack') {
      const baseDatasetResult = evaluationRun.dataset_results?.find(
        (dr: any) => dr.dataset_type === 'base'
      )

      if (baseDatasetResult?.metrics_summary) {
        console.log('[EvaluationDetailView] ✓ Using base dataset_results metrics:', baseDatasetResult.metrics_summary)
        return normalizeMetrics(baseDatasetResult.metrics_summary)
      }

      // Fallback to original_metrics in eval_run.metrics_summary
      if (evaluationRun.metrics_summary?.original_metrics) {
        console.log('[EvaluationDetailView] ⚠ Using original_metrics (fallback):', evaluationRun.metrics_summary.original_metrics)
        return normalizeMetrics(evaluationRun.metrics_summary.original_metrics as MetricsSummary)
      }

      console.warn('[EvaluationDetailView] ⚠ No base dataset metrics found for post_attack, using eval_run.metrics_summary')
    }

    // For pre_attack or fallback, use eval_run.metrics_summary
    if (evaluationRun.metrics_summary) {
      console.log('[EvaluationDetailView] ✓ Using eval_run.metrics_summary for base metrics')
        return normalizeMetrics(evaluationRun.metrics_summary as MetricsSummary)
    }

    console.error('[EvaluationDetailView] ✗ No base dataset metrics found!')
    return null
  }

  // Derived metrics (updated each render from current evaluationRun)
  const baseMetrics = getBaseDatasetMetrics()
  const attackMetrics = getAttackDatasetMetrics()
  const headerMetrics = evaluationRun?.phase === 'post_attack' ? attackMetrics : baseMetrics

  useEffect(() => {
    if (runId) {
      loadEvaluationData()
    }
  }, [runId])

  const loadEvaluationData = async () => {
    setLoading(true)
    try {
      // Load evaluation run
      const run: any = await apiClient.getEvaluationRun(runId)

      // Debug logging
      console.log('[EvaluationDetailView] Loaded evaluation run:', {
        id: run.id,
        name: run.name,
        phase: run.phase,
        has_dataset_results: !!run.dataset_results,
        dataset_results_count: run.dataset_results?.length || 0,
        dataset_results: run.dataset_results,
        metrics_summary_keys: run.metrics_summary ? Object.keys(run.metrics_summary) : []
      })

      let datasetResults = run.dataset_results
      if (!datasetResults || datasetResults.length === 0) {
        try {
          const res = await fetch(`/api/evaluations/${runId}/dataset-results`)
          if (res.ok) {
            datasetResults = await res.json()
            console.log('[EvaluationDetailView] Loaded dataset_results via fallback endpoint:', datasetResults?.length)
          }
        } catch (error) {
          console.error('[EvaluationDetailView] Failed to load dataset_results fallback:', error)
        }
      }

      setEvaluationRun({ ...run, dataset_results: datasetResults })

      // Load model
      if (run.model_id) {
        try {
          const modelData: any = await apiClient.getModel(run.model_id)
          setModel(modelData)
        } catch (error) {
          console.error("Failed to load model:", error)
        }
      }

      // Load base dataset
      if (run.base_dataset_id) {
        try {
          const datasetData: any = await apiClient.getDataset(run.base_dataset_id)
          setBaseDataset(datasetData)
        } catch (error) {
          console.error("Failed to load base dataset:", error)
        }
      }

      // Load attack dataset
      if (run.attack_dataset_id) {
        try {
          const attackData: any = await apiClient.getAttackDataset(run.attack_dataset_id)
          setAttackDataset(attackData)
        } catch (error) {
          console.error("Failed to load attack dataset:", error)
        }
      }

    } catch (error) {
      console.error("Failed to load evaluation data:", error)
      toast.error("평가 데이터를 불러오는데 실패했습니다")
    } finally {
      setLoading(false)
    }
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-tertiary-container text-on-tertiary-container border-tertiary'
      case 'failed': return 'bg-error-container text-on-error-container border-error'
      case 'running': return 'bg-primary-container text-on-primary-container border-primary'
      case 'pending': return 'bg-secondary-container text-on-secondary-container border-outline-variant'
      default: return 'bg-surface-variant text-on-surface-variant border-outline'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-5 h-5 text-success" />
      case 'failed': return <XCircle className="w-5 h-5 text-error" />
      case 'running': return <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      case 'pending': return <AlertTriangle className="w-5 h-5 text-warning" />
      default: return <AlertTriangle className="w-5 h-5 text-on-surface-variant" />
    }
  }

  const getPhaseBadge = (phase: string) => {
    switch (phase) {
      case 'pre_attack': return 'bg-primary-container text-on-primary-container border-primary'
      case 'post_attack': return 'bg-error-container text-on-error-container border-error'
      default: return 'bg-surface-variant text-on-surface-variant border-outline'
    }
  }

  const getPhaseLabel = (phase: string) => {
    switch (phase) {
      case 'pre_attack': return '기준 데이터'
      case 'post_attack': return '공격 데이터'
      default: return phase
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto" />
          <p className="text-muted">평가 데이터를 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (!evaluationRun) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <AlertTriangle className="w-12 h-12 text-error mx-auto" />
          <p className="text-muted">평가 데이터를 찾을 수 없습니다</p>
          <Button onClick={onBack} variant="outline">
            돌아가기
          </Button>
        </div>
      </div>
    )
  }

  return (
    <AdversarialToolLayout
      title={evaluationRun.name}
      description={evaluationRun.description}
      icon={BarChart3}
      headerStats={
        <div className="flex items-center gap-2 sm:gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="text-muted hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            돌아가기
          </Button>
          <Badge className={getPhaseBadge(evaluationRun.phase)}>
            {getPhaseLabel(evaluationRun.phase)}
          </Badge>
          <Badge className={getStatusBadge(evaluationRun.status)}>
            <div className="flex items-center gap-2">
              {getStatusIcon(evaluationRun.status)}
              {evaluationRun.status}
            </div>
          </Badge>
        </div>
      }
      rightPanel={{
        title: "평가 결과",
        icon: FileText,
        description: "상세 평가 지표 및 이미지 비교",
        children: (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className={`grid grid-cols-1 gap-4 ${
              evaluationRun.phase === 'post_attack' ? 'md:grid-cols-4' : 'md:grid-cols-3'
            }`}>
              <Card className="bg-surface-container/50 border-border">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <Brain className="w-10 h-10 text-primary" />
                    <div>
                      <p className="text-sm text-muted">모델</p>
                      <p className="text-lg font-semibold text-foreground">{model?.name || 'Unknown'}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-surface-container/50 border-border">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <Database className="w-10 h-10 text-tertiary" />
                    <div>
                      <p className="text-sm text-muted">기준 데이터셋</p>
                      <p className="text-lg font-semibold text-foreground">
                        {baseDataset?.name || 'N/A'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {evaluationRun.phase === 'post_attack' && (
                <Card className="bg-surface-container/50 border-border">
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <Images className="w-10 h-10 text-error" />
                      <div>
                        <p className="text-sm text-muted">공격 데이터셋</p>
                        <p className="text-lg font-semibold text-foreground">
                          {attackDataset?.name || 'N/A'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card className="bg-surface-container/50 border-border">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <Target className="w-10 h-10 text-info" />
                    <div>
                      <p className="text-sm text-muted">타겟 클래스</p>
                      <p className="text-lg font-semibold text-foreground">
                        {evaluationRun.params?.target_class || 'All'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Main Content Tabs */}
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="grid w-full grid-cols-2 bg-surface-container/50">
                <TabsTrigger value="overview">개요</TabsTrigger>
                <TabsTrigger value="images">이미지 비교</TabsTrigger>
              </TabsList>

              {/* Overview Tab */}
              <TabsContent value="overview" className="space-y-6 mt-6">

                  {/* 신뢰성 표시 */}
                  {evaluationRun.phase === 'post_attack' && evaluationRun.metrics_summary?.robustness && (() => {
                    const dropPercentage = evaluationRun.metrics_summary.robustness.drop_percentage
                    let reliabilityLevel: string
                    let reliabilityColor: string
                    let reliabilityBgColor: string

                    if (dropPercentage < 5) {
                      reliabilityLevel = '신뢰성 높음'
                      reliabilityColor = 'text-tertiary'
                      reliabilityBgColor = 'bg-tertiary-container border-tertiary'
                    } else if (dropPercentage < 15) {
                      reliabilityLevel = '신뢰성 낮음'
                      reliabilityColor = 'text-tertiary'
                      reliabilityBgColor = 'bg-warning-container border-warning'
                    } else {
                      reliabilityLevel = '신뢰성 매우 낮음'
                      reliabilityColor = 'text-error'
                      reliabilityBgColor = 'bg-error-container border-error'
                    }

                    return (
                      <Card className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border-primary">
                        <CardHeader>
                          <CardTitle className="text-primary flex items-center gap-2">
                            <Target className="w-5 h-5 text-primary" />
                            신뢰성 분석
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-surface-container rounded p-4">
                              <p className="text-sm font-medium text-foreground mb-2">
                                성능 유지율
                              </p>
                              <p className="text-2xl font-bold text-primary">
                                {(evaluationRun.metrics_summary.robustness.robustness_ratio * 100).toFixed(1)}%
                              </p>
                              <p className="text-xs text-muted mt-1">
                                공격 후 남은 성능
                              </p>
                            </div>
                            <div className="bg-surface-container rounded p-4">
                              <p className="text-sm font-medium text-foreground mb-2">
                                성능 하락률
                              </p>
                              <p className="text-2xl font-bold text-error">
                                {dropPercentage.toFixed(1)}%
                              </p>
                              <p className="text-xs text-muted mt-1">
                                공격으로 인한 감소
                              </p>
                            </div>
                            <div className="bg-surface-container rounded p-4">
                              <p className="text-sm font-medium text-foreground mb-2">
                                {evaluationRun.params?.target_class ? 'AP@50 감소량' : 'mAP@50 감소량'}
                              </p>
                              <p className="text-2xl font-bold text-tertiary">
                                {Math.abs((evaluationRun.metrics_summary.robustness.delta_map50 || 0) * 100).toFixed(2)}%
                              </p>
                              <p className="text-xs text-muted mt-1">
                                절대 감소폭
                              </p>
                            </div>
                            <div className={`rounded p-4 border-2 ${reliabilityBgColor}`}>
                              <p className="text-sm font-medium text-foreground mb-2">
                                신뢰성 평가
                              </p>
                              <p className={`text-2xl font-bold ${reliabilityColor}`}>
                                {reliabilityLevel}
                              </p>
                              <p className="text-xs text-muted mt-1">
                                {dropPercentage < 5 ? '성능 하락율 5% 미만' : dropPercentage < 15 ? '성능 하락율 5-15%' : '성능 하락율 15% 이상'}
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )
                  })()}

                  {/* 성능 지표 비교: 기준 vs 공격 */}
                  {evaluationRun.metrics_summary && (
                    <div className="space-y-6">
                      {/* AP 메트릭과 Classification 메트릭을 2열로 배치 */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* AP 메트릭 카드 (IoU 독립적) */}
                        <Card className="bg-surface-container/50 border-border">
                        <CardHeader>
                          <CardTitle className="text-primary">Average Precision (AP) 메트릭</CardTitle>
                          <p className="text-sm text-muted mt-2">
                            전체 confidence threshold 범위에서 계산된 Average Precision
                          </p>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            {/* 기준 데이터셋 AP 메트릭 */}
                            <div className="space-y-3">
                              <div className="flex items-center gap-2">
                                <Database className="w-4 h-4 text-tertiary" />
                                <h3 className="text-sm font-semibold text-tertiary">기준 데이터셋</h3>
                              </div>
                              <div className="grid grid-cols-3 gap-3">
                                <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                  <p className="text-xs text-muted mb-1">
                                    {evaluationRun.params?.target_class ? `AP@50` : 'mAP@50'}
                                  </p>
                                  <p className="text-xl font-bold text-secondary">
                                    {((baseMetrics?.ap50 || baseMetrics?.map50 || 0) * 100).toFixed(2)}%
                                  </p>
                                </div>
                                <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                  <p className="text-xs text-muted mb-1">
                                    {evaluationRun.params?.target_class ? `AP@75` : 'mAP@75'}
                                  </p>
                                  <p className="text-xl font-bold text-secondary">
                                    {((baseMetrics?.ap75 || baseMetrics?.map75 || 0) * 100).toFixed(2)}%
                                  </p>
                                </div>
                                <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                  <p className="text-xs text-muted mb-1">
                                    {evaluationRun.params?.target_class ? `AP@50:95` : 'mAP@50:95'}
                                  </p>
                                  <p className="text-xl font-bold text-secondary">
                                    {((baseMetrics?.ap || baseMetrics?.map || 0) * 100).toFixed(2)}%
                                  </p>
                                </div>
                              </div>
                            </div>

                            {/* 공격 데이터셋 AP 메트릭 */}
                            {evaluationRun.phase === 'post_attack' && (
                              <div className="space-y-3">
                                <div className="flex items-center gap-2">
                                  <Images className="w-4 h-4 text-error" />
                                  <h3 className="text-sm font-semibold text-error">공격 데이터셋</h3>
                                </div>
                                <div className="grid grid-cols-3 gap-3">
                                  <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                    <p className="text-xs text-muted mb-1">
                                      {evaluationRun.params?.target_class ? `AP@50` : 'mAP@50'}
                                    </p>
                                    <p className="text-xl font-bold text-secondary">
                                      {((attackMetrics?.ap50 || attackMetrics?.map50 || 0) * 100).toFixed(2)}%
                                    </p>
                                  </div>
                                  <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                    <p className="text-xs text-muted mb-1">
                                      {evaluationRun.params?.target_class ? `AP@75` : 'mAP@75'}
                                    </p>
                                    <p className="text-xl font-bold text-secondary">
                                      {((attackMetrics?.ap75 || attackMetrics?.map75 || 0) * 100).toFixed(2)}%
                                    </p>
                                  </div>
                                  <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                    <p className="text-xs text-muted mb-1">
                                      {evaluationRun.params?.target_class ? `AP@50:95` : 'mAP@50:95'}
                                    </p>
                                    <p className="text-xl font-bold text-secondary">
                                      {((attackMetrics?.ap || attackMetrics?.map || 0) * 100).toFixed(2)}%
                                    </p>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>

                      {/* IoU-dependent 메트릭 카드 */}
                      <Card className="bg-surface-container/50 border-border">
                        <CardHeader>
                          <div className="flex items-center justify-between">
                            <div>
                              <CardTitle className="text-primary">Classification 메트릭 (IoU Threshold별)</CardTitle>
                              <p className="text-sm text-muted mt-2">
                                선택한 IoU threshold에서의 Precision, Recall, F1 Score
                              </p>
                            </div>
                            <Tabs value={selectedIoU} onValueChange={(value) => setSelectedIoU(value as any)}>
                              <TabsList className="bg-surface-container">
                                <TabsTrigger value="0.50" className="text-xs">IoU 0.50</TabsTrigger>
                                <TabsTrigger value="0.75" className="text-xs">IoU 0.75</TabsTrigger>
                                <TabsTrigger value="0.95" className="text-xs">IoU 0.95</TabsTrigger>
                              </TabsList>
                            </Tabs>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-4">
                            {/* 기준 데이터셋 Classification 메트릭 */}
                            <div className="space-y-3">
                              <div className="flex items-center gap-2">
                                <Database className="w-4 h-4 text-tertiary" />
                                <h3 className="text-sm font-semibold text-tertiary">기준 데이터셋</h3>
                              </div>
                              <div className="grid grid-cols-3 gap-3">
                                <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                  <p className="text-xs text-muted mb-1">F1 Score</p>
                                  <p className="text-xl font-bold text-secondary">
                                    {(() => {
                                      const iouKey = selectedIoU.replace('.', '')
                                      const f1 = baseMetrics?.[`f1_iou${iouKey}` as keyof typeof baseMetrics] || baseMetrics?.f1 || 0
                                      return ((f1 as number) * 100).toFixed(2)
                                    })()}%
                                  </p>
                                </div>
                                <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                  <p className="text-xs text-muted mb-1">Precision</p>
                                  <p className="text-xl font-bold text-secondary">
                                    {(() => {
                                      const iouKey = selectedIoU.replace('.', '')
                                      const precision = baseMetrics?.[`precision_iou${iouKey}` as keyof typeof baseMetrics] || baseMetrics?.precision || 0
                                      return ((precision as number) * 100).toFixed(2)
                                    })()}%
                                  </p>
                                </div>
                                <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                  <p className="text-xs text-muted mb-1">Recall</p>
                                  <p className="text-xl font-bold text-secondary">
                                    {(() => {
                                      const iouKey = selectedIoU.replace('.', '')
                                      const recall = baseMetrics?.[`recall_iou${iouKey}` as keyof typeof baseMetrics] || baseMetrics?.recall || 0
                                      return ((recall as number) * 100).toFixed(2)
                                    })()}%
                                  </p>
                                </div>
                              </div>
                            </div>

                            {/* 공격 데이터셋 Classification 메트릭 */}
                            {evaluationRun.phase === 'post_attack' && (
                              <div className="space-y-3">
                                <div className="flex items-center gap-2">
                                  <Images className="w-4 h-4 text-error" />
                                  <h3 className="text-sm font-semibold text-error">공격 데이터셋</h3>
                                </div>
                                <div className="grid grid-cols-3 gap-3">
                                  <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                    <p className="text-xs text-muted mb-1">F1 Score</p>
                                    <p className="text-xl font-bold text-secondary">
                                      {(() => {
                                        const iouKey = selectedIoU.replace('.', '')
                                        const f1 = attackMetrics?.[`f1_iou${iouKey}` as keyof typeof attackMetrics] || attackMetrics?.f1 || 0
                                        return ((f1 as number) * 100).toFixed(2)
                                      })()}%
                                    </p>
                                  </div>
                                  <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                    <p className="text-xs text-muted mb-1">Precision</p>
                                    <p className="text-xl font-bold text-secondary">
                                      {(() => {
                                        const iouKey = selectedIoU.replace('.', '')
                                        const precision = attackMetrics?.[`precision_iou${iouKey}` as keyof typeof attackMetrics] || attackMetrics?.precision || 0
                                        return ((precision as number) * 100).toFixed(2)
                                      })()}%
                                    </p>
                                  </div>
                                  <div className="bg-surface-container rounded-lg p-3 border border-border hover:border-secondary transition-colors">
                                    <p className="text-xs text-muted mb-1">Recall</p>
                                    <p className="text-xl font-bold text-secondary">
                                      {(() => {
                                        const iouKey = selectedIoU.replace('.', '')
                                        const recall = attackMetrics?.[`recall_iou${iouKey}` as keyof typeof attackMetrics] || attackMetrics?.recall || 0
                                        return ((recall as number) * 100).toFixed(2)
                                      })()}%
                                    </p>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                      </div>

                      {/* 성능 비교 차트 */}
                      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Bar Chart - Base vs Attack */}
                        <Card className="bg-surface-container/50 border-border">
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted">
                              <BarChart3 className="w-4 h-4" />
                              성능 지표 비교
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <ResponsiveContainer width="100%" height={350}>
                              <BarChart
                                data={(() => {
                                  const targetClass = evaluationRun.params?.target_class;
                                  const ap50Label = targetClass ? `AP@50` : "mAP@50";
                                  const ap75Label = targetClass ? `AP@75` : "mAP@75";
                                  const apLabel = targetClass ? `AP@50:95` : "mAP@50:95";

                                  const baseData = {
                                    f1: ((baseMetrics?.f1 || 0) * 100),
                                    ap50: ((baseMetrics?.ap50 || baseMetrics?.map50 || 0) * 100),
                                    ap75: ((baseMetrics?.ap75 || baseMetrics?.map75 || 0) * 100),
                                    ap: ((baseMetrics?.ap || baseMetrics?.map || 0) * 100),
                                    precision: ((baseMetrics?.precision || 0) * 100),
                                    recall: ((baseMetrics?.recall || 0) * 100),
                                  };

                                  const attackData = evaluationRun.phase === 'post_attack' && attackMetrics ? {
                                    f1: ((attackMetrics.f1 || 0) * 100),
                                    ap50: ((attackMetrics.ap50 || attackMetrics.map50 || 0) * 100),
                                    ap75: ((attackMetrics.ap75 || attackMetrics.map75 || 0) * 100),
                                    ap: ((attackMetrics.ap || attackMetrics.map || 0) * 100),
                                    precision: ((attackMetrics.precision || 0) * 100),
                                    recall: ((attackMetrics.recall || 0) * 100),
                                  } : null;

                                  return [
                                    {
                                      metric: ap50Label,
                                      base: baseData.ap50,
                                      attack: attackData?.ap50
                                    },
                                    {
                                      metric: ap75Label,
                                      base: baseData.ap75,
                                      attack: attackData?.ap75
                                    },
                                    {
                                      metric: apLabel,
                                      base: baseData.ap,
                                      attack: attackData?.ap
                                    },
                                    {
                                      metric: "F1 Score",
                                      base: baseData.f1,
                                      attack: attackData?.f1
                                    },
                                    {
                                      metric: "Precision",
                                      base: baseData.precision,
                                      attack: attackData?.precision
                                    },
                                    {
                                      metric: "Recall",
                                      base: baseData.recall,
                                      attack: attackData?.recall
                                    },
                                  ];
                                })()}
                                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="metric" stroke="#94a3b8" />
                                <YAxis stroke="#94a3b8" domain={[0, 100]} />
                                <Tooltip
                                  contentStyle={{
                                    backgroundColor: '#1e293b',
                                    border: '1px solid #334155',
                                    borderRadius: '8px',
                                    color: '#fff',
                                  }}
                                  formatter={(value: number) => `${value.toFixed(2)}%`}
                                />
                                <Legend
                                  wrapperStyle={{ fontSize: 12 }}
                                  formatter={(value) => {
                                    if (value === "base") return "기준 데이터"
                                    if (value === "attack") return "공격 데이터"
                                    return value
                                  }}
                                />
                                <Bar dataKey="base" fill="#3b82f6" radius={[8, 8, 0, 0]} name="base" />
                                {evaluationRun.phase === 'post_attack' && (
                                  <Bar dataKey="attack" fill="#ef4444" radius={[8, 8, 0, 0]} name="attack" />
                                )}
                              </BarChart>
                            </ResponsiveContainer>
                          </CardContent>
                        </Card>

                        {/* PR Curve Chart */}
                        <PRCurveChart
                          data={prCurveQuery.data}
                          isLoading={prCurveQuery.isLoading}
                          showAttackCurve={evaluationRun.phase === 'post_attack'}
                        />

                        {/* IoU Distribution Chart */}
                        <IoUDistributionChart
                          baseDistribution={iouDistQuery.data?.base || null}
                          attackDistribution={iouDistQuery.data?.attack || null}
                          isLoading={iouDistQuery.isLoading}
                          showAttackData={evaluationRun.phase === 'post_attack'}
                        />
                      </div>
                    </div>
                  )}
              </TabsContent>

              {/* Images Tab */}
              <TabsContent value="images" className="space-y-6 mt-6">
                  <EvaluationImageComparison
                    runId={evaluationRun.id}
                    phase={evaluationRun.phase}
                    baseDatasetId={evaluationRun.base_dataset_id}
                    attackDatasetId={evaluationRun.attack_dataset_id}
                    targetClass={evaluationRun.params?.target_class}
                  />
              </TabsContent>
            </Tabs>
          </div>
        )
      }}
    />
  )
}
