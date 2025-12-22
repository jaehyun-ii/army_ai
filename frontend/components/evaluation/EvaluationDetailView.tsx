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
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
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
      setEvaluationRun(run)

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

      // Class metrics removed - not used
      // loadClassMetrics()
    } catch (error) {
      console.error("Failed to load evaluation data:", error)
      toast.error("평가 데이터를 불러오는데 실패했습니다")
    } finally {
      setLoading(false)
    }
  }

  // Class metrics removed - not used
  // const loadClassMetrics = async () => {
  //   setLoadingClassMetrics(true)
  //   try {
  //     const response: any = await apiClient.getEvaluationClassMetrics(runId)
  //     setClassMetrics(response.items || [])
  //   } catch (error) {
  //     console.error("Failed to load class metrics:", error)
  //   } finally {
  //     setLoadingClassMetrics(false)
  //   }
  // }

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
              evaluationRun.phase === 'post_attack' ? 'md:grid-cols-5' : 'md:grid-cols-4'
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

              <Card className="bg-surface-container/50 border-border">
                <CardContent className="pt-6">
                  <div className="space-y-3">
                    <p className="text-sm font-semibold text-foreground">평가 파라미터</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-muted">Confidence Threshold</p>
                        <p className="text-base font-mono font-semibold text-foreground">
                          {evaluationRun.params?.conf_threshold?.toFixed(2) || '0.25'}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted">IOU Threshold</p>
                        <p className="text-base font-mono font-semibold text-primary">
                          {evaluationRun.params?.iou_threshold?.toFixed(2) || '0.45'}
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {evaluationRun.metrics_summary && (
                <Card className="bg-surface-container/50 border-border">
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <BarChart3 className="w-10 h-10 text-secondary" />
                      <div>
                        <p className="text-sm text-muted">
                          {evaluationRun.params?.target_class ? `${evaluationRun.params.target_class} AP@50` : 'mAP@50'}
                        </p>
                        <p className="text-2xl font-bold text-secondary">
                          {((evaluationRun.metrics_summary.ap50 || evaluationRun.metrics_summary.map50 || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
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
                    <Card className="bg-surface-container/50 border-border">
                      <CardHeader>
                        <CardTitle className="text-primary">주요 성능 지표</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className={`grid gap-6 ${evaluationRun.phase === 'post_attack' ? 'grid-cols-2' : 'grid-cols-1'}`}>
                          {/* 좌측: 기준 데이터셋 */}
                          <div className="space-y-3">
                            <div className="flex items-center gap-2 pb-2 border-b border-tertiary">
                              <Database className="w-5 h-5 text-tertiary" />
                              <h3 className="text-lg font-semibold text-tertiary">기준 데이터셋</h3>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                              <div className="bg-surface-container rounded p-3">
                                <p className="text-xs text-muted mb-1">F1 Score</p>
                                <p className="text-xl font-bold text-tertiary">
                                  {((evaluationRun.metrics_summary.original_metrics?.f1 || evaluationRun.metrics_summary.f1) * 100).toFixed(2)}%
                                </p>
                              </div>
                              <div className="bg-surface-container rounded p-3">
                                <p className="text-xs text-muted mb-1">
                                  {evaluationRun.params?.target_class ? `${evaluationRun.params.target_class} AP@50` : 'mAP@50'}
                                </p>
                                <p className="text-xl font-bold text-secondary">
                                  {((evaluationRun.metrics_summary.original_metrics?.ap50 || evaluationRun.metrics_summary.original_metrics?.map50 || evaluationRun.metrics_summary.ap50 || evaluationRun.metrics_summary.map50 || 0) * 100).toFixed(2)}%
                                </p>
                              </div>
                              <div className="bg-surface-container rounded p-3">
                                <p className="text-xs text-muted mb-1">Precision</p>
                                <p className="text-xl font-bold text-tertiary">
                                  {((evaluationRun.metrics_summary.original_metrics?.precision || evaluationRun.metrics_summary.precision) * 100).toFixed(2)}%
                                </p>
                              </div>
                              <div className="bg-surface-container rounded p-3">
                                <p className="text-xs text-muted mb-1">Recall</p>
                                <p className="text-xl font-bold text-tertiary">
                                  {((evaluationRun.metrics_summary.original_metrics?.recall || evaluationRun.metrics_summary.recall) * 100).toFixed(2)}%
                                </p>
                              </div>
                            </div>

                            {/* Overall Performance Metrics Chart - 기준 */}
                            <div className="mt-4">
                              <ResponsiveContainer width="100%" height={250}>
                                <BarChart
                                  data={(() => {
                                    const targetClass = evaluationRun.params?.target_class;
                                    const apMetricLabel = targetClass ? targetClass + " AP@50" : "mAP@50";
                                    const apMetricValue = (evaluationRun.metrics_summary.original_metrics?.ap50 || evaluationRun.metrics_summary.original_metrics?.map50 || evaluationRun.metrics_summary.ap50 || evaluationRun.metrics_summary.map50 || 0) * 100;
                                    return [
                                      { metric: "F1 Score", value: ((evaluationRun.metrics_summary.original_metrics?.f1 || evaluationRun.metrics_summary.f1) * 100), color: "#f97316" },
                                      { metric: apMetricLabel, value: apMetricValue, color: "#8b5cf6" },
                                      { metric: "Precision", value: ((evaluationRun.metrics_summary.original_metrics?.precision || evaluationRun.metrics_summary.precision) * 100), color: "var(--color-success)" },
                                      { metric: "Recall", value: ((evaluationRun.metrics_summary.original_metrics?.recall || evaluationRun.metrics_summary.recall) * 100), color: "var(--color-warning)" },
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
                                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                                    {(() => {
                                      const targetClass = evaluationRun.params?.target_class;
                                      const apMetricLabel = targetClass ? targetClass + " AP@50" : "mAP@50";
                                      const apMetricValue = (evaluationRun.metrics_summary.original_metrics?.ap50 || evaluationRun.metrics_summary.original_metrics?.map50 || evaluationRun.metrics_summary.ap50 || evaluationRun.metrics_summary.map50 || 0) * 100;
                                      return [
                                        { metric: "F1 Score", value: ((evaluationRun.metrics_summary.original_metrics?.f1 || evaluationRun.metrics_summary.f1) * 100), color: "#f97316" },
                                        { metric: apMetricLabel, value: apMetricValue, color: "#8b5cf6" },
                                        { metric: "Precision", value: ((evaluationRun.metrics_summary.original_metrics?.precision || evaluationRun.metrics_summary.precision) * 100), color: "var(--color-success)" },
                                        { metric: "Recall", value: ((evaluationRun.metrics_summary.original_metrics?.recall || evaluationRun.metrics_summary.recall) * 100), color: "var(--color-warning)" },
                                      ].map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                      ));
                                    })()}
                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>

                          {/* 우측: 공격 데이터셋 (post_attack만) */}
                          {evaluationRun.phase === 'post_attack' && (
                            <div className="space-y-3">
                              <div className="flex items-center gap-2 pb-2 border-b border-error">
                                <Images className="w-5 h-5 text-error" />
                                <h3 className="text-lg font-semibold text-error">공격 데이터셋</h3>
                              </div>
                              <div className="grid grid-cols-2 gap-3">
                                <div className="bg-surface-container rounded p-3">
                                  <p className="text-xs text-muted mb-1">F1 Score</p>
                                  <p className="text-xl font-bold text-tertiary">
                                    {(evaluationRun.metrics_summary.f1 * 100).toFixed(2)}%
                                  </p>
                                </div>
                                <div className="bg-surface-container rounded p-3">
                                  <p className="text-xs text-muted mb-1">{evaluationRun.params?.target_class ? `${evaluationRun.params.target_class} AP@50` : "mAP@50"}</p>
                                  <p className="text-xl font-bold text-secondary">
                                    {((evaluationRun.metrics_summary.ap50 || evaluationRun.metrics_summary.map50 || 0) * 100).toFixed(2)}%
                                  </p>
                                </div>
                                <div className="bg-surface-container rounded p-3">
                                  <p className="text-xs text-muted mb-1">Precision</p>
                                  <p className="text-xl font-bold text-tertiary">
                                    {(evaluationRun.metrics_summary.precision * 100).toFixed(2)}%
                                  </p>
                                </div>
                                <div className="bg-surface-container rounded p-3">
                                  <p className="text-xs text-muted mb-1">Recall</p>
                                  <p className="text-xl font-bold text-tertiary">
                                    {(evaluationRun.metrics_summary.recall * 100).toFixed(2)}%
                                  </p>
                                </div>
                              </div>

                              {/* Overall Performance Metrics Chart - 공격 */}
                              <div className="mt-4">
                                <ResponsiveContainer width="100%" height={250}>
                                  <BarChart
                                    data={(() => {
                                      const targetClass = evaluationRun.params?.target_class;
                                      const apMetricLabel = targetClass ? targetClass + " AP@50" : "mAP@50";
                                      const apMetricValue = ((evaluationRun.metrics_summary.ap50 || evaluationRun.metrics_summary.map50 || 0) * 100);
                                      return [
                                        { metric: "F1 Score", value: (evaluationRun.metrics_summary.f1 * 100), color: "#f97316" },
                                        { metric: apMetricLabel, value: apMetricValue, color: "#8b5cf6" },
                                        { metric: "Precision", value: (evaluationRun.metrics_summary.precision * 100), color: "var(--color-success)" },
                                        { metric: "Recall", value: (evaluationRun.metrics_summary.recall * 100), color: "var(--color-warning)" },
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
                                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                                      {(() => {
                                        const targetClass = evaluationRun.params?.target_class;
                                        const apMetricLabel = targetClass ? targetClass + " AP@50" : "mAP@50";
                                        const apMetricValue = ((evaluationRun.metrics_summary.ap50 || evaluationRun.metrics_summary.map50 || 0) * 100);
                                        return [
                                          { metric: "F1 Score", value: (evaluationRun.metrics_summary.f1 * 100), color: "#f97316" },
                                          { metric: apMetricLabel, value: apMetricValue, color: "#8b5cf6" },
                                          { metric: "Precision", value: (evaluationRun.metrics_summary.precision * 100), color: "var(--color-success)" },
                                          { metric: "Recall", value: (evaluationRun.metrics_summary.recall * 100), color: "var(--color-warning)" },
                                        ].map((entry, index) => (
                                          <Cell key={`cell-attack-${index}`} fill={entry.color} />
                                        ));
                                      })()}
                                    </Bar>
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
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
