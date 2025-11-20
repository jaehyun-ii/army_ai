"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
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
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts"

interface MetricsSummary {
  map: number
  map50: number
  map75: number
  precision: number
  recall: number
  f1: number
  ar_100: number
  original_metrics?: {
    map: number
    map50: number
    map75: number
    precision: number
    recall: number
    f1: number
    ar_100: number
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
      case 'completed': return 'bg-green-900/40 text-green-300 border-green-500/40'
      case 'failed': return 'bg-red-900/40 text-red-300 border-red-500/40'
      case 'running': return 'bg-blue-900/40 text-blue-300 border-blue-500/40'
      case 'pending': return 'bg-yellow-900/40 text-yellow-300 border-yellow-500/40'
      default: return 'bg-slate-900/40 text-slate-300 border-slate-500/40'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-5 h-5 text-green-400" />
      case 'failed': return <XCircle className="w-5 h-5 text-red-400" />
      case 'running': return <div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
      case 'pending': return <AlertTriangle className="w-5 h-5 text-yellow-400" />
      default: return <AlertTriangle className="w-5 h-5 text-slate-400" />
    }
  }

  const getPhaseBadge = (phase: string) => {
    switch (phase) {
      case 'pre_attack': return 'bg-blue-900/40 text-blue-300 border-blue-500/40'
      case 'post_attack': return 'bg-red-900/40 text-red-300 border-red-500/40'
      default: return 'bg-slate-900/40 text-slate-300 border-slate-500/40'
    }
  }

  const getPhaseLabel = (phase: string) => {
    switch (phase) {
      case 'pre_attack': return '기준 데이터'
      case 'post_attack': return '공격 데이터'
      default: return phase
    }
  }

  const formatDate = (dateString: string) => {
    if (!dateString) return '-'
    const date = new Date(dateString)
    return date.toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 animate-spin text-blue-400 mx-auto" />
          <p className="text-slate-400">평가 데이터를 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (!evaluationRun) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <AlertTriangle className="w-12 h-12 text-red-400 mx-auto" />
          <p className="text-slate-400">평가 데이터를 찾을 수 없습니다</p>
          <Button onClick={onBack} variant="outline">
            돌아가기
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-[1800px] mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={onBack}
              className="text-slate-400 hover:text-white"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              돌아가기
            </Button>
            <div>
              <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                <BarChart3 className="w-7 h-7 text-blue-400" />
                {evaluationRun.name}
              </h1>
              <p className="text-sm text-slate-400 mt-1">{evaluationRun.description}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
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
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="bg-slate-800/50 border-white/10">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <Brain className="w-10 h-10 text-blue-400" />
                <div>
                  <p className="text-sm text-slate-400">모델</p>
                  <p className="text-lg font-semibold text-white">{model?.name || 'Unknown'}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <Database className="w-10 h-10 text-green-400" />
                <div>
                  <p className="text-sm text-slate-400">기준 데이터셋</p>
                  <p className="text-lg font-semibold text-white">
                    {baseDataset?.name || 'N/A'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <Images className="w-10 h-10 text-red-400" />
                <div>
                  <p className="text-sm text-slate-400">공격 데이터셋</p>
                  <p className="text-lg font-semibold text-white">
                    {attackDataset?.name || 'N/A'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {evaluationRun.metrics_summary && (
            <Card className="bg-slate-800/50 border-white/10">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3">
                  <BarChart3 className="w-10 h-10 text-purple-400" />
                  <div>
                    <p className="text-sm text-slate-400">mAP@50</p>
                    <p className="text-2xl font-bold text-purple-400">
                      {(evaluationRun.metrics_summary.map50 * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-2 bg-slate-800/50">
            <TabsTrigger value="overview">개요</TabsTrigger>
            <TabsTrigger value="images">이미지 비교</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6 mt-6">


            {/* 성능 지표 비교: 기준 vs 공격 */}
            {evaluationRun.metrics_summary && (
              <Card className="bg-slate-800/50 border-white/10">
                <CardHeader>
                  <CardTitle className="text-white">주요 성능 지표</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-6">
                    {/* 좌측: 기준 데이터셋 */}
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 pb-2 border-b border-green-500/30">
                        <Database className="w-5 h-5 text-green-400" />
                        <h3 className="text-lg font-semibold text-green-400">기준 데이터셋</h3>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-slate-900/50 rounded p-3">
                          <p className="text-xs text-slate-400 mb-1">F1 Score</p>
                          <p className="text-xl font-bold text-orange-400">
                            {((evaluationRun.metrics_summary.original_metrics?.f1 || evaluationRun.metrics_summary.f1) * 100).toFixed(2)}%
                          </p>
                        </div>
                        <div className="bg-slate-900/50 rounded p-3">
                          <p className="text-xs text-slate-400 mb-1">mAP@50</p>
                          <p className="text-xl font-bold text-purple-400">
                            {((evaluationRun.metrics_summary.original_metrics?.map50 || evaluationRun.metrics_summary.map50) * 100).toFixed(2)}%
                          </p>
                        </div>
                        <div className="bg-slate-900/50 rounded p-3">
                          <p className="text-xs text-slate-400 mb-1">Precision</p>
                          <p className="text-xl font-bold text-green-400">
                            {((evaluationRun.metrics_summary.original_metrics?.precision || evaluationRun.metrics_summary.precision) * 100).toFixed(2)}%
                          </p>
                        </div>
                        <div className="bg-slate-900/50 rounded p-3">
                          <p className="text-xs text-slate-400 mb-1">Recall</p>
                          <p className="text-xl font-bold text-yellow-400">
                            {((evaluationRun.metrics_summary.original_metrics?.recall || evaluationRun.metrics_summary.recall) * 100).toFixed(2)}%
                          </p>
                        </div>
                      </div>

                      {/* Overall Performance Metrics Chart - 기준 */}
                      <div className="mt-4">
                        <ResponsiveContainer width="100%" height={250}>
                          <BarChart
                            data={[
                              { metric: "F1 Score", value: ((evaluationRun.metrics_summary.original_metrics?.f1 || evaluationRun.metrics_summary.f1) * 100), color: "#f97316" },
                              { metric: "mAP@50", value: ((evaluationRun.metrics_summary.original_metrics?.map50 || evaluationRun.metrics_summary.map50) * 100), color: "#8b5cf6" },
                              { metric: "Precision", value: ((evaluationRun.metrics_summary.original_metrics?.precision || evaluationRun.metrics_summary.precision) * 100), color: "#10b981" },
                              { metric: "Recall", value: ((evaluationRun.metrics_summary.original_metrics?.recall || evaluationRun.metrics_summary.recall) * 100), color: "#f59e0b" },
                            ]}
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
                              {[
                                { metric: "F1 Score", value: ((evaluationRun.metrics_summary.original_metrics?.f1 || evaluationRun.metrics_summary.f1) * 100), color: "#f97316" },
                                { metric: "mAP@50", value: ((evaluationRun.metrics_summary.original_metrics?.map50 || evaluationRun.metrics_summary.map50) * 100), color: "#8b5cf6" },
                                { metric: "Precision", value: ((evaluationRun.metrics_summary.original_metrics?.precision || evaluationRun.metrics_summary.precision) * 100), color: "#10b981" },
                                { metric: "Recall", value: ((evaluationRun.metrics_summary.original_metrics?.recall || evaluationRun.metrics_summary.recall) * 100), color: "#f59e0b" },
                              ].map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* 우측: 공격 데이터셋 (post_attack만) */}
                    {evaluationRun.phase === 'post_attack' && (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2 pb-2 border-b border-red-500/30">
                          <Images className="w-5 h-5 text-red-400" />
                          <h3 className="text-lg font-semibold text-red-400">공격 데이터셋</h3>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-slate-900/50 rounded p-3">
                            <p className="text-xs text-slate-400 mb-1">F1 Score</p>
                            <p className="text-xl font-bold text-orange-400">
                              {(evaluationRun.metrics_summary.f1 * 100).toFixed(2)}%
                            </p>
                          </div>
                          <div className="bg-slate-900/50 rounded p-3">
                            <p className="text-xs text-slate-400 mb-1">mAP@50</p>
                            <p className="text-xl font-bold text-purple-400">
                              {(evaluationRun.metrics_summary.map50 * 100).toFixed(2)}%
                            </p>
                          </div>
                          <div className="bg-slate-900/50 rounded p-3">
                            <p className="text-xs text-slate-400 mb-1">Precision</p>
                            <p className="text-xl font-bold text-green-400">
                              {(evaluationRun.metrics_summary.precision * 100).toFixed(2)}%
                            </p>
                          </div>
                          <div className="bg-slate-900/50 rounded p-3">
                            <p className="text-xs text-slate-400 mb-1">Recall</p>
                            <p className="text-xl font-bold text-yellow-400">
                              {(evaluationRun.metrics_summary.recall * 100).toFixed(2)}%
                            </p>
                          </div>
                        </div>

                        {/* Overall Performance Metrics Chart - 공격 */}
                        <div className="mt-4">
                          <ResponsiveContainer width="100%" height={250}>
                            <BarChart
                              data={[
                                { metric: "F1 Score", value: (evaluationRun.metrics_summary.f1 * 100), color: "#f97316" },
                                { metric: "mAP@50", value: (evaluationRun.metrics_summary.map50 * 100), color: "#8b5cf6" },
                                { metric: "Precision", value: (evaluationRun.metrics_summary.precision * 100), color: "#10b981" },
                                { metric: "Recall", value: (evaluationRun.metrics_summary.recall * 100), color: "#f59e0b" },
                              ]}
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
                                {[
                                  { metric: "F1 Score", value: (evaluationRun.metrics_summary.f1 * 100), color: "#f97316" },
                                  { metric: "mAP@50", value: (evaluationRun.metrics_summary.map50 * 100), color: "#8b5cf6" },
                                  { metric: "Precision", value: (evaluationRun.metrics_summary.precision * 100), color: "#10b981" },
                                  { metric: "Recall", value: (evaluationRun.metrics_summary.recall * 100), color: "#f59e0b" },
                                ].map((entry, index) => (
                                  <Cell key={`cell-attack-${index}`} fill={entry.color} />
                                ))}
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
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
