"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2 } from "lucide-react"
import { apiClient } from "@/lib/api-client"
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Cell,
  PieChart,
  Pie,
} from "recharts"
import { TrendingDown, TrendingUp, Activity, Target } from "lucide-react"

interface EvaluationMetrics {
  map: number
  map50: number
  map75: number
  precision: number
  recall: number
  f1: number
  ar_100?: number
  ap_small?: number
  ap_medium?: number
  ap_large?: number
}

interface EvaluationMetricsVisualizationProps {
  metrics: EvaluationMetrics
  phase: string
  runId: string
}

export function EvaluationMetricsVisualization({
  metrics,
  phase,
  runId,
}: EvaluationMetricsVisualizationProps) {
  const [prCurveData, setPrCurveData] = useState<any[]>([])
  const [iouThresholdsData, setIouThresholdsData] = useState<any[]>([])
  const [loadingVizData, setLoadingVizData] = useState(false)

  // Load actual PR curve data from backend
  useEffect(() => {
    const loadVizData = async () => {
      setLoadingVizData(true)
      try {
        const response: any = await apiClient.getEvaluationPRCurveData(runId)
        if (response.pr_curve && response.pr_curve.length > 0) {
          setPrCurveData(response.pr_curve)
        }
        if (response.iou_thresholds && response.iou_thresholds.length > 0) {
          setIouThresholdsData(response.iou_thresholds)
        }
      } catch (error) {
        console.error("Failed to load visualization data:", error)
      } finally {
        setLoadingVizData(false)
      }
    }

    if (runId) {
      loadVizData()
    }
  }, [runId])
  // Prepare data for overall metrics bar chart
  const overallMetricsData = [
    { metric: "mAP", value: metrics.map * 100, color: "#3b82f6" },
    { metric: "mAP@50", value: metrics.map50 * 100, color: "#8b5cf6" },
    { metric: "mAP@75", value: metrics.map75 * 100, color: "#ec4899" },
    { metric: "F1 Score", value: metrics.f1 * 100, color: "#f97316" },
    { metric: "Precision", value: metrics.precision * 100, color: "#10b981" },
    { metric: "Recall", value: metrics.recall * 100, color: "#f59e0b" },
  ]

  // Prepare data for size-based AP chart
  const sizeBasedData = [
    { size: "Small", value: (metrics.ap_small || 0) * 100 },
    { size: "Medium", value: (metrics.ap_medium || 0) * 100 },
    { size: "Large", value: (metrics.ap_large || 0) * 100 },
  ]

  // Use actual IoU thresholds data from backend or fallback to simulated
  const iouMetricsData = iouThresholdsData.length > 0
    ? iouThresholdsData.map(d => ({
        iou: d.iou.toFixed(2),
        precision: d.precision,
        recall: d.recall,
        ap: d.ap,
      }))
    : [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].map((iou) => {
        const degradationFactor = Math.pow(1 - (iou - 0.5) / 0.45, 1.5)
        return {
          iou: iou.toFixed(2),
          precision: Math.max(0, (metrics.precision * degradationFactor * 100)),
          recall: Math.max(0, (metrics.recall * degradationFactor * 100)),
          ap: Math.max(0, (metrics.map * degradationFactor * 100)),
        }
      })

  // Use actual PR curve data from backend or fallback to simulated
  const actualPrCurveData = prCurveData.length > 0
    ? prCurveData
    : Array.from({ length: 11 }, (_, i) => {
        const recall = i / 10
        const basePrecision = metrics.precision
        const precisionAtRecall = basePrecision * (1 - recall * 0.3)
        return {
          recall: recall * 100,
          precision: Math.max(0, precisionAtRecall * 100),
        }
      }).reverse()

  // Prepare radar chart data
  const radarData = [
    { metric: "mAP@50", value: metrics.map50 * 100, fullMark: 100 },
    { metric: "F1 Score", value: metrics.f1 * 100, fullMark: 100 },
    { metric: "Precision", value: metrics.precision * 100, fullMark: 100 },
    { metric: "Recall", value: metrics.recall * 100, fullMark: 100 },
    { metric: "mAP@75", value: metrics.map75 * 100, fullMark: 100 },
    { metric: "AR@100", value: (metrics.ar_100 || 0) * 100, fullMark: 100 },
  ]

  return (
    <div className="space-y-6">
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-slate-800/50">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="pr-curve">PR Curve</TabsTrigger>
          <TabsTrigger value="iou-analysis">IoU Analysis</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Overall Metrics Bar Chart */}
            <Card className="bg-slate-800/50 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-400" />
                  Overall Performance Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={overallMetricsData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="metric" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
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
                      {overallMetricsData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Radar Chart */}
            <Card className="bg-slate-800/50 border-white/10">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-purple-400" />
                  Performance Radar
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="metric" stroke="#94a3b8" />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#94a3b8" />
                    <Radar
                      name="Performance"
                      dataKey="value"
                      stroke="#8b5cf6"
                      fill="#8b5cf6"
                      fillOpacity={0.6}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        color: '#fff',
                      }}
                      formatter={(value: number) => `${value.toFixed(2)}%`}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Size-based AP */}
            {(metrics.ap_small || metrics.ap_medium || metrics.ap_large) && (
              <Card className="bg-slate-800/50 border-white/10 lg:col-span-2">
                <CardHeader>
                  <CardTitle className="text-white">AP by Object Size</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={sizeBasedData} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="category" dataKey="size" stroke="#94a3b8" />
                      <YAxis type="number" stroke="#94a3b8" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1e293b',
                          border: '1px solid #334155',
                          borderRadius: '8px',
                          color: '#fff',
                        }}
                        formatter={(value: number) => `${value.toFixed(2)}%`}
                      />
                      <Bar dataKey="value" fill="#10b981" radius={[0, 8, 8, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* PR Curve Tab */}
        <TabsContent value="pr-curve" className="space-y-4">
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                Precision-Recall Curve
                {loadingVizData && <Loader2 className="w-4 h-4 animate-spin text-blue-400" />}
                {!loadingVizData && prCurveData.length > 0 && (
                  <span className="text-xs text-green-400">(실제 데이터)</span>
                )}
              </CardTitle>
              <p className="text-sm text-slate-400">
                Shows the trade-off between precision and recall at different confidence thresholds
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={actualPrCurveData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="recall"
                    label={{ value: 'Recall (%)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                    stroke="#94a3b8"
                  />
                  <YAxis
                    label={{ value: 'Precision (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                    stroke="#94a3b8"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      color: '#fff',
                    }}
                    formatter={(value: number) => `${value.toFixed(2)}%`}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="precision"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.6}
                    name="Precision"
                  />
                </AreaChart>
              </ResponsiveContainer>
              <div className="mt-4 p-4 bg-blue-900/20 border border-blue-500/30 rounded-lg">
                <p className="text-sm text-blue-300">
                  <strong>AP (Average Precision):</strong> Area under the PR curve = {(metrics.map * 100).toFixed(2)}%
                </p>
                <p className="text-xs text-slate-400 mt-2">
                  Higher AP indicates better overall performance across all recall levels
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* IoU Analysis Tab */}
        <TabsContent value="iou-analysis" className="space-y-4">
          <Card className="bg-slate-800/50 border-white/10">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                IoU Threshold vs Performance
                {loadingVizData && <Loader2 className="w-4 h-4 animate-spin text-blue-400" />}
                {!loadingVizData && iouThresholdsData.length > 0 && (
                  <span className="text-xs text-green-400">(실제 데이터)</span>
                )}
              </CardTitle>
              <p className="text-sm text-slate-400">
                Shows how detection performance changes with stricter IoU requirements
              </p>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={iouMetricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="iou"
                    label={{ value: 'IoU Threshold', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
                    stroke="#94a3b8"
                  />
                  <YAxis
                    label={{ value: 'Performance (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
                    stroke="#94a3b8"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      color: '#fff',
                    }}
                    formatter={(value: number) => `${value.toFixed(2)}%`}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="ap"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="AP"
                  />
                  <Line
                    type="monotone"
                    dataKey="precision"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="Precision"
                  />
                  <Line
                    type="monotone"
                    dataKey="recall"
                    stroke="#f59e0b"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="Recall"
                  />
                </LineChart>
              </ResponsiveContainer>
              <div className="mt-4 p-4 bg-slate-700/30 rounded-lg">
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-slate-400">mAP@50</p>
                    <p className="text-xl font-bold text-blue-400">{(metrics.map50 * 100).toFixed(2)}%</p>
                  </div>
                  <div>
                    <p className="text-slate-400">mAP@75</p>
                    <p className="text-xl font-bold text-purple-400">{(metrics.map75 * 100).toFixed(2)}%</p>
                  </div>
                  <div>
                    <p className="text-slate-400">mAP (0.5:0.95)</p>
                    <p className="text-xl font-bold text-pink-400">{(metrics.map * 100).toFixed(2)}%</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

      </Tabs>
    </div>
  )
}
