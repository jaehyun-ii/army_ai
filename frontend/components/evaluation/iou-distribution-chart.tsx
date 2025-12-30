"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { Activity, Loader2 } from "lucide-react"

interface IoUDistribution {
  iou_histogram: {
    bins: number[]
    counts: number[]
  }
  iou_mean: number
  iou_median: number
  iou_std: number
  iou_max: number
  iou_min: number
  matched_pairs_count: number
}

interface IoUDistributionChartProps {
  baseDistribution: IoUDistribution | null
  attackDistribution?: IoUDistribution | null
  isLoading?: boolean
  showAttackData?: boolean
}

export function IoUDistributionChart({
  baseDistribution,
  attackDistribution,
  isLoading = false,
  showAttackData = false,
}: IoUDistributionChartProps) {
  if (isLoading) {
    return (
      <Card className="bg-surface-container/50 border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted">
            <Activity className="w-4 h-4" />
            IoU 분포
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[350px] flex items-center justify-center">
          <div className="flex flex-col items-center gap-2 text-muted">
            <Loader2 className="w-8 h-8 animate-spin" />
            <p className="text-sm">IoU 분포 데이터를 불러오는 중...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!baseDistribution) {
    return (
      <Card className="bg-surface-container/50 border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted">
            <Activity className="w-4 h-4" />
            IoU 분포
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[350px] flex items-center justify-center">
          <p className="text-muted text-sm">IoU 분포 데이터가 없습니다</p>
        </CardContent>
      </Card>
    )
  }

  // Convert histogram data to chart format
  const chartData = baseDistribution.iou_histogram.bins.map((bin, index) => ({
    iou: bin.toFixed(2),
    base: baseDistribution.iou_histogram.counts[index] || 0,
    attack: showAttackData && attackDistribution
      ? attackDistribution.iou_histogram.counts[index] || 0
      : null,
  }))

  return (
    <Card className="bg-surface-container/50 border-border">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted">
            <Activity className="w-4 h-4" />
            IoU 분포
          </CardTitle>
        </div>
        {/* Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          <div className="bg-surface-container rounded-lg p-3 border border-border">
            <p className="text-xs text-muted mb-1">평균 IoU</p>
            <div className="flex items-center gap-2">
              <p className="text-lg font-bold text-tertiary">
                {baseDistribution.iou_mean.toFixed(3)}
              </p>
              {showAttackData && attackDistribution && (
                <p className="text-sm font-semibold text-error">
                  {attackDistribution.iou_mean.toFixed(3)}
                </p>
              )}
            </div>
          </div>
          <div className="bg-surface-container rounded-lg p-3 border border-border">
            <p className="text-xs text-muted mb-1">중앙값 IoU</p>
            <div className="flex items-center gap-2">
              <p className="text-lg font-bold text-tertiary">
                {baseDistribution.iou_median.toFixed(3)}
              </p>
              {showAttackData && attackDistribution && (
                <p className="text-sm font-semibold text-error">
                  {attackDistribution.iou_median.toFixed(3)}
                </p>
              )}
            </div>
          </div>
          <div className="bg-surface-container rounded-lg p-3 border border-border">
            <p className="text-xs text-muted mb-1">최대 IoU</p>
            <div className="flex items-center gap-2">
              <p className="text-lg font-bold text-tertiary">
                {baseDistribution.iou_max.toFixed(3)}
              </p>
              {showAttackData && attackDistribution && (
                <p className="text-sm font-semibold text-error">
                  {attackDistribution.iou_max.toFixed(3)}
                </p>
              )}
            </div>
          </div>
          <div className="bg-surface-container rounded-lg p-3 border border-border">
            <p className="text-xs text-muted mb-1">매칭 개수</p>
            <div className="flex items-center gap-2">
              <p className="text-lg font-bold text-tertiary">
                {baseDistribution.matched_pairs_count}
              </p>
              {showAttackData && attackDistribution && (
                <p className="text-sm font-semibold text-error">
                  {attackDistribution.matched_pairs_count}
                </p>
              )}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="iou"
              stroke="#94a3b8"
              label={{
                value: "IoU",
                position: "insideBottom",
                offset: -5,
                style: { fill: "#94a3b8", fontSize: 12 },
              }}
            />
            <YAxis
              stroke="#94a3b8"
              label={{
                value: "Count",
                angle: -90,
                position: "insideLeft",
                style: { fill: "#94a3b8", fontSize: 12 },
              }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
                color: "#fff",
              }}
              labelStyle={{ color: "#94a3b8" }}
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
            {showAttackData && attackDistribution && (
              <Bar dataKey="attack" fill="#ef4444" radius={[8, 8, 0, 0]} name="attack" />
            )}
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
