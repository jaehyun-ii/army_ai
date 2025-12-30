"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from "recharts"
import { TrendingUp, Loader2 } from "lucide-react"

interface PRCurveData {
  base?: {
    "iou_0.50"?: PRCurvePoints
    "iou_0.75"?: PRCurvePoints
    "iou_0.95"?: PRCurvePoints
  }
  attack?: {
    "iou_0.50"?: PRCurvePoints
    "iou_0.75"?: PRCurvePoints
    "iou_0.95"?: PRCurvePoints
  }
}

interface PRCurvePoints {
  precisions: number[]
  recalls: number[]
  confidence_thresholds: number[]
  ap: number
}

interface PRCurveChartProps {
  data: PRCurveData | null
  isLoading?: boolean
  showAttackCurve?: boolean
}

export function PRCurveChart({ data, isLoading = false, showAttackCurve = true }: PRCurveChartProps) {
  const [selectedIoU, setSelectedIoU] = useState<"0.50" | "0.75" | "0.95">("0.50")

  if (isLoading) {
    return (
      <Card className="bg-surface-container/50 border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted">
            <TrendingUp className="w-4 h-4" />
            PR 커브
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[350px] flex items-center justify-center">
          <div className="flex flex-col items-center gap-2 text-muted">
            <Loader2 className="w-8 h-8 animate-spin" />
            <p className="text-sm">PR 커브 데이터를 불러오는 중...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!data || (!data.base && !data.attack)) {
    return (
      <Card className="bg-surface-container/50 border-border">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted">
            <TrendingUp className="w-4 h-4" />
            PR 커브
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[350px] flex items-center justify-center">
          <p className="text-muted text-sm">PR 커브 데이터가 없습니다</p>
        </CardContent>
      </Card>
    )
  }

  const iouKey = `iou_${selectedIoU}` as const
  const baseCurve = data.base?.[iouKey]
  const attackCurve = data.attack?.[iouKey]

  // Convert PR curve data to chart format
  const chartData = baseCurve
    ? baseCurve.recalls.map((recall, index) => ({
        recall: recall, // Keep as 0.0-1.0
        basePrecision: baseCurve.precisions[index],
        attackPrecision:
          showAttackCurve && attackCurve ? attackCurve.precisions[index] : null,
      }))
    : []

  return (
    <Card className="bg-surface-container/50 border-border">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted">
            <TrendingUp className="w-4 h-4" />
            PR 커브
          </CardTitle>
          <Tabs value={selectedIoU} onValueChange={(value) => setSelectedIoU(value as any)}>
            <TabsList className="bg-surface-container">
              <TabsTrigger value="0.50" className="text-xs">
                IoU 0.50
              </TabsTrigger>
              <TabsTrigger value="0.75" className="text-xs">
                IoU 0.75
              </TabsTrigger>
              <TabsTrigger value="0.95" className="text-xs">
                IoU 0.95
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        {baseCurve && (
          <div className="flex gap-4 text-xs text-muted mt-2">
            <div>
              기준 AP: <span className="font-semibold" style={{ color: "#3b82f6" }}>{baseCurve.ap.toFixed(3)}</span>
            </div>
            {showAttackCurve && attackCurve && (
              <div>
                공격 AP:{" "}
                <span className="font-semibold" style={{ color: "#ef4444" }}>{attackCurve.ap.toFixed(3)}</span>
              </div>
            )}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis
              type="number"
              dataKey="recall"
              label={{
                value: "Recall",
                position: "insideBottom",
                offset: -5,
                style: { fill: "hsl(var(--muted-foreground))", fontSize: 12 },
              }}
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              domain={[0, 1]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
              tickFormatter={(value: number) => value.toFixed(1)}
              allowDataOverflow={false}
              scale="linear"
            />
            <YAxis
              label={{
                value: "Precision",
                angle: -90,
                position: "insideLeft",
                style: { fill: "hsl(var(--muted-foreground))", fontSize: 12 },
              }}
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              domain={[0, 1]}
              ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
              tickFormatter={(value: number) => value.toFixed(1)}
              allowDataOverflow={false}
              scale="linear"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--surface-container))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "6px",
                fontSize: 12,
              }}
              labelStyle={{ color: "hsl(var(--foreground))" }}
              formatter={(value: number) => value.toFixed(3)}
            />
            <Legend
              wrapperStyle={{ fontSize: 12 }}
              iconType="rect"
              formatter={(value) => {
                if (value === "basePrecision") return "기준 데이터"
                if (value === "attackPrecision") return "공격 데이터"
                return value
              }}
            />
            {/* Base dataset area and line - Blue tones */}
            <Area
              type="monotone"
              dataKey="basePrecision"
              fill="#3b82f6"
              fillOpacity={0.2}
              stroke="#3b82f6"
              strokeWidth={3}
              dot={false}
              name="basePrecision"
            />
            {/* Attack dataset area and line - Red tones */}
            {showAttackCurve && attackCurve && (
              <Area
                type="monotone"
                dataKey="attackPrecision"
                fill="#ef4444"
                fillOpacity={0.2}
                stroke="#ef4444"
                strokeWidth={3}
                dot={false}
                name="attackPrecision"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
