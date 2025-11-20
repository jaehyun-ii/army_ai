import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"
import { LucideIcon } from "lucide-react"

interface MetricDisplayProps {
  label: string
  value: string | number
  unit?: string
  progress?: number
  icon?: LucideIcon
  iconColor?: string
  status?: "success" | "warning" | "error" | "info"
  className?: string
}

export function MetricDisplay({
  label,
  value,
  unit,
  progress,
  icon: Icon,
  iconColor,
  status = "info",
  className
}: MetricDisplayProps) {
  const statusColors = {
    success: "text-green-400",
    warning: "text-yellow-400",
    error: "text-red-400",
    info: "text-blue-400"
  }

  const progressColors = {
    success: "bg-green-400",
    warning: "bg-yellow-400",
    error: "bg-red-400",
    info: "bg-blue-400"
  }

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {Icon && (
            <Icon className={cn("w-4 h-4", iconColor || statusColors[status])} />
          )}
          <span className="text-sm text-slate-400">{label}</span>
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-lg font-semibold text-white">{value}</span>
          {unit && <span className="text-sm text-slate-400">{unit}</span>}
        </div>
      </div>
      {progress !== undefined && (
        <Progress
          value={progress}
          className="h-2"
          indicatorClassName={progressColors[status]}
        />
      )}
    </div>
  )
}

interface MetricGridProps {
  metrics: MetricDisplayProps[]
  columns?: 1 | 2 | 3 | 4
  className?: string
}

export function MetricGrid({ metrics, columns = 2, className }: MetricGridProps) {
  const gridCols = {
    1: "grid-cols-1",
    2: "grid-cols-1 md:grid-cols-2",
    3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4"
  }

  return (
    <div className={cn(`grid ${gridCols[columns]} gap-4`, className)}>
      {metrics.map((metric, index) => (
        <MetricDisplay key={index} {...metric} />
      ))}
    </div>
  )
}