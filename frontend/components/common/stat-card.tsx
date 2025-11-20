import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { styles } from "@/lib/styles"

interface StatCardProps {
  title: string
  value: string | number
  description?: string
  icon?: LucideIcon
  iconColor?: string
  trend?: string
  trendIcon?: LucideIcon
  className?: string
}

export function StatCard({
  title,
  value,
  description,
  icon: Icon,
  iconColor = "text-white",
  trend,
  trendIcon: TrendIcon,
  className
}: StatCardProps) {
  return (
    <Card className={cn(styles.card.base, className)}>
      <CardHeader className="py-3">
        <CardTitle className="text-white text-base flex items-center gap-2">
          {Icon && <Icon className={cn("w-5 h-5", iconColor)} />}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="pb-3">
        <div className="text-2xl font-bold text-white">{value}</div>
        {description && (
          <p className="text-slate-400 text-xs">{description}</p>
        )}
        {trend && (
          <div className="flex items-center gap-1 text-green-400 text-sm mt-1">
            {TrendIcon && <TrendIcon className="w-4 h-4" />}
            <span>{trend}</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}