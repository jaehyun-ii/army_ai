import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { styles } from "@/lib/styles"
import { type VariantProps } from "class-variance-authority"
import { cardVariants } from "@/components/ui/card"

interface StatCardProps {
  title: string
  value: string | number
  description?: string
  icon?: LucideIcon
  iconColor?: string
  trend?: string
  trendIcon?: LucideIcon
  className?: string
  variant?: VariantProps<typeof cardVariants>['variant']
  valueColor?: string
  trendColor?: string
}

export function StatCard({
  title,
  value,
  description,
  icon: Icon,
  iconColor = "text-foreground",
  trend,
  trendIcon: TrendIcon,
  className,
  variant = "default",
  valueColor = "text-foreground",
  trendColor = "text-on-surface-variant"
}: StatCardProps) {
  return (
    <Card variant={variant} className={cn(styles.card.interactive, className)}>
      <CardHeader className="py-3">
        <CardTitle className="text-base flex items-center gap-2">
          {Icon && <Icon className={cn("w-5 h-5", iconColor)} />}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="pb-3">
        <div className={cn("text-2xl font-bold", valueColor)}>{value}</div>
        {description && (
          <p className="text-muted text-xs mt-1">{description}</p>
        )}
        {trend && (
          <div className={cn("flex items-center gap-1 text-sm mt-2", trendColor)}>
            {TrendIcon && <TrendIcon className="w-4 h-4" />}
            <span>{trend}</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}