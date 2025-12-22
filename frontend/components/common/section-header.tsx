import { Badge } from "@/components/ui/badge"
import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

interface SectionHeaderProps {
  title: string
  description?: string
  icon?: LucideIcon
  badge?: {
    text: string
    variant?: "default" | "secondary" | "destructive" | "outline"
  }
  actions?: React.ReactNode
  className?: string
}

export function SectionHeader({
  title,
  description,
  icon: Icon,
  badge,
  actions,
  className
}: SectionHeaderProps) {
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {Icon && (
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
              <Icon className="w-6 h-6 text-foreground" />
            </div>
          )}
          <div>
            <h2 className="text-primary">{title}</h2>
            {description && (
              <p className="text-muted text-sm">{description}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {badge && (
            <Badge variant={badge.variant || "default"}>
              {badge.text}
            </Badge>
          )}
          {actions}
        </div>
      </div>
    </div>
  )
}