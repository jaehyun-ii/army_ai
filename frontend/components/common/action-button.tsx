import { Button } from "@/components/ui/button"
import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { styles } from "@/lib/styles"

interface ActionButtonProps {
  label: string
  icon?: LucideIcon
  onClick?: () => void
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link"
  size?: "default" | "sm" | "lg" | "icon"
  className?: string
  disabled?: boolean
  loading?: boolean
  type?: "button" | "submit" | "reset"
  gradient?: boolean
}

export function ActionButton({
  label,
  icon: Icon,
  onClick,
  variant = "default",
  size = "default",
  className,
  disabled,
  loading,
  type = "button",
  gradient = false
}: ActionButtonProps) {
  return (
    <Button
      type={type}
      variant={variant}
      size={size}
      onClick={onClick}
      disabled={disabled || loading}
      className={cn(
        gradient && styles.gradient.primary + " text-white hover:opacity-90",
        className
      )}
    >
      {loading ? (
        <>
          <span className="animate-spin mr-2">⏳</span>
          처리중...
        </>
      ) : (
        <>
          {Icon && <Icon className="w-4 h-4 mr-2" />}
          {label}
        </>
      )}
    </Button>
  )
}