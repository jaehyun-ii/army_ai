import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { LucideIcon } from "lucide-react"

interface FormFieldProps {
  label: string
  name: string
  type?: "text" | "email" | "password" | "number" | "textarea" | "select"
  value?: string | number
  onChange?: (value: string) => void
  placeholder?: string
  required?: boolean
  disabled?: boolean
  icon?: LucideIcon
  options?: { value: string; label: string }[]
  rows?: number
  className?: string
  error?: string
}

export function FormField({
  label,
  name,
  type = "text",
  value,
  onChange,
  placeholder,
  required,
  disabled,
  icon: Icon,
  options,
  rows = 4,
  className,
  error
}: FormFieldProps) {
  return (
    <div className={cn("space-y-2", className)}>
      <Label htmlFor={name} className="text-sm font-medium flex items-center gap-2 text-foreground">
        {Icon && <Icon className="w-4 h-4 text-primary" />}
        {label}
        {required && <span className="text-error">*</span>}
      </Label>

      {type === "textarea" ? (
        <Textarea
          id={name}
          name={name}
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          rows={rows}
          className={cn(
            "bg-surface-container/70 border-border text-foreground placeholder:text-muted",
            "focus:border-primary focus:bg-surface-container/90",
            error && "border-error"
          )}
          required={required}
        />
      ) : type === "select" && options ? (
        <Select
          value={value as string}
          onValueChange={onChange}
          disabled={disabled}
        >
          <SelectTrigger
            className={cn(
              "bg-surface-container/70 border-border text-foreground",
              "focus:border-primary focus:bg-surface-container/90",
              error && "border-error"
            )}
          >
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent className="bg-surface-container border-border">
            {options.map((option) => (
              <SelectItem
                key={option.value}
                value={option.value}
                className="text-foreground hover:bg-surface-container-high"
              >
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <Input
          id={name}
          name={name}
          type={type}
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className={cn(
            "bg-surface-container/70 border-border text-foreground placeholder:text-muted",
            "focus:border-primary focus:bg-surface-container/90",
            error && "border-error"
          )}
          required={required}
        />
      )}

      {error && (
        <p className="text-error text-xs">{error}</p>
      )}
    </div>
  )
}