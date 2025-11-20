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
      <Label htmlFor={name} className="text-sm font-medium flex items-center gap-2 text-slate-50">
        {Icon && <Icon className="w-4 h-4 text-primary" />}
        {label}
        {required && <span className="text-red-400">*</span>}
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
            "bg-slate-800/70 border-white/20 text-white placeholder:text-slate-400",
            "focus:border-primary focus:bg-slate-800/90",
            error && "border-red-400"
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
              "bg-slate-800/70 border-white/20 text-white",
              "focus:border-primary focus:bg-slate-800/90",
              error && "border-red-400"
            )}
          >
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent className="bg-slate-800 border-white/20">
            {options.map((option) => (
              <SelectItem
                key={option.value}
                value={option.value}
                className="text-white hover:bg-slate-700"
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
            "bg-slate-800/70 border-white/20 text-white placeholder:text-slate-400",
            "focus:border-primary focus:bg-slate-800/90",
            error && "border-red-400"
          )}
          required={required}
        />
      )}

      {error && (
        <p className="text-red-400 text-xs">{error}</p>
      )}
    </div>
  )
}