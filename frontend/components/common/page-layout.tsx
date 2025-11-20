import { cn } from "@/lib/utils"

interface PageLayoutProps {
  children: React.ReactNode
  className?: string
}

export function PageLayout({ children, className }: PageLayoutProps) {
  return (
    <div className={cn("h-full flex flex-col gap-4", className)}>
      {children}
    </div>
  )
}

interface PageHeaderProps {
  children: React.ReactNode
  className?: string
}

export function PageHeader({ children, className }: PageHeaderProps) {
  return (
    <div className={cn("flex-shrink-0", className)}>
      {children}
    </div>
  )
}

interface PageContentProps {
  children: React.ReactNode
  className?: string
  scrollable?: boolean
}

export function PageContent({ children, className, scrollable = true }: PageContentProps) {
  return (
    <div className={cn(
      "flex-1 min-h-0",
      scrollable && "overflow-y-auto scrollbar-thin",
      className
    )}>
      {children}
    </div>
  )
}