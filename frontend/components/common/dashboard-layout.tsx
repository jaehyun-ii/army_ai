import React from 'react'
import { cn } from '@/lib/utils'

interface DashboardLayoutProps {
  children: React.ReactNode
  className?: string
}

/**
 * 대시보드 레이아웃 HOC
 */
export function DashboardLayout({ children, className }: DashboardLayoutProps) {
  return (
    <div className={cn("h-full flex flex-col gap-4 p-6", className)}>
      {children}
    </div>
  )
}

interface DashboardGridProps {
  children: React.ReactNode
  cols?: 1 | 2 | 3 | 4
  className?: string
}

/**
 * 대시보드 그리드 레이아웃
 */
export function DashboardGrid({
  children,
  cols = 3,
  className
}: DashboardGridProps) {
  const gridCols = {
    1: "grid-cols-1",
    2: "grid-cols-1 md:grid-cols-2",
    3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 md:grid-cols-2 lg:grid-cols-4"
  }

  return (
    <div className={cn(`grid ${gridCols[cols]} gap-4`, className)}>
      {children}
    </div>
  )
}

interface DashboardSectionProps {
  title?: string
  description?: string
  children: React.ReactNode
  className?: string
}

/**
 * 대시보드 섹션 레이아웃
 */
export function DashboardSection({
  title,
  description,
  children,
  className
}: DashboardSectionProps) {
  return (
    <div className={cn("space-y-4", className)}>
      {(title || description) && (
        <div className="space-y-1">
          {title && (
            <h2 className="text-2xl font-bold text-white">{title}</h2>
          )}
          {description && (
            <p className="text-slate-400">{description}</p>
          )}
        </div>
      )}
      {children}
    </div>
  )
}