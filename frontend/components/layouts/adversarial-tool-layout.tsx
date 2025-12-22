"use client"

import { ReactNode } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LucideIcon } from "lucide-react"

interface ToolSection {
  title: string
  icon?: LucideIcon
  description?: string
  children: ReactNode
  className?: string
}

interface AdversarialToolLayoutProps {
  title: string
  description?: string
  icon: LucideIcon
  headerStats?: ReactNode
  leftPanel?: ToolSection  // Optional
  rightPanel: ToolSection
  actionButtons?: ReactNode
  leftPanelWidth?: "xs" | "sm" | "md" | "lg" | "xl" | "2xl" | "3xl" | "4xl" | "5xl"  // Deprecated: 이제 reversePanels로 제어
  reversePanels?: boolean  // true: LEFT 67%, RIGHT 33% | false: LEFT 33%, RIGHT 67% (기본값)
  disabled?: boolean  // 작업 진행 중 비활성화 여부
}

export function AdversarialToolLayout({
  title,
  description,
  icon: Icon,
  headerStats,
  leftPanel,
  rightPanel,
  actionButtons,
  leftPanelWidth = "md",
  reversePanels = false,
  disabled = false
}: AdversarialToolLayoutProps) {
  const LeftIcon = leftPanel?.icon
  const RightIcon = rightPanel.icon

  // 패널 너비 설정 - reversePanels에 따라 비율 전환
  // min-w 제거하여 겹침 방지, 대신 브레이크포인트를 lg(1024px)로 통일
  // false(기본): LEFT 33%, RIGHT 67% - lg(1024px)부터 가로 배치
  // true(역전): LEFT 67%, RIGHT 33% - lg(1024px)부터 가로 배치
  const gridClass = leftPanel ? "lg:grid-cols-12" : ""

  const leftColSpan = leftPanel
    ? reversePanels
      ? "lg:col-span-8"  // 8/12 = 67%
      : "lg:col-span-4"  // 4/12 = 33%
    : ""

  const rightColSpan = leftPanel
    ? reversePanels
      ? "lg:col-span-4"  // 4/12 = 33%
      : "lg:col-span-8"  // 8/12 = 67%
    : "lg:col-span-12"

  const minHeightBreakpoint = "lg:min-h-0"

  return (
    <div className="h-full flex flex-col gap-4 overflow-hidden">
      {/* Header */}
      <div className="bg-surface-container rounded-lg sm:rounded-xl p-3 sm:p-4 border border-border shadow-sm flex-shrink-0">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex-shrink-0">
            <h1 className="text-primary flex items-center gap-2">
              <Icon className="w-5 h-5 sm:w-6 sm:h-6 text-primary" />
              {title}
            </h1>
            {description && (
              <p className="text-xs sm:text-sm text-muted mt-1">{description}</p>
            )}
          </div>
          {headerStats && (
            <div className="flex-shrink-0">
              {headerStats}
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <div className={`grid grid-cols-1 ${gridClass} gap-4 h-full`}>
          {/* Left Panel - Optional, responsive stacking */}
          {leftPanel && (
            <div className={`${leftColSpan} h-full overflow-hidden min-h-[300px] ${minHeightBreakpoint} ${disabled ? 'pointer-events-none opacity-50' : ''}`}>
              <Card className="bg-surface-container/50 border-border h-full flex flex-col">
              <CardHeader className="flex-shrink-0">
                <CardTitle className="text-primary flex items-center gap-2">
                    {LeftIcon && <LeftIcon className="w-4 h-4 sm:w-5 sm:h-5" />}
                    <span className="truncate">{leftPanel.title}</span>
                  </CardTitle>
                  {leftPanel.description && (
                    <CardDescription className="text-xs sm:text-sm text-muted">
                      {leftPanel.description}
                    </CardDescription>
                  )}
                </CardHeader>
                <CardContent className={`flex-1 overflow-hidden px-4 pt-0 pb-0 ${leftPanel.className || ""}`}>
                  <div className="h-full overflow-y-auto scrollbar-thin pb-2">
                    {leftPanel.children}
                  </div>
                </CardContent>

                {/* Action Buttons (optional) */}
                {actionButtons && (
                  <div className="flex-shrink-0 p-4 border-t border-border">
                    {actionButtons}
                  </div>
                )}
              </Card>
            </div>
          )}

          {/* Right Panel - Takes full width if no left panel, responsive */}
          <div className={`${rightColSpan} h-full overflow-hidden min-h-[400px] ${minHeightBreakpoint}`}>
            <Card className="bg-surface-container/50 border-border h-full flex flex-col">
              <CardHeader className="flex-shrink-0">
                <CardTitle className="text-primary flex items-center gap-2">
                  {RightIcon && <RightIcon className="w-4 h-4 sm:w-5 sm:h-5" />}
                  {rightPanel.title}
                </CardTitle>
                {rightPanel.description && (
                  <CardDescription className="text-xs sm:text-sm text-muted">
                    {rightPanel.description}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className={`flex-1 overflow-hidden px-4 pt-0 pb-0 ${rightPanel.className || ""}`}>
                <div className="h-full overflow-y-auto scrollbar-thin pb-2">
                  {rightPanel.children}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

interface StatCardProps {
  icon: LucideIcon
  title: string
  value: string | number
  subtitle?: string
  iconColor?: string
  compact?: boolean  // 컴팩트 모드 (헤더에 사용)
}

export function StatCard({
  icon: Icon,
  title,
  value,
  subtitle,
  iconColor = "text-foreground",
  compact = false
}: StatCardProps) {
  if (compact) {
    // 컴팩트 모드: 헤더에 인라인으로 표시
    return (
      <div className="flex items-center gap-3 px-3 py-2 bg-surface-container-high/30 rounded-lg border border-border">
        <Icon className={`w-4 h-4 sm:w-5 sm:h-5 ${iconColor}`} />
        <div>
          <p className="text-xs text-muted">{title}</p>
          <p className="text-sm sm:text-base font-bold text-secondary">{value}</p>
        </div>
      </div>
    )
  }

  return (
    <Card className="bg-surface-container/50 border-border">
      <CardHeader className="pb-3 p-4">
        <CardTitle className="text-primary flex items-center gap-2">
          <Icon className={`w-4 h-4 sm:w-5 sm:h-5 ${iconColor}`} />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <p className="text-lg sm:text-xl font-bold text-secondary">{value}</p>
        {subtitle && (
          <p className="text-xs sm:text-sm text-muted mt-1">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  )
}