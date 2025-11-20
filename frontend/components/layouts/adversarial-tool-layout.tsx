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
  leftPanelWidth?: "xs" | "sm" | "md" | "lg" | "xl" | "2xl" | "3xl" | "4xl" | "5xl"  // 좌측 패널 너비 옵션
}

export function AdversarialToolLayout({
  title,
  description,
  icon: Icon,
  headerStats,
  leftPanel,
  rightPanel,
  actionButtons,
  leftPanelWidth = "md"
}: AdversarialToolLayoutProps) {
  const LeftIcon = leftPanel?.icon
  const RightIcon = rightPanel.icon

  // 좌측 패널 너비 설정
  const leftColSpan = leftPanel ? {
    xs: "lg:col-span-1",   // 1/12 = 약 8%
    sm: "lg:col-span-2",   // 2/12 = 약 17%
    md: "lg:col-span-3",   // 3/12 = 25%
    lg: "lg:col-span-4",   // 4/12 = 약 33%
    xl: "lg:col-span-5",   // 5/12 = 약 42%
    "2xl": "lg:col-span-6", // 6/12 = 50%
    "3xl": "lg:col-span-7", // 7/12 = 약 58%
    "4xl": "lg:col-span-8", // 8/12 = 약 67%
    "5xl": "lg:col-span-9"  // 9/12 = 75%
  }[leftPanelWidth] : ""

  const rightColSpan = leftPanel ? {
    xs: "lg:col-span-11",  // 11/12 = 약 92%
    sm: "lg:col-span-10",  // 10/12 = 약 83%
    md: "lg:col-span-9",   // 9/12 = 75%
    lg: "lg:col-span-8",   // 8/12 = 약 67%
    xl: "lg:col-span-7",   // 7/12 = 약 58%
    "2xl": "lg:col-span-6", // 6/12 = 50%
    "3xl": "lg:col-span-5", // 5/12 = 약 42%
    "4xl": "lg:col-span-4", // 4/12 = 약 33%
    "5xl": "lg:col-span-3"  // 3/12 = 25%
  }[leftPanelWidth] : "lg:col-span-12"

  return (
    <div className="h-full flex flex-col gap-2 sm:gap-4 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800/80 to-slate-900/80 rounded-lg sm:rounded-xl p-3 sm:p-4 border border-white/10 shadow-xl flex-shrink-0">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex-shrink-0">
            <h1 className="text-lg sm:text-xl lg:text-2xl font-bold text-white flex items-center gap-2">
              <Icon className="w-5 h-5 sm:w-6 sm:h-6 text-red-400" />
              {title}
            </h1>
            {description && (
              <p className="text-xs sm:text-sm text-slate-400 mt-1">{description}</p>
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
        <div className={`grid grid-cols-1 ${leftPanel ? 'lg:grid-cols-12' : ''} gap-3 sm:gap-4 h-full`}>
          {/* Left Panel - Optional, responsive stacking */}
          {leftPanel && (
            <div className={`${leftColSpan} h-full overflow-hidden`}>
              <Card className="bg-slate-800/50 border-white/10 h-full flex flex-col">
              <CardHeader className="flex-shrink-0">
                <CardTitle className="text-base sm:text-lg text-white flex items-center gap-2">
                    {LeftIcon && <LeftIcon className="w-4 h-4 sm:w-5 sm:h-5" />}
                    <span className="truncate">{leftPanel.title}</span>
                  </CardTitle>
                  {leftPanel.description && (
                    <CardDescription className="text-xs sm:text-sm text-slate-400">
                      {leftPanel.description}
                    </CardDescription>
                  )}
                </CardHeader>
                <CardContent className={`flex-1 overflow-hidden px-3 sm:px-4 pt-0 pb-0 ${leftPanel.className || ""}`}>
                  <div className="h-full overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800 pb-2">
                    {leftPanel.children}
                  </div>
                </CardContent>

                {/* Action Buttons (optional) */}
                {actionButtons && (
                  <div className="flex-shrink-0 p-3 sm:p-4 border-t border-white/10">
                    {actionButtons}
                  </div>
                )}
              </Card>
            </div>
          )}

          {/* Right Panel - Takes full width if no left panel, responsive */}
          <div className={`${rightColSpan} h-full overflow-hidden`}>
            <Card className="bg-slate-800/50 border-white/10 h-full flex flex-col">
              <CardHeader className="flex-shrink-0">
                <CardTitle className="text-base sm:text-lg text-white flex items-center gap-2">
                  {RightIcon && <RightIcon className="w-4 h-4 sm:w-5 sm:h-5" />}
                  {rightPanel.title}
                </CardTitle>
                {rightPanel.description && (
                  <CardDescription className="text-xs sm:text-sm text-slate-400">
                    {rightPanel.description}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className={`flex-1 overflow-hidden px-3 sm:px-4 pt-0 pb-0 ${rightPanel.className || ""}`}>
                <div className="h-full overflow-y-auto scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-slate-800 pb-2">
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
  iconColor = "text-white",
  compact = false
}: StatCardProps) {
  if (compact) {
    // 컴팩트 모드: 헤더에 인라인으로 표시
    return (
      <div className="flex items-center gap-3 px-3 py-2 bg-slate-700/30 rounded-lg border border-white/10">
        <Icon className={`w-4 h-4 sm:w-5 sm:h-5 ${iconColor}`} />
        <div>
          <p className="text-xs text-slate-400">{title}</p>
          <p className="text-sm sm:text-base font-bold text-white">{value}</p>
        </div>
      </div>
    )
  }

  return (
    <Card className="bg-slate-800/50 border-white/10">
      <CardHeader className="pb-2 sm:pb-3 p-3 sm:p-4">
        <CardTitle className="text-base sm:text-lg text-white flex items-center gap-2">
          <Icon className={`w-4 h-4 sm:w-5 sm:h-5 ${iconColor}`} />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-3 sm:p-4">
        <p className="text-lg sm:text-xl font-bold text-white">{value}</p>
        {subtitle && (
          <p className="text-xs sm:text-sm text-slate-400 mt-1">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  )
}