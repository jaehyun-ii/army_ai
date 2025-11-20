import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { styles } from '@/lib/styles'
import { cn } from '@/lib/utils'
import { LucideIcon } from 'lucide-react'

interface CardLayoutProps {
  title: string
  description?: string
  icon?: LucideIcon
  iconColor?: string
  actions?: React.ReactNode
  children: React.ReactNode
  className?: string
  contentClassName?: string
}

/**
 * 공통 카드 레이아웃 컴포넌트
 */
export function CardLayout({
  title,
  description,
  icon: Icon,
  iconColor = "text-white",
  actions,
  children,
  className,
  contentClassName
}: CardLayoutProps) {
  return (
    <Card className={cn(styles.card.base, className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2 text-white">
              {Icon && <Icon className={cn("w-5 h-5", iconColor)} />}
              {title}
            </CardTitle>
            {description && (
              <CardDescription className="text-slate-400">
                {description}
              </CardDescription>
            )}
          </div>
          {actions && (
            <div className="flex items-center gap-2">
              {actions}
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className={contentClassName}>
        {children}
      </CardContent>
    </Card>
  )
}

interface SimpleCardLayoutProps {
  title?: string
  children: React.ReactNode
  className?: string
}

/**
 * 간단한 카드 레이아웃 (헤더 없음)
 */
export function SimpleCardLayout({
  title,
  children,
  className
}: SimpleCardLayoutProps) {
  return (
    <Card className={cn(styles.card.base, className)}>
      {title && (
        <CardHeader>
          <CardTitle className="text-white">{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent className={title ? undefined : "pt-6"}>
        {children}
      </CardContent>
    </Card>
  )
}