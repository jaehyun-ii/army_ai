/**
 * 스타일 관련 커스텀 훅
 */

import { useMemo } from 'react'
import { styles } from '@/lib/styles'
import { theme } from '@/lib/theme'
import { cn } from '@/lib/utils'

/**
 * 컴포넌트 스타일 훅
 */
export function useComponentStyles(
  componentType: keyof typeof styles,
  variant?: string,
  className?: string
) {
  return useMemo(() => {
    const componentStyles = styles[componentType] as any
    let baseStyles = ''

    if (typeof componentStyles === 'string') {
      baseStyles = componentStyles
    } else if (componentStyles.base) {
      baseStyles = componentStyles.base
    }

    if (variant && componentStyles.variants?.[variant]) {
      baseStyles = cn(baseStyles, componentStyles.variants[variant])
    }

    return cn(baseStyles, className)
  }, [componentType, variant, className])
}

/**
 * 테마 색상 훅
 */
export function useThemeColor(colorPath: string) {
  return useMemo(() => {
    const paths = colorPath.split('.')
    let value: any = theme.colors

    for (const path of paths) {
      value = value?.[path]
    }

    return value || '#000000'
  }, [colorPath])
}

/**
 * 반응형 스타일 훅
 */
export function useResponsiveStyles(
  mobile?: string,
  tablet?: string,
  desktop?: string
) {
  return useMemo(() => {
    const classes = []

    if (mobile) classes.push(mobile)
    if (tablet) classes.push(`md:${tablet}`)
    if (desktop) classes.push(`lg:${desktop}`)

    return classes.join(' ')
  }, [mobile, tablet, desktop])
}

/**
 * 조건부 스타일 훅
 */
export function useConditionalStyles(
  conditions: Record<string, boolean>,
  className?: string
) {
  return useMemo(() => {
    const activeStyles = Object.entries(conditions)
      .filter(([_, isActive]) => isActive)
      .map(([style]) => style)

    return cn(...activeStyles, className)
  }, [conditions, className])
}