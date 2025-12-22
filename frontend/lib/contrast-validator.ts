/**
 * 색상 대비 검증 유틸리티
 * WCAG 2.1 AA/AAA 기준 준수 확인
 */

/**
 * 올바른 배경-텍스트 색상 조합
 * Material Design 3 "on-" 색상 시스템 기반
 */
export const VALID_CONTRAST_PAIRS: Record<string, string[]> = {
  // Primary 시스템
  'bg-primary': ['text-primary-foreground', 'text-on-primary'],
  'bg-primary-container': ['text-on-primary-container'],
  'bg-primary-fixed': ['text-on-primary-fixed'],
  'bg-primary-fixed-dim': ['text-on-primary-fixed'],

  // Secondary 시스템
  'bg-secondary': ['text-secondary-foreground', 'text-on-secondary'],
  'bg-secondary-container': ['text-on-secondary-container'],
  'bg-secondary-fixed': ['text-on-secondary-fixed'],
  'bg-secondary-fixed-dim': ['text-on-secondary-fixed'],

  // Tertiary 시스템
  'bg-tertiary': ['text-tertiary-foreground', 'text-on-tertiary'],
  'bg-tertiary-container': ['text-on-tertiary-container'],
  'bg-tertiary-fixed': ['text-on-tertiary-fixed'],
  'bg-tertiary-fixed-dim': ['text-on-tertiary-fixed'],

  // Error 시스템
  'bg-error': ['text-error-foreground', 'text-on-error'],
  'bg-error-container': ['text-on-error-container'],
  'bg-destructive': ['text-destructive-foreground', 'text-error-foreground'],

  // Surface 시스템
  'bg-surface': ['text-on-surface', 'text-foreground'],
  'bg-surface-variant': ['text-on-surface-variant'],
  'bg-surface-container': ['text-on-surface', 'text-foreground'],
  'bg-surface-container-low': ['text-on-surface', 'text-foreground'],
  'bg-surface-container-high': ['text-on-surface', 'text-foreground'],
  'bg-surface-container-highest': ['text-on-surface', 'text-foreground'],
  'bg-surface-container-lowest': ['text-on-surface', 'text-foreground'],
  'bg-surface-dim': ['text-on-surface-variant'],
  'bg-surface-bright': ['text-on-surface'],

  // Base colors
  'bg-background': ['text-foreground', 'text-on-background'],

  // Inverse 시스템
  'bg-inverse-surface': ['text-inverse-on-surface'],

  // Status colors
  'bg-success': ['text-success-foreground'],
  'bg-warning': ['text-warning-foreground'],
  'bg-info': ['text-info-foreground'],
}

/**
 * 잘못된 조합 (절대 사용 금지)
 */
export const INVALID_CONTRAST_PAIRS: Record<string, string[]> = {
  'bg-primary': ['text-primary'],
  'bg-primary-container': ['text-primary'],
  'bg-secondary': ['text-secondary'],
  'bg-secondary-container': ['text-secondary'],
  'bg-tertiary': ['text-tertiary'],
  'bg-tertiary-container': ['text-tertiary'],
  'bg-error': ['text-error'],
  'bg-error-container': ['text-error'],
}

/**
 * 배경-텍스트 색상 조합이 올바른지 검증
 * @param bgClass 배경 색상 클래스 (예: 'bg-primary')
 * @param textClass 텍스트 색상 클래스 (예: 'text-primary-foreground')
 * @returns 올바른 조합이면 true, 아니면 false
 */
export function validateContrast(bgClass: string, textClass: string): boolean {
  // 1. 유효한 조합 확인
  const validTexts = VALID_CONTRAST_PAIRS[bgClass]
  if (validTexts && validTexts.includes(textClass)) {
    return true
  }

  // 2. 명시적으로 금지된 조합 확인
  const invalidTexts = INVALID_CONTRAST_PAIRS[bgClass]
  if (invalidTexts && invalidTexts.includes(textClass)) {
    return false
  }

  // 3. 알 수 없는 조합은 경고
  if (process.env.NODE_ENV === 'development') {
    console.warn(
      `⚠️ 검증되지 않은 색상 조합: ${bgClass} + ${textClass}\n` +
      `올바른 조합을 확인하세요: /docs/ACCESSIBILITY_CONTRAST_GUIDE.md`
    )
  }

  return false
}

/**
 * 배경 색상에 대한 올바른 텍스트 색상 추천
 * @param bgClass 배경 색상 클래스
 * @returns 추천 텍스트 색상 클래스 배열
 */
export function getRecommendedTextColor(bgClass: string): string[] {
  return VALID_CONTRAST_PAIRS[bgClass] || ['text-foreground']
}

/**
 * className 문자열에서 대비 문제 검출
 * @param className Tailwind CSS 클래스 문자열
 * @returns 문제가 있으면 에러 메시지, 없으면 null
 */
export function checkContrastIssues(className: string): string | null {
  const classes = className.split(/\s+/)

  // 배경 색상 찾기
  const bgClass = classes.find(c => c.startsWith('bg-'))
  if (!bgClass) return null

  // 텍스트 색상 찾기
  const textClass = classes.find(c => c.startsWith('text-'))
  if (!textClass) return null

  // 검증
  if (!validateContrast(bgClass, textClass)) {
    const recommended = getRecommendedTextColor(bgClass)
    return `❌ 대비 문제: ${bgClass} + ${textClass}\n` +
           `✅ 권장 조합: ${bgClass} + ${recommended.join(' 또는 ')}`
  }

  return null
}

/**
 * React 컴포넌트용 대비 검증 Hook (개발 환경 전용)
 */
export function useContrastValidation(className: string) {
  if (process.env.NODE_ENV === 'development') {
    const issue = checkContrastIssues(className)
    if (issue) {
      console.error(issue)
    }
  }
}

/**
 * 대비 문제 자동 수정
 * @param className 원본 클래스 문자열
 * @returns 수정된 클래스 문자열
 */
export function autoFixContrast(className: string): string {
  const classes = className.split(/\s+/)

  // 배경 색상 찾기
  const bgClass = classes.find(c => c.startsWith('bg-'))
  if (!bgClass) return className

  // 텍스트 색상 찾기
  const textClass = classes.find(c => c.startsWith('text-'))
  if (!textClass) return className

  // 이미 올바른 조합이면 그대로 반환
  if (validateContrast(bgClass, textClass)) {
    return className
  }

  // 잘못된 조합이면 자동 수정
  const recommended = getRecommendedTextColor(bgClass)[0]
  const fixedClasses = classes.map(c =>
    c === textClass ? recommended : c
  )

  console.info(
    `🔧 자동 수정: ${textClass} → ${recommended} (배경: ${bgClass})`
  )

  return fixedClasses.join(' ')
}

/**
 * 타입 안전한 색상 조합 생성기
 */
export const contrastSafeClasses = {
  // Primary
  primary: {
    solid: 'bg-primary text-primary-foreground',
    container: 'bg-primary-container text-on-primary-container',
    fixed: 'bg-primary-fixed text-on-primary-fixed',
  },

  // Secondary
  secondary: {
    solid: 'bg-secondary text-secondary-foreground',
    container: 'bg-secondary-container text-on-secondary-container',
    fixed: 'bg-secondary-fixed text-on-secondary-fixed',
  },

  // Tertiary
  tertiary: {
    solid: 'bg-tertiary text-tertiary-foreground',
    container: 'bg-tertiary-container text-on-tertiary-container',
    fixed: 'bg-tertiary-fixed text-on-tertiary-fixed',
  },

  // Error
  error: {
    solid: 'bg-error text-error-foreground',
    container: 'bg-error-container text-on-error-container',
  },

  // Surface
  surface: {
    default: 'bg-surface text-on-surface',
    variant: 'bg-surface-variant text-on-surface-variant',
    container: 'bg-surface-container text-on-surface',
    containerHigh: 'bg-surface-container-high text-on-surface',
    containerHighest: 'bg-surface-container-highest text-on-surface',
    dim: 'bg-surface-dim text-on-surface-variant',
    bright: 'bg-surface-bright text-on-surface',
  },

  // Inverse
  inverse: 'bg-inverse-surface text-inverse-on-surface',

  // Status
  success: 'bg-success text-success-foreground',
  warning: 'bg-warning text-warning-foreground',
  info: 'bg-info text-info-foreground',
} as const

/**
 * 사용 예시:
 *
 * // ✅ 올바른 사용
 * <div className={contrastSafeClasses.primary.container}>
 *   Content
 * </div>
 *
 * // ✅ 검증
 * if (!validateContrast('bg-primary', 'text-secondary')) {
 *   console.error('대비 문제!')
 * }
 *
 * // ✅ 자동 수정
 * const fixed = autoFixContrast('bg-primary text-primary')
 * // → 'bg-primary text-primary-foreground'
 *
 * // ✅ 권장 색상 확인
 * const recommended = getRecommendedTextColor('bg-tertiary-container')
 * // → ['text-on-tertiary-container']
 */
