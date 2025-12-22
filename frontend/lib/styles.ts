/**
 * 중앙 집중식 스타일 상수
 * 모든 컴포넌트에서 재사용 가능한 스타일 정의
 */

export const styles = {
  // 카드 스타일 - 5단계 elevation 시스템 활용
  card: {
    base: "bg-surface-container border-border backdrop-blur-sm rounded-lg",
    elevated: "bg-surface-container-high border-border shadow-sm rounded-lg",
    floating: "bg-surface-container-highest border-outline-variant shadow-md rounded-lg",
    sunken: "bg-surface-container-low border-outline-variant/50 rounded-lg",
    flat: "bg-surface-container-lowest border-outline-variant/30 rounded-lg",
    hover: "hover:bg-surface-container-high transition-all duration-300",
    interactive: "hover:bg-surface-container-high active:bg-surface-container-highest transition-all duration-200",
    dark: "bg-surface-container border-white/5"
  },

  // 버튼 스타일 - Full Color Spectrum
  button: {
    base: "px-4 py-2 rounded-lg font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-ring",
    primary: "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm hover:shadow-md",
    primaryContainer: "bg-primary-container text-on-primary-container hover:bg-primary-container/80 border border-outline-variant",
    secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/90 shadow-sm",
    secondaryContainer: "bg-secondary-container text-on-secondary-container hover:bg-secondary-container/80 border border-outline-variant",
    tertiary: "bg-tertiary text-tertiary-foreground hover:bg-tertiary/90 shadow-sm",
    tertiaryContainer: "bg-tertiary-container text-on-tertiary-container hover:bg-tertiary-container/80 border border-outline-variant",
    outline: "border border-outline bg-transparent text-foreground hover:bg-surface-variant",
    outlineVariant: "border border-outline-variant bg-transparent text-on-surface-variant hover:bg-surface-container-high",
    ghost: "text-foreground hover:bg-surface-container-high",
    danger: "bg-error text-error-foreground hover:bg-error/90 shadow-sm",
    dangerContainer: "bg-error-container text-on-error-container hover:bg-error-container/80 border border-outline-variant",
    success: "bg-success text-success-foreground hover:bg-success/90 shadow-sm",
    inverse: "bg-inverse-surface text-inverse-on-surface hover:bg-inverse-surface/90",
    tonal: "bg-surface-container-highest text-foreground hover:bg-surface-container-high",
    sizes: {
      sm: "px-3 py-1.5 text-sm",
      md: "px-4 py-2",
      lg: "px-6 py-3 text-lg"
    }
  },

  // 입력 필드 스타일
  input: {
    base: "bg-surface-container-high border-input text-foreground placeholder:text-muted rounded-lg transition-all duration-300",
    focus: "focus:border-primary focus:ring-1 focus:ring-primary outline-none",
    error: "border-error focus:border-error focus:ring-error",
    sizes: {
      sm: "px-3 py-1.5 text-sm",
      md: "px-3 py-2",
      lg: "px-4 py-3 text-lg"
    }
  },

  // 텍스트 스타일
  text: {
    heading: {
      h1: "text-3xl font-bold text-foreground",
      h2: "text-2xl font-bold text-foreground",
      h3: "text-xl font-semibold text-foreground",
      h4: "text-lg font-semibold text-foreground",
      h5: "text-base font-medium text-foreground",
      h6: "text-sm font-medium text-foreground"
    },
    body: {
      base: "text-foreground",
      muted: "text-muted",
      small: "text-sm text-muted",
      tiny: "text-xs text-muted/80"
    }
  },

  // 배지 스타일 - Full Spectrum
  badge: {
    base: "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium transition-colors duration-200",
    variants: {
      default: "bg-primary/20 text-primary border border-primary/30",
      primary: "bg-primary-container text-on-primary-container border border-outline-variant",
      secondary: "bg-secondary-container text-on-secondary-container border border-outline-variant",
      tertiary: "bg-tertiary-container text-on-tertiary-container border border-outline-variant",
      error: "bg-error-container text-on-error-container border border-outline-variant",
      success: "bg-success/20 text-success border border-success/30",
      warning: "bg-warning/20 text-warning border border-warning/30",
      info: "bg-info/20 text-info border border-info/30",
      surface: "bg-surface-container-high text-on-surface border border-outline-variant",
      inverse: "bg-inverse-surface text-inverse-on-surface",
      outline: "bg-transparent text-foreground border border-outline",
    }
  },

  // 레이아웃 스타일
  layout: {
    container: "container mx-auto px-4 sm:px-6 lg:px-8",
    section: "py-6 sm:py-8 lg:py-12",
    grid: {
      cols2: "grid grid-cols-1 md:grid-cols-2 gap-4",
      cols3: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4",
      cols4: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
    }
  },

  // 애니메이션 스타일
  animation: {
    fadeIn: "animate-fade-in",
    pulse: "animate-pulse",
    spin: "animate-spin",
    bounce: "animate-bounce"
  },

  // 그라디언트 스타일
  gradient: {
    primary: "bg-gradient-to-r from-primary to-tertiary", // Accent 대신 Tertiary 사용
    dark: "bg-gradient-to-br from-surface to-surface-container",
    card: "bg-gradient-to-br from-surface-container to-surface-container-high"
  },

  // 오버레이 스타일
  overlay: {
    base: "fixed inset-0 bg-scrim/50 backdrop-blur-sm z-50",
    light: "fixed inset-0 bg-scrim/10 backdrop-blur-sm z-50",
    heavy: "fixed inset-0 bg-scrim/70 backdrop-blur-md z-50",
    subtle: "fixed inset-0 bg-scrim/20 backdrop-blur-sm z-50"
  },

  // 하이라이트 및 강조 스타일
  highlight: {
    primary: "bg-primary-fixed-dim text-on-primary-fixed border-l-4 border-primary",
    secondary: "bg-secondary-fixed-dim text-on-secondary-fixed border-l-4 border-secondary",
    tertiary: "bg-tertiary-fixed-dim text-on-tertiary-fixed border-l-4 border-tertiary",
    error: "bg-error-container text-on-error-container border-l-4 border-error",
    info: "bg-surface-container-high text-on-surface border-l-4 border-info",
    success: "bg-surface-container-high text-on-surface border-l-4 border-success"
  },

  // 상태 표시 스타일
  status: {
    active: "bg-primary-fixed text-on-primary-fixed ring-2 ring-primary/20",
    inactive: "bg-surface-variant text-on-surface-variant",
    success: "bg-success/10 text-success ring-2 ring-success/20",
    warning: "bg-warning/10 text-warning ring-2 ring-warning/20",
    error: "bg-error-container text-on-error-container ring-2 ring-error/20",
    pending: "bg-secondary-container text-on-secondary-container ring-2 ring-secondary/20"
  },

  // 분할선 및 구분자
  divider: {
    default: "border-outline",
    variant: "border-outline-variant",
    subtle: "border-outline-variant/50",
    heavy: "border-outline",
    primary: "border-primary/30",
    gradient: "border-transparent bg-gradient-to-r from-transparent via-outline to-transparent"
  },

  // 리스트 아이템 스타일
  listItem: {
    base: "p-3 rounded-lg transition-all duration-200",
    interactive: "hover:bg-surface-container-high active:bg-surface-container-highest cursor-pointer",
    selected: "bg-primary-container text-on-primary-container",
    disabled: "opacity-50 cursor-not-allowed",
    withIcon: "flex items-center gap-3"
  },

  // 칩 스타일 (Material Design Chips)
  chip: {
    base: "inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all duration-200",
    filled: {
      primary: "bg-primary text-primary-foreground hover:bg-primary/90",
      secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/90",
      tertiary: "bg-tertiary text-tertiary-foreground hover:bg-tertiary/90",
      error: "bg-error text-error-foreground hover:bg-error/90"
    },
    outlined: {
      primary: "border-2 border-primary text-primary hover:bg-primary-container",
      secondary: "border-2 border-secondary text-secondary hover:bg-secondary-container",
      tertiary: "border-2 border-tertiary text-tertiary hover:bg-tertiary-container",
      error: "border-2 border-error text-error hover:bg-error-container"
    },
    tonal: {
      primary: "bg-primary-container text-on-primary-container hover:bg-primary-container/80",
      secondary: "bg-secondary-container text-on-secondary-container hover:bg-secondary-container/80",
      tertiary: "bg-tertiary-container text-on-tertiary-container hover:bg-tertiary-container/80",
      error: "bg-error-container text-on-error-container hover:bg-error-container/80"
    }
  },

  // 알림 및 Alert 스타일
  alert: {
    base: "p-4 rounded-lg border-l-4",
    variants: {
      info: "bg-surface-container-high border-info text-foreground",
      success: "bg-surface-container-high border-success text-foreground",
      warning: "bg-surface-container-high border-warning text-foreground",
      error: "bg-error-container border-error text-on-error-container",
      primary: "bg-primary-container border-primary text-on-primary-container",
      secondary: "bg-secondary-container border-secondary text-on-secondary-container"
    }
  },

  // 툴팁 스타일
  tooltip: {
    base: "px-2 py-1 text-xs rounded shadow-lg backdrop-blur-sm",
    default: "bg-inverse-surface text-inverse-on-surface",
    light: "bg-surface-container-highest text-on-surface border border-outline-variant"
  }
} as const

/**
 * 스타일 조합 헬퍼 함수
 */
export function combineStyles(...styles: (string | undefined | false)[]): string {
  return styles.filter(Boolean).join(" ")
}