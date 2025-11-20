/**
 * 중앙 집중식 스타일 상수
 * 모든 컴포넌트에서 재사용 가능한 스타일 정의
 */

export const styles = {
  // 카드 스타일
  card: {
    base: "bg-slate-800/50 border-white/10 backdrop-blur-sm rounded-lg",
    hover: "hover:bg-slate-800/70 transition-all duration-300",
    dark: "bg-slate-900/50 border-white/5"
  },

  // 버튼 스타일
  button: {
    base: "px-4 py-2 rounded-lg font-medium transition-all duration-300 focus:outline-none focus:ring-2",
    primary: "bg-gradient-to-r from-primary to-accent text-white hover:shadow-lg",
    secondary: "bg-slate-700/50 text-white hover:bg-slate-700/70",
    outline: "border border-white/20 text-white hover:bg-white/10",
    ghost: "text-white hover:bg-white/10",
    danger: "bg-red-500 text-white hover:bg-red-600",
    success: "bg-green-500 text-white hover:bg-green-600",
    sizes: {
      sm: "px-3 py-1.5 text-sm",
      md: "px-4 py-2",
      lg: "px-6 py-3 text-lg"
    }
  },

  // 입력 필드 스타일
  input: {
    base: "bg-slate-800/70 border-white/20 text-white placeholder:text-slate-400 rounded-lg transition-all duration-300",
    focus: "focus:border-primary focus:bg-slate-800/90 focus:ring-2 focus:ring-primary/20",
    error: "border-red-400 focus:border-red-400",
    sizes: {
      sm: "px-3 py-1.5 text-sm",
      md: "px-3 py-2",
      lg: "px-4 py-3 text-lg"
    }
  },

  // 텍스트 스타일
  text: {
    heading: {
      h1: "text-3xl font-bold text-white",
      h2: "text-2xl font-bold text-white",
      h3: "text-xl font-semibold text-white",
      h4: "text-lg font-semibold text-white",
      h5: "text-base font-medium text-white",
      h6: "text-sm font-medium text-white"
    },
    body: {
      base: "text-slate-200",
      muted: "text-slate-400",
      small: "text-sm text-slate-400",
      tiny: "text-xs text-slate-500"
    }
  },

  // 배지 스타일
  badge: {
    base: "inline-flex items-center px-2 py-1 rounded-full text-xs font-medium",
    variants: {
      default: "bg-primary/20 text-primary border border-primary/30",
      success: "bg-green-900/30 text-green-400 border border-green-500/30",
      warning: "bg-yellow-900/30 text-yellow-400 border border-yellow-500/30",
      error: "bg-red-900/30 text-red-400 border border-red-500/30",
      info: "bg-blue-900/30 text-blue-400 border border-blue-500/30"
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
    primary: "bg-gradient-to-r from-primary to-accent",
    dark: "bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900",
    card: "bg-gradient-to-br from-slate-800/30 to-slate-900/30"
  },

  // 오버레이 스타일
  overlay: {
    base: "fixed inset-0 bg-black/50 backdrop-blur-sm z-50",
    light: "fixed inset-0 bg-white/10 backdrop-blur-sm z-50"
  }
} as const

/**
 * 스타일 조합 헬퍼 함수
 */
export function combineStyles(...styles: (string | undefined | false)[]): string {
  return styles.filter(Boolean).join(" ")
}