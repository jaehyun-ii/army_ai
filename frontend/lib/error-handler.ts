/**
 * Unified Error Handler
 * API 에러를 일관되게 처리
 */

import { toast } from 'sonner'

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string,
    public details?: any
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

interface ErrorHandlerOptions {
  showToast?: boolean
  customMessage?: string
  logError?: boolean
}

/**
 * API 에러를 처리하고 사용자에게 표시
 */
export function handleApiError(
  error: unknown,
  options: ErrorHandlerOptions = {}
): string {
  const {
    showToast = true,
    customMessage,
    logError = true,
  } = options

  let message: string
  let status: number | undefined

  if (error instanceof ApiError) {
    message = error.message
    status = error.status
  } else if (error instanceof Error) {
    message = error.message
  } else if (typeof error === 'string') {
    message = error
  } else {
    message = 'Unknown error occurred'
  }

  const displayMessage = customMessage || message

  // 로그 출력
  if (logError) {
    console.error('[API Error]', {
      message,
      status,
      error,
      timestamp: new Date().toISOString(),
    })
  }

  // Toast 표시
  if (showToast) {
    if (status && status >= 500) {
      toast.error('서버 오류가 발생했습니다', {
        description: displayMessage,
      })
    } else if (status === 404) {
      toast.error('요청한 리소스를 찾을 수 없습니다', {
        description: displayMessage,
      })
    } else if (status === 401 || status === 403) {
      toast.error('권한이 없습니다', {
        description: displayMessage,
      })
    } else {
      toast.error(displayMessage)
    }
  }

  return displayMessage
}

/**
 * 특정 상태 코드에 대한 에러 메시지 매핑
 */
export const ERROR_MESSAGES: Record<number, string> = {
  400: '잘못된 요청입니다',
  401: '인증이 필요합니다',
  403: '접근 권한이 없습니다',
  404: '요청한 리소스를 찾을 수 없습니다',
  409: '이미 존재하는 데이터입니다',
  422: '입력 데이터가 올바르지 않습니다',
  429: '너무 많은 요청을 보냈습니다. 잠시 후 다시 시도해주세요',
  500: '서버 내부 오류가 발생했습니다',
  502: '게이트웨이 오류가 발생했습니다',
  503: '서비스를 일시적으로 사용할 수 없습니다',
  504: '게이트웨이 시간 초과가 발생했습니다',
}

/**
 * HTTP 상태 코드로 에러 메시지 가져오기
 */
export function getErrorMessage(status: number, defaultMessage?: string): string {
  return ERROR_MESSAGES[status] || defaultMessage || '오류가 발생했습니다'
}

/**
 * 네트워크 에러 체크
 */
export function isNetworkError(error: unknown): boolean {
  if (error instanceof Error) {
    return (
      error.message.includes('fetch') ||
      error.message.includes('network') ||
      error.message.includes('Failed to fetch')
    )
  }
  return false
}

/**
 * Validation 에러 포맷팅
 */
export function formatValidationErrors(errors: Array<{ loc: string[]; msg: string }>): string {
  return errors.map(err => `${err.loc.join('.')}: ${err.msg}`).join('\n')
}
