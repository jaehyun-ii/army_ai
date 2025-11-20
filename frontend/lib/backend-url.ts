/**
 * 백엔드 API URL을 가져옵니다.
 *
 * - 서버사이드: BACKEND_API_URL 환경변수 사용 (http://backend:8000)
 * - 클라이언트사이드: NEXT_PUBLIC_BACKEND_API_URL 환경변수 사용 (http://localhost:54321)
 */
export function getBackendUrl(): string {
  // 서버사이드에서는 BACKEND_API_URL 우선 사용
  if (typeof window === 'undefined') {
    return process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
  }

  // 클라이언트사이드에서는 NEXT_PUBLIC_BACKEND_API_URL 사용
  return process.env.NEXT_PUBLIC_BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
}

/**
 * 백엔드 API URL (상수)
 *
 * 주의: API 라우트에서 사용할 때는 getBackendUrl() 함수를 사용하세요.
 */
export const BACKEND_API_URL = getBackendUrl()
