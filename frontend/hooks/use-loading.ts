import { useState, useCallback } from 'react'

/**
 * 로딩 상태 관리를 위한 커스텀 훅
 */
export function useLoading(initialState = false) {
  const [isLoading, setIsLoading] = useState(initialState)
  const [progress, setProgress] = useState(0)

  const startLoading = useCallback(() => {
    setIsLoading(true)
    setProgress(0)
  }, [])

  const stopLoading = useCallback(() => {
    setIsLoading(false)
    setProgress(0)
  }, [])

  const updateProgress = useCallback((value: number) => {
    setProgress(Math.min(100, Math.max(0, value)))
  }, [])

  return {
    isLoading,
    progress,
    setIsLoading,
    setProgress,
    startLoading,
    stopLoading,
    updateProgress
  }
}