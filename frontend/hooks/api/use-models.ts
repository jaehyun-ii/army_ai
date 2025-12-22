/**
 * Models API Hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { handleApiError } from '@/lib/error-handler'
import { queryKeys } from './query-keys'
import type { Model } from '@/types/api'

/**
 * 모델 목록 조회
 */
export function useModels(skip = 0, limit = 100) {
  return useQuery({
    queryKey: queryKeys.models.list(skip, limit),
    queryFn: () => apiClient.getModels(skip, limit),
  })
}

/**
 * 개별 모델 조회
 */
export function useModel(id: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.models.detail(id),
    queryFn: () => apiClient.getModel(id),
    enabled: enabled && !!id,
  })
}

/**
 * 모델 생성
 */
export function useCreateModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: Partial<Model>) => apiClient.createModel(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.models.lists() })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '모델 생성에 실패했습니다' })
    },
  })
}
