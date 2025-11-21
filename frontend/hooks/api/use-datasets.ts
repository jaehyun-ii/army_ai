/**
 * Datasets API Hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { handleApiError } from '@/lib/error-handler'
import { queryKeys } from './query-keys'
import type { Dataset } from '@/types/api'

/**
 * 데이터셋 목록 조회
 */
export function useDatasets(skip = 0, limit = 100) {
  return useQuery({
    queryKey: queryKeys.datasets.list(skip, limit),
    queryFn: () => apiClient.getDatasets(skip, limit),
  })
}

/**
 * 개별 데이터셋 조회
 */
export function useDataset(id: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.datasets.detail(id),
    queryFn: () => apiClient.getDataset(id),
    enabled: enabled && !!id,
  })
}

/**
 * 데이터셋 이미지 목록 조회
 */
export function useDatasetImages(datasetId: string, skip = 0, limit = 100, enabled = true) {
  return useQuery({
    queryKey: queryKeys.datasets.images(datasetId, skip, limit),
    queryFn: () => apiClient.getDatasetImages(datasetId, skip, limit),
    enabled: enabled && !!datasetId,
  })
}

/**
 * 데이터셋 생성
 */
export function useCreateDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: Partial<Dataset>) => apiClient.createDataset(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets.lists() })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '데이터셋 생성에 실패했습니다' })
    },
  })
}

/**
 * 데이터셋 업데이트
 */
export function useUpdateDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<Dataset> }) =>
      apiClient.updateDataset(id, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets.detail(variables.id) })
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets.lists() })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '데이터셋 업데이트에 실패했습니다' })
    },
  })
}

/**
 * 데이터셋 삭제
 */
export function useDeleteDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => apiClient.deleteDataset(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.datasets.lists() })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '데이터셋 삭제에 실패했습니다' })
    },
  })
}
