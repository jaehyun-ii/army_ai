/**
 * Annotations API Hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { handleApiError } from '@/lib/error-handler'
import { queryKeys } from './query-keys'
import type { Annotation, CreateAnnotationRequest } from '@/types/api'

/**
 * 이미지 어노테이션 조회
 */
export function useImageAnnotations(
  imageId: string,
  options?: {
    annotation_type?: 'bbox' | 'polygon' | 'keypoint' | 'segmentation'
    min_confidence?: number
  },
  enabled = true
) {
  return useQuery({
    queryKey: queryKeys.annotations.image(imageId, options),
    queryFn: () => apiClient.getImageAnnotations(imageId, options),
    enabled: enabled && !!imageId,
  })
}

/**
 * 데이터셋 어노테이션 요약 조회
 */
export function useDatasetAnnotationsSummary(
  datasetId: string,
  minConfidence?: number,
  enabled = true
) {
  return useQuery({
    queryKey: queryKeys.annotations.dataset(datasetId, minConfidence),
    queryFn: () => apiClient.getDatasetAnnotationsSummary(datasetId, minConfidence),
    enabled: enabled && !!datasetId,
  })
}

/**
 * 어노테이션 일괄 생성
 */
export function useCreateAnnotations() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      imageId,
      annotations,
    }: {
      imageId: string
      annotations: CreateAnnotationRequest[]
    }) => apiClient.createAnnotationsBulk(imageId, annotations),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.annotations.image(variables.imageId),
      })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '어노테이션 생성에 실패했습니다' })
    },
  })
}

/**
 * 이미지 어노테이션 삭제
 */
export function useDeleteImageAnnotations() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      imageId,
      annotationType,
    }: {
      imageId: string
      annotationType?: 'bbox' | 'polygon' | 'keypoint' | 'segmentation'
    }) => apiClient.deleteImageAnnotations(imageId, annotationType),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.annotations.image(variables.imageId),
      })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '어노테이션 삭제에 실패했습니다' })
    },
  })
}
