/**
 * Evaluations API Hooks
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { handleApiError } from '@/lib/error-handler'
import { queryKeys } from './query-keys'
import type { EvaluationRun, CreateEvaluationRunRequest } from '@/types/api'

/**
 * 평가 실행 목록 조회
 */
export function useEvaluations(params: {
  page?: number
  page_size?: number
  phase?: string
  status?: string
  model_id?: string
  base_dataset_id?: string
  attack_dataset_id?: string
} = {}) {
  return useQuery({
    queryKey: queryKeys.evaluations.list(params),
    queryFn: () => apiClient.listEvaluationRuns(params),
  })
}

/**
 * 개별 평가 실행 조회
 */
export function useEvaluation(id: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.evaluations.detail(id),
    queryFn: () => apiClient.getEvaluationRun(id),
    enabled: enabled && !!id,
  })
}

/**
 * PR Curve 데이터 조회
 */
export function usePRCurveData(runId: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.evaluations.prCurve(runId),
    queryFn: () => apiClient.getEvaluationPRCurveData(runId),
    enabled: enabled && !!runId,
  })
}

/**
 * 평가 실행 생성
 */
export function useCreateEvaluation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: CreateEvaluationRunRequest) => apiClient.createEvaluationRun(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.evaluations.lists() })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '평가 생성에 실패했습니다' })
    },
  })
}

/**
 * 평가 실행
 */
export function useExecuteEvaluation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      runId,
      params,
    }: {
      runId: string
      params?: {
        conf_threshold?: number
        iou_threshold?: number
        session_id?: string
      }
    }) => apiClient.executeEvaluationRun(runId, params),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.evaluations.detail(variables.runId) })
      queryClient.invalidateQueries({ queryKey: queryKeys.evaluations.lists() })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '평가 실행에 실패했습니다' })
    },
  })
}

/**
 * 평가 삭제
 */
export function useDeleteEvaluation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (runId: string) => apiClient.deleteEvaluationRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.evaluations.lists() })
    },
    onError: (error) => {
      handleApiError(error, { customMessage: '평가 삭제에 실패했습니다' })
    },
  })
}
