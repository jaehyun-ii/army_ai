/**
 * Query Keys Factory
 * React Query 쿼리 키 관리
 */

export const queryKeys = {
  // Models
  models: {
    all: ['models'] as const,
    lists: () => [...queryKeys.models.all, 'list'] as const,
    list: (skip?: number, limit?: number) =>
      [...queryKeys.models.lists(), { skip, limit }] as const,
    details: () => [...queryKeys.models.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.models.details(), id] as const,
  },

  // Datasets
  datasets: {
    all: ['datasets'] as const,
    lists: () => [...queryKeys.datasets.all, 'list'] as const,
    list: (skip?: number, limit?: number) =>
      [...queryKeys.datasets.lists(), { skip, limit }] as const,
    details: () => [...queryKeys.datasets.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.datasets.details(), id] as const,
    images: (id: string, skip?: number, limit?: number) =>
      [...queryKeys.datasets.detail(id), 'images', { skip, limit }] as const,
    stats: (id: string) => [...queryKeys.datasets.detail(id), 'stats'] as const,
  },

  // Attack Datasets
  attackDatasets: {
    all: ['attackDatasets'] as const,
    lists: () => [...queryKeys.attackDatasets.all, 'list'] as const,
    list: (skip?: number, limit?: number, attackType?: string) =>
      [...queryKeys.attackDatasets.lists(), { skip, limit, attackType }] as const,
    details: () => [...queryKeys.attackDatasets.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.attackDatasets.details(), id] as const,
  },

  // Adversarial Patches
  patches: {
    all: ['patches'] as const,
    lists: () => [...queryKeys.patches.all, 'list'] as const,
    list: (skip?: number, limit?: number, targetClass?: string) =>
      [...queryKeys.patches.lists(), { skip, limit, targetClass }] as const,
    details: () => [...queryKeys.patches.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.patches.details(), id] as const,
  },

  // Evaluations
  evaluations: {
    all: ['evaluations'] as const,
    lists: () => [...queryKeys.evaluations.all, 'list'] as const,
    list: (params?: Record<string, any>) =>
      [...queryKeys.evaluations.lists(), params] as const,
    details: () => [...queryKeys.evaluations.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.evaluations.details(), id] as const,
    prCurve: (id: string) => [...queryKeys.evaluations.detail(id), 'prCurve'] as const,
    items: (id: string, page?: number, datasetType?: string) =>
      [...queryKeys.evaluations.detail(id), 'items', { page, datasetType }] as const,
  },

  // Annotations
  annotations: {
    all: ['annotations'] as const,
    image: (imageId: string, options?: Record<string, any>) =>
      [...queryKeys.annotations.all, 'image', imageId, options] as const,
    dataset: (datasetId: string, minConfidence?: number) =>
      [...queryKeys.annotations.all, 'dataset', datasetId, { minConfidence }] as const,
  },

  // Experiments
  experiments: {
    all: ['experiments'] as const,
    lists: () => [...queryKeys.experiments.all, 'list'] as const,
    list: (skip?: number, limit?: number) =>
      [...queryKeys.experiments.lists(), { skip, limit }] as const,
    details: () => [...queryKeys.experiments.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.experiments.details(), id] as const,
  },

  // Tags
  tags: {
    all: ['tags'] as const,
    lists: () => [...queryKeys.tags.all, 'list'] as const,
    list: (options?: Record<string, any>) =>
      [...queryKeys.tags.lists(), options] as const,
    details: () => [...queryKeys.tags.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.tags.details(), id] as const,
  },
} as const
