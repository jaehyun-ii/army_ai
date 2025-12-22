import { useState, useEffect, useCallback } from 'react'

interface EvaluationRecord {
  id: string
  modelName: string
  modelVersion: string
  evaluationType: string
  accuracy?: number
  precision?: number
  recall?: number
  f1Score?: number
  processingTime?: number
  datasetSize?: number
  successRate?: number
  notes?: string
  metadata?: any
  createdAt: string
  updatedAt: string
  adversarialTests?: any[]
  performanceTests?: any[]
}

interface Dataset {
  id: string
  name: string
  type: string
  source?: string
  size: number
  storageLocation?: string
  description?: string
  metadata?: any
  createdAt: string
  updatedAt: string
  dataGenerations?: any[]
}

export const useEvaluations = () => {
  const [evaluations, setEvaluations] = useState<EvaluationRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchEvaluations = useCallback(async (params?: {
    modelName?: string
    evaluationType?: string
    limit?: number
  }) => {
    setLoading(true)
    setError(null)
    try {
      const queryParams = new URLSearchParams()
      if (params?.modelName) queryParams.append('modelName', params.modelName)
      if (params?.evaluationType) queryParams.append('evaluationType', params.evaluationType)
      if (params?.limit) queryParams.append('limit', params.limit.toString())

      const response = await fetch(`/api/evaluations?${queryParams}`)
      if (!response.ok) throw new Error('Failed to fetch evaluations')

      const data = await response.json()
      setEvaluations(data)
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('Error fetching evaluations:', err)
      return []
    } finally {
      setLoading(false)
    }
  }, [])

  const createEvaluation = useCallback(async (evaluation: Omit<EvaluationRecord, 'id' | 'createdAt' | 'updatedAt'>) => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/evaluations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(evaluation),
      })
      if (!response.ok) throw new Error('Failed to create evaluation')

      const data = await response.json()
      await fetchEvaluations()
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('Error creating evaluation:', err)
      return null
    } finally {
      setLoading(false)
    }
  }, [fetchEvaluations])

  useEffect(() => {
    fetchEvaluations()
  }, [fetchEvaluations])

  return { evaluations, loading, error, fetchEvaluations, createEvaluation }
}

export const useDatasets = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchDatasets = useCallback(async (params?: {
    type?: string
    source?: string
  }) => {
    setLoading(true)
    setError(null)
    try {
      const queryParams = new URLSearchParams()
      if (params?.type) queryParams.append('type', params.type)
      if (params?.source) queryParams.append('source', params.source)

      const response = await fetch(`/api/datasets?${queryParams}`)
      if (!response.ok) throw new Error('Failed to fetch datasets')

      const data = await response.json()
      setDatasets(data)
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('Error fetching datasets:', err)
      return []
    } finally {
      setLoading(false)
    }
  }, [])

  const createDataset = useCallback(async (dataset: Omit<Dataset, 'id' | 'createdAt' | 'updatedAt'>) => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dataset),
      })
      if (!response.ok) throw new Error('Failed to create dataset')

      const data = await response.json()
      await fetchDatasets()
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('Error creating dataset:', err)
      return null
    } finally {
      setLoading(false)
    }
  }, [fetchDatasets])

  useEffect(() => {
    fetchDatasets()
  }, [fetchDatasets])

  return { datasets, loading, error, fetchDatasets, createDataset }
}

export const useSystemLogs = () => {
  const createLog = useCallback(async (log: {
    level?: string
    category: string
    message: string
    details?: any
    userId?: string
    sessionId?: string
  }) => {
    try {
      const response = await fetch('/api/system-logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(log),
      })
      if (!response.ok) throw new Error('Failed to create log')
      return await response.json()
    } catch (err) {
      console.error('Error creating log:', err)
      return null
    }
  }, [])

  const fetchLogs = useCallback(async (params?: {
    level?: string
    category?: string
    limit?: number
  }) => {
    try {
      const queryParams = new URLSearchParams()
      if (params?.level) queryParams.append('level', params.level)
      if (params?.category) queryParams.append('category', params.category)
      if (params?.limit) queryParams.append('limit', params.limit.toString())

      const response = await fetch(`/api/system-logs?${queryParams}`)
      if (!response.ok) throw new Error('Failed to fetch logs')
      return await response.json()
    } catch (err) {
      console.error('Error fetching logs:', err)
      return []
    }
  }, [])

  return { createLog, fetchLogs }
}