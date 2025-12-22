"use client"

import React, { createContext, useContext, useState, ReactNode } from 'react'

interface OperationContextType {
  isOperationInProgress: boolean
  operationType: string | null
  setOperationInProgress: (inProgress: boolean, type?: string) => void
}

const OperationContext = createContext<OperationContextType | undefined>(undefined)

export function OperationProvider({ children }: { children: ReactNode }) {
  const [isOperationInProgress, setIsOperationInProgress] = useState(false)
  const [operationType, setOperationType] = useState<string | null>(null)

  const setOperationInProgress = (inProgress: boolean, type?: string) => {
    setIsOperationInProgress(inProgress)
    setOperationType(inProgress ? (type || null) : null)
  }

  return (
    <OperationContext.Provider value={{ isOperationInProgress, operationType, setOperationInProgress }}>
      {children}
    </OperationContext.Provider>
  )
}

export function useOperation() {
  const context = useContext(OperationContext)
  if (context === undefined) {
    throw new Error('useOperation must be used within an OperationProvider')
  }
  return context
}
