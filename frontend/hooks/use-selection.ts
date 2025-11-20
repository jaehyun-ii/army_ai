import { useState, useCallback } from 'react'

/**
 * 아이템 선택 관리를 위한 커스텀 훅
 */
export function useSelection<T>() {
  const [selectedItem, setSelectedItem] = useState<T | null>(null)
  const [selectedItems, setSelectedItems] = useState<T[]>([])

  const selectItem = useCallback((item: T) => {
    setSelectedItem(item)
  }, [])

  const clearSelection = useCallback(() => {
    setSelectedItem(null)
  }, [])

  const toggleItemSelection = useCallback((item: T, identifier?: keyof T) => {
    setSelectedItems(prev => {
      const key = identifier || 'id' as keyof T
      const exists = prev.some(i => i[key] === item[key])

      if (exists) {
        return prev.filter(i => i[key] !== item[key])
      }
      return [...prev, item]
    })
  }, [])

  const selectAll = useCallback((items: T[]) => {
    setSelectedItems(items)
  }, [])

  const clearAll = useCallback(() => {
    setSelectedItems([])
  }, [])

  return {
    selectedItem,
    selectedItems,
    setSelectedItem,
    setSelectedItems,
    selectItem,
    clearSelection,
    toggleItemSelection,
    selectAll,
    clearAll,
    isSelected: (item: T, identifier?: keyof T) => {
      const key = identifier || 'id' as keyof T
      return selectedItems.some(i => i[key] === item[key])
    }
  }
}