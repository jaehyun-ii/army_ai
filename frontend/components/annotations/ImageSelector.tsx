'use client'

import React, { useState, useEffect } from 'react'
import { apiClient } from '@/lib/api-client'
import { AnnotatedImageViewer } from './AnnotatedImageViewer'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface Image {
  id: string
  filename: string
  storage_path: string
  file_size_bytes: number
  width?: number
  height?: number
}

interface ImageSelectorProps {
  datasetId: string
  className?: string
}

/**
 * 데이터셋의 이미지를 선택하면 해당 이미지의 어노테이션을 표시하는 컴포넌트
 *
 * 사용법:
 * ```tsx
 * <ImageSelector datasetId="dataset-uuid" />
 * ```
 */
export function ImageSelector({ datasetId, className = '' }: ImageSelectorProps) {
  const [images, setImages] = useState<Image[]>([])
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // 데이터셋의 이미지 목록 가져오기
  useEffect(() => {
    const fetchImages = async () => {
      try {
        setLoading(true)
        setError(null)

        const response = await apiClient.getDatasetImages(datasetId) as any
        setImages(response.images || response || [])

        // 첫 번째 이미지를 자동으로 선택
        if (response.images && response.images.length > 0) {
          setSelectedImageId(response.images[0].id)
        } else if (response.length > 0) {
          setSelectedImageId(response[0].id)
        }
      } catch (err) {
        console.error('Failed to fetch images:', err)
        setError(err instanceof Error ? err.message : 'Failed to load images')
      } finally {
        setLoading(false)
      }
    }

    fetchImages()
  }, [datasetId])

  const selectedImage = images.find((img) => img.id === selectedImageId)

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-gray-500">이미지 로딩 중...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div className="p-4 text-center text-gray-500">
        데이터셋에 이미지가 없습니다
      </div>
    )
  }

  return (
    <div className={`grid grid-cols-1 lg:grid-cols-4 gap-4 ${className}`}>
      {/* 이미지 목록 (왼쪽) */}
      <Card className="lg:col-span-1 p-4 max-h-[600px] overflow-y-auto">
        <h3 className="font-semibold mb-4">이미지 목록 ({images.length})</h3>
        <div className="space-y-2">
          {images.map((image) => (
            <Button
              key={image.id}
              variant={selectedImageId === image.id ? 'default' : 'outline'}
              className="w-full justify-start text-left truncate"
              onClick={() => setSelectedImageId(image.id)}
            >
              <div className="truncate">
                {image.filename}
              </div>
            </Button>
          ))}
        </div>
      </Card>

      {/* 선택된 이미지와 어노테이션 (오른쪽) */}
      <div className="lg:col-span-3">
        {selectedImage && selectedImageId && (
          <div className="space-y-4">
            <Card className="p-4">
              <h3 className="font-semibold mb-2">선택된 이미지</h3>
              <div className="text-sm text-gray-600">
                <div>파일명: {selectedImage.filename}</div>
                {selectedImage.width && selectedImage.height && (
                  <div>크기: {selectedImage.width} x {selectedImage.height}</div>
                )}
                <div>용량: {(selectedImage.file_size_bytes / 1024).toFixed(2)} KB</div>
              </div>
            </Card>

            <AnnotatedImageViewer
              imageId={selectedImageId}
              imageUrl={`http://localhost:8000/api/v1/storage/file?file_path=${encodeURIComponent(selectedImage.storage_path)}`}
              minConfidence={0.3}
              showLabels={true}
            />
          </div>
        )}
      </div>
    </div>
  )
}

export default ImageSelector
