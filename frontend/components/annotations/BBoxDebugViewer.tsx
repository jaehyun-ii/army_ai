'use client'

import React, { useEffect, useState, useRef } from 'react'
import { apiClient } from '@/lib/api-client'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface Annotation {
  id: string
  class_name: string
  class_index?: number
  confidence?: string | number
  annotation_type: string
  bbox_x?: string | number
  bbox_y?: string | number
  bbox_width?: string | number
  bbox_height?: string | number
}

interface BBoxDebugViewerProps {
  imageId: string
  imageUrl: string
  minConfidence?: number
  showLabels?: boolean
  className?: string
}

/**
 * 바운딩 박스 렌더링 디버그 버전
 * 콘솔에 상세한 좌표 정보를 출력합니다
 */
export function BBoxDebugViewer({
  imageId,
  imageUrl,
  minConfidence = 0.0,
  showLabels = true,
  className = '',
}: BBoxDebugViewerProps) {
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    const fetchAnnotations = async () => {
      try {
        setLoading(true)
        setError(null)

        const data = await apiClient.getImageAnnotations(imageId, {
          annotation_type: 'bbox',
          min_confidence: minConfidence,
        }) as Annotation[]

        console.log('[BBoxDebug] Fetched annotations:', data)
        setAnnotations(data)
      } catch (err) {
        console.error('[BBoxDebug] Failed to fetch annotations:', err)
        setError(err instanceof Error ? err.message : 'Failed to load annotations')
      } finally {
        setLoading(false)
      }
    }

    fetchAnnotations()
  }, [imageId, minConfidence])

  useEffect(() => {
    const image = imageRef.current
    const canvas = canvasRef.current

    if (!image || !canvas) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const drawAnnotations = () => {
      const displayWidth = image.clientWidth
      const displayHeight = image.clientHeight

      console.log('[BBoxDebug] Display size:', { displayWidth, displayHeight })

      if (!displayWidth || !displayHeight) {
        return
      }

      canvas.width = displayWidth
      canvas.height = displayHeight
      ctx.clearRect(0, 0, displayWidth, displayHeight)

      if (annotations.length === 0) {
        console.log('[BBoxDebug] No annotations to draw')
        return
      }

      const naturalWidth = image.naturalWidth || 1
      const naturalHeight = image.naturalHeight || 1

      console.log('[BBoxDebug] Natural size:', { naturalWidth, naturalHeight })

      const imageAspect = naturalWidth / naturalHeight
      const displayAspect = displayWidth / displayHeight

      console.log('[BBoxDebug] Aspect ratios:', { imageAspect, displayAspect })

      let drawWidth = displayWidth
      let drawHeight = displayHeight
      let offsetX = 0
      let offsetY = 0

      if (displayAspect > imageAspect) {
        drawHeight = displayHeight
        drawWidth = drawHeight * imageAspect
        offsetX = (displayWidth - drawWidth) / 2
      } else {
        drawWidth = displayWidth
        drawHeight = drawWidth / imageAspect
        offsetY = (displayHeight - drawHeight) / 2
      }

      console.log('[BBoxDebug] Draw dimensions:', {
        drawWidth,
        drawHeight,
        offsetX,
        offsetY
      })

      annotations.forEach((ann, idx) => {
        if (
          ann.annotation_type !== 'bbox' ||
          ann.bbox_x === undefined ||
          ann.bbox_y === undefined ||
          ann.bbox_width === undefined ||
          ann.bbox_height === undefined
        ) {
          console.log('[BBoxDebug] Skipping annotation (missing bbox):', ann)
          return
        }

        const xCenter = Number(ann.bbox_x)
        const yCenter = Number(ann.bbox_y)
        const width = Number(ann.bbox_width)
        const height = Number(ann.bbox_height)

        console.log(`[BBoxDebug] Annotation ${idx} (${ann.class_name}):`, {
          normalized: { xCenter, yCenter, width, height },
          raw: {
            bbox_x: ann.bbox_x,
            bbox_y: ann.bbox_y,
            bbox_width: ann.bbox_width,
            bbox_height: ann.bbox_height
          }
        })

        if (
          Number.isNaN(xCenter) ||
          Number.isNaN(yCenter) ||
          Number.isNaN(width) ||
          Number.isNaN(height)
        ) {
          console.error('[BBoxDebug] Invalid coordinates:', { xCenter, yCenter, width, height })
          return
        }

        const x = offsetX + (xCenter - width / 2) * drawWidth
        const y = offsetY + (yCenter - height / 2) * drawHeight
        const w = width * drawWidth
        const h = height * drawHeight

        console.log(`[BBoxDebug] Annotation ${idx} rendered at:`, {
          x, y, w, h,
          topLeft: { x, y },
          bottomRight: { x: x + w, y: y + h }
        })

        const hue = (idx * 137.508) % 360
        const color = `hsl(${hue}, 70%, 50%)`

        ctx.strokeStyle = color
        ctx.lineWidth = 3  // Thicker for debugging
        ctx.strokeRect(x, y, w, h)

        if (showLabels) {
          const label = `${ann.class_name} ${ann.confidence ? (Number(ann.confidence) * 100).toFixed(1) : ''}%`
          const fontSize = 14
          ctx.font = `${fontSize}px Arial`
          const textWidth = ctx.measureText(label).width
          const padding = 4

          ctx.fillStyle = color
          ctx.fillRect(x, Math.max(0, y - fontSize - padding * 2), textWidth + padding * 2, fontSize + padding * 2)

          ctx.fillStyle = 'white'
          ctx.fillText(label, x + padding, Math.max(fontSize, y - padding))
        }
      })

      console.log('[BBoxDebug] Drawing complete')
    }

    const handleLoad = () => {
      console.log('[BBoxDebug] Image loaded')
      drawAnnotations()
    }

    if (image.complete && image.naturalWidth > 0) {
      drawAnnotations()
    } else {
      image.addEventListener('load', handleLoad)
    }

    return () => {
      image.removeEventListener('load', handleLoad)
    }
  }, [annotations, showLabels])

  return (
    <Card className={`relative ${className}`}>
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="text-white">어노테이션 로딩 중...</div>
        </div>
      )}

      {error && (
        <div className="absolute top-4 left-4 right-4 z-10">
          <div className="bg-red-500 text-white p-4 rounded">
            {error}
          </div>
        </div>
      )}

      <div className="relative w-full" style={{ aspectRatio: '1' }}>
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Annotated image (debug)"
          className="w-full h-full object-contain bg-slate-950"
          crossOrigin="anonymous"
        />

        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {!loading && annotations.length > 0 && (
        <div className="p-4 border-t">
          <div className="text-sm text-gray-600 mb-2">
            탐지된 객체: {annotations.length}개
          </div>
          <div className="flex flex-wrap gap-2">
            {annotations.map((ann, idx) => (
              <Badge key={ann.id} variant="outline">
                {ann.class_name} ({ann.confidence ? (Number(ann.confidence) * 100).toFixed(1) : '0'}%)
              </Badge>
            ))}
          </div>
        </div>
      )}

      {!loading && annotations.length === 0 && (
        <div className="p-4 text-center text-gray-500">
          탐지된 객체가 없습니다
        </div>
      )}
    </Card>
  )
}

export default BBoxDebugViewer
