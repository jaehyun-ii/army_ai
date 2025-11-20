'use client'

import React, { useEffect, useState, useRef } from 'react'
import { Annotation } from '@/types/common'
import { apiClient } from '@/lib/api-client'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface AnnotatedImageViewerProps {
  imageId: string
  imageUrl: string
  minConfidence?: number
  showLabels?: boolean
  className?: string
  aspectRatio?: string | number // 기본값: 'auto' (이미지 원본 비율 유지)
  showAnnotationInfo?: boolean // 하단에 어노테이션 정보 표시 여부 (기본값: true)
}

/**
 * 클래스 이름을 기반으로 일관된 색상(hue)을 생성
 */
function getColorFromClassName(className: string): number {
  let hash = 0
  for (let i = 0; i < className.length; i++) {
    hash = className.charCodeAt(i) + ((hash << 5) - hash)
    hash = hash & hash // Convert to 32bit integer
  }
  return Math.abs(hash % 360)
}

/**
 * 이미지에 탐지된 객체의 바운딩 박스와 라벨을 렌더링하는 컴포넌트
 *
 * 사용법:
 * ```tsx
 * <AnnotatedImageViewer
 *   imageId="image-uuid"
 *   imageUrl="http://localhost:8000/storage/..."
 *   minConfidence={0.5}
 *   showLabels={true}
 * />
 * ```
 */
export function AnnotatedImageViewer({
  imageId,
  imageUrl,
  minConfidence = 0.0,
  showLabels = true,
  className = '',
  aspectRatio = 'auto',
  showAnnotationInfo = true,
}: AnnotatedImageViewerProps) {
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  // 어노테이션 가져오기
  useEffect(() => {
    const fetchAnnotations = async () => {
      try {
        setLoading(true)
        setError(null)

        const data = await apiClient.getImageAnnotations(imageId, {
          annotation_type: 'bbox',
          min_confidence: minConfidence,
        }) as Annotation[]

        setAnnotations(data)
      } catch (err) {
        console.error('Failed to fetch annotations:', err)
        setError(err instanceof Error ? err.message : 'Failed to load annotations')
      } finally {
        setLoading(false)
      }
    }

    fetchAnnotations()
  }, [imageId, minConfidence])

  // 이미지 로드 및 어노테이션 그리기
  useEffect(() => {
    const image = imageRef.current
    const canvas = canvasRef.current

    if (!image || !canvas) {
      if (canvas?.getContext) {
        const ctx = canvas.getContext('2d')
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height)
        }
      }
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const drawAnnotations = () => {
      // 이미지 컨테이너의 표시 크기
      const displayWidth = image.clientWidth
      const displayHeight = image.clientHeight

      if (!displayWidth || !displayHeight) {
        return
      }

      canvas.width = displayWidth
      canvas.height = displayHeight
      ctx.clearRect(0, 0, displayWidth, displayHeight)

      if (annotations.length === 0) {
        return
      }

      // 원본 이미지의 자연 크기
      const naturalWidth = image.naturalWidth || 1
      const naturalHeight = image.naturalHeight || 1

      // 종횡비 계산
      const imageAspect = naturalWidth / naturalHeight
      const displayAspect = displayWidth / displayHeight

      // object-contain 동작을 시뮬레이션: 레터박싱 계산
      let drawWidth = displayWidth
      let drawHeight = displayHeight
      let offsetX = 0
      let offsetY = 0

      if (displayAspect > imageAspect) {
        // 이미지가 세로로 긴 경우 - 양옆에 레터박스
        drawHeight = displayHeight
        drawWidth = drawHeight * imageAspect
        offsetX = (displayWidth - drawWidth) / 2
      } else {
        // 이미지가 가로로 긴 경우 - 위아래에 레터박스
        drawWidth = displayWidth
        drawHeight = drawWidth / imageAspect
        offsetY = (displayHeight - drawHeight) / 2
      }

      // 각 어노테이션 그리기
      annotations.forEach((ann) => {
        if (
          ann.annotation_type !== 'bbox' ||
          ann.bbox_x === undefined ||
          ann.bbox_y === undefined ||
          ann.bbox_width === undefined ||
          ann.bbox_height === undefined
        ) {
          return
        }

        let x: number, y: number, w: number, h: number

        // Check for pixel-based coordinates first
        if (ann.x1 !== undefined && ann.y1 !== undefined && ann.x2 !== undefined && ann.y2 !== undefined) {
          const scaleX = drawWidth / naturalWidth
          const scaleY = drawHeight / naturalHeight
          x = offsetX + ann.x1 * scaleX
          y = offsetY + ann.y1 * scaleY
          w = (ann.x2 - ann.x1) * scaleX
          h = (ann.y2 - ann.y1) * scaleY
        } else {
          // Fallback to normalized coordinates
          const xCenter = Number(ann.bbox_x)
          const yCenter = Number(ann.bbox_y)
          const width = Number(ann.bbox_width)
          const height = Number(ann.bbox_height)

          if (
            Number.isNaN(xCenter) ||
            Number.isNaN(yCenter) ||
            Number.isNaN(width) ||
            Number.isNaN(height)
          ) {
            return
          }

          x = offsetX + (xCenter - width / 2) * drawWidth
          y = offsetY + (yCenter - height / 2) * drawHeight
          w = width * drawWidth
          h = height * drawHeight
        }

        // 클래스 이름을 기반으로 일관된 색상 생성
        const hue = getColorFromClassName(ann.class_name)
        const color = `hsl(${hue}, 70%, 50%)`

        // 바운딩 박스 그리기
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, w, h)

        // 라벨 그리기
        if (showLabels) {
          const label = `${ann.class_name} ${ann.confidence ? (Number(ann.confidence) * 100).toFixed(1) : ''}%`
          const fontSize = 14
          ctx.font = `${fontSize}px Arial`
          const textWidth = ctx.measureText(label).width
          const padding = 4

          // 배경
          ctx.fillStyle = color
          ctx.fillRect(x, Math.max(0, y - fontSize - padding * 2), textWidth + padding * 2, fontSize + padding * 2)

          // 텍스트
          ctx.fillStyle = 'white'
          ctx.fillText(label, x + padding, Math.max(fontSize, y - padding))
        }
      })
    }

    const handleLoad = () => drawAnnotations()

    if (image.complete && image.naturalWidth > 0) {
      drawAnnotations()
    } else {
      image.addEventListener('load', handleLoad)
    }

    // ResizeObserver로 리사이즈 감지
    let resizeObserver: ResizeObserver | undefined
    if (typeof window !== 'undefined' && 'ResizeObserver' in window) {
      resizeObserver = new ResizeObserver(() => drawAnnotations())
      resizeObserver.observe(image)
    } else {
      if (typeof window !== 'undefined' && window && typeof (window as any).addEventListener === 'function') {
        (window as Window).addEventListener('resize', drawAnnotations)
      }
    }

    return () => {
      image.removeEventListener('load', handleLoad)
      if (resizeObserver) {
        resizeObserver.disconnect()
      } else {
        window.removeEventListener('resize', drawAnnotations)
      }
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

      <div className="relative w-full h-[400px]">
        {/* 원본 이미지 */}
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Annotated image"
          className="w-full h-full object-contain bg-slate-950"
          crossOrigin="anonymous"
        />

        {/* 어노테이션 캔버스 오버레이 */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>


      {!loading && annotations.length === 0 && (
        <div className="p-4 text-center text-gray-500">
          탐지된 객체가 없습니다
        </div>
      )}
    </Card>
  )
}

export default AnnotatedImageViewer
