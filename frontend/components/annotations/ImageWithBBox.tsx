'use client'

import React, { useEffect, useState, useRef } from 'react'
import { apiClient } from '@/lib/api-client'
import { ImageIcon } from 'lucide-react'

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
  // For pixel-based coordinates
  x1?: number
  y1?: number
  x2?: number
  y2?: number
}

interface ImageWithBBoxProps {
  imageId: string
  imageData?: string // Base64 encoded image data
  imageMimeType?: string
  imageWidth?: number
  imageHeight?: number
  className?: string
  targetClass?: string // Only show bounding boxes for this class
  minConfidence?: number
  patchImageUrl?: string // URL of the adversarial patch to render
  patchScale?: number // Patch scale as percentage of bbox area (e.g., 30 for 30%)
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
 * 간단한 이미지 + 바운딩 박스 렌더링 컴포넌트
 * 이미지 그리드에서 사용하기 위한 경량 버전
 */
export function ImageWithBBox({
  imageId,
  imageData,
  imageMimeType = 'image/jpeg',
  imageWidth = 1,
  imageHeight = 1,
  className = '',
  targetClass,
  minConfidence = 0.0,
  patchImageUrl,
  patchScale = 30,
}: ImageWithBBoxProps) {
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [patchImage, setPatchImage] = useState<HTMLImageElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // 패치 이미지 로드
  useEffect(() => {
    if (!patchImageUrl) {
      setPatchImage(null)
      return
    }

    console.log('[ImageWithBBox] Loading patch image from:', patchImageUrl)
    console.log('[ImageWithBBox] Patch scale:', patchScale)

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      console.log('[ImageWithBBox] Patch image loaded successfully')
      setPatchImage(img)
    }
    img.onerror = (err) => {
      console.error('[ImageWithBBox] Failed to load patch image:', err)
      console.error('[ImageWithBBox] Patch URL:', patchImageUrl)
      setPatchImage(null)
    }
    img.src = patchImageUrl
  }, [patchImageUrl])

  // 어노테이션 가져오기
  useEffect(() => {
    // targetClass가 변경될 때 즉시 annotations를 초기화하여 깜빡임 방지
    setAnnotations([])

    const fetchAnnotations = async () => {
      try {
        const data = await apiClient.getImageAnnotations(imageId, {
          annotation_type: 'bbox',
          min_confidence: minConfidence,
        }) as Annotation[]

        // 타겟 클래스가 지정되어 있으면 필터링
        const filteredData = targetClass
          ? data.filter(ann => ann.class_name === targetClass)
          : data

        setAnnotations(filteredData)
      } catch (err) {
        console.error('Failed to fetch annotations:', err)
      }
    }

    if (imageId) {
      fetchAnnotations()
    }
  }, [imageId, minConfidence, targetClass])

  // 이미지 로드 및 어노테이션 그리기
  useEffect(() => {
    const image = imageRef.current
    const canvas = canvasRef.current
    const container = containerRef.current

    if (!image || !canvas || !container) {
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
      const displayWidth = image.clientWidth || container.clientWidth
      const displayHeight = image.clientHeight || container.clientHeight

      if (!displayWidth || !displayHeight) {
        return
      }

      canvas.width = displayWidth
      canvas.height = displayHeight
      ctx.clearRect(0, 0, displayWidth, displayHeight)

      if (annotations.length === 0) {
        return
      }

      const naturalWidth = image.naturalWidth || imageWidth || 1
      const naturalHeight = image.naturalHeight || imageHeight || 1

      const imageAspect = naturalWidth / naturalHeight
      const displayAspect = displayWidth / displayHeight

      let drawWidth = displayWidth
      let drawHeight = displayHeight
      let offsetX = 0
      let offsetY = 0

      if (displayAspect > imageAspect) {
        // Letter boxing on sides
        drawHeight = displayHeight
        drawWidth = drawHeight * imageAspect
        offsetX = (displayWidth - drawWidth) / 2
      } else {
        // Letter boxing on top/bottom
        drawWidth = displayWidth
        drawHeight = drawWidth / imageAspect
        offsetY = (displayHeight - drawHeight) / 2
      }

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
        const hue = targetClass && ann.class_name === targetClass ? 0 : getColorFromClassName(ann.class_name)
        const color = `hsl(${hue}, 70%, 50%)`

        // Draw bounding box
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, w, h)

        // Draw patch if available and this is the target class
        if (patchImage && patchImageUrl && (!targetClass || ann.class_name === targetClass)) {
          // Calculate patch size based on bbox area and patchScale
          // Patch area = bbox_area * (patchScale / 100)
          const bboxArea = w * h
          const patchArea = bboxArea * (patchScale / 100)

          // Assuming square patch, side length = sqrt(patchArea)
          const patchSize = Math.sqrt(patchArea)

          // Calculate patch position (center of bbox using already calculated x, y, w, h)
          const bboxCenterX = x + w / 2
          const bboxCenterY = y + h / 2
          const patchX = bboxCenterX - patchSize / 2
          const patchY = bboxCenterY - patchSize / 2

          console.log('[ImageWithBBox] Drawing patch:', {
            bboxArea,
            patchArea,
            patchSize,
            bboxCenterX,
            bboxCenterY,
            patchX,
            patchY,
            className: ann.class_name,
            targetClass
          })

          // Draw the patch (opaque)
          ctx.drawImage(patchImage, patchX, patchY, patchSize, patchSize)
        } else if (patchImageUrl) {
          console.log('[ImageWithBBox] Patch NOT drawn:', {
            hasPatchImage: !!patchImage,
            hasPatchUrl: !!patchImageUrl,
            className: ann.class_name,
            targetClass,
            matches: !targetClass || ann.class_name === targetClass
          })
        }
      })
    }

    const handleLoad = () => drawAnnotations()

    if (image.complete) {
      drawAnnotations()
    } else {
      image.addEventListener('load', handleLoad)
    }

    let resizeObserver: ResizeObserver | undefined
    const win: Window | null = typeof window !== 'undefined' ? window as Window : null
    
    if (win && 'ResizeObserver' in win) {
      resizeObserver = new ResizeObserver(() => drawAnnotations())
      if (container) {
        resizeObserver.observe(container)
      }
    } else if (win) {
      win.addEventListener('resize', drawAnnotations)
    }

    return () => {
      image.removeEventListener('load', handleLoad)
      if (resizeObserver) {
        resizeObserver.disconnect()
      } else if (win) {
        win.removeEventListener('resize', drawAnnotations)
      }
    }
  }, [annotations, targetClass, imageWidth, imageHeight, patchImage, patchImageUrl, patchScale])

  return (
    <div ref={containerRef} className={`relative w-full h-full ${className}`}>
      {imageData ? (
        <>
          <img
            ref={imageRef}
            src={`data:${imageMimeType};base64,${imageData}`}
            alt="Image with annotations"
            className="w-full h-full object-contain bg-slate-900"
            crossOrigin="anonymous"
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
          />
        </>
      ) : (
        <div className="w-full h-full flex items-center justify-center bg-slate-700">
          <ImageIcon className="w-8 h-8 text-slate-500" />
        </div>
      )}
    </div>
  )
}

export default ImageWithBBox
