import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    // Handle both sync and async params for Next.js compatibility
    const params = await Promise.resolve(context.params)
    const datasetId = params.id
    const { searchParams } = new URL(request.url)
    // Support both 'skip' and 'offset' query parameters
    const skip = searchParams.get('skip') || searchParams.get('offset') || '0'
    const limit = searchParams.get('limit') || '100'

    console.log('[/api/datasets/[id]/images] GET request - id:', datasetId, 'skip:', skip, 'limit:', limit)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/datasets-2d/${datasetId}/images?skip=${skip}&limit=${limit}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets/[id]/images] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch images' },
        { status: backendResponse.status }
      )
    }

    const images = await backendResponse.json()
    console.log('[/api/datasets/[id]/images] Fetched images count:', images.length)

    // If no images from backend (all deleted), return empty list
    if (!images || images.length === 0) {
      return NextResponse.json({
        total: 0,
        images: [],
      })
    }

    // Enrich images with base64 data and fetch annotations from backend
    const enrichedImages = await Promise.all(
      images.map(async (image: any) => {
        let base64Data = null
        let detections = null
        let visualization = null

        // Try to read image file and convert to base64
        if (image.storage_key) {
          try {
            let imagePath = image.storage_key
            console.log(`[/api/datasets/[id]/images] Original storage_key: ${image.storage_key}`)
            console.log(`[/api/datasets/[id]/images] isAbsolute: ${path.isAbsolute(imagePath)}`)

            // If storage_key is relative, convert to absolute path using STORAGE_ROOT
            if (!path.isAbsolute(imagePath)) {
              // storage_key is now relative to STORAGE_ROOT (e.g., "datasets/test_dataset_20251029_114112/sample_image.jpg")
              const cwd = process.cwd()
              const storageRoot = path.join(cwd, '..', 'database', 'storage')
              imagePath = path.join(storageRoot, imagePath)
              console.log(`[/api/datasets/[id]/images] process.cwd(): ${cwd}`)
              console.log(`[/api/datasets/[id]/images] storageRoot: ${storageRoot}`)
              console.log(`[/api/datasets/[id]/images] Final imagePath: ${imagePath}`)
            }

            const imageBuffer = await fs.readFile(imagePath)
            base64Data = imageBuffer.toString('base64')
            console.log(`[/api/datasets/[id]/images] Successfully read image ${image.file_name}, size: ${imageBuffer.length} bytes`)
          } catch (err) {
            console.error(`[/api/datasets/[id]/images] Failed to read image ${image.file_name}:`, err)
            console.error(`[/api/datasets/[id]/images] Original storage_key: ${image.storage_key}`)
          }
        }

        // Fetch annotations from backend API
        try {
          const annotationsResponse = await fetch(
            `${BACKEND_API_URL}/api/v1/annotations/image/${image.id}`,
            {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
              },
            }
          )

          if (annotationsResponse.ok) {
            const annotations = await annotationsResponse.json()

            // Convert annotations to detections format (YOLO-compatible)
            if (annotations && Array.isArray(annotations) && annotations.length > 0) {
              console.log(`[API Route] Image ${image.file_name} - Raw annotations from DB:`,
                annotations.map((a: any) => ({
                  class: a.class_name,
                  bbox_x: a.bbox_x,
                  bbox_y: a.bbox_y,
                  bbox_width: a.bbox_width,
                  bbox_height: a.bbox_height
                }))
              )

              detections = annotations.map((ann: any) => {
                const x1 = parseFloat(ann.bbox_x) || 0
                const y1 = parseFloat(ann.bbox_y) || 0
                const width = parseFloat(ann.bbox_width) || 0
                const height = parseFloat(ann.bbox_height) || 0
                const x2 = x1 + width
                const y2 = y1 + height

                console.log(`[API Route] Converting: bbox_x=${x1}, bbox_y=${y1}, bbox_width=${width}, bbox_height=${height} -> x1=${x1}, y1=${y1}, x2=${x2}, y2=${y2}`)

                return {
                  class: ann.class_name,
                  class_id: ann.class_index,
                  confidence: parseFloat(ann.confidence) || 1.0,
                  bbox: { x1, y1, x2, y2 }
                }
              })
            }
          }
        } catch (err) {
          console.error(`[/api/datasets/[id]/images] Failed to fetch annotations for image ${image.id}:`, err)
        }

        // Extract visualization from metadata (if any)
        if (image.metadata?.visualization) {
          visualization = image.metadata.visualization
        }

        return {
          id: image.id,
          datasetId: image.dataset_id,
          filename: image.file_name,
          data: base64Data,
          width: image.width,
          height: image.height,
          format: path.extname(image.file_name).slice(1).toUpperCase() || 'UNKNOWN',
          mimeType: image.mime_type || 'image/jpeg',
          metadata: image.metadata,
          createdAt: image.created_at,
          detections,
          visualization,
        }
      })
    )

    return NextResponse.json({
      total: enrichedImages.length,
      images: enrichedImages,
    })
  } catch (error) {
    console.error('[/api/datasets/[id]/images] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
