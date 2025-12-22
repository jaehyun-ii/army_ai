import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

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

    const data = await backendResponse.json()
    const images = Array.isArray(data) ? data : data.items || data.images || []
    const total = typeof data.total === 'number' ? data.total : images.length
    console.log('[/api/datasets/[id]/images] Fetched images count:', images.length, 'total:', total)

    // If no images from backend (all deleted), return empty list
    if (!images || images.length === 0) {
      return NextResponse.json({
        total: 0,
        images: [],
      })
    }

    // Return lightweight image metadata with storage_key (no base64 data)
    // Images will be loaded via /api/storage/[...path] endpoint
    const enrichedImages = images.map((image: any) => {
      const storageKey = image.storage_key || image.storageKey
      return {
        id: image.id,
        datasetId: image.dataset_id || image.datasetId,
        filename: image.file_name || image.filename,
        storage_key: storageKey, // Keep storage_key for image loading via /api/storage
        previewUrl: storageKey ? `/api/storage/${storageKey}` : null,
        width: image.width,
        height: image.height,
        format: path.extname(image.file_name || image.filename || '').slice(1).toUpperCase() || 'UNKNOWN',
        mimeType: image.mime_type || image.mimeType || 'image/jpeg',
        metadata: image.metadata || image.metadata_,
        createdAt: image.created_at || image.createdAt,
      }
    })

    return NextResponse.json({
      total,
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
