/**
 * YOLO dataset upload proxy API route
 * Proxies YOLO file uploads to the backend
 *
 * Usage: POST /api/datasets/upload-yolo
 * Body: FormData with images, labels, and classes file
 */
import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  console.log("[/api/datasets/upload-yolo] POST request received")

  try {
    // Get the form data from the request
    const formData = await request.formData()

    console.log("[/api/datasets/upload-yolo] FormData keys:", Array.from(formData.keys()))
    console.log("[/api/datasets/upload-yolo] Dataset name:", formData.get('dataset_name'))
    console.log("[/api/datasets/upload-yolo] Images count:", formData.getAll('images').length)
    console.log("[/api/datasets/upload-yolo] Labels count:", formData.getAll('labels').length)
    console.log("[/api/datasets/upload-yolo] Classes file:", formData.get('classes_file') ? 'Present' : 'Missing')

    // Validate required fields
    if (!formData.get('dataset_name')) {
      return NextResponse.json(
        { error: 'Dataset name is required' },
        { status: 400 }
      )
    }

    if (!formData.get('classes_file')) {
      return NextResponse.json(
        { error: 'Classes file is required' },
        { status: 400 }
      )
    }

    const images = formData.getAll('images')
    if (!images || images.length === 0) {
      return NextResponse.json(
        { error: 'At least one image is required' },
        { status: 400 }
      )
    }

    console.log("[/api/datasets/upload-yolo] Forwarding to backend:", `${BACKEND_API_URL}/api/v1/dataset-service/upload-yolo-files`)

    // Forward the entire FormData to backend
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/dataset-service/upload-yolo-files`,
      {
        method: 'POST',
        body: formData,
      }
    )

    console.log("[/api/datasets/upload-yolo] Backend response status:", backendResponse.status)

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets/upload-yolo] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || errorData.error || 'Failed to upload YOLO dataset' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    console.log("[/api/datasets/upload-yolo] Upload successful:", result)

    return NextResponse.json(result)
  } catch (error) {
    console.error('[/api/datasets/upload-yolo] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
