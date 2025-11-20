import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  console.log("[/api/datasets/upload-folder] POST request received")

  try {
    const formData = await request.formData()
    const datasetName = formData.get('name') as string
    const description = formData.get('description') as string | null
    const files = formData.getAll('files') as File[]
    const metadataFile = formData.get('metadata') as File | null

    console.log("[/api/datasets/upload-folder] Dataset name:", datasetName)
    console.log("[/api/datasets/upload-folder] Description:", description)
    console.log("[/api/datasets/upload-folder] Files count:", files.length)
    console.log("[/api/datasets/upload-folder] Metadata file:", metadataFile?.name || 'None')

    if (!datasetName) {
      return NextResponse.json(
        { error: 'Dataset name is required' },
        { status: 400 }
      )
    }

    if (!files || files.length === 0) {
      return NextResponse.json(
        { error: 'No files provided' },
        { status: 400 }
      )
    }

    // Create FormData for backend API
    const backendFormData = new FormData()
    backendFormData.append('dataset_name', datasetName)
    if (description) {
      backendFormData.append('description', description)
    }

    // Add all image files
    for (const file of files) {
      backendFormData.append('images', file)
    }

    // Add metadata file if present
    if (metadataFile) {
      backendFormData.append('metadata', metadataFile)
      console.log("[/api/datasets/upload-folder] Including metadata file in upload")
    }

    console.log("[/api/datasets/upload-folder] Forwarding to backend:", `${BACKEND_API_URL}/api/v1/dataset-service/upload-multipart`)

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/dataset-service/upload-multipart`, {
      method: 'POST',
      body: backendFormData,
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error("[/api/datasets/upload-folder] Backend error:", errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to upload dataset' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    console.log("[/api/datasets/upload-folder] Upload successful, image count:", result.image_count)

    return NextResponse.json({
      success: true,
      dataset: result.dataset,
      message: result.message,
      imageCount: result.image_count,
    })

  } catch (error) {
    console.error('[/api/datasets/upload-folder] Error:', error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Internal server error',
        details: 'Failed to upload dataset to backend'
      },
      { status: 500 }
    )
  }
}
