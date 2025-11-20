import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const skip = searchParams.get('skip') || '0'
    const limit = searchParams.get('limit') || '100'

    console.log('[/api/datasets] GET request - skip:', skip, 'limit:', limit)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/datasets-2d?skip=${skip}&limit=${limit}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch datasets' },
        { status: backendResponse.status }
      )
    }

    const datasets = await backendResponse.json()
    console.log('[/api/datasets] Fetched datasets count:', datasets.length)

    // Transform backend response to match frontend expectations
    const transformedDatasets = datasets.map((dataset: any) => ({
      id: dataset.id,
      name: dataset.name,
      description: dataset.description,
      type: '2D_IMAGE',
      source: 'CUSTOM',
      size: dataset.image_count || 0, // Map image_count to size
      storageLocation: dataset.storage_path,
      metadata: {
        totalSizeBytes: dataset.metadata?.total_size_bytes || 0,
        model: dataset.metadata?.model,
        ...dataset.metadata,
      },
      createdAt: dataset.created_at,
      updatedAt: dataset.updated_at,
    }))

    return NextResponse.json(transformedDatasets)
  } catch (error) {
    console.error('[/api/datasets] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    console.log('[/api/datasets] POST request - body:', body)

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/datasets-2d`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to create dataset' },
        { status: backendResponse.status }
      )
    }

    const dataset = await backendResponse.json()
    console.log('[/api/datasets] Created dataset:', dataset.id)

    return NextResponse.json(dataset, { status: 201 })
  } catch (error) {
    console.error('[/api/datasets] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
