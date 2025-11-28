import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params

    console.log('[/api/attack-datasets/[id]] GET request - id:', id)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/attack-datasets/${id}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/attack-datasets/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch attack dataset' },
        { status: backendResponse.status }
      )
    }

    const dataset = await backendResponse.json()
    console.log('[/api/attack-datasets/[id]] Fetched dataset:', dataset.id)

    return NextResponse.json(dataset)
  } catch (error) {
    console.error('[/api/attack-datasets/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params

    console.log('[/api/attack-datasets/[id]] DELETE request - id:', id)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/attack-datasets/${id}`,
      {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/attack-datasets/[id]] Backend delete error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to delete attack dataset' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json().catch(() => ({ message: 'Deleted successfully' }))
    console.log('[/api/attack-datasets/[id]] Deleted dataset:', id)

    return NextResponse.json(result)
  } catch (error) {
    console.error('[/api/attack-datasets/[id]] Delete error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
