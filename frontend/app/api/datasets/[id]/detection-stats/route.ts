import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const body = await request.json()

    console.log('[/api/datasets/[id]/detection-stats] POST request - datasetId:', id, 'body:', body)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/dataset-service/${id}/detection-stats`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets/[id]/detection-stats] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch detection stats' },
        { status: backendResponse.status }
      )
    }

    const stats = await backendResponse.json()
    return NextResponse.json(stats)
  } catch (error) {
    console.error('[/api/datasets/[id]/detection-stats] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
