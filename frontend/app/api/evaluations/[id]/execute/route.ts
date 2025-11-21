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
    const { searchParams } = new URL(request.url)
    const body = await request.json().catch(() => ({}))

    const confThreshold = searchParams.get('conf_threshold')
    const iouThreshold = searchParams.get('iou_threshold')

    let queryString = ''
    const queryParams = new URLSearchParams()

    if (confThreshold) queryParams.append('conf_threshold', confThreshold)
    if (iouThreshold) queryParams.append('iou_threshold', iouThreshold)

    if (queryParams.toString()) {
      queryString = `?${queryParams.toString()}`
    }

    console.log('[/api/evaluations/[id]/execute] POST request - runId:', id, 'query:', queryString)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/evaluation/runs/${id}/execute${queryString}`,
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
      console.error('[/api/evaluations/[id]/execute] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to execute evaluation run' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error('[/api/evaluations/[id]/execute] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
