import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const body = await request.json()

    console.log('[/api/models/[id]/predict] POST request - modelId:', id)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/models/${id}/predict`,
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
      console.error('[/api/models/[id]/predict] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to run prediction' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    console.log('[/api/models/[id]/predict] Prediction completed successfully')

    return NextResponse.json(data)
  } catch (error) {
    console.error('[/api/models/[id]/predict] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
