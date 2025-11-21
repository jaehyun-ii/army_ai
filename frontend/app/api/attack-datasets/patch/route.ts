import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    console.log('[/api/attack-datasets/patch] POST request - body:', body)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/attack-datasets/patch`,
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
      console.error('[/api/attack-datasets/patch] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to generate patch attack dataset' },
        { status: backendResponse.status }
      )
    }

    const dataset = await backendResponse.json()
    return NextResponse.json(dataset, { status: 201 })
  } catch (error) {
    console.error('[/api/attack-datasets/patch] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
