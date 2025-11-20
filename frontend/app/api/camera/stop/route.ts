import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    console.log('[/api/camera/stop] POST request')

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/camera/stop`,
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
      console.error('[/api/camera/stop] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || errorData.error || 'Failed to stop camera' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    console.log('[/api/camera/stop] Camera stopped successfully')

    return NextResponse.json(data)
  } catch (error) {
    console.error('[/api/camera/stop] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
