import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    console.log('[/api/camera/stats/stream] SSE stream request')

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/camera/stats/stream`,
      {
        method: 'GET',
      }
    )

    if (!backendResponse.ok) {
      console.error('[/api/camera/stats/stream] Backend error:', backendResponse.status)
      return new NextResponse('Stream error', { status: backendResponse.status })
    }

    // Stream SSE response directly
    return new NextResponse(backendResponse.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    })
  } catch (error) {
    console.error('[/api/camera/stats/stream] Error:', error)
    return new NextResponse('Internal server error', { status: 500 })
  }
}
