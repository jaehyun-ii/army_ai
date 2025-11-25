import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    // Get interval parameter from query string
    const searchParams = request.nextUrl.searchParams
    const interval = searchParams.get('interval') || '1.0'

    console.log('[/api/system-stats/stream] SSE stream request with interval:', interval)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/system/stats/stream?interval=${interval}`,
      {
        method: 'GET',
      }
    )

    if (!backendResponse.ok) {
      console.error('[/api/system-stats/stream] Backend error:', backendResponse.status)
      return new NextResponse('Stream error', { status: backendResponse.status })
    }

    // Stream SSE response directly with proper headers
    return new NextResponse(backendResponse.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    })
  } catch (error) {
    console.error('[/api/system-stats/stream] Error:', error)
    return new NextResponse('Internal server error', { status: 500 })
  }
}
