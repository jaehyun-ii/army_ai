import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const draw_boxes = searchParams.get('draw_boxes') || 'false'
    const t = searchParams.get('t') || Date.now().toString()

    console.log('[/api/camera/stream] GET request - draw_boxes:', draw_boxes)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/camera/stream?draw_boxes=${draw_boxes}&t=${t}`,
      {
        method: 'GET',
      }
    )

    if (!backendResponse.ok) {
      console.error('[/api/camera/stream] Backend error:', backendResponse.status)
      return new NextResponse('Stream error', { status: backendResponse.status })
    }

    // Stream the response directly
    return new NextResponse(backendResponse.body, {
      headers: {
        'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
    })
  } catch (error) {
    console.error('[/api/camera/stream] Error:', error)
    return new NextResponse('Internal server error', { status: 500 })
  }
}
