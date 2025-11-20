import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    console.log('[/api/system-stats] GET request - forwarding to backend')

    // Forward to backend API for system statistics
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/system/stats`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      // If endpoint doesn't exist, return informative error
      if (backendResponse.status === 404) {
        return NextResponse.json({
          error: 'System stats endpoint not implemented in backend.',
          note: 'Backend should implement /api/v1/system/stats endpoint'
        }, { status: 501 })
      }

      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/system-stats] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch system stats' },
        { status: backendResponse.status }
      )
    }

    const stats = await backendResponse.json()
    return NextResponse.json(stats)
  } catch (error) {
    console.error('[/api/system-stats] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
