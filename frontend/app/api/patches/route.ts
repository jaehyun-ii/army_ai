import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = searchParams.get('limit') || '100'
    const skip = searchParams.get('skip') || '0'

    console.log('[/api/patches] GET request - limit:', limit, 'skip:', skip)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/patches/?limit=${limit}&skip=${skip}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/patches] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch patches' },
        { status: backendResponse.status }
      )
    }

    const patches = await backendResponse.json()
    console.log('[/api/patches] Fetched patches count:', patches.length)

    return NextResponse.json(patches)
  } catch (error) {
    console.error('[/api/patches] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
