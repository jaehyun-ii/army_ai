import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    console.log('[/api/adversarial-patches] GET request - forwarding to backend')

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/adversarial-patch/patches`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/adversarial-patches] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch patches' },
        { status: backendResponse.status }
      )
    }

    const patches = await backendResponse.json()
    return NextResponse.json(patches)
  } catch (error) {
    console.error('[/api/adversarial-patches] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
