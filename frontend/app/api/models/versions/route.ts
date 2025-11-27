import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    console.log('[/api/models/versions] GET request')

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/models/versions/`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/models/versions] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch model versions' },
        { status: backendResponse.status }
      )
    }

    const versions = await backendResponse.json()
    console.log('[/api/models/versions] Fetched versions count:', versions.length)

    return NextResponse.json(versions)
  } catch (error) {
    console.error('[/api/models/versions] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
