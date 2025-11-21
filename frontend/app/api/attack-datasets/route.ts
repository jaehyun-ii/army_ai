import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const skip = searchParams.get('skip') || '0'
    const limit = searchParams.get('limit') || '100'
    const attackType = searchParams.get('attack_type')

    let queryString = `skip=${skip}&limit=${limit}`
    if (attackType) {
      queryString += `&attack_type=${attackType}`
    }

    console.log('[/api/attack-datasets] GET request - query:', queryString)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/attack-datasets?${queryString}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/attack-datasets] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch attack datasets' },
        { status: backendResponse.status }
      )
    }

    const datasets = await backendResponse.json()
    console.log('[/api/attack-datasets] Fetched attack datasets count:', datasets.length)

    return NextResponse.json(datasets)
  } catch (error) {
    console.error('[/api/attack-datasets] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
