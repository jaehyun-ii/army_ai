import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)

    const cleanRunId = searchParams.get('clean_run_id')
    const advRunId = searchParams.get('adv_run_id')
    const page = searchParams.get('page') || '1'
    const pageSize = searchParams.get('page_size') || '20'

    if (!cleanRunId || !advRunId) {
      return NextResponse.json(
        { error: 'clean_run_id and adv_run_id are required' },
        { status: 400 }
      )
    }

    const queryString = `clean_run_id=${cleanRunId}&adv_run_id=${advRunId}&page=${page}&page_size=${pageSize}`

    console.log('[/api/evaluations/compare-images] GET request - query:', queryString)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/evaluation/runs/compare-images?${queryString}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/evaluations/compare-images] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to compare evaluation images' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error('[/api/evaluations/compare-images] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
