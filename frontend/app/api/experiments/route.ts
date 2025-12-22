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

    console.log('[/api/experiments] GET request - skip:', skip, 'limit:', limit)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/experiments?skip=${skip}&limit=${limit}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/experiments] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch experiments' },
        { status: backendResponse.status }
      )
    }

    const experiments = await backendResponse.json()
    return NextResponse.json(experiments)
  } catch (error) {
    console.error('[/api/experiments] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    console.log('[/api/experiments] POST request - body:', body)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/experiments`,
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
      console.error('[/api/experiments] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to create experiment' },
        { status: backendResponse.status }
      )
    }

    const experiment = await backendResponse.json()
    return NextResponse.json(experiment, { status: 201 })
  } catch (error) {
    console.error('[/api/experiments] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
