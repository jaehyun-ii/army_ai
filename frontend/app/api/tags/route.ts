import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const skip = searchParams.get('skip')
    const limit = searchParams.get('limit')
    const search = searchParams.get('search')

    let queryString = ''
    const queryParams = new URLSearchParams()

    if (skip) queryParams.append('skip', skip)
    if (limit) queryParams.append('limit', limit)
    if (search) queryParams.append('search', search)

    if (queryParams.toString()) {
      queryString = `?${queryParams.toString()}`
    }

    console.log('[/api/tags] GET request - query:', queryString)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/tags${queryString}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/tags] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch tags' },
        { status: backendResponse.status }
      )
    }

    const tags = await backendResponse.json()
    return NextResponse.json(tags)
  } catch (error) {
    console.error('[/api/tags] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    console.log('[/api/tags] POST request - body:', body)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/tags`,
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
      console.error('[/api/tags] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to create tag' },
        { status: backendResponse.status }
      )
    }

    const tag = await backendResponse.json()
    return NextResponse.json(tag, { status: 201 })
  } catch (error) {
    console.error('[/api/tags] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
