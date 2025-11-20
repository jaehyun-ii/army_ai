import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    console.log('[/api/evaluations] GET request - forwarding to backend')

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/evaluation/runs`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/evaluations] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch evaluations' },
        { status: backendResponse.status }
      )
    }

    const evaluations = await backendResponse.json()
    return NextResponse.json(evaluations)
  } catch (error) {
    console.error('[/api/evaluations] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    console.log('[/api/evaluations] POST request - forwarding to backend')

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/evaluation/runs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/evaluations] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to create evaluation' },
        { status: backendResponse.status }
      )
    }

    const evaluation = await backendResponse.json()
    return NextResponse.json(evaluation, { status: 201 })
  } catch (error) {
    console.error('[/api/evaluations] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
