import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params

    console.log('[/api/experiments/[id]] GET request - experimentId:', id)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/experiments/${id}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/experiments/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch experiment' },
        { status: backendResponse.status }
      )
    }

    const experiment = await backendResponse.json()
    return NextResponse.json(experiment)
  } catch (error) {
    console.error('[/api/experiments/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const body = await request.json()

    console.log('[/api/experiments/[id]] PATCH request - experimentId:', id)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/experiments/${id}`,
      {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/experiments/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to update experiment' },
        { status: backendResponse.status }
      )
    }

    const experiment = await backendResponse.json()
    return NextResponse.json(experiment)
  } catch (error) {
    console.error('[/api/experiments/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params

    console.log('[/api/experiments/[id]] DELETE request - experimentId:', id)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/experiments/${id}`,
      {
        method: 'DELETE',
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/experiments/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to delete experiment' },
        { status: backendResponse.status }
      )
    }

    return new NextResponse(null, { status: 204 })
  } catch (error) {
    console.error('[/api/experiments/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
