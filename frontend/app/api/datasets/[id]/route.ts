import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    // Handle both sync and async params for Next.js compatibility
    const params = await Promise.resolve(context.params)
    const datasetId = params.id
    console.log('[/api/datasets/[id]] GET request - id:', datasetId)

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/datasets-2d/${datasetId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Dataset not found' },
        { status: backendResponse.status }
      )
    }

    const dataset = await backendResponse.json()
    return NextResponse.json(dataset)
  } catch (error) {
    console.error('[/api/datasets/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    // Handle both sync and async params for Next.js compatibility
    const params = await Promise.resolve(context.params)
    const datasetId = params.id
    const body = await request.json()
    console.log('[/api/datasets/[id]] PATCH request - id:', datasetId)

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/datasets-2d/${datasetId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to update dataset' },
        { status: backendResponse.status }
      )
    }

    const dataset = await backendResponse.json()
    return NextResponse.json(dataset)
  } catch (error) {
    console.error('[/api/datasets/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    // Handle both sync and async params for Next.js compatibility
    const params = await Promise.resolve(context.params)
    const datasetId = params.id
    console.log('[/api/datasets/[id]] DELETE request - id:', datasetId)

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/datasets-2d/${datasetId}`, {
      method: 'DELETE',
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to delete dataset' },
        { status: backendResponse.status }
      )
    }

    // 204 No Content should not have a body
    return new NextResponse(null, { status: 204 })
  } catch (error) {
    console.error('[/api/datasets/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
