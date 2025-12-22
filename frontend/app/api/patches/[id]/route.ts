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
    const patchId = params.id
    console.log('[/api/patches/[id]] GET request - patchId:', patchId)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/patches/${patchId}/`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/patches/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch patch' },
        { status: backendResponse.status }
      )
    }

    const patch = await backendResponse.json()
    return NextResponse.json(patch)
  } catch (error) {
    console.error('[/api/patches/[id]] Error:', error)
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
    const patchId = params.id
    console.log('[/api/patches/[id]] DELETE request - patchId:', patchId)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/patches/${patchId}/`,
      {
        method: 'DELETE',
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/patches/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to delete patch' },
        { status: backendResponse.status }
      )
    }

    return new NextResponse(null, { status: 204 })
  } catch (error) {
    console.error('[/api/patches/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
