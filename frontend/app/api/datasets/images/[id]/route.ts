import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    const params = await Promise.resolve(context.params)
    const imageId = params.id
    console.log('[/api/datasets/images/[id]] GET request - id:', imageId)

    // Note: Backend may not have this exact endpoint
    // This is a placeholder - adjust based on actual backend API
    return NextResponse.json({
      error: 'Image detail endpoint not implemented. Use /api/datasets/[id]/images instead.'
    }, { status: 501 })
  } catch (error) {
    console.error('[/api/datasets/images/[id]] Error:', error)
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
    const params = await Promise.resolve(context.params)
    const imageId = params.id
    console.log('[/api/datasets/images/[id]] DELETE request - id:', imageId)

    // Forward to backend API
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/datasets-2d/images/${imageId}`, {
      method: 'DELETE',
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/datasets/images/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to delete image' },
        { status: backendResponse.status }
      )
    }

    // 204 No Content should not have a body
    return new NextResponse(null, { status: 204 })
  } catch (error) {
    console.error('[/api/datasets/images/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
