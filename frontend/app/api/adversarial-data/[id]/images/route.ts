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
    const dataId = params.id
    console.log('[/api/adversarial-data/[id]/images] GET request - id:', dataId)

    // This may map to attack datasets endpoint
    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/adversarial-patch/attack-datasets/${dataId}/images`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      // If endpoint doesn't exist, return informative error
      if (backendResponse.status === 404) {
        return NextResponse.json({
          error: 'Adversarial data images endpoint not implemented in backend.',
          note: 'Add endpoint or use /api/v1/datasets-2d/[id]/images'
        }, { status: 501 })
      }

      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/adversarial-data/[id]/images] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch images' },
        { status: backendResponse.status }
      )
    }

    const images = await backendResponse.json()
    return NextResponse.json(images)
  } catch (error) {
    console.error('[/api/adversarial-data/[id]/images] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
