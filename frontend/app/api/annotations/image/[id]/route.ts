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
    const { searchParams } = new URL(request.url)

    const annotationType = searchParams.get('annotation_type')
    const minConfidence = searchParams.get('min_confidence')

    let queryString = ''
    const queryParams = new URLSearchParams()

    if (annotationType) queryParams.append('annotation_type', annotationType)
    if (minConfidence) queryParams.append('min_confidence', minConfidence)

    if (queryParams.toString()) {
      queryString = `?${queryParams.toString()}`
    }

    console.log('[/api/annotations/image/[id]] GET request - imageId:', id, 'query:', queryString)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/annotations/image/${id}${queryString}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/annotations/image/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch image annotations' },
        { status: backendResponse.status }
      )
    }

    const annotations = await backendResponse.json()
    return NextResponse.json(annotations)
  } catch (error) {
    console.error('[/api/annotations/image/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
