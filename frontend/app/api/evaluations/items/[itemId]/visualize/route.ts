import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { itemId: string } }
) {
  try {
    const { itemId } = params
    const backendUrl = `${BACKEND_API_URL}/api/v1/evaluation/items/${itemId}/visualize`

    console.log('[/api/evaluations/items/visualize] GET request - itemId:', itemId)
    console.log('[/api/evaluations/items/visualize] Backend URL:', backendUrl)

    // Forward to backend API
    const backendResponse = await fetch(backendUrl, {
      method: 'GET',
      headers: {
        // Don't forward Content-Type for images
      },
    })

    if (!backendResponse.ok) {
      console.error('[/api/evaluations/items/visualize] Backend error:', backendResponse.status, backendResponse.statusText)
      return NextResponse.json(
        { error: 'Visualization not found' },
        { status: backendResponse.status }
      )
    }

    // Get the content type from backend (should be image/jpeg)
    const contentType = backendResponse.headers.get('content-type') || 'image/jpeg'

    // Get the image as buffer
    const imageBuffer = await backendResponse.arrayBuffer()

    // Return the image with appropriate headers
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    })
  } catch (error) {
    console.error('[/api/evaluations/items/visualize] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
