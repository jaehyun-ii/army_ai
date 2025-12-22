import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const backendUrl = `${BACKEND_API_URL}/api/v1/manual/download`

    console.log('[/api/manual/download] GET request')
    console.log('[/api/manual/download] Backend URL:', backendUrl)

    // Forward to backend API
    const backendResponse = await fetch(backendUrl, {
      method: 'GET',
    })

    if (!backendResponse.ok) {
      console.error('[/api/manual/download] Backend error:', backendResponse.status)
      return NextResponse.json(
        { error: 'Manual file not found' },
        { status: backendResponse.status }
      )
    }

    // Get the content type from backend
    const contentType = backendResponse.headers.get('content-type') || 'application/pdf'
    const contentDisposition = backendResponse.headers.get('content-disposition')

    // Get the file as buffer
    const fileBuffer = await backendResponse.arrayBuffer()

    // Return the file with appropriate headers for download
    const headers: HeadersInit = {
      'Content-Type': contentType,
    }

    // Preserve content-disposition from backend (contains Korean filename)
    if (contentDisposition) {
      headers['Content-Disposition'] = contentDisposition
    } else {
      headers['Content-Disposition'] = 'attachment; filename="AI_모델_신뢰성_검증_시스템_운용자_메뉴얼.pdf"'
    }

    return new NextResponse(fileBuffer, {
      status: 200,
      headers,
    })
  } catch (error) {
    console.error('[/api/manual/download] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
