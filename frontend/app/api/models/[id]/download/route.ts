import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    console.log('[/api/models/download] Downloading model:', id)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/models/${id}/download`,
      {
        method: 'GET',
      }
    )

    if (!backendResponse.ok) {
      console.error('[/api/models/download] Backend error:', backendResponse.status)
      const error = await backendResponse.json().catch(() => ({ detail: 'Download failed' }))
      return NextResponse.json(error, { status: backendResponse.status })
    }

    // Get filename from Content-Disposition header or use default
    const contentDisposition = backendResponse.headers.get('content-disposition')
    let filename = `model_${id}.zip`

    if (contentDisposition) {
      const match = contentDisposition.match(/filename="?(.+?)"?$/)
      if (match) {
        filename = match[1]
      }
    }

    // Stream the file response
    return new NextResponse(backendResponse.body, {
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': `attachment; filename="${filename}"`,
        'Cache-Control': 'no-cache',
      },
    })
  } catch (error) {
    console.error('[/api/models/download] Error:', error)
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    )
  }
}
