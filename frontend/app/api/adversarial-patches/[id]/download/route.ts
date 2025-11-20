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
    const patchId = params.id
    console.log('[/api/adversarial-patches/[id]/download] GET request - id:', patchId)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/adversarial-patch/patches/${patchId}/download`,
      {
        method: 'GET',
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/adversarial-patches/[id]/download] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to download patch' },
        { status: backendResponse.status }
      )
    }

    // Forward the binary response
    const blob = await backendResponse.blob()
    const headers = new Headers()

    // Copy content-type and content-disposition from backend
    const contentType = backendResponse.headers.get('content-type')
    const contentDisposition = backendResponse.headers.get('content-disposition')

    if (contentType) headers.set('Content-Type', contentType)
    if (contentDisposition) headers.set('Content-Disposition', contentDisposition)

    return new NextResponse(blob, { headers })
  } catch (error) {
    console.error('[/api/adversarial-patches/[id]/download] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
