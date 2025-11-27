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
    const backendUrl = `${BACKEND_API_URL}/api/v1/patches/${patchId}/download`

    console.log('[/api/patches/[id]/download] GET request - patchId:', patchId)
    console.log('[/api/patches/[id]/download] Backend URL:', backendUrl)

    // Forward to backend API
    const backendResponse = await fetch(backendUrl, {
      method: 'GET',
    })

    if (!backendResponse.ok) {
      console.error('[/api/patches/[id]/download] Backend error:', backendResponse.status)
      return NextResponse.json(
        { error: 'Patch not found' },
        { status: backendResponse.status }
      )
    }

    // Get the content type from backend
    const contentType = backendResponse.headers.get('content-type') || 'image/png'

    // Get the file as buffer
    const fileBuffer = await backendResponse.arrayBuffer()

    // Return the file with appropriate headers for download
    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="patch_${patchId}.png"`,
      },
    })
  } catch (error) {
    console.error('[/api/patches/[id]/download] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
