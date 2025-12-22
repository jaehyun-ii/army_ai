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
    const datasetId = params.id
    const backendUrl = `${BACKEND_API_URL}/api/v1/attack-datasets/${datasetId}/download`

    console.log('[/api/attack-datasets/[id]/download] GET request - datasetId:', datasetId)
    console.log('[/api/attack-datasets/[id]/download] Backend URL:', backendUrl)

    // Forward to backend API
    const backendResponse = await fetch(backendUrl, {
      method: 'GET',
    })

    if (!backendResponse.ok) {
      console.error('[/api/attack-datasets/[id]/download] Backend error:', backendResponse.status)
      return NextResponse.json(
        { error: 'Attack dataset not found' },
        { status: backendResponse.status }
      )
    }

    // Get the content type from backend
    const contentType = backendResponse.headers.get('content-type') || 'application/zip'

    // Get the file as buffer
    const fileBuffer = await backendResponse.arrayBuffer()

    // Return the file with appropriate headers for download
    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="attack_dataset_${datasetId}.zip"`,
      },
    })
  } catch (error) {
    console.error('[/api/attack-datasets/[id]/download] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
