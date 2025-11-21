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

    console.log('[/api/attack-datasets/[id]/download] GET request - attackDatasetId:', id)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/attack-datasets/${id}/download`,
      {
        method: 'GET',
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/attack-datasets/[id]/download] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to download attack dataset' },
        { status: backendResponse.status }
      )
    }

    // Stream the file response
    const blob = await backendResponse.blob()
    const contentDisposition = backendResponse.headers.get('content-disposition') || `attachment; filename="attack_dataset_${id}.zip"`

    return new NextResponse(blob, {
      status: 200,
      headers: {
        'Content-Type': backendResponse.headers.get('content-type') || 'application/zip',
        'Content-Disposition': contentDisposition,
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
