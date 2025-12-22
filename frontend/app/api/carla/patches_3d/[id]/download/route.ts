import { NextRequest, NextResponse } from "next/server"

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const backendUrl = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

    const response = await fetch(`${backendUrl}/api/v1/carla/patches_3d/${params.id}/download`, {
      method: 'GET',
    })

    if (!response.ok) {
      return NextResponse.json(
        { error: `Failed to download patch: ${response.statusText}` },
        { status: response.status }
      )
    }

    // Get filename from Content-Disposition header
    const contentDisposition = response.headers.get('Content-Disposition')
    const filename = contentDisposition
      ? contentDisposition.split('filename=')[1]?.replace(/"/g, '')
      : `patch_${params.id}.png`

    // Get content type
    const contentType = response.headers.get('Content-Type') || 'image/png'

    // Stream the file
    const blob = await response.blob()

    return new NextResponse(blob, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="${filename}"`,
      },
    })
  } catch (error) {
    console.error('Error downloading 3D patch:', error)
    return NextResponse.json(
      { error: 'Failed to download patch' },
      { status: 500 }
    )
  }
}
