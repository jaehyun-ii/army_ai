import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string; frame_number: string } }
) {
  try {
    const { id, frame_number } = params
    console.log(`[/api/camera/captures/${id}/image/${frame_number}] GET request`)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/camera/captures/${id}/image/${frame_number}`,
      {
        method: 'GET',
      }
    )

    // Handle 501 Not Implemented (images not stored yet)
    if (backendResponse.status === 501) {
      console.log(`[/api/camera/captures/${id}/image/${frame_number}] Image storage not implemented`)

      // Return a placeholder SVG image
      const placeholderSvg = `
        <svg width="640" height="480" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="480" fill="#1e293b"/>
          <text x="320" y="220" font-family="Arial" font-size="16" fill="#94a3b8" text-anchor="middle">
            Frame #${frame_number}
          </text>
          <text x="320" y="250" font-family="Arial" font-size="14" fill="#64748b" text-anchor="middle">
            Image storage not yet implemented
          </text>
          <text x="320" y="280" font-family="Arial" font-size="12" fill="#475569" text-anchor="middle">
            Only metadata is stored in database
          </text>
        </svg>
      `

      return new NextResponse(placeholderSvg, {
        headers: {
          'Content-Type': 'image/svg+xml',
          'Cache-Control': 'no-cache',
        },
      })
    }

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error(`[/api/camera/captures/${id}/image/${frame_number}] Backend error:`, errorData)

      // Return placeholder for any error
      const errorSvg = `
        <svg width="640" height="480" xmlns="http://www.w3.org/2000/svg">
          <rect width="640" height="480" fill="#1e293b"/>
          <text x="320" y="240" font-family="Arial" font-size="16" fill="#ef4444" text-anchor="middle">
            Error loading image
          </text>
        </svg>
      `

      return new NextResponse(errorSvg, {
        headers: {
          'Content-Type': 'image/svg+xml',
          'Cache-Control': 'no-cache',
        },
      })
    }

    // If image exists, stream it
    const contentType = backendResponse.headers.get('content-type') || 'image/jpeg'

    return new NextResponse(backendResponse.body, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    })
  } catch (error) {
    console.error(`[/api/camera/captures/${params.id}/image/${params.frame_number}] Error:`, error)

    // Return error placeholder
    const errorSvg = `
      <svg width="640" height="480" xmlns="http://www.w3.org/2000/svg">
        <rect width="640" height="480" fill="#1e293b"/>
        <text x="320" y="240" font-family="Arial" font-size="16" fill="#ef4444" text-anchor="middle">
          Error: ${error instanceof Error ? error.message : 'Unknown error'}
        </text>
      </svg>
    `

    return new NextResponse(errorSvg, {
      headers: {
        'Content-Type': 'image/svg+xml',
        'Cache-Control': 'no-cache',
      },
    })
  }
}
