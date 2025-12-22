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
    console.log('[/api/admin/backups/export] Exporting backup:', id)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/admin/backups/${id}/export`,
      {
        method: 'GET',
        cache: 'no-store',  // Prevent Next.js from caching large files
      }
    )

    if (!backendResponse.ok) {
      console.error('[/api/admin/backups/export] Backend error:', backendResponse.status)
      const error = await backendResponse.json().catch(() => ({ detail: 'Export failed' }))
      return NextResponse.json(error, { status: backendResponse.status })
    }

    // Get filename from Content-Disposition header or use default
    const contentDisposition = backendResponse.headers.get('content-disposition')
    let filename = `backup_export_${id}.zip`

    if (contentDisposition) {
      const match = contentDisposition.match(/filename="?(.+?)"?$/)
      if (match) {
        filename = match[1]
      }
    }

    console.log('[/api/admin/backups/export] Streaming file:', filename)

    // Stream the file response directly
    return new NextResponse(backendResponse.body, {
      status: 200,
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': `attachment; filename="${filename}"`,
        'Cache-Control': 'no-store, no-cache, must-revalidate, private',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
    })
  } catch (error) {
    console.error('[/api/admin/backups/export] Error:', error)
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    )
  }
}
