import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

// POST - Restore from backup
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    // Await params if it's a Promise (Next.js 15+)
    const resolvedParams = await Promise.resolve(params)
    const backupId = resolvedParams.id
    console.log(`[/api/admin/backups/${backupId}/restore] POST request`)

    // Get token from Authorization header or cookie
    const authHeader = request.headers.get('authorization')
    let token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null

    if (!token) {
      const cookieToken = request.cookies.get('token')
      token = cookieToken?.value || null
    }

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    }

    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/admin/backups/${backupId}/restore`,
      {
        method: 'POST',
        headers,
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error(`[/api/admin/backups/${backupId}/restore] Backend error:`, errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to restore backup' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    console.log(`[/api/admin/backups/${backupId}/restore] Restore started successfully`)
    return NextResponse.json(data)
  } catch (error) {
    console.error(`[/api/admin/backups/restore] Error:`, error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
