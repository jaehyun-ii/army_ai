import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

// GET - List all backups
export async function GET(request: NextRequest) {
  try {
    console.log('[/api/admin/backups] GET request')

    // Get token from Authorization header or cookie
    const authHeader = request.headers.get('authorization')
    let token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null

    if (!token) {
      const cookieToken = request.cookies.get('token')
      token = cookieToken?.value || null
    }

    const { searchParams } = new URL(request.url)
    const skip = searchParams.get('skip') || '0'
    const limit = searchParams.get('limit') || '100'

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    }

    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/admin/backups?skip=${skip}&limit=${limit}`,
      {
        method: 'GET',
        headers,
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/admin/backups] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to get backups' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('[/api/admin/backups] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

// POST - Create new backup
export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}))
    console.log('[/api/admin/backups] POST request - body:', body)

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
      `${BACKEND_API_URL}/api/v1/admin/backups`,
      {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/admin/backups] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to create backup' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    console.log('[/api/admin/backups] Backup created successfully')
    return NextResponse.json(data)
  } catch (error) {
    console.error('[/api/admin/backups] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
