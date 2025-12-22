import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0
export const maxDuration = 3600 // 1 hour for long patch generation

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    console.log('[/api/patches/generate] POST request - body:', body)

    // Forward to backend API with NO timeout
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/patches/generate/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      // @ts-ignore - undici-specific options for UNLIMITED timeout
      headersTimeout: 0, // 0 = unlimited
      bodyTimeout: 0, // 0 = unlimited
      keepAlive: true,
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/patches/generate] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to generate patch' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    console.log('[/api/patches/generate] Patch generation started:', result)

    return NextResponse.json(result)
  } catch (error) {
    console.error('[/api/patches/generate] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
