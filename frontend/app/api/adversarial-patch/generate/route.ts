import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    console.log('[/api/adversarial-patch/generate] POST request - forwarding to backend')

    const formData = await request.formData()

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/adversarial-patch/generate`,
      {
        method: 'POST',
        body: formData,
      }
    )

    if (!backendResponse.ok) {
      // If endpoint doesn't exist, return informative error
      if (backendResponse.status === 404) {
        return NextResponse.json({
          error: 'Adversarial patch generation endpoint not implemented in backend.',
          note: 'Use /api/v1/adversarial-patch/patches endpoint to create patches'
        }, { status: 501 })
      }

      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/adversarial-patch/generate] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to generate patch' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error('[/api/adversarial-patch/generate] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    console.log('[/api/adversarial-patch/generate] GET request - forwarding to backend')

    // Forward to backend API for attack methods/config info
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/adversarial-patch/methods`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      // If endpoint doesn't exist, return informative error
      if (backendResponse.status === 404) {
        return NextResponse.json({
          error: 'Attack methods endpoint not implemented in backend.',
          note: 'Backend should implement /api/v1/adversarial-patch/methods'
        }, { status: 501 })
      }

      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/adversarial-patch/generate] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch attack methods' },
        { status: backendResponse.status }
      )
    }

    const methods = await backendResponse.json()
    return NextResponse.json(methods)
  } catch (error) {
    console.error('[/api/adversarial-patch/generate] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}