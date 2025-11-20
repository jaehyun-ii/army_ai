import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    console.log('[/api/adversarial-patch/batch] POST request - forwarding to backend')

    const formData = await request.formData()

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/adversarial-patch/batch`,
      {
        method: 'POST',
        body: formData,
      }
    )

    if (!backendResponse.ok) {
      // If endpoint doesn't exist, return informative error
      if (backendResponse.status === 404) {
        return NextResponse.json({
          error: 'Batch patch generation endpoint not implemented in backend.',
          note: 'Backend should implement /api/v1/adversarial-patch/batch'
        }, { status: 501 })
      }

      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/adversarial-patch/batch] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to generate batch patches' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error('[/api/adversarial-patch/batch] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const jobId = searchParams.get('jobId')

    console.log('[/api/adversarial-patch/batch] GET request - jobId:', jobId)

    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      )
    }

    // Forward to backend API for batch job status
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/adversarial-patch/batch/${jobId}/status`,
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
          error: 'Batch job status endpoint not implemented in backend.',
          note: 'Backend should implement /api/v1/adversarial-patch/batch/{jobId}/status'
        }, { status: 501 })
      }

      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/adversarial-patch/batch] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch batch job status' },
        { status: backendResponse.status }
      )
    }

    const status = await backendResponse.json()
    return NextResponse.json(status)
  } catch (error) {
    console.error('[/api/adversarial-patch/batch] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}