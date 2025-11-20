import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    console.log('[/api/models/upload] POST request')

    // Get FormData from request
    const formData = await request.formData()

    // Forward FormData to backend
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/models/upload`,
      {
        method: 'POST',
        body: formData, // FormData is sent directly
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/models/upload] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to upload model' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    console.log('[/api/models/upload] Model uploaded successfully')

    return NextResponse.json(data, { status: 201 })
  } catch (error) {
    console.error('[/api/models/upload] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
