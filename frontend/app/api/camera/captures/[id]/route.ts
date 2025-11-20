import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    console.log(`[/api/camera/captures/${id}] GET request`)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/camera/captures/${id}`,
      {
        method: 'GET',
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error(`[/api/camera/captures/${id}] Backend error:`, errorData)
      return NextResponse.json(
        { error: errorData.detail || errorData.error || 'Failed to get capture details' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    console.log(`[/api/camera/captures/${id}] Capture details retrieved successfully`)

    return NextResponse.json(data)
  } catch (error) {
    console.error(`[/api/camera/captures/${id}] Error:`, error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    console.log(`[/api/camera/captures/${id}] DELETE request`)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/camera/captures/${id}`,
      {
        method: 'DELETE',
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error(`[/api/camera/captures/${id}] Backend error:`, errorData)
      return NextResponse.json(
        { error: errorData.detail || errorData.error || 'Failed to delete capture' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    console.log(`[/api/camera/captures/${id}] Capture deleted successfully`)

    return NextResponse.json(data)
  } catch (error) {
    console.error(`[/api/camera/captures/${id}] Error:`, error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
