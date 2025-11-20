import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const token = request.headers.get('Authorization')

    console.log('[/api/users/[id]] DELETE request - userId:', id)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/users/${id}`,
      {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': token }),
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/users/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to delete user' },
        { status: backendResponse.status }
      )
    }

    console.log('[/api/users/[id]] User deleted successfully:', id)

    return new NextResponse(null, { status: 204 })
  } catch (error) {
    console.error('[/api/users/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const body = await request.json()
    const token = request.headers.get('Authorization')

    console.log('[/api/users/[id]] PATCH request - userId:', id)

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/users/${id}`,
      {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': token }),
        },
        body: JSON.stringify(body),
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/users/[id]] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to update user' },
        { status: backendResponse.status }
      )
    }

    const data = await backendResponse.json()
    console.log('[/api/users/[id]] User updated successfully:', id)

    return NextResponse.json(data)
  } catch (error) {
    console.error('[/api/users/[id]] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
