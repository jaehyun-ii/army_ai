import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

// GET: 사용자 목록 조회
// Note: Backend doesn't have a list users endpoint yet,
// so this returns an informational message
export async function GET(request: NextRequest) {
  try {
    console.log('[/api/users] GET request - users list not implemented in backend')

    // TODO: Implement user list endpoint in backend
    return NextResponse.json({
      success: false,
      error: 'User list endpoint not implemented in backend. Please add /api/v1/users endpoint.',
      note: 'Use /api/v1/auth/register for user creation'
    }, { status: 501 })
  } catch (error) {
    console.error('[/api/users] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

// POST: 새 사용자 생성 (register)
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { username, name, email, password, rank, unit, role = 'user' } = body

    console.log('[/api/users] POST request - registering user:', username)

    // Forward to backend registration endpoint
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username,
        email,
        password,
        role: role.toLowerCase(), // Backend expects lowercase
      }),
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/users] Backend error:', errorData)
      return NextResponse.json(
        {
          success: false,
          error: errorData.detail || 'Failed to create user'
        },
        { status: backendResponse.status }
      )
    }

    const user = await backendResponse.json()
    console.log('[/api/users] User created:', user.id)

    return NextResponse.json({
      success: true,
      user,
      message: '사용자가 성공적으로 생성되었습니다.'
    })
  } catch (error) {
    console.error('[/api/users] Error:', error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error'
      },
      { status: 500 }
    )
  }
}

// PATCH: 사용자 상태 업데이트
// Note: This requires a backend endpoint
export async function PATCH(request: NextRequest) {
  try {
    console.log('[/api/users] PATCH request - user update not implemented in backend')

    return NextResponse.json({
      success: false,
      error: 'User update endpoint not implemented in backend.',
      note: 'Please add PATCH /api/v1/users/{id} endpoint'
    }, { status: 501 })
  } catch (error) {
    console.error('[/api/users] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

// DELETE: 사용자 삭제
// Note: This requires a backend endpoint
export async function DELETE(request: NextRequest) {
  try {
    console.log('[/api/users] DELETE request - user deletion not implemented in backend')

    return NextResponse.json({
      success: false,
      error: 'User deletion endpoint not implemented in backend.',
      note: 'Please add DELETE /api/v1/users/{id} endpoint'
    }, { status: 501 })
  } catch (error) {
    console.error('[/api/users] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
