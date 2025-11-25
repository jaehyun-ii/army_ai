import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

// GET: 사용자 목록 조회
export async function GET(request: NextRequest) {
  try {
    const token = request.headers.get('Authorization')

    if (!token) {
      return NextResponse.json(
        { error: 'Authorization required' },
        { status: 401 }
      )
    }

    console.log('[/api/users] GET request - fetching users list')

    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/users`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': token,
      },
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/users] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch users' },
        { status: backendResponse.status }
      )
    }

    const users = await backendResponse.json()
    console.log('[/api/users] Fetched users:', users.length)

    return NextResponse.json(users)
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
