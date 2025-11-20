import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => null)
    const { username, password } = (body ?? {}) as { username?: unknown; password?: unknown }

    if (typeof username !== 'string' || typeof password !== 'string') {
      return NextResponse.json({ success: false, error: '필수 정보가 누락되었습니다.' }, { status: 400 })
    }

    if (!username.trim() || !password.trim()) {
      return NextResponse.json({ success: false, error: '사용자 정보를 확인할 수 없습니다.' }, { status: 400 })
    }

    // Call FastAPI backend login endpoint
    const backendResponse = await fetch(`${BACKEND_API_URL}/api/v1/auth/login-json`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({}))
      return NextResponse.json({
        success: false,
        error: errorData.detail || '아이디 또는 비밀번호가 올바르지 않습니다.'
      }, { status: backendResponse.status })
    }

    const data = await backendResponse.json()

    // Backend returns: { access_token: string, token_type: string }
    const token = data.access_token

    if (!token) {
      return NextResponse.json({
        success: false,
        error: '인증 토큰을 받지 못했습니다.'
      }, { status: 500 })
    }

    // Decode JWT to get user info (without verification, just for display)
    // Token format: header.payload.signature
    const payloadBase64 = token.split('.')[1]
    const payloadJson = Buffer.from(payloadBase64, 'base64').toString()
    const payload = JSON.parse(payloadJson)

    // Create user object from JWT payload
    const user = {
      id: payload.sub || payload.user_id || '',
      username: payload.username || username,
      email: payload.email || '',
      name: payload.name || username,
      rank: payload.rank || null,
      unit: payload.unit || null,
      role: payload.role || 'user',
      lastLoginAt: new Date().toISOString()
    }

    const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000)

    const response = NextResponse.json({
      success: true,
      token,
      user
    })

    response.cookies.set('token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      expires: expiresAt
    })

    return response
  } catch (error) {
    console.error('Login error:', error)
    return NextResponse.json({
      success: false,
      error: '로그인 처리 중 오류가 발생했습니다.'
    }, { status: 500 })
  }
}
