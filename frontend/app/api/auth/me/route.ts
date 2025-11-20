import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function GET(request: NextRequest) {
  try {
    // Get token from Authorization header or cookie
    const authHeader = request.headers.get('authorization')
    let token = authHeader?.startsWith('Bearer ') ? authHeader.substring(7) : null

    if (!token) {
      const cookieToken = request.cookies.get('token')
      token = cookieToken?.value || null
    }

    if (!token) {
      return NextResponse.json({ error: '인증이 필요합니다.' }, { status: 401 })
    }

    // Decode JWT to get user info (basic validation)
    try {
      const payloadBase64 = token.split('.')[1]
      if (!payloadBase64) {
        return NextResponse.json({ error: '토큰이 유효하지 않습니다.' }, { status: 401 })
      }

      const payloadJson = Buffer.from(payloadBase64, 'base64').toString()
      const payload = JSON.parse(payloadJson)

      // Check if token is expired
      if (payload.exp && payload.exp * 1000 < Date.now()) {
        return NextResponse.json({ error: '세션이 만료되었습니다.' }, { status: 401 })
      }

      // Return user info from JWT payload
      const user = {
        id: payload.sub || payload.user_id || '',
        username: payload.username || '',
        email: payload.email || '',
        name: payload.name || payload.username || '',
        rank: payload.rank || null,
        unit: payload.unit || null,
        role: payload.role || 'user',
        lastLoginAt: payload.iat ? new Date(payload.iat * 1000).toISOString() : new Date().toISOString()
      }

      return NextResponse.json(user)
    } catch (decodeError) {
      console.error('Token decode error:', decodeError)
      return NextResponse.json({ error: '토큰이 유효하지 않습니다.' }, { status: 401 })
    }
  } catch (error) {
    console.error('Auth me error:', error)
    return NextResponse.json({ error: '사용자 정보를 가져오는 중 오류가 발생했습니다.' }, { status: 500 })
  }
}
