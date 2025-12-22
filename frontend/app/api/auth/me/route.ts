import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

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

    // Proxy to backend /api/v1/auth/me
    const backendUrl = `${BACKEND_API_URL}/api/v1/auth/me`
    const response = await fetch(backendUrl, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Authentication failed' }))
      return NextResponse.json(errorData, { status: response.status })
    }

    const userData = await response.json()
    return NextResponse.json(userData)
  } catch (error) {
    console.error('Auth me error:', error)
    return NextResponse.json({ error: '운용자 정보를 가져오는 중 오류가 발생했습니다.' }, { status: 500 })
  }
}
