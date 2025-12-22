import { NextRequest, NextResponse } from 'next/server'
import { cookies } from 'next/headers'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const queryString = searchParams.toString()
    const url = `${BACKEND_API_URL}/api/v1/system-logs/statistics${queryString ? `?${queryString}` : ''}`
    
    console.log(`[/api/system-logs/statistics] Proxying GET request to ${url}`)

    const cookieStore = cookies()
    const token = cookieStore.get('token')?.value || request.headers.get('authorization')?.replace('Bearer ', '')

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    }

    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    const response = await fetch(url, {
      method: 'GET',
      headers: headers,
      cache: 'no-store'
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`[/api/system-logs/statistics] Backend error ${response.status}: ${errorText}`)
      return NextResponse.json(
        { error: `Backend error: ${response.status}`, details: errorText },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)

  } catch (error) {
    console.error('[/api/system-logs/statistics] Internal Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
