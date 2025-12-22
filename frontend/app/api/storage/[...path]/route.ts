'use server'

import { NextRequest, NextResponse } from 'next/server'

// Resolve the backend base URL from env (containers set BACKEND_API_URL)
function getBackendBaseUrl() {
  return (
    process.env.BACKEND_API_URL ||
    process.env.NEXT_PUBLIC_BACKEND_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    'http://localhost:8000'
  ).replace(/\/$/, '')
}

export async function GET(
  req: NextRequest,
  { params }: { params: { path?: string[] } }
) {
  const path = (params.path || []).join('/')
  const search = req.nextUrl.search || ''
  const backendUrl = `${getBackendBaseUrl()}/api/v1/storage/${path}${search}`

  try {
    const backendRes = await fetch(backendUrl, {
      // Pass through headers that matter for auth/cache; avoid Host
      headers: {
        ...(req.headers.get('authorization')
          ? { authorization: req.headers.get('authorization')! }
          : {}),
      },
    })

    // Stream body and forward status/content-type
    const headers = new Headers()
    const contentType = backendRes.headers.get('content-type')
    if (contentType) {
      headers.set('content-type', contentType)
    }
    const contentDisposition = backendRes.headers.get('content-disposition')
    if (contentDisposition) {
      headers.set('content-disposition', contentDisposition)
    }
    // CORS headers to allow browser fetch
    headers.set('access-control-allow-origin', '*')
    headers.set('access-control-allow-methods', 'GET, OPTIONS')
    headers.set('access-control-allow-headers', '*')

    return new NextResponse(backendRes.body, {
      status: backendRes.status,
      statusText: backendRes.statusText,
      headers,
    })
  } catch (error) {
    console.error('[api/storage] Proxy error:', error)
    return NextResponse.json(
      { message: 'Failed to fetch storage file' },
      { status: 502 }
    )
  }
}
