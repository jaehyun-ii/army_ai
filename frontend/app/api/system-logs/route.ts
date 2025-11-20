import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    console.log('[/api/system-logs] GET request - forwarding to backend')

    // Backend may have audit logs endpoint
    return NextResponse.json({
      error: 'System logs endpoint not yet implemented in backend.',
      note: 'Consider using audit_logs table or implementing /api/v1/logs endpoint'
    }, { status: 501 })
  } catch (error) {
    console.error('[/api/system-logs] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
