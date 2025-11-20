import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    console.log('[/api/yolo-metadata] GET request - forwarding to backend')

    // This endpoint may need to be implemented in backend
    return NextResponse.json({
      error: 'YOLO metadata endpoint not yet mapped to backend.',
      note: 'YOLO metadata is included in dataset upload responses'
    }, { status: 501 })
  } catch (error) {
    console.error('[/api/yolo-metadata] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
