import { NextRequest, NextResponse } from 'next/server'

/**
 * Get class-level metrics for an evaluation run
 * Proxy to backend: GET /api/v1/evaluation/runs/{run_id}/class-metrics
 */
export async function GET(
  request: NextRequest,
  { params }: { params: { runId: string } }
) {
  try {
    const { runId } = params
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'

    const response = await fetch(
      `${backendUrl}/api/v1/evaluation/runs/${runId}/class-metrics`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!response.ok) {
      const error = await response.text()
      console.error(`[/api/evaluation/runs/${runId}/class-metrics] Backend error:`, error)
      return NextResponse.json(
        { error: `Backend error: ${response.statusText}` },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error: any) {
    console.error('[/api/evaluation/runs/[runId]/class-metrics] Error:', error)
    return NextResponse.json(
      { error: error.message || 'Failed to fetch class metrics' },
      { status: 500 }
    )
  }
}
