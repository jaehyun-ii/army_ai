import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const { id } = params
  const datasetType = request.nextUrl.searchParams.get('dataset_type')

  try {
    const url = new URL(`${BACKEND_API_URL}/api/v1/evaluation/runs/${id}/dataset-results`)
    if (datasetType) {
      url.searchParams.set('dataset_type', datasetType)
    }

    const backendResponse = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/evaluations/[id]/dataset-results] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch dataset results' },
        { status: backendResponse.status }
      )
    }

    const results = await backendResponse.json()
    return NextResponse.json(results)
  } catch (error) {
    console.error('[/api/evaluations/[id]/dataset-results] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
