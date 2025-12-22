import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { id } = params
    const { searchParams } = new URL(request.url)

    const page = searchParams.get('page') || '1'
    const pageSize = searchParams.get('page_size') || '20'
    const datasetType = searchParams.get('dataset_type')

    let queryString = `page=${page}&page_size=${pageSize}`
    if (datasetType) {
      queryString += `&dataset_type=${datasetType}`
    }

    console.log('[/api/evaluations/[id]/images-with-predictions] GET request - runId:', id, 'query:', queryString)

    // Forward to backend API
    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/evaluation/runs/${id}/images-with-predictions?${queryString}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Unknown error' }))
      console.error('[/api/evaluations/[id]/images-with-predictions] Backend error:', errorData)
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch images with predictions' },
        { status: backendResponse.status }
      )
    }

    const result = await backendResponse.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error('[/api/evaluations/[id]/images-with-predictions] Error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}
