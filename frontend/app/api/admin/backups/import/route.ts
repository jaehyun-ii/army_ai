import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    // Forward multipart form data to backend
    const backendUrl = process.env.BACKEND_URL || process.env.BACKEND_API_URL || 'http://backend:8000'
    const response = await fetch(`${backendUrl}/api/v1/admin/backups/import`, {
      method: 'POST',
      body: formData, // Forward the FormData as-is
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(
        { error: data.detail || 'Failed to import backup' },
        { status: response.status }
      )
    }

    return NextResponse.json(data, { status: 200 })
  } catch (error) {
    console.error('Import backup error:', error)
    return NextResponse.json(
      { error: 'Failed to import backup' },
      { status: 500 }
    )
  }
}
