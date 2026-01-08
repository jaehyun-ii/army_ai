import { NextRequest, NextResponse } from "next/server"

// Disable caching for this route
export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function POST(request: NextRequest) {
  try {
    const backendUrl = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
    const url = `${backendUrl}/api/v1/carla/vehicles/upload`

    console.log(`[CARLA API] Forwarding vehicles upload to: ${url}`)

    // Get FormData from request
    const formData = await request.formData()

    // Forward to backend
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    })

    console.log(`[CARLA API] Upload response status: ${response.status}`)

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error('[CARLA API] Error uploading vehicles:', error)
    return NextResponse.json(
      { state: 500, message: 'Failed to upload vehicles', detail: String(error) },
      { status: 500 }
    )
  }
}
