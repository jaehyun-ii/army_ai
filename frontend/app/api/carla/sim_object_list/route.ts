import { NextResponse } from "next/server"

// Disable caching for this route
export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function GET() {
  try {
    const backendUrl = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
    const url = `${backendUrl}/api/v1/carla/sim_object_list`

    console.log(`[CARLA API] Forwarding request to: ${url}`)

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    console.log(`[CARLA API] Response status: ${response.status}`)

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error('[CARLA API] Error fetching object list:', error)
    return NextResponse.json(
      { state: 500, message: 'Failed to fetch object list', result: [] },
      { status: 500 }
    )
  }
}
