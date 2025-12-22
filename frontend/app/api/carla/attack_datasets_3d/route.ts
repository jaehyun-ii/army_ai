import { NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  try {
    const backendUrl = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
    const searchParams = request.nextUrl.searchParams
    const queryString = searchParams.toString()

    const response = await fetch(`${backendUrl}/api/v1/carla/attack_datasets_3d${queryString ? `?${queryString}` : ''}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error('Error fetching 3D attack datasets:', error)
    return NextResponse.json(
      { message: 'Failed to fetch 3D attack datasets', datasets: [] },
      { status: 500 }
    )
  }
}
