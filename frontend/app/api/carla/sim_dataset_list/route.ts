import { NextResponse } from "next/server"

export async function GET() {
  try {
    const backendUrl = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
    const response = await fetch(`${backendUrl}/api/v1/carla/sim_dataset_list`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error('Error fetching dataset list:', error)
    return NextResponse.json(
      { state: 500, message: 'Failed to fetch dataset list', result: [] },
      { status: 500 }
    )
  }
}
