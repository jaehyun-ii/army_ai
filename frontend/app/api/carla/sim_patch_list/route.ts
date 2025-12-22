import { NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const objectName = searchParams.get('object_name')
    const attackMethod = searchParams.get('attack_method')

    if (!objectName || !attackMethod) {
      return NextResponse.json(
        { state: 400, message: 'object_name and attack_method are required', result: [] },
        { status: 400 }
      )
    }

    const backendUrl = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'
    const response = await fetch(
      `${backendUrl}/api/v1/carla/sim_patch_list?object_name=${encodeURIComponent(objectName)}&attack_method=${encodeURIComponent(attackMethod)}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      }
    )

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error('Error fetching patch list:', error)
    return NextResponse.json(
      { state: 500, message: 'Failed to fetch patch list', result: [] },
      { status: 500 }
    )
  }
}
