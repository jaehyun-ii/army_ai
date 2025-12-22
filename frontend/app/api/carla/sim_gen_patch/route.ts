import { NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const backendUrl = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000'

    const response = await fetch(`${backendUrl}/api/v1/carla/sim_gen_patch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })

    // Stream the response back to the client
    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json(errorData, { status: response.status })
    }

    // Forward the streaming response with correct headers from backend
    return new Response(response.body, {
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'application/x-ndjson',
        'Transfer-Encoding': 'chunked',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      }
    })
  } catch (error) {
    console.error('Error generating patch:', error)
    return NextResponse.json(
      { state: 500, message: 'Failed to generate patch', result: null },
      { status: 500 }
    )
  }
}
