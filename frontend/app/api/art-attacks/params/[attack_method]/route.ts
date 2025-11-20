/**
 * ART Attacks - Get Attack Parameters Schema
 *
 * Proxies request to backend to get parameter schema for a specific attack method.
 */
import { NextRequest, NextResponse } from 'next/server';

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000';

export async function GET(
  request: NextRequest,
  { params }: { params: { attack_method: string } }
) {
  try {
    const { attack_method } = params;

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/art-attacks/params/${attack_method}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error('Backend error:', errorText);
      return NextResponse.json(
        { error: `Failed to fetch parameters for attack method: ${attack_method}` },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching attack parameters:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
