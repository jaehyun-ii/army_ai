/**
 * ART Attacks - Generate Adversarial Attack
 *
 * Proxies request to backend to generate adversarial attack using specified method.
 */
import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0
export const maxDuration = 3600 // 1 hour for long attack generation

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_BACKEND_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const backendResponse = await fetch(
      `${BACKEND_API_URL}/api/v1/art-attacks/generate`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        // @ts-ignore - undici-specific options for UNLIMITED timeout
        headersTimeout: 0, // 0 = unlimited
        bodyTimeout: 0, // 0 = unlimited
        keepAlive: true,
      }
    );

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      console.error('Backend error:', errorText);
      return NextResponse.json(
        { error: 'Failed to generate adversarial attack' },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error generating adversarial attack:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
