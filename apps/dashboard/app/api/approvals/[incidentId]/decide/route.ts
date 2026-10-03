import { NextRequest, NextResponse } from "next/server";

// Server-side only -- CONTROL_API_KEY (no NEXT_PUBLIC_ prefix) is never
// sent to the browser. The Approve/Deny buttons call this route; this
// route attaches the real secret when calling the control API. This is
// the actual security boundary: a write-capable credential must never
// live in client-side JavaScript, no matter how "internal" the tool is.
const CONTROL_API_URL = process.env.CONTROL_API_URL || process.env.NEXT_PUBLIC_CONTROL_API_URL || "http://localhost:8101";
const CONTROL_API_KEY = process.env.CONTROL_API_KEY;

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ incidentId: string }> }
) {
  const { incidentId } = await params;
  const body = await request.json();

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (CONTROL_API_KEY) {
    headers["X-API-Key"] = CONTROL_API_KEY;
  }

  try {
    const res = await fetch(`${CONTROL_API_URL}/approvals/${incidentId}/decide`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15_000),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { detail: "Could not reach the control API. Is it running?" },
      { status: 502 }
    );
  }
}
