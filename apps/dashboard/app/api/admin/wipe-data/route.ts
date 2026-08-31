import { NextRequest, NextResponse } from "next/server";

// Same pattern as app/api/approvals/[incidentId]/decide/route.ts --
// CONTROL_API_KEY (no NEXT_PUBLIC_ prefix) never reaches the browser.
// This route exists specifically so the Danger Zone button on /status
// can call something without the page itself holding a write-capable
// secret.
const CONTROL_API_URL = process.env.CONTROL_API_URL || process.env.NEXT_PUBLIC_CONTROL_API_URL || "http://localhost:8001";
const CONTROL_API_KEY = process.env.CONTROL_API_KEY;

export async function POST(request: NextRequest) {
  const body = await request.json();

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (CONTROL_API_KEY) {
    headers["X-API-Key"] = CONTROL_API_KEY;
  }

  try {
    const res = await fetch(`${CONTROL_API_URL}/admin/wipe-data`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30_000), // deleting 6 tables can take a moment
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
