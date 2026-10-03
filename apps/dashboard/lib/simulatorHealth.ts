const SIMULATOR_URL = process.env.NEXT_PUBLIC_SIMULATOR_URL || "http://localhost:8100";
const TIMEOUT_MS = 5_000;

export async function fetchSimulatorHealth(): Promise<{ status: string } | null> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${SIMULATOR_URL}/health`, { signal: controller.signal });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  } finally {
    clearTimeout(timeoutId);
  }
}
