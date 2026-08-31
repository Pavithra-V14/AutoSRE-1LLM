"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-neutral-50 px-6 text-center">
      <div className="max-w-md">
        <h2 className="text-lg font-semibold text-neutral-900">
          Couldn&apos;t load data from Supabase
        </h2>
        <p className="mt-2 text-sm text-neutral-500">
          {error.message.includes("abort") || error.message.includes("timeout")
            ? "The request to Supabase timed out. This usually means a network issue (proxy, VPN, or firewall) between this machine and Supabase -- check dashboard/.env.local has the right project URL and anon key, and that Supabase is reachable."
            : error.message}
        </p>
      </div>
      <button
        onClick={reset}
        className="rounded-md bg-neutral-900 px-4 py-2 text-sm text-white hover:bg-neutral-700"
      >
        Try again
      </button>
    </div>
  );
}
