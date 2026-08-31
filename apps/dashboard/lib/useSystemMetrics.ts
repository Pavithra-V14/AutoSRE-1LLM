"use client";

import { useSyncExternalStore } from "react";
import { fetchSystemMetrics, type SystemMetrics } from "@/lib/controlApi";

const POLL_INTERVAL_MS = 20_000;

let cachedMetrics: SystemMetrics | null = null;
const subscribers = new Set<() => void>();
let intervalId: ReturnType<typeof setInterval> | null = null;

function notifySubscribers() {
  subscribers.forEach((callback) => callback());
}

async function poll() {
  cachedMetrics = await fetchSystemMetrics();
  notifySubscribers();
}

function subscribe(callback: () => void): () => void {
  subscribers.add(callback);

  if (subscribers.size === 1) {
    poll();
    intervalId = setInterval(poll, POLL_INTERVAL_MS);
  }

  return () => {
    subscribers.delete(callback);
    if (subscribers.size === 0 && intervalId !== null) {
      clearInterval(intervalId);
      intervalId = null;
    }
  };
}

function getSnapshot(): SystemMetrics | null {
  return cachedMetrics;
}

export function useSystemMetrics(): SystemMetrics | null {
  return useSyncExternalStore(subscribe, getSnapshot, () => null);
}
