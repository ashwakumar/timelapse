"use client";
import { useEffect, useSyncExternalStore } from "react";
import { useReplayStore } from "./replay-store";
const query = "(prefers-reduced-motion: reduce)";
const subscribe = (callback: () => void) => {
  const media = window.matchMedia(query);
  media.addEventListener("change", callback);
  return () => media.removeEventListener("change", callback);
};
export const motionDuration = (reduced: boolean, duration: number) =>
  reduced ? 0 : duration;
export function useReducedMotion() {
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => true,
  );
}
export function useReplayClock() {
  const playing = useReplayStore((s) => s.playing);
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    let last = performance.now();
    let accumulated = 0;
    const advance = (now: number) => {
      accumulated +=
        Math.min(now - last, 250) * useReplayStore.getState().speed;
      last = now;
      if (accumulated >= 100) {
        const position = Math.min(
          20,
          useReplayStore.getState().position + accumulated / 1000,
        );
        useReplayStore.setState({ position, playing: position < 20 });
        accumulated = 0;
        if (position >= 20) return;
      }
      frame = requestAnimationFrame(advance);
    };
    frame = requestAnimationFrame(advance);
    return () => cancelAnimationFrame(frame);
  }, [playing]);
}
