import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { motionDuration, useReducedMotion } from "../motion";
import { useReplayStore } from "../replay-store";

const initialState = {
  playing: false,
  position: 20,
  speed: 1,
  view: "replay" as const,
  selectedSignals: [],
  guideStep: null,
  evidenceOpen: false,
};

afterEach(() => useReplayStore.setState(initialState));

describe("replay store", () => {
  it("starts replay from the beginning after reaching the analysis cutoff", () => {
    act(() => useReplayStore.getState().play());

    expect(useReplayStore.getState()).toMatchObject({
      playing: true,
      position: 0,
    });
  });

  it("pauses, restarts, and clamps timeline scrubbing to the 20-second window", () => {
    act(() => useReplayStore.getState().scrub(8));
    expect(useReplayStore.getState()).toMatchObject({
      playing: false,
      position: 8,
    });

    act(() => useReplayStore.getState().scrub(99));
    expect(useReplayStore.getState().position).toBe(20);

    act(() => useReplayStore.getState().scrub(-10));
    expect(useReplayStore.getState().position).toBe(0);

    act(() => useReplayStore.getState().restart());
    expect(useReplayStore.getState()).toMatchObject({
      playing: false,
      position: 0,
    });
  });

  it("accepts only supported replay speeds", () => {
    act(() => useReplayStore.getState().setSpeed(4));
    expect(useReplayStore.getState().speed).toBe(4);

    act(() => useReplayStore.getState().setSpeed(3));
    expect(useReplayStore.getState().speed).toBe(1);
  });

  it("keeps at least one telemetry signal selected", () => {
    act(() => useReplayStore.getState().setSignals(["map"]));
    act(() => useReplayStore.getState().toggleSignal("map"));
    expect(useReplayStore.getState().selectedSignals).toEqual(["map"]);

    act(() => useReplayStore.getState().toggleSignal("bis"));
    act(() => useReplayStore.getState().toggleSignal("map"));
    expect(useReplayStore.getState().selectedSignals).toEqual(["bis"]);
  });

  it("supports view and evidence transitions without exposing evaluation data itself", () => {
    act(() => {
      useReplayStore.getState().setView("evaluation");
      useReplayStore.getState().setEvidenceOpen(true);
    });

    expect(useReplayStore.getState()).toMatchObject({
      view: "evaluation",
      evidenceOpen: true,
    });
  });
});

describe("reduced motion", () => {
  it("reduces animation duration to zero", () => {
    expect(motionDuration(true, 500)).toBe(0);
    expect(motionDuration(false, 500)).toBe(500);
  });

  it("reads the system reduced-motion setting", () => {
    const { result } = renderHook(() => useReducedMotion());
    expect(result.current).toBe(false);
  });
});
