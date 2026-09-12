import { create } from "zustand";
export type View = "replay" | "analysis" | "evaluation" | "evidence";
type ReplayState = {
  playing: boolean;
  position: number;
  speed: number;
  view: View;
  selectedSignals: string[];
  guideStep: number | null;
  evidenceOpen: boolean;
  play: () => void;
  pause: () => void;
  restart: () => void;
  scrub: (value: number) => void;
  setSpeed: (speed: number) => void;
  setView: (view: View) => void;
  toggleSignal: (id: string) => void;
  setSignals: (ids: string[]) => void;
  setGuideStep: (step: number | null) => void;
  setEvidenceOpen: (open: boolean) => void;
};
export const useReplayStore = create<ReplayState>((set) => ({
  playing: false,
  position: 20,
  speed: 1,
  view: "replay",
  selectedSignals: [],
  guideStep: null,
  evidenceOpen: false,
  play: () =>
    set((s) => ({
      playing: true,
      position: s.position >= 20 ? 0 : s.position,
    })),
  pause: () => set({ playing: false }),
  restart: () => set({ position: 0, playing: false }),
  scrub: (value) =>
    set({ position: Math.max(0, Math.min(20, value)), playing: false }),
  setSpeed: (speed) =>
    set({ speed: [0.5, 1, 2, 4].includes(speed) ? speed : 1 }),
  setView: (view) => set({ view }),
  toggleSignal: (id) =>
    set((s) => ({
      selectedSignals: s.selectedSignals.includes(id)
        ? s.selectedSignals.length > 1
          ? s.selectedSignals.filter((x) => x !== id)
          : s.selectedSignals
        : [...s.selectedSignals, id],
    })),
  setSignals: (selectedSignals) => set({ selectedSignals }),
  setGuideStep: (guideStep) => set({ guideStep }),
  setEvidenceOpen: (evidenceOpen) => set({ evidenceOpen }),
}));
