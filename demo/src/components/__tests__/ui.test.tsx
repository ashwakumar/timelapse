import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { SignalSeries } from "@/lib/contracts";
import { ForecastResponseSchema } from "@/lib/contracts";
import {
  FixtureInferenceProvider,
  FixtureTelemetryProvider,
  EvaluationProvider,
} from "@/lib/providers";
import { createTelemetryProvider } from "@/lib/configuration";
import { useReplayStore } from "@/lib/replay-store";
import {
  ForecastHorizon,
  horizonLabels,
  modelStatus,
  EvidenceDrawer,
} from "../forecast";
import { BaselineComparison } from "../evaluation";
import { TelemetryChart, TelemetryWorkspace } from "../telemetry";
import evaluation from "../../../public/evaluation/results.v1.json";

beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.spyOn(window, "matchMedia").mockImplementation((query) => ({
    matches: true,
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  }));
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  useReplayStore.setState({
    position: 20,
    playing: false,
    evidenceOpen: false,
    selectedSignals: [],
  });
});
const windowFor = (id = "fixture-no-event") =>
  new FixtureTelemetryProvider().loadCase(id);
const forecastFor = async () =>
  new FixtureInferenceProvider().forecast({
    requestId: "ui-forecast",
    window: await windowFor(),
  });

describe("truthful and readable output", () => {
  it("uses all five exact forecast categories and never implies an event after a no-event horizon", async () => {
    expect(Object.values(horizonLabels)).toEqual([
      "Hypotension within 3 minutes",
      "Hypotension within 5 minutes",
      "Hypotension within 10 minutes",
      "Hypotension within 15 minutes",
      "No hypotension within 15 minutes",
    ]);
    render(<ForecastHorizon forecast={await forecastFor()} previous={null} />);
    expect(screen.getByRole("heading", { level: 3 })).toHaveTextContent(
      "None within 15 minutes",
    );
    expect(screen.getByRole("status")).toHaveTextContent(
      "No hypotension within 15 minutes",
    );
    expect(screen.queryByText(/Beyond/)).not.toBeInTheDocument();
  });
  it("distinguishes fixture, baseline, generic pretrained, task-trained and unavailable states", async () => {
    const forecast = await forecastFor();
    expect(modelStatus(forecast)).toBe("Fixture Replay");
    expect(modelStatus({ ...forecast, provider: "baseline" })).toBe("Baseline");
    expect(modelStatus({ ...forecast, provider: "opentslm_pretrained" })).toBe(
      "OpenTSLM Pretrained",
    );
    expect(
      modelStatus({
        ...forecast,
        provider: "opentslm_task_trained",
        checkpointId: "adapter-v1",
      }),
    ).toBe("Task-Trained");
    expect(
      modelStatus({ ...forecast, provider: "opentslm_task_trained" }),
    ).toBe("Model Unavailable");
    expect(modelStatus(null, false, "backend")).toBe("Model Unavailable");
    expect(modelStatus(forecast, true)).toBe("Model Unavailable");
  });
  it("rejects a missing reasoning section, invalid horizon and unsafe treatment language independently", async () => {
    const forecast = await forecastFor();
    for (const section of ["interpret", "anticipate", "act"] as const)
      expect(
        ForecastResponseSchema.safeParse({ ...forecast, [section]: undefined })
          .success,
      ).toBe(false);
    expect(
      ForecastResponseSchema.safeParse({ ...forecast, horizon: "within_20" })
        .success,
    ).toBe(false);
    expect(
      ForecastResponseSchema.safeParse({
        ...forecast,
        act: "Give 100 mcg phenylephrine.",
      }).success,
    ).toBe(false);
    expect(
      ForecastResponseSchema.safeParse({
        ...forecast,
        interpret: "Administer a bolus.",
      }).success,
    ).toBe(false);
  });
  it("shows physical units and observed precision for normalized exports, preserving gaps", async () => {
    const base = (await windowFor()).signals[6];
    const signal: SignalSeries = {
      ...base,
      unit: "z-score",
      values: Array(10).fill(null),
      normalizedValues: [0, 0, null, 0, 0, 0, 0, 0, 0, 0.425],
      observed: [true, true, false, true, true, true, true, true, true, true],
      normalization: {
        mean: 30,
        std: 10,
        sourceUnit: "z-score",
        physicalUnit: "mmHg",
      },
    };
    const { container } = render(
      <TelemetryChart
        signal={signal}
        cutoff={480}
        visibleCount={10}
        crosshair={null}
        setCrosshair={() => {}}
      />,
    );
    expect(screen.getByLabelText("EtCO2: 34.25 mmHg")).toBeVisible();
    expect(screen.getByRole("img")).toHaveAttribute(
      "aria-label",
      expect.stringContaining("missing"),
    );
    expect(container.innerHTML).not.toMatch(/NaN|Infinity/);
  });
  it("does not display the first sample before its source time at +2 seconds", async () => {
    const window = await windowFor();
    render(
      <TelemetryChart
        signal={window.signals[0]}
        cutoff={480}
        visibleCount={0}
        crosshair={null}
        setCrosshair={() => {}}
      />,
    );
    expect(screen.getByLabelText("MAP: — mmHg")).toBeVisible();
    expect(screen.getByRole("img")).toHaveAttribute(
      "aria-label",
      expect.stringContaining("0 of ten"),
    );
  });
  it("calculates displayed coverage from the actual parameter count", async () => {
    const source = await windowFor();
    const window = { ...source, signals: source.signals.slice(0, 2) };
    act(() =>
      useReplayStore.setState({
        selectedSignals: window.signals.map((s) => s.channel),
      }),
    );
    render(
      <TelemetryWorkspace
        window={window}
        analyzing={false}
        onAnalyze={() => {}}
        onPrevious={() => {}}
        onNext={() => {}}
      />,
    );
    expect(screen.getByText(/0 missing of 20/)).toHaveTextContent(
      "100% observed",
    );
  });
  it("reports insufficient observations rather than a fabricated zero delta", async () => {
    const source = await windowFor();
    const forecast = await forecastFor();
    const window = {
      ...source,
      signals: [
        {
          ...source.signals[0],
          values: Array(10).fill(null),
          observed: Array(10).fill(false),
        },
      ],
    };
    act(() => useReplayStore.setState({ evidenceOpen: true }));
    render(<EvidenceDrawer window={window} forecast={forecast} />);
    expect(screen.getByText("Insufficient observations")).toBeVisible();
  });
  it("labels fixture comparison honestly and uses each models supplied lead time", async () => {
    const outcome = await new EvaluationProvider({
      bundled: evaluation,
    }).getOutcome("fixture-early-warning");
    const { rerender } = render(<BaselineComparison outcome={outcome} />);
    expect(screen.getByText("Fixture comparison")).toBeVisible();
    expect(screen.getAllByText("3m 52s")).toHaveLength(2);
    rerender(
      <BaselineComparison
        outcome={{
          ...outcome,
          source: "vitaldb_held_out_export",
          datasetSplit: "held_out_test",
        }}
      />,
    );
    expect(screen.getByText("Held-out comparison")).toBeVisible();
    expect(screen.queryByText("Fixture")).not.toBeInTheDocument();
  });
  it("loads a configured input-only export through the same telemetry interface", async () => {
    const source = await windowFor();
    const window = {
      ...source,
      source: "vitaldb_held_out_export",
      datasetSplit: "held_out_test",
      fixtureDisclaimer: undefined,
    };
    const metadata = {
      sampleId: window.sampleId,
      caseId: window.caseId,
      title: "Held-out case",
      description: "Observed input",
      source: window.source,
      datasetSplit: window.datasetSplit,
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            schemaVersion: "telemetry-export.v1",
            source: window.source,
            cases: [{ metadata, window }],
          }),
        ),
      );
    vi.stubGlobal("fetch", fetchMock);
    const provider = createTelemetryProvider("/inputs/held-out.v1.json");
    expect(await provider.listCases()).toHaveLength(1);
    expect((await provider.loadCase(window.sampleId)).datasetSplit).toBe(
      "held_out_test",
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
