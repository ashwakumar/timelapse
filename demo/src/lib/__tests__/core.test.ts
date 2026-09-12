import { afterEach, describe, expect, it, vi } from "vitest";
import { calculateInputQuality, physicalValues } from "../adapters";
import {
  ForecastResponseSchema,
  EvaluationResultsSchema,
  SignalSeriesSchema,
  InferenceRequestSchema,
  TelemetryWindowSchema,
} from "../contracts";
import {
  BaselineInferenceProvider,
  EvaluationProvider,
  FixtureInferenceProvider,
  FixtureTelemetryProvider,
  HeldOutTelemetryProvider,
  loadHeldOutTelemetryExport,
  OpenTSLMInferenceProvider,
  StaleInferenceResponseError,
  scenarioMetadata,
} from "../providers";
import evaluationResults from "../../../public/evaluation/results.v1.json";
import { POST } from "../../app/api/inference/route";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.useRealTimers();
});

describe("TimeLapse core providers", () => {
  it("exposes four deterministic, input-safe fixture scenarios", async () => {
    const telemetry = new FixtureTelemetryProvider();
    expect(scenarioMetadata).toHaveLength(4);

    for (const scenario of scenarioMetadata) {
      const window = await telemetry.loadCase(scenario.sampleId);
      expect(TelemetryWindowSchema.parse(window).signals).toHaveLength(10);
      expect(window.timestamps).toHaveLength(10);
      expect(JSON.stringify(window)).not.toContain("groundTruth");
      expect(JSON.stringify(window)).not.toContain("actualOnset");
    }
  });

  it("calculates coverage from explicit masks rather than interpolating missing samples", async () => {
    const telemetry = new FixtureTelemetryProvider();
    const window = await telemetry.loadCase("fixture-early-warning");
    const quality = calculateInputQuality(window);

    expect(quality.observedFraction).toBeLessThan(1);
    expect(quality.missingSignals).toContain("bis_signal_quality_index");
    expect(quality.signalCoverage.bis_signal_quality_index).toBeLessThan(1);
  });

  it("returns an exact, typed prerecorded fixture forecast for the current window", async () => {
    const telemetry = new FixtureTelemetryProvider();
    const window = await telemetry.loadCase("fixture-no-event");
    const request = InferenceRequestSchema.parse({
      requestId: "test-request-1",
      window,
    });
    const forecast = await new FixtureInferenceProvider().forecast(request);

    expect(ForecastResponseSchema.parse(forecast)).toMatchObject({
      requestId: "test-request-1",
      sampleId: "fixture-no-event",
      horizon: "none_within_15",
      provider: "fixture",
    });
    expect(forecast.interpret).not.toEqual("");
    expect(forecast.anticipate).not.toEqual("");
    expect(forecast.act).toMatch(/signal quality/i);
  });

  it("keeps baseline outputs separate from fixture output and labels unavailable experiment metrics", async () => {
    const telemetry = new FixtureTelemetryProvider();
    const window = await telemetry.loadCase("fixture-early-warning");
    const baseline = await new BaselineInferenceProvider().forecast({
      requestId: "baseline-1",
      window,
    });
    const evaluation = await new EvaluationProvider({
      bundled: evaluationResults,
    }).loadResults();

    expect(baseline.provider).toBe("baseline");
    expect(evaluation.status).toBe("pending_actual_results");
    expect(evaluation.metrics.kind).toBe("pending");
    await expect(
      new EvaluationProvider({ bundled: evaluationResults }).getOutcome(
        "fixture-early-warning",
      ),
    ).resolves.toMatchObject({
      groundTruthHorizon: "within_5",
      source: "fixture",
    });
  });

  it("rejects invalid sampling alignment and task-trained output without a checkpoint", async () => {
    const telemetry = new FixtureTelemetryProvider();
    const window = await telemetry.loadCase("fixture-no-event");
    const misaligned = {
      ...window,
      timestamps: [
        ...window.timestamps.slice(0, 4),
        "2026-09-12T09:00:09.000Z",
        ...window.timestamps.slice(5),
      ],
    };
    expect(TelemetryWindowSchema.safeParse(misaligned).success).toBe(false);

    const forecast = await new FixtureInferenceProvider().forecast({
      requestId: "fixture-2",
      window,
    });
    expect(
      ForecastResponseSchema.safeParse({
        ...forecast,
        provider: "opentslm_task_trained",
        checkpointId: undefined,
        interpret: "   ",
      }).success,
    ).toBe(false);
  });

  it("accepts normalized-only held-out values and rejects labels in model input", async () => {
    const fixtureWindow = await new FixtureTelemetryProvider().loadCase(
      "fixture-no-event",
    );
    const map = fixtureWindow.signals[0];
    const normalizedMap = {
      ...map,
      values: Array<number | null>(10).fill(null),
      normalizedValues: map.values.map((value) =>
        value === null ? null : (value - 70) / 10,
      ),
      normalization: {
        mean: 70,
        std: 10,
        sourceUnit: "z-score",
        physicalUnit: "mmHg",
      },
    };
    const window = {
      ...fixtureWindow,
      source: "vitaldb_held_out_export" as const,
      datasetSplit: "held_out_test" as const,
      selectedChannels: ["map"],
      signals: [normalizedMap],
      fixtureDisclaimer: undefined,
    };
    const exportData = {
      schemaVersion: "telemetry-export.v1" as const,
      source: "vitaldb_held_out_export" as const,
      cases: [
        {
          metadata: {
            ...scenarioMetadata[1],
            source: "vitaldb_held_out_export" as const,
            datasetSplit: "held_out_test" as const,
            fixtureDisclaimer: undefined,
          },
          window,
        },
      ],
    };

    const provider = new HeldOutTelemetryProvider(exportData);
    const realWindow = await provider.loadCase("fixture-no-event");
    expect(physicalValues(realWindow.signals[0]).at(-1)).toBe(78);
    expect(() =>
      loadHeldOutTelemetryExport({
        ...exportData,
        cases: [
          {
            ...exportData.cases[0],
            window: { ...window, groundTruthHorizon: "within_5" },
          },
        ],
      }),
    ).toThrow();

    const leakedMaskedValue = {
      ...normalizedMap,
      observed: [false, ...normalizedMap.observed.slice(1)],
      normalizedValues: [0, ...normalizedMap.normalizedValues!.slice(1)],
    };
    expect(SignalSeriesSchema.safeParse(leakedMaskedValue).success).toBe(false);
    expect(physicalValues(leakedMaskedValue).at(0)).toBeNull();
  });

  it("enforces fixture identity, disclaimer, and a non-hypotensive replay cutoff", async () => {
    const window = await new FixtureTelemetryProvider().loadCase(
      "fixture-no-event",
    );
    expect(
      TelemetryWindowSchema.safeParse({
        ...window,
        fixtureDisclaimer: undefined,
      }).success,
    ).toBe(false);
    expect(
      TelemetryWindowSchema.safeParse({ ...window, datasetSplit: "validation" })
        .success,
    ).toBe(false);
    expect(
      TelemetryWindowSchema.safeParse({
        ...window,
        signals: [
          {
            ...window.signals[0],
            values: [...window.signals[0].values.slice(0, 9), 64],
          },
          ...window.signals.slice(1),
        ],
      }).success,
    ).toBe(false);
  });

  it("rejects malformed horizons, whitespace reasoning, and non-comparable available metrics", async () => {
    const fixture = await new FixtureTelemetryProvider().loadCase(
      "fixture-no-event",
    );
    const forecast = await new FixtureInferenceProvider().forecast({
      requestId: "invalid-output",
      window: fixture,
    });
    expect(
      ForecastResponseSchema.safeParse({
        ...forecast,
        horizon: "tomorrow",
        interpret: "  ",
      }).success,
    ).toBe(false);
    expect(
      EvaluationResultsSchema.safeParse({
        ...evaluationResults,
        status: "available",
      }).success,
    ).toBe(false);
    expect(
      EvaluationResultsSchema.safeParse({
        ...evaluationResults,
        status: "available",
        modelComparison: {
          heldOutCohortId: "cohort-01",
          inputRevision: "input-v1",
          baseline: { ...evaluationResults.metrics, kind: "fixture" },
          opentslm: { ...evaluationResults.metrics, kind: "experiment" },
        },
      }).success,
    ).toBe(false);
    expect(
      EvaluationResultsSchema.safeParse({
        ...evaluationResults,
        status: "available",
        modelComparison: {
          heldOutCohortId: "cohort-01",
          inputRevision: "input-v1",
          baseline: {
            ...evaluationResults.metrics,
            kind: "experiment",
            confusionMatrix: [
              [1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
            ],
          },
          opentslm: {
            ...evaluationResults.metrics,
            kind: "experiment",
            confusionMatrix: [
              [1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1],
            ],
          },
        },
      }).success,
    ).toBe(false);
  });

  it("discards an older concurrent OpenTSLM response", async () => {
    const window = await new FixtureTelemetryProvider().loadCase(
      "fixture-no-event",
    );
    const first = await new FixtureInferenceProvider().forecast({
      requestId: "first",
      window,
    });
    const second = await new FixtureInferenceProvider().forecast({
      requestId: "second",
      window,
    });
    const requests: Array<{ resolve: (response: Response) => void }> = [];
    const fetchImpl = vi.fn(
      () => new Promise<Response>((resolve) => requests.push({ resolve })),
    ) as unknown as typeof fetch;
    const provider = new OpenTSLMInferenceProvider({ fetchImpl });
    const firstRequest = provider.forecast({ requestId: "first", window });
    const secondRequest = provider.forecast({ requestId: "second", window });
    requests[1].resolve(
      new Response(
        JSON.stringify({ ...second, provider: "opentslm_pretrained" }),
        { status: 200 },
      ),
    );
    await expect(secondRequest).resolves.toMatchObject({ requestId: "second" });
    requests[0].resolve(
      new Response(
        JSON.stringify({ ...first, provider: "opentslm_pretrained" }),
        { status: 200 },
      ),
    );
    await expect(firstRequest).rejects.toBeInstanceOf(
      StaleInferenceResponseError,
    );
  });

  it("protects the inference route from invalid input, unavailable backend, stale and malformed replies", async () => {
    const request = (body: unknown) =>
      new Request("http://localhost/api/inference", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    vi.stubEnv("OPENTSLM_API_URL", "");
    expect((await POST(request({}))).status).toBe(503);
    vi.stubEnv("OPENTSLM_API_URL", "https://model.example/infer");
    expect((await POST(request({}))).status).toBe(400);

    const window = await new FixtureTelemetryProvider().loadCase(
      "fixture-no-event",
    );
    const input = { requestId: "route-request", window };
    const response = await new FixtureInferenceProvider().forecast(input);
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              ...response,
              provider: "opentslm_pretrained",
              requestId: "old",
            }),
          ),
        ),
    );
    expect((await POST(request(input))).status).toBe(409);
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              ...response,
              provider: "opentslm_pretrained",
              horizon: "bad",
            }),
          ),
        ),
    );
    expect((await POST(request(input))).status).toBe(502);
  });

  it("returns a clear unavailable response when the upstream inference times out", async () => {
    vi.useFakeTimers();
    vi.stubEnv("OPENTSLM_API_URL", "https://model.example/infer");
    const window = await new FixtureTelemetryProvider().loadCase(
      "fixture-no-event",
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string | URL | Request, init?: RequestInit) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => {
              const timeout = new Error("timed out");
              timeout.name = "AbortError";
              reject(timeout);
            });
          }),
      ),
    );

    const pending = POST(
      new Request("http://localhost/api/inference", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ requestId: "timeout", window }),
      }),
    );
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(12_000);
    const response = await pending;
    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toMatchObject({
      error: "model_unavailable",
      message: "OpenTSLM inference timed out.",
    });
  });
});
