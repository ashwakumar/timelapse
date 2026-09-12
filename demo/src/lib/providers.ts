import scenarioMetadataExport from "../fixtures/scenario-metadata.v1.json";
import { calculateInputQuality, parseTelemetryWindow } from "./adapters";
import {
  type EvaluationOutcome,
  type EvaluationResults,
  EvaluationResultsSchema,
  type ForecastProvider,
  type ForecastResponse,
  ForecastResponseSchema,
  type HeldOutTelemetryExport,
  HeldOutTelemetryExportSchema,
  type InferenceRequest,
  type ScenarioMetadata,
  ScenarioMetadataSchema,
  type TelemetryExport,
  type TelemetryWindow,
  TelemetryExportSchema,
} from "./contracts";

export interface TelemetryProvider {
  listCases(): Promise<ScenarioMetadata[]>;
  loadCase(sampleId: string): Promise<TelemetryWindow>;
}

export interface InferenceProvider {
  readonly provider: ForecastProvider;
  forecast(request: InferenceRequest): Promise<ForecastResponse>;
}

export interface EvaluationDataProvider {
  loadResults(): Promise<EvaluationResults>;
  getOutcome(sampleId: string): Promise<EvaluationOutcome>;
}

export class UnknownCaseError extends Error {
  constructor(sampleId: string) {
    super(`No fixture scenario exists for sample '${sampleId}'.`);
    this.name = "UnknownCaseError";
  }
}

export class InferenceUnavailableError extends Error {
  constructor(message = "Model unavailable — telemetry replay remains active") {
    super(message);
    this.name = "InferenceUnavailableError";
  }
}

export class StaleInferenceResponseError extends Error {
  constructor() {
    super("Discarded an inference response for a superseded request.");
    this.name = "StaleInferenceResponseError";
  }
}

/** Metadata intended for scenario selectors; it never contains future outcomes. */
export const scenarioMetadata: readonly ScenarioMetadata[] =
  zodScenarioMetadata(scenarioMetadataExport);

function zodScenarioMetadata(input: unknown): ScenarioMetadata[] {
  return ScenarioMetadataSchema.array().parse(input);
}

async function scenarioFor(sampleId: string) {
  const { default: fixtureExport } =
    await import("../fixtures/scenarios.v1.json");
  const parsedScenarioExport: TelemetryExport =
    TelemetryExportSchema.parse(fixtureExport);
  const scenario = parsedScenarioExport.scenarios.find(
    (candidate) => candidate.metadata.sampleId === sampleId,
  );
  if (!scenario) throw new UnknownCaseError(sampleId);
  return scenario;
}

function validateResponseIdentity(
  response: ForecastResponse,
  request: InferenceRequest,
): ForecastResponse {
  if (
    response.requestId !== request.requestId ||
    response.sampleId !== request.window.sampleId ||
    response.caseId !== request.window.caseId ||
    response.cutoffSeconds !== request.window.cutoffSeconds ||
    response.datasetSplit !== request.window.datasetSplit
  ) {
    throw new StaleInferenceResponseError();
  }
  return response;
}

async function fixtureForecast(
  request: InferenceRequest,
  provider: ForecastProvider,
): Promise<ForecastResponse> {
  const scenario = await scenarioFor(request.window.sampleId);
  const response = scenario.forecasts.find(
    (forecast) => forecast.provider === provider,
  );
  if (!response) {
    throw new InferenceUnavailableError(
      `No ${provider} fixture output is available for this replay.`,
    );
  }

  return validateResponseIdentity(
    ForecastResponseSchema.parse({
      ...response,
      requestId: request.requestId,
      sampleId: request.window.sampleId,
      caseId: request.window.caseId,
      cutoffSeconds: request.window.cutoffSeconds,
      inputQuality: calculateInputQuality(request.window),
    }),
    request,
  );
}

/** Deterministic local case source for jury replay. */
export class FixtureTelemetryProvider implements TelemetryProvider {
  async listCases(): Promise<ScenarioMetadata[]> {
    return [...scenarioMetadata];
  }

  async loadCase(sampleId: string): Promise<TelemetryWindow> {
    return parseTelemetryWindow((await scenarioFor(sampleId)).window);
  }
}

/** Parses a real, input-only VitalDB export and exposes it through the same UI boundary. */
export function loadHeldOutTelemetryExport(
  input: unknown,
): HeldOutTelemetryExport {
  return HeldOutTelemetryExportSchema.parse(input);
}

/**
 * Provider for held-out telemetry exports. Its accepted schema is strict and has
 * no forecast, label, onset, or other future-outcome fields.
 */
export class HeldOutTelemetryProvider implements TelemetryProvider {
  private readonly exportData: HeldOutTelemetryExport;

  constructor(input: unknown) {
    this.exportData = loadHeldOutTelemetryExport(input);
  }

  async listCases(): Promise<ScenarioMetadata[]> {
    return this.exportData.cases.map((entry) => entry.metadata);
  }

  async loadCase(sampleId: string): Promise<TelemetryWindow> {
    const entry = this.exportData.cases.find(
      (candidate) => candidate.window.sampleId === sampleId,
    );
    if (!entry) throw new UnknownCaseError(sampleId);
    return parseTelemetryWindow(entry.window);
  }
}

/** Deterministic prerecorded forecast used only while the UI is in fixture mode. */
export class FixtureInferenceProvider implements InferenceProvider {
  readonly provider = "fixture" as const;

  async forecast(request: InferenceRequest): Promise<ForecastResponse> {
    return await fixtureForecast(request, this.provider);
  }
}

/** Deterministic baseline output using the same fixture input window. */
export class BaselineInferenceProvider implements InferenceProvider {
  readonly provider = "baseline" as const;

  async forecast(request: InferenceRequest): Promise<ForecastResponse> {
    return await fixtureForecast(request, this.provider);
  }
}

type FetchLike = typeof fetch;

/** Browser adapter for a configured OpenTSLM backend; credentials remain in route.ts. */
export class OpenTSLMInferenceProvider implements InferenceProvider {
  readonly provider = "opentslm_pretrained" as const;
  private latestRequestId: string | undefined;

  constructor(
    private readonly options: { endpoint?: string; fetchImpl?: FetchLike } = {},
  ) {}

  async forecast(request: InferenceRequest): Promise<ForecastResponse> {
    this.latestRequestId = request.requestId;
    const response = await (this.options.fetchImpl ?? fetch)(
      this.options.endpoint ?? "/api/inference",
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
        signal: undefined,
      },
    );

    if (response.status === 503) throw new InferenceUnavailableError();
    if (!response.ok)
      throw new Error(`Inference request failed (${response.status}).`);

    const validation = ForecastResponseSchema.safeParse(await response.json());
    if (!validation.success)
      throw new Error(
        "The model returned an invalid forecast. Retry analysis or inspect the backend response contract.",
      );
    const parsed = validation.data;
    if (this.latestRequestId !== request.requestId)
      throw new StaleInferenceResponseError();
    if (!parsed.provider.startsWith("opentslm_")) {
      throw new Error(
        "Configured inference endpoint returned a non-OpenTSLM provider.",
      );
    }
    return validateResponseIdentity(parsed, request);
  }
}

/**
 * Sole provider of fixture/held-out labels. Live telemetry and inference providers
 * deliberately have no API for future onset or true horizon.
 */
export class EvaluationProvider implements EvaluationDataProvider {
  private cached: EvaluationResults | undefined;

  constructor(
    private readonly options: {
      url?: string;
      fetchImpl?: FetchLike;
      bundled?: unknown;
    } = {},
  ) {}

  async loadResults(): Promise<EvaluationResults> {
    if (this.cached) return this.cached;
    if (this.options.bundled)
      return (this.cached = EvaluationResultsSchema.parse(
        this.options.bundled,
      ));

    const url =
      this.options.url ??
      process.env.NEXT_PUBLIC_EVALUATION_URL ??
      "/evaluation/results.v1.json";
    const fetchImpl =
      this.options.fetchImpl ??
      (typeof window === "undefined" ? undefined : fetch);
    if (!fetchImpl) {
      throw new Error(
        "Evaluation results require a URL or a bundled test fixture when rendered on the server.",
      );
    }

    const response = await fetchImpl(url, {
      headers: { accept: "application/json" },
    });
    if (!response.ok)
      throw new Error(
        `Could not load evaluation results (${response.status}).`,
      );
    return (this.cached = EvaluationResultsSchema.parse(await response.json()));
  }

  async getOutcome(sampleId: string): Promise<EvaluationOutcome> {
    const outcome = (await this.loadResults()).outcomes.find(
      (item) => item.sampleId === sampleId,
    );
    if (!outcome) throw new UnknownCaseError(sampleId);
    return outcome;
  }
}
