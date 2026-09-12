import { z } from "zod";

/** The fixed prediction buckets used by the TimeLapse research task. */
export const ForecastHorizonSchema = z.enum([
  "within_3",
  "within_5",
  "within_10",
  "within_15",
  "none_within_15",
]);
export type ForecastHorizon = z.infer<typeof ForecastHorizonSchema>;

export const ForecastProviderSchema = z.enum([
  "fixture",
  "baseline",
  "opentslm_pretrained",
  "opentslm_task_trained",
]);
export type ForecastProvider = z.infer<typeof ForecastProviderSchema>;

export const DatasetSplitSchema = z.enum([
  "fixture",
  "validation",
  "held_out_test",
]);
export type DatasetSplit = z.infer<typeof DatasetSplitSchema>;

export const SourceTypeSchema = z.enum(["fixture", "vitaldb_held_out_export"]);
export type SourceType = z.infer<typeof SourceTypeSchema>;

export const ChannelIdSchema = z.enum([
  "map",
  "remifentanil_effect_site",
  "respiratory_rate_co2",
  "remifentanil_infusion_rate",
  "bis_signal_quality_index",
  "inspired_co2",
  "etco2",
  "bis_emg",
  "bis",
  "bis_suppression_ratio",
]);
export type ChannelId = z.infer<typeof ChannelIdSchema>;

export const CHANNEL_ORDER: readonly ChannelId[] = [
  "map",
  "remifentanil_effect_site",
  "respiratory_rate_co2",
  "remifentanil_infusion_rate",
  "bis_signal_quality_index",
  "inspired_co2",
  "etco2",
  "bis_emg",
  "bis",
  "bis_suppression_ratio",
] as const;

export const CHANNEL_LABELS: Record<ChannelId, string> = {
  map: "MAP",
  remifentanil_effect_site: "Remifentanil effect-site concentration",
  respiratory_rate_co2: "Respiratory rate from CO2",
  remifentanil_infusion_rate: "Remifentanil infusion rate",
  bis_signal_quality_index: "BIS signal-quality index",
  inspired_co2: "Inspired CO2",
  etco2: "EtCO2",
  bis_emg: "BIS EMG",
  bis: "BIS",
  bis_suppression_ratio: "BIS suppression ratio",
};

export const InputQualitySchema = z.object({
  observedFraction: z.number().min(0).max(1),
  missingSignals: z.array(z.string()),
  signalCoverage: z.record(z.string(), z.number().min(0).max(1)),
});
export type InputQuality = z.infer<typeof InputQualitySchema>;

const RequiredTextSchema = z.string().trim().min(1);

/** The demonstration accepts only this non-prescriptive reassessment prompt. */
export const SAFE_ACT =
  "Verify signal quality and reassess the current haemodynamic state.";

/** Exact frontend-to-provider forecast result contract from the demo brief. */
export const ForecastResponseSchema = z
  .object({
    requestId: RequiredTextSchema,
    generatedAt: z.string().datetime(),
    provider: ForecastProviderSchema,
    modelId: RequiredTextSchema,
    checkpointId: RequiredTextSchema.optional(),
    datasetSplit: DatasetSplitSchema,
    sampleId: RequiredTextSchema,
    caseId: RequiredTextSchema,
    cutoffSeconds: z.number().nonnegative(),
    horizon: ForecastHorizonSchema,
    interpret: RequiredTextSchema,
    anticipate: RequiredTextSchema,
    act: z.literal(SAFE_ACT),
    inputQuality: InputQualitySchema,
  })
  .superRefine((response, ctx) => {
    const text = `${response.interpret} ${response.anticipate}`;
    if (
      /\b(?:vasodilatory|hypovolemic|hypovolaemic) shock\b|\b(?:caused by|causes|because of)\b|\b(?:give|administer|inject|bolus|titrate|prescribe)\b/i.test(
        text,
      )
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message:
          "Unsupported causal, diagnostic, or treatment language is not accepted.",
        path: ["interpret"],
      });
    }
    if (
      response.provider === "opentslm_task_trained" &&
      !response.checkpointId
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Task-trained OpenTSLM responses require a checkpointId.",
        path: ["checkpointId"],
      });
    }
  });
export type ForecastResponse = z.infer<typeof ForecastResponseSchema>;

export const NormalizationMetadataSchema = z.object({
  mean: z.number(),
  std: z.number().positive(),
  sourceUnit: z.string().min(1),
  physicalUnit: z.string().min(1),
});
export type NormalizationMetadata = z.infer<typeof NormalizationMetadataSchema>;

const SamplesSchema = z.array(z.number().nullable()).length(10);
const ObservationMaskSchema = z.array(z.boolean()).length(10);

export const SignalSeriesSchema = z
  .object({
    channel: ChannelIdSchema,
    label: z.string().min(1),
    unit: z.string().min(1),
    values: SamplesSchema,
    observed: ObservationMaskSchema,
    normalizedValues: SamplesSchema.optional(),
    normalization: NormalizationMetadataSchema.optional(),
  })
  .strict()
  .superRefine((series, ctx) => {
    if (series.normalizedValues && !series.normalization) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "normalizedValues require training normalization metadata.",
        path: ["normalization"],
      });
    }
    series.observed.forEach((isObserved, index) => {
      if (
        !isObserved &&
        (series.values[index] !== null ||
          (series.normalizedValues?.[index] !== null &&
            series.normalizedValues?.[index] !== undefined))
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message:
            "Missing samples must be represented by null in physical and normalized values.",
          path: ["values", index],
        });
      }
      if (
        isObserved &&
        series.values[index] === null &&
        (!series.normalizedValues || series.normalizedValues[index] === null)
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message:
            "Observed samples must have a physical or normalized numeric value.",
          path: ["values", index],
        });
      }
    });
  });
export type SignalSeries = z.infer<typeof SignalSeriesSchema>;

export const StaticContextSchema = z
  .object({
    ageBand: z.string().min(1),
    asaClass: z.string().min(1),
    procedureCategory: z.string().min(1),
    casePhase: z.string().min(1),
    elapsedCaseMinutes: z.number().nonnegative(),
    signalQualityStatus: z.enum(["adequate", "limited", "review_required"]),
  })
  .strict();
export type StaticContext = z.infer<typeof StaticContextSchema>;

/**
 * Live/input-safe telemetry schema. It deliberately excludes labels and future
 * onset information; those are available only from EvaluationProvider.
 */
export const TelemetryWindowSchema = z
  .object({
    schemaVersion: z.literal("telemetry.v1"),
    source: SourceTypeSchema,
    sampleId: z.string().min(1),
    caseId: z.string().min(1),
    datasetSplit: DatasetSplitSchema,
    cutoffTimestamp: z.string().datetime(),
    cutoffSeconds: z.number().nonnegative(),
    sampleIntervalSeconds: z.literal(2),
    timestamps: z.array(z.string().datetime()).length(10),
    staticContext: StaticContextSchema,
    /** A UI may select any available known channels without inventing alternatives. */
    selectedChannels: z
      .array(ChannelIdSchema)
      .min(1)
      .max(CHANNEL_ORDER.length)
      .optional(),
    signals: z.array(SignalSeriesSchema).min(1).max(CHANNEL_ORDER.length),
    fixtureDisclaimer: RequiredTextSchema.optional(),
  })
  .strict()
  .superRefine((window, ctx) => {
    const actualOrder = window.signals.map((signal) => signal.channel);
    if (new Set(actualOrder).size !== actualOrder.length) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Telemetry signals cannot repeat a channel.",
        path: ["signals"],
      });
    }
    if (
      actualOrder.some(
        (channel, index) =>
          channel !==
          CHANNEL_ORDER.filter((id) => actualOrder.includes(id))[index],
      )
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Signals must preserve the published parameter ordering.",
        path: ["signals"],
      });
    }
    if (
      window.selectedChannels &&
      window.selectedChannels.some((channel) => !actualOrder.includes(channel))
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Selected channels must be present in the telemetry export.",
        path: ["selectedChannels"],
      });
    }
    if (window.source === "fixture") {
      if (window.datasetSplit !== "fixture") {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Fixture telemetry must use the fixture split.",
          path: ["datasetSplit"],
        });
      }
      if (!window.fixtureDisclaimer) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Fixture telemetry requires a disclaimer.",
          path: ["fixtureDisclaimer"],
        });
      }
    }
    const map = window.signals.find((signal) => signal.channel === "map");
    if (!map) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message:
          "MAP is required to exclude windows already hypotensive at cutoff.",
        path: ["signals"],
      });
    }
    const latestMapIndex = map ? map.observed.lastIndexOf(true) : -1;
    if (map && latestMapIndex >= 0) {
      const value =
        map.values[latestMapIndex] ??
        (map.normalizedValues?.[latestMapIndex] !== null &&
        map.normalizedValues?.[latestMapIndex] !== undefined &&
        map.normalization
          ? map.normalizedValues[latestMapIndex]! * map.normalization.std +
            map.normalization.mean
          : null);
      if (value !== null && value < 65) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message:
            "Input windows already hypotensive at the replay cutoff are excluded.",
          path: [
            "signals",
            window.signals.indexOf(map),
            "values",
            latestMapIndex,
          ],
        });
      }
    }
    const timestamps = window.timestamps.map((timestamp) =>
      Date.parse(timestamp),
    );
    if (timestamps.some(Number.isNaN)) return;
    if (
      timestamps[timestamps.length - 1] !== Date.parse(window.cutoffTimestamp)
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "The final source timestamp must equal cutoffTimestamp.",
        path: ["cutoffTimestamp"],
      });
    }
    for (let index = 1; index < timestamps.length; index += 1) {
      if (
        timestamps[index] - timestamps[index - 1] !==
        window.sampleIntervalSeconds * 1_000
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message:
            "Telemetry timestamps must match the declared sampling interval.",
          path: ["timestamps", index],
        });
      }
    }
  });
export type TelemetryWindow = z.infer<typeof TelemetryWindowSchema>;

export const InferenceRequestSchema = z
  .object({
    requestId: RequiredTextSchema,
    window: TelemetryWindowSchema,
  })
  .strict();
export type InferenceRequest = z.infer<typeof InferenceRequestSchema>;

export const ScenarioMetadataSchema = z
  .object({
    sampleId: z.string().min(1),
    caseId: z.string().min(1),
    title: z.string().min(1),
    description: z.string().min(1),
    datasetSplit: DatasetSplitSchema,
    source: SourceTypeSchema,
    fixtureDisclaimer: RequiredTextSchema.optional(),
  })
  .strict()
  .superRefine((metadata, ctx) => {
    if (metadata.source === "fixture" && metadata.datasetSplit !== "fixture") {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Fixture metadata must use the fixture split.",
        path: ["datasetSplit"],
      });
    }
    if (metadata.source === "fixture" && !metadata.fixtureDisclaimer) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Fixture metadata requires a disclaimer.",
        path: ["fixtureDisclaimer"],
      });
    }
  });
export type ScenarioMetadata = z.infer<typeof ScenarioMetadataSchema>;

export const TelemetryExportSchema = z
  .object({
    schemaVersion: z.literal("telemetry-export.v1"),
    scenarios: z
      .array(
        z
          .object({
            metadata: ScenarioMetadataSchema,
            window: TelemetryWindowSchema,
            forecasts: z.array(ForecastResponseSchema),
          })
          .strict(),
      )
      .min(1),
  })
  .strict()
  .superRefine((exportData, ctx) => {
    exportData.scenarios.forEach((scenario, index) => {
      if (
        scenario.metadata.sampleId !== scenario.window.sampleId ||
        scenario.metadata.caseId !== scenario.window.caseId
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Scenario metadata and telemetry identity must match.",
          path: ["scenarios", index],
        });
      }
    });
  });
export type TelemetryExport = z.infer<typeof TelemetryExportSchema>;

/** Safe schema for a real held-out export: input telemetry only, no labels or forecasts. */
export const HeldOutTelemetryExportSchema = z
  .object({
    schemaVersion: z.literal("telemetry-export.v1"),
    source: z.literal("vitaldb_held_out_export"),
    cases: z
      .array(
        z
          .object({
            metadata: ScenarioMetadataSchema,
            window: TelemetryWindowSchema,
          })
          .strict(),
      )
      .min(1),
  })
  .strict()
  .superRefine((exportData, ctx) => {
    exportData.cases.forEach((entry, index) => {
      if (
        entry.metadata.source !== "vitaldb_held_out_export" ||
        entry.window.source !== "vitaldb_held_out_export"
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Held-out exports must declare VitalDB export source.",
          path: ["cases", index],
        });
      }
      if (
        entry.metadata.sampleId !== entry.window.sampleId ||
        entry.metadata.caseId !== entry.window.caseId
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Held-out metadata and telemetry identity must match.",
          path: ["cases", index],
        });
      }
    });
  });
export type HeldOutTelemetryExport = z.infer<
  typeof HeldOutTelemetryExportSchema
>;

export const EvaluationOutcomeSchema = z
  .object({
    sampleId: z.string().min(1),
    caseId: z.string().min(1),
    source: SourceTypeSchema,
    datasetSplit: DatasetSplitSchema,
    groundTruthHorizon: ForecastHorizonSchema,
    actualOnsetTimestamp: z.string().datetime().nullable(),
    episodeReceivedWarning: z.boolean(),
    leadTimeSeconds: z.number().nonnegative().nullable(),
    baseline: z.object({
      horizon: ForecastHorizonSchema,
      correct: z.boolean(),
      leadTimeSeconds: z.number().nonnegative().nullable().optional(),
    }),
    opentslm: z.object({
      horizon: ForecastHorizonSchema,
      correct: z.boolean(),
      leadTimeSeconds: z.number().nonnegative().nullable().optional(),
    }),
    errorType: z.enum([
      "none",
      "false_warning",
      "missed_event",
      "late_warning",
    ]),
    fixtureDisclaimer: z.string().min(1).optional(),
  })
  .strict()
  .superRefine((outcome, ctx) => {
    if (
      outcome.source === "fixture" &&
      (outcome.datasetSplit !== "fixture" || !outcome.fixtureDisclaimer)
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Fixture outcomes require the fixture split and disclaimer.",
        path: ["fixtureDisclaimer"],
      });
    }
    if (
      outcome.source === "vitaldb_held_out_export" &&
      (outcome.datasetSplit === "fixture" || outcome.fixtureDisclaimer)
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Real held-out outcomes cannot carry fixture labels.",
        path: ["source"],
      });
    }
  });
export type EvaluationOutcome = z.infer<typeof EvaluationOutcomeSchema>;

const EvaluationMetricSetSchema = z.object({
  kind: z.enum(["fixture", "experiment", "pending"]),
  episodeSensitivity: z.number().min(0).max(1).nullable(),
  medianLeadTimeSeconds: z.number().nonnegative().nullable(),
  falseWarningsPerCase: z.number().nonnegative().nullable(),
  macroF1: z.number().min(0).max(1).nullable(),
  balancedAccuracy: z.number().min(0).max(1).nullable(),
  confusionMatrix: z
    .array(z.array(z.number().int().nonnegative()).length(5))
    .length(5)
    .nullable(),
});

export const EvaluationResultsSchema = z
  .object({
    schemaVersion: z.literal("evaluation-results.v1"),
    status: z.enum(["pending_actual_results", "available"]),
    source: SourceTypeSchema,
    datasetSplit: DatasetSplitSchema,
    evaluationRevision: z.string().min(1),
    disclaimer: z.string().min(1).optional(),
    metrics: EvaluationMetricSetSchema,
    modelComparison: z
      .object({
        heldOutCohortId: RequiredTextSchema,
        inputRevision: RequiredTextSchema,
        baseline: EvaluationMetricSetSchema,
        opentslm: EvaluationMetricSetSchema,
      })
      .optional(),
    outcomes: z.array(EvaluationOutcomeSchema),
  })
  .superRefine((results, ctx) => {
    if (
      results.source === "fixture" &&
      (results.datasetSplit !== "fixture" ||
        results.status !== "pending_actual_results" ||
        results.metrics.kind === "experiment")
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message:
          "Fixture evaluation cannot be presented as available experiment results.",
        path: ["status"],
      });
    }
    if (
      results.source === "vitaldb_held_out_export" &&
      results.datasetSplit === "fixture"
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Held-out evaluation cannot use the fixture split.",
        path: ["datasetSplit"],
      });
    }
    results.outcomes.forEach((outcome, index) => {
      if (
        outcome.source !== results.source ||
        outcome.datasetSplit !== results.datasetSplit
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message:
            "Outcome source and split must match the evaluation results.",
          path: ["outcomes", index],
        });
      }
    });
    if (results.status === "available" && !results.modelComparison) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message:
          "Available evaluation results require a same-cohort modelComparison.",
        path: ["modelComparison"],
      });
    }
    if (results.status === "available" && results.modelComparison) {
      for (const provider of [
        results.modelComparison.baseline,
        results.modelComparison.opentslm,
      ]) {
        if (provider.kind !== "experiment") {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message:
              "Available model comparison metrics must be experiment results.",
            path: ["modelComparison"],
          });
        }
      }
    }
  });
export type EvaluationResults = z.infer<typeof EvaluationResultsSchema>;
