import {
  type InputQuality,
  type SignalSeries,
  type TelemetryWindow,
  TelemetryWindowSchema,
} from "./contracts";

/** Parses a held-out VitalDB or fixture export into the input-safe UI shape. */
export function parseTelemetryWindow(input: unknown): TelemetryWindow {
  return TelemetryWindowSchema.parse(input);
}

/**
 * Returns physical-scale values. Exporters may include normalized training values
 * alongside the source-scale `values`; charts always use the latter when present.
 */
export function physicalValues(series: SignalSeries): Array<number | null> {
  return series.values.map((physical, index) => {
    if (!series.observed[index]) return null;
    if (physical !== null) return physical;
    const normalized = series.normalizedValues?.[index];
    if (
      normalized === null ||
      normalized === undefined ||
      !series.normalization
    )
      return null;
    return normalized * series.normalization.std + series.normalization.mean;
  });
}

export function calculateInputQuality(window: TelemetryWindow): InputQuality {
  const signalCoverage: Record<string, number> = {};
  const missingSignals: string[] = [];
  let observedCount = 0;

  for (const signal of window.signals) {
    const coverage =
      signal.observed.filter(Boolean).length / signal.observed.length;
    signalCoverage[signal.channel] = coverage;
    observedCount += signal.observed.filter(Boolean).length;
    if (coverage < 1) missingSignals.push(signal.channel);
  }

  return {
    observedFraction:
      observedCount / (window.signals.length * window.timestamps.length),
    missingSignals,
    signalCoverage,
  };
}

/** Makes a serializable export where only chart-safe, current-window data exists. */
export function toTelemetryRequestWindow(
  window: TelemetryWindow,
): TelemetryWindow {
  return TelemetryWindowSchema.parse(JSON.parse(JSON.stringify(window)));
}
