import {
  FixtureTelemetryProvider,
  HeldOutTelemetryProvider,
  type TelemetryProvider,
} from "./providers";

export const usesHeldOutTelemetry = Boolean(
  process.env.NEXT_PUBLIC_TELEMETRY_URL,
);

/** The UI consumes the same interface for deterministic fixtures and input-only exports. */
export function createTelemetryProvider(
  url = process.env.NEXT_PUBLIC_TELEMETRY_URL,
): TelemetryProvider {
  if (!url) return new FixtureTelemetryProvider();
  let source: Promise<HeldOutTelemetryProvider> | undefined;
  const load = () =>
    (source ??= fetch(url, { headers: { accept: "application/json" } }).then(
      async (response) => {
        if (!response.ok)
          throw new Error(
            `Telemetry export unavailable (${response.status}). Check NEXT_PUBLIC_TELEMETRY_URL.`,
          );
        return new HeldOutTelemetryProvider(await response.json());
      },
    ));
  return {
    listCases: async () => (await load()).listCases(),
    loadCase: async (id) => (await load()).loadCase(id),
  };
}
