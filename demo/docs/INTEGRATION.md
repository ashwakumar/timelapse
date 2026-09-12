# OpenTSLM integration

The frontend separates current telemetry, model output and evaluation-only outcomes. UI components consume provider interfaces in `src/lib/providers.ts`.

## Configure a model

`OpenTSLMInferenceProvider` posts a typed request to `POST /api/inference` in `src/app/api/inference/route.ts`. Set `OPENTSLM_API_URL` to your inference service and optionally set `OPENTSLM_API_TOKEN`. The Next route attaches the bearer token server-side; it is never sent to the browser.

The route validates input and output with Zod, applies a 12-second timeout, disables response caching, and checks request ID, sample ID, case ID, cutoff and dataset split. It returns 400 for invalid configured requests, 409 for mismatched response identity, 502 for malformed output, and 503 for an unconfigured, failed or timed-out backend. The browser also discards superseded requests when changing case or provider.

Use `NEXT_PUBLIC_DEMO_MODE=backend` to select the backend initially. `fixture` and `baseline` select deterministic prerecorded outputs. Those two local providers support fixture cases only.

## Telemetry contract

`InferenceRequestSchema` in `src/lib/contracts.ts` is authoritative:

```ts
{ requestId: string, window: TelemetryWindow }
```

A window declares `schemaVersion: 'telemetry.v1'`, source, sample and case IDs, dataset split, cutoff timestamp and elapsed cutoff seconds, `sampleIntervalSeconds: 2`, ten UTC timestamps, de-identified `staticContext`, and one to ten available `signals`. The optional `selectedChannels` array may name only supplied channels.

Signal channels preserve the published top-10 order: MAP, remifentanil effect-site concentration, respiratory rate from CO2, remifentanil infusion rate, BIS signal-quality index, inspired CO2, EtCO2, BIS EMG, BIS and BIS suppression ratio. Subsets are supported. No unavailable channel is fabricated or renamed.

Each signal contains its channel ID, label, unit, ten nullable values and ten boolean observation masks. A missing sample must have null physical and normalized values. For normalized-only exports, supply null physical values plus `normalizedValues` and training metadata `{ mean, std, sourceUnit, physicalUnit }`; `physicalValues()` converts observed values back to physical scale. Masks are checked before conversion.

The analysis interval is `(cutoff − 20 seconds, cutoff]`, sampled at +2, +4, …, +20 seconds. Thus the first and last source timestamps span 18 seconds; ten samples occupy the 20-second analysis window. The final timestamp must equal the cutoff. Windows already hypotensive at cutoff are rejected.

## Load held-out VitalDB inputs

Set `NEXT_PUBLIC_TELEMETRY_URL` to a JSON URL. `createTelemetryProvider()` in `src/lib/configuration.ts` loads it through `HeldOutTelemetryProvider`, then populates the same case selector and telemetry workspace. This configuration selects backend inference and disables fixture-only inference choices. Restart or rebuild after changing public environment variables.

The strict `HeldOutTelemetryExportSchema` accepts:

```ts
{
  schemaVersion: 'telemetry-export.v1',
  source: 'vitaldb_held_out_export',
  cases: [{ metadata: ScenarioMetadata, window: TelemetryWindow }]
}
```

Both metadata and window must declare the VitalDB export source and matching sample and case IDs. Use `datasetSplit: 'held_out_test'` for held-out test inputs. Do not include forecasts, true onset, future MAP or ground-truth labels in this file. These extra fields are rejected. The separate `TelemetryExportSchema` with `scenarios: [{ metadata, window, forecasts }]` is the internal fixture format, not the real input format.

## Forecast response

`ForecastResponseSchema` requires `requestId`, `generatedAt`, `provider`, `modelId`, `datasetSplit`, `sampleId`, `caseId`, `cutoffSeconds`, `horizon`, `interpret`, `anticipate`, `act` and `inputQuality`. `inputQuality` carries the observed fraction, missing signal IDs and per-signal coverage. The optional `checkpointId` is mandatory for `opentslm_task_trained`.

The five exact horizon values are `within_3`, `within_5`, `within_10`, `within_15` and `none_within_15`. The UI constructs ANSWER from this category. Generic upstream checkpoints must use `opentslm_pretrained`; only an actual task adapter may use `opentslm_task_trained` with its checkpoint identifier. Model revision is reported through `modelId`.

For this research demonstration, ACT must equal the approved non-prescriptive prompt:

> Verify signal quality and reassess the current haemodynamic state.

Missing or empty reasoning is rejected. A conservative wording guard also rejects explicit treatment directives and several unsupported causal or shock claims. This is not a clinical safety certification; the service remains responsible for safe, grounded model output. INTERPRET and ANTICIPATE must describe supplied observations without diagnosis, causation, medication or dosing advice.

## Load evaluation results

Replace **`public/evaluation/results.v1.json`** with real held-out metrics, or set `NEXT_PUBLIC_EVALUATION_URL` to a versioned replacement. `EvaluationProvider` loads it only after Evaluation mode is opened. The default file explicitly has `pending_actual_results`, null metrics and labeled synthetic outcomes.

Real results must pass `EvaluationResultsSchema`: `status: 'available'`, `source: 'vitaldb_held_out_export'`, the matching dataset split, an evaluation revision and experiment metrics. `modelComparison` is required for available results and identifies one shared `heldOutCohortId` and `inputRevision`, with independent baseline and OpenTSLM metric sets from the identical patients and inputs.

Metric sets include episode sensitivity, median lead time, false warnings per case, macro F1, balanced accuracy and a 5×5 integer confusion matrix. The interface displays both model metric sets when supplied. It never substitutes feature-screen metrics.

Each outcome includes sample ID, case ID, source, dataset split, true horizon, optional actual onset, episode warning, legacy episode lead time, error type and baseline/OpenTSLM forecast results. Each model result may supply `leadTimeSeconds` alongside its horizon and correctness flag. Missing per-model lead time is displayed as unavailable, not inferred. Outcome identity, source and split must match the current telemetry window before it is rendered.

No real VitalDB export, task-trained adapter or experiment result is bundled. All default forecasts and case outcomes are deterministic fixtures.
