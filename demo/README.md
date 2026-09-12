# TimeLapse

TimeLapse is a research demonstration for replaying a 20-second intraoperative telemetry window and presenting a categorical forecast of sustained hypotension. It is a research prototype, not a medical device. It does not diagnose a cause, prescribe treatment, or provide dosing guidance.

## Run locally

```bash
cd demo
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

The supported checks are:

```bash
pnpm lint
pnpm typecheck
pnpm test
pnpm test:e2e
pnpm build
```

`pnpm test:e2e` builds and starts a production preview on localhost:3100 unless that preview is already running. `pnpm build` uses Next.js’s supported webpack builder for reproducible local builds. The Playwright suite also writes review captures to `../.impeccable/review/` when it runs.

## Demonstration data

The default mode is deterministic synthetic fixture replay (`NEXT_PUBLIC_DEMO_MODE=fixture`). Four scenarios are stored in [scenarios.v1.json](src/fixtures/scenarios.v1.json). They provide ten 2-second samples for the fixed top-10 signal order, explicit observation masks, and prerecorded fixture, baseline, and demonstration OpenTSLM outputs.

Fixture telemetry and outcomes are synthetic, are labeled in the interface, and must not guide clinical care. The checked-in evaluation file, [results.v1.json](public/evaluation/results.v1.json), is explicitly `pending_actual_results`; its metrics are null. It is not an experiment result.

The application keeps current-window telemetry separate from evaluation data. True future onset and ground-truth horizon belong only to `EvaluationProvider` and must appear only after opening Evaluation.

## Configuration

Copy [`.env.example`](.env.example) to `.env.local` when configuring a backend:

```dotenv
OPENTSLM_API_URL=https://your-service.example/forecast
OPENTSLM_API_TOKEN=server-side-secret
NEXT_PUBLIC_DEMO_MODE=fixture
NEXT_PUBLIC_EVALUATION_URL=/evaluation/results.v1.json
NEXT_PUBLIC_TELEMETRY_URL=
```

`OPENTSLM_API_TOKEN` is server-only. Do not prefix it with `NEXT_PUBLIC_`. With no `OPENTSLM_API_URL`, the inference route returns HTTP 503 and the replay remains usable.

## Integration boundaries

The stable browser-facing contracts and provider interfaces live in [contracts.ts](src/lib/contracts.ts) and [providers.ts](src/lib/providers.ts). The Next route at [route.ts](src/app/api/inference/route.ts) validates the telemetry request, forwards it with a server-side bearer token, enforces a 12-second timeout, validates the response, and rejects mismatched request identity.

See [INTEGRATION.md](docs/INTEGRATION.md) for the exact OpenTSLM request and response boundary, telemetry-export shape, and the evaluation-results file to replace when held-out results are available.

## Current limitations

- Research prototype; not for clinical use or prospective decision-making.
- The real task uses MAP-derived, non-adjudicated labels; this demo uses labeled synthetic outcomes.
- Device identities are unavailable, so device-disjoint performance cannot be claimed.
- Missing future MAP can hide qualifying events; boundary events may lack follow-up needed to establish 60 seconds below MAP 65 mmHg.
- Generated reasoning reports observed relationships and is not causal evidence.
- Prospective clinical validation has not been performed; small held-out cohorts can produce unstable estimates.

## Interface

The white clinical workspace prioritizes readable telemetry, manual replay, and one timed forecast. Live Replay shows observed relationships and the reassessment prompt; Model Analysis exposes INTERPRET, ANTICIPATE, ACT and ANSWER. Evidence, checkpoint metadata and the research narrative are progressively disclosed. The full demonstration supports keyboard navigation and reduced motion.

With `NEXT_PUBLIC_TELEMETRY_URL`, held-out input-only exports replace fixtures through the same provider interface. See the integration guide for the strict schema. No real held-out data, fine-tuned checkpoint or experiment metrics are included.
