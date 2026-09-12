# TimeLapse final review

## 1. Disposition

**Ship the rebuilt research demonstration.** The user's later request for a white background, larger readable type, simpler composition and polished motion supersedes the original dark direction. This was a structural rebuild, followed by corrections and final production recapture. No commit was made by the reviewer.

## 2. Material findings and corrections

- Replaced the dense dark interface with white working surfaces, readable Geist prose, prominent MAP and one exact timed forecast. Live Replay now has a short observed/reassess summary; Model Analysis retains INTERPRET, ANTICIPATE, ACT and ANSWER. Evidence and the research story are progressive disclosures.
- Removed repeated information instead of shrinking prose to fit. Default Live Replay and Model Analysis fit the 1440×900 viewport. Tablet and portrait layouts preserve readable content and scroll naturally.
- Corrected signal precision, physical-unit conversion, normalized-only data display, observation masks, source-sample timing, shared crosshair labels and top-N coverage. Missing data remains missing. Incomplete observations do not produce invented change values.
- Added a configurable held-out telemetry loader, preserved strict input-only inference, invalidated stale responses on case/provider changes, and made unavailable/malformed responses recoverable. Model and checkpoint labels reflect verified response provenance.
- Corrected no-event language to “None within 15 minutes.” Evaluation alone fetches and exposes future outcomes. Per-model lead time and matched-cohort metrics use the actual versioned evaluation contract; fixture metrics stay pending.
- Restricted ACT to signal verification and haemodynamic reassessment. Added conservative rejection of explicit treatment directives and causal assertions. This is a contract safeguard, not clinical certification.
- Added restrained GSAP transitions and keyboard navigation. Reduced motion suppresses animation and WebGL. Unsupported and software-rendered WebGL use the static fallback, avoiding GPU-stall console warnings.
- Formatted source and documented the implemented light system in PRODUCT.md, DESIGN.md, the surface contract and design.json. Integration and demonstration documentation describe the actual fixture/real-data boundary.

## 3. Validation evidence

Final production verification on `http://localhost:3100`:

- `pnpm lint`: passed.
- `pnpm typecheck`: passed.
- `pnpm test`: 27 tests passed across replay, schemas/API/provider behavior and UI contracts.
- `pnpm build`: optimized Next.js webpack production build passed.
- `pnpm test:e2e`: all 9 Chromium tests passed. Covers real replay progression to cutoff, scenario changes, loading, unavailable backend recovery, stale/malformed output, keyboard tabs/signals/crosshair, evaluation-only outcomes, all guided steps, reduced motion and responsive sizing.
- Complete browser flow: no page errors or browser warning/error console messages. No evaluation file was requested before entering Evaluation.
- Impeccable detector on the rebuilt UI returned `[]`.
- Root's independent production dependency audit reported no known vulnerabilities.

Final captures were inspected as a batch: `desktop.png` (1440×900), `tablet.png` (1024×768 viewport), `portrait.png` (768×1024 viewport), plus `analysis.png`, `evaluation.png` and `evidence.png`. All are in this directory. Primary desktop views pass the ≤900px document-height assertion; all required viewports pass the no-horizontal-overflow assertion. Screenshot capture waits for the selected tab's visible active state.

## 4. Visual quality verdict

The rebuilt interface has a clear reading order, substantially larger reading text, coherent semantic color, quiet grouping and legible physical measurements. The main task is understandable without opening secondary research detail. All four views share the same light visual system; empty, error and reduced-motion states remain usable. No stock imagery or generated raster assets are shipped. No material visual blockers remain in the reviewed viewports.

## 5. Remaining limitations

Included telemetry, forecasts and comparison outcomes are deterministic synthetic fixtures. A verified held-out export, live inference service, task-trained checkpoint and final experiment metrics have not been supplied; their integration boundaries are implemented and tested. The prototype has no prospective clinical validation. Browser automation was run in Chromium; this is not an exhaustive assistive-technology or cross-browser certification. Expanded evidence, evaluation and smaller viewports intentionally use vertical scrolling.

Production preview remains running at `http://localhost:3100` for the root's independent verification and the user's review.
