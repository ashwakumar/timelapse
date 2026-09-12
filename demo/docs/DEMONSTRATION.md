# Presenting TimeLapse

Use the **Guided Demo** control for a presentation advanced one step at a time. It never runs the whole presentation automatically. At 1440×900, begin in **Live Replay**.

1. Introduce the target: a new episode of MAP below 65 mmHg sustained for at least 60 seconds. This is a research forecast, not a treatment instruction.
2. Identify the input honestly. The included scenarios are deterministic synthetic fixtures, not exported patient measurements. The fixture banner remains visible.
3. Choose a demonstration scenario. Inspect the separate patient-context strip and the five initially displayed signals. **Signals** enables the other supplied parameters.
4. Start replay. The transport supports pause, restart, four speeds, scrubbing, and previous/next cases. Missing observations stay visibly missing.
5. Pause at the 20-second decision cutoff and select **Analyze window**.
6. Read the exact forecast category. The segmented horizon expresses a category, not a probability or calibrated confidence.
7. Review **INTERPRET**, **ANTICIPATE**, **ACT**, and **ANSWER**. The explanation describes observed relationships; it does not establish causality.
8. Open **Why this forecast?** to inspect the sample identity, input timestamps, units, masks, measured changes, model state, and checkpoint provenance.
9. Open **Evaluation** to reveal the eventual outcome. The future event label is not part of the live input.
10. Compare baseline and OpenTSLM-format fixture outputs on the same input. Final experiment metrics remain **Evaluation pending** until a valid versioned held-out evaluation is supplied.
11. Select **TimeNet** in the research workflow to explain the shared data contract and patient-disjoint validation design.
12. Finish in **Evidence & Limitations**. The prototype has no prospective clinical validation and does not prescribe medication or treatment.

## What is real and what is illustrative

| Content | Provenance |
| --- | --- |
| Replay engine, chart synchronization, masks, controls, API validation | Implemented application behavior |
| Included telemetry and de-identified context | Synthetic deterministic fixtures |
| Included timed forecasts and four-part reasoning | Prerecorded fixture content |
| Baseline versus OpenTSLM case outcomes | Authored demonstration fixtures |
| Final model metrics | Not supplied; evaluation pending |
| Task-trained surgical checkpoint | Not supplied or loaded by this frontend |
| VitalDB input export adapter | Implemented boundary; requires a verified held-out export |
| OpenTSLM inference | Implemented server proxy; requires a configured service |

## Failure-state demonstration

With no backend configured, choose **OpenTSLM backend** and request analysis. The interface reports model unavailability while telemetry replay remains active. Select the fixture provider to return to the deterministic demonstration.

Keyboard users can reach navigation, scenario selection, all replay controls, signal selection and the evidence disclosure with standard focus navigation. Focus a telemetry plot and use the left/right arrows to inspect source samples. Reduced-motion settings suppress decorative movement and reveal reasoning immediately.
