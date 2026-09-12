"use client";

import { useEffect, useState } from "react";
import {
  ArrowRight,
  Check,
  Clock3,
  Database,
  Eye,
  FileCheck2,
  ShieldCheck,
  TriangleAlert,
  X,
} from "lucide-react";
import type {
  EvaluationOutcome,
  EvaluationResults,
  TelemetryWindow,
} from "@/lib/contracts";
import { EvaluationProvider } from "@/lib/providers";
import { formatTime } from "@/lib/utils";
import { horizonKeys, horizonLabels } from "./forecast";

const evaluationProvider = new EvaluationProvider();
const shortClasses = ["≤3", "≤5", "≤10", "≤15", "None"];
type Metrics = EvaluationResults["metrics"];
const percent = (value: number | null) =>
  value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const leadTime = (seconds: number | null | undefined) =>
  seconds == null
    ? "Not supplied"
    : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
const metricCells = (metrics: Metrics) => [
  percent(metrics.episodeSensitivity),
  metrics.medianLeadTimeSeconds == null
    ? "—"
    : `${(metrics.medianLeadTimeSeconds / 60).toFixed(1)} min`,
  metrics.falseWarningsPerCase?.toFixed(2) ?? "—",
  metrics.macroF1?.toFixed(3) ?? "—",
  percent(metrics.balancedAccuracy),
];
const metricLabels = [
  "Episode sensitivity",
  "Median lead time",
  "False warnings / case",
  "Five-class macro F1",
  "Balanced accuracy",
];

export function ConfusionMatrix({ matrix }: { matrix: number[][] | null }) {
  return (
    <div className="confusion-panel">
      <div className="section-heading">
        <h3>OpenTSLM confusion matrix</h3>
        <span>{matrix ? "Versioned evaluation" : "Evaluation pending"}</span>
      </div>
      <div className="matrix-layout">
        <span className="matrix-y-title">ACTUAL</span>
        <div>
          <span className="matrix-x-title">PREDICTED</span>
          <table className="matrix-table">
            <thead>
              <tr>
                <th aria-label="Actual class" />
                {shortClasses.map((label) => (
                  <th key={label}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {horizonKeys.map((key, i) => (
                <tr key={key}>
                  <th>{shortClasses[i]}</th>
                  {horizonKeys.map((predicted, j) => (
                    <td
                      key={predicted}
                      className={i === j ? "diagonal" : ""}
                      aria-label={`${horizonLabels[key]}, predicted ${horizonLabels[predicted]}: ${matrix ? matrix[i][j] : "pending"}`}
                    >
                      {matrix ? matrix[i][j] : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p>
        Five exact forecast categories. Feature-screen metrics are excluded.
      </p>
    </div>
  );
}

export function BaselineComparison({
  outcome,
}: {
  outcome: EvaluationOutcome;
}) {
  const fixture = outcome.source === "fixture";
  return (
    <div className="baseline-comparison">
      <div className="section-heading">
        <h3>Same window. Two approaches.</h3>
        <span>{fixture ? "Fixture comparison" : "Held-out comparison"}</span>
      </div>
      <div className="comparison-grid">
        {(["baseline", "opentslm"] as const).map((key) => {
          const result = outcome[key];
          const baseline = key === "baseline";
          const Icon = baseline ? Database : FileCheck2;
          const warningLead =
            result.leadTimeSeconds !== undefined
              ? result.leadTimeSeconds
              : baseline
                ? undefined
                : outcome.leadTimeSeconds;
          return (
            <div key={key} className="comparison-model">
              <div className="comparison-title">
                <Icon size={18} />
                <h4>{baseline ? "Baseline" : "OpenTSLM"}</h4>
                <span>{fixture ? "Fixture" : "Evaluation result"}</span>
              </div>
              <p className="comparison-forecast">
                {horizonLabels[result.horizon]}
              </p>
              <div
                className={`correctness ${result.correct ? "correct" : "incorrect"}`}
              >
                {result.correct ? <Check size={16} /> : <X size={16} />}
                {result.correct ? "Correct category" : "Incorrect category"}
              </div>
              <dl>
                <div>
                  <dt>Warning lead time</dt>
                  <dd>
                    {result.horizon === "none_within_15" ||
                    !outcome.actualOnsetTimestamp
                      ? "No qualifying warning"
                      : leadTime(warningLead)}
                  </dd>
                </div>
                <div>
                  <dt>{baseline ? "Explanation" : "Explanation format"}</dt>
                  <dd>
                    {baseline
                      ? "Structured explanation unavailable"
                      : "INTERPRET / ANTICIPATE / ACT"}
                  </dd>
                </div>
              </dl>
            </div>
          );
        })}
      </div>
      <p className="comparison-note">
        {fixture
          ? "Prerecorded outputs illustrate this comparison. "
          : "Both outputs use this same evaluation window. "}
        Explanation quality does not establish prediction quality.
      </p>
    </div>
  );
}

export function EvaluationPanel({ window }: { window: TelemetryWindow }) {
  const [data, setData] = useState<{
    results: EvaluationResults;
    outcome: EvaluationOutcome;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    Promise.all([
      evaluationProvider.loadResults(),
      evaluationProvider.getOutcome(window.sampleId),
    ])
      .then(([results, outcome]) => {
        if (
          results.datasetSplit !== window.datasetSplit ||
          outcome.datasetSplit !== window.datasetSplit ||
          outcome.caseId !== window.caseId ||
          outcome.source !== window.source
        )
          throw new Error(
            "Evaluation identity or split does not match the current telemetry window.",
          );
        if (active) setData({ results, outcome });
      })
      .catch((error) => {
        if (active)
          setError(
            error instanceof Error ? error.message : "Evaluation unavailable",
          );
      });
    return () => {
      active = false;
    };
  }, [window.sampleId, window.datasetSplit, window.caseId, window.source]);
  if (error)
    return (
      <section className="evaluation-empty panel" role="alert">
        <TriangleAlert size={26} />
        <h2>Evaluation pending</h2>
        <p>{error}</p>
        <p>
          Telemetry remains available. Supply a matching versioned evaluation
          file to reveal results.
        </p>
      </section>
    );
  if (!data)
    return (
      <section className="evaluation-empty panel" role="status">
        <h2>Loading versioned evaluation…</h2>
        <div className="skeleton" />
      </section>
    );
  const { results, outcome } = data;
  const onsetSeconds = outcome.actualOnsetTimestamp
    ? (Date.parse(outcome.actualOnsetTimestamp) -
        Date.parse(window.cutoffTimestamp)) /
      1000
    : null;
  const eventPosition =
    onsetSeconds !== null && onsetSeconds >= 0 && onsetSeconds <= 900
      ? 20 + (onsetSeconds / 900) * 80
      : null;
  const metrics = results.modelComparison?.opentslm ?? results.metrics;
  const cells = metricCells(metrics);
  const ticks: [number, string][] = [
    [0, formatTime(window.cutoffSeconds - 20)],
    [20, formatTime(window.cutoffSeconds)],
    [36, "+3 min"],
    [20 + 80 / 3, "+5 min"],
    [20 + 160 / 3, "+10 min"],
    [100, "+15 min"],
  ];
  return (
    <section className="evaluation-view">
      <div className="evaluation-intro">
        <div>
          <h2>Prediction meets outcome</h2>
          <p>
            The future is visible here. It is never included in the model’s
            input.
          </p>
        </div>
        <span className="evaluation-mode-badge">
          <Eye size={16} /> Evaluation mode
        </span>
      </div>
      <div className="outcome-strip panel">
        <div className="ground-truth">
          <span>
            <ShieldCheck size={17} /> Ground truth{" "}
            <small>
              {outcome.source === "fixture" ? "Fixture" : "Held-out label"}
            </small>
          </span>
          <strong>{horizonLabels[outcome.groundTruthHorizon]}</strong>
          <p>
            {outcome.actualOnsetTimestamp
              ? `Actual onset ${outcome.actualOnsetTimestamp.slice(11, 19)} UTC · sustained MAP <65 mmHg ≥60 s`
              : "No qualifying event in the following 15 minutes."}
          </p>
        </div>
        <div className="outcome-facts">
          <span>
            Episode warning
            <strong>
              {outcome.episodeReceivedWarning ? "Received" : "Not received"}
            </strong>
          </span>
          <span>
            Error type
            <strong>
              {outcome.errorType === "none"
                ? "None"
                : outcome.errorType.replaceAll("_", " ")}
            </strong>
          </span>
          <span>
            Sample<strong>{outcome.sampleId}</strong>
          </span>
        </div>
      </div>
      <div className="outcome-timeline panel">
        <div className="timeline-track">
          <div className="history-range">Observed 20 s</div>
          <div className="decision-line">
            <span>Decision cutoff</span>
          </div>
          <span className="timeline-future-label">
            Future · evaluation only
          </span>
          {eventPosition !== null && (
            <div className="onset-marker" style={{ left: `${eventPosition}%` }}>
              <span>
                <TriangleAlert size={13} /> True onset
              </span>
            </div>
          )}
        </div>
        <div className="timeline-scale">
          {ticks.map(([left, label]) => (
            <span key={label} style={{ left: `${left}%` }}>
              {label}
            </span>
          ))}
        </div>
      </div>
      <div className="evaluation-body panel">
        <BaselineComparison outcome={outcome} />
        <ConfusionMatrix matrix={metrics.confusionMatrix} />
      </div>
      <div className="evaluation-metrics panel">
        <div className="metrics-status">
          <Clock3 size={18} />
          <span>
            {results.status === "pending_actual_results"
              ? "Evaluation pending"
              : metrics.kind === "fixture"
                ? "Fixture metrics"
                : "Held-out evaluation"}
            <small>{results.evaluationRevision}</small>
          </span>
        </div>
        {metricLabels.map((label, index) => (
          <div className="evaluation-metric" key={label}>
            <strong>{cells[index]}</strong>
            <span>{label}</span>
          </div>
        ))}
      </div>
      {results.modelComparison && (
        <div className="evaluation-comparison-metrics panel">
          <div className="section-heading">
            <h3>Matched held-out model metrics</h3>
            <span>
              {results.modelComparison.heldOutCohortId} ·{" "}
              {results.modelComparison.inputRevision}
            </span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  {metricLabels.map((label) => (
                    <th key={label}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(["baseline", "opentslm"] as const).map((key) => (
                  <tr key={key}>
                    <th>{key === "baseline" ? "Baseline" : "OpenTSLM"}</th>
                    {metricCells(results.modelComparison![key]).map(
                      (cell, index) => (
                        <td key={index}>{cell}</td>
                      ),
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <p className="evaluation-footnote">
        {results.disclaimer ||
          "Versioned research evaluation. Prospective clinical validation has not been performed."}
      </p>
    </section>
  );
}

const limitations = [
  [
    "Research prototype",
    "Not for clinical use. Prospective clinical validation has not been performed.",
  ],
  [
    "Algorithmic event labels",
    "Labels are generated from MAP and are not clinician-adjudicated.",
  ],
  [
    "Unverified device generalization",
    "Physical device identities are unavailable, so device-disjoint performance cannot currently be claimed.",
  ],
  [
    "Incomplete future observations",
    "Missing future MAP can hide qualifying events. Events near the prediction boundary may lack enough follow-up to confirm 60 seconds of hypotension.",
  ],
  [
    "Observations, not causality",
    "Generated explanations describe observed relationships and are not causal evidence. The system does not prescribe medication or treatment.",
  ],
  [
    "Limited evaluation certainty",
    "Performance on a small held-out cohort may be unstable. Performance claims require a versioned held-out evaluation.",
  ],
];
export function ResearchDisclaimer() {
  return (
    <span className="research-disclaimer">
      <ShieldCheck size={14} /> Research prototype · not for clinical use
    </span>
  );
}
export function LimitationsPanel({
  isFixture = true,
}: {
  isFixture?: boolean;
}) {
  return (
    <section className="limitations-view">
      <div className="limitations-intro">
        <div>
          <h2>Know what the evidence supports</h2>
          <p>
            Understand what is observed, what is predicted, and what remains
            unproven.
          </p>
        </div>
        <ResearchDisclaimer />
      </div>
      <div className="evidence-layout">
        <div className="evidence-method panel">
          <h3>
            One observed window.
            <br />
            One timed forecast.
          </h3>
          <p>
            TimeLapse uses a short telemetry history to anticipate a new
            sustained hypotension event. Every sample, mask, and model state
            remains inspectable.
          </p>
          <dl>
            <div>
              <dt>Source dataset</dt>
              <dd>
                VitalDB surgical telemetry
                <small>
                  {isFixture
                    ? "Current interface uses deterministic fixtures"
                    : "Current interface uses the configured held-out export"}
                </small>
              </dd>
            </div>
            <div>
              <dt>Observed history</dt>
              <dd>20 seconds · 10 samples · 2 s interval</dd>
            </div>
            <div>
              <dt>Target event</dt>
              <dd>MAP &lt;65 mmHg for ≥60 seconds</dd>
            </div>
            <div>
              <dt>Exclusions</dt>
              <dd>Windows already hypotensive at cutoff</dd>
            </div>
            <div>
              <dt>Required validation design</dt>
              <dd>
                Patient-disjoint, identical held-out cohort
                <small>
                  {isFixture
                    ? "Held-out export and evaluation pending"
                    : "Verify cohort and revision in Evaluation mode"}
                </small>
              </dd>
            </div>
            <div>
              <dt>Language output</dt>
              <dd>INTERPRET → ANTICIPATE → ACT → ANSWER</dd>
            </div>
          </dl>
          <a
            href="https://vitaldb.net"
            target="_blank"
            rel="noreferrer"
            className="text-button"
          >
            Explore the source dataset <ArrowRight size={15} />
          </a>
        </div>
        <div className="limitations-list panel">
          <h3>Read before interpreting a forecast</h3>
          {limitations.map(([heading, body]) => (
            <article key={heading}>
              <TriangleAlert size={18} />
              <div>
                <h4>{heading}</h4>
                <p>{body}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
