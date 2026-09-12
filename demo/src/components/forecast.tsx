"use client";

import { useEffect, useRef } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { gsap } from "gsap";
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleAlert,
  Clock3,
  Layers,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import type {
  ForecastHorizon as Horizon,
  ForecastResponse,
  TelemetryWindow,
} from "@/lib/contracts";
import { physicalValues } from "@/lib/adapters";
import { useReducedMotion } from "@/lib/motion";
import { useReplayStore } from "@/lib/replay-store";
import { formatTime } from "@/lib/utils";

export const horizonLabels: Record<Horizon, string> = {
  within_3: "Hypotension within 3 minutes",
  within_5: "Hypotension within 5 minutes",
  within_10: "Hypotension within 10 minutes",
  within_15: "Hypotension within 15 minutes",
  none_within_15: "No hypotension within 15 minutes",
};
export const horizonKeys: Horizon[] = [
  "within_3",
  "within_5",
  "within_10",
  "within_15",
  "none_within_15",
];
const horizonSegments = ["0–3", "3–5", "5–10", "10–15", "No event"];

export function modelStatus(
  forecast: ForecastResponse | null,
  unavailable = false,
  selectedProvider: "fixture" | "baseline" | "backend" = "fixture",
) {
  if (unavailable) return "Model Unavailable";
  if (!forecast)
    return selectedProvider === "backend"
      ? "Model Unavailable"
      : selectedProvider === "baseline"
        ? "Baseline"
        : "Fixture Replay";
  if (forecast.provider === "fixture") return "Fixture Replay";
  if (forecast.provider === "baseline") return "Baseline";
  if (forecast.provider === "opentslm_pretrained") return "OpenTSLM Pretrained";
  return forecast.checkpointId ? "Task-Trained" : "Model Unavailable";
}
export function ModelStatusBadge({
  forecast,
  unavailable = false,
  selectedProvider = "fixture",
}: {
  forecast: ForecastResponse | null;
  unavailable?: boolean;
  selectedProvider?: "fixture" | "baseline" | "backend";
}) {
  return (
    <span className={`model-badge ${unavailable ? "model-error" : ""}`}>
      <span className="status-dot" />
      {modelStatus(forecast, unavailable, selectedProvider)}
    </span>
  );
}
export function ForecastChangeIndicator({
  forecast,
  previous,
}: {
  forecast: ForecastResponse;
  previous: Horizon | null;
}) {
  if (!previous || previous === forecast.horizon)
    return (
      <span className="forecast-change">
        <Check size={12} />{" "}
        {previous
          ? "Unchanged from previous forecast"
          : "First forecast for this window"}
      </span>
    );
  const sooner =
    horizonKeys.indexOf(forecast.horizon) < horizonKeys.indexOf(previous);
  return (
    <span className="forecast-change">
      {sooner ? <ArrowDown size={12} /> : <ArrowUpRight size={12} />}{" "}
      {sooner ? "Earlier" : "Later"} horizon than previous forecast
    </span>
  );
}

export function ForecastHorizon({
  forecast,
  previous,
}: {
  forecast: ForecastResponse;
  previous: Horizon | null;
}) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  const none = forecast.horizon === "none_within_15";
  const immediate = forecast.horizon === "within_3";
  const horizon = forecast.horizon.replace("within_", "");
  useEffect(() => {
    if (reduced) return;
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".horizon-selected",
        { y: 5, opacity: 0.6 },
        { y: 0, opacity: 1, duration: 0.3, ease: "power2.out" },
      );
    }, ref);
    return () => ctx.revert();
  }, [forecast.horizon, reduced]);
  return (
    <section
      ref={ref}
      className={`forecast-panel ${none ? "stable-forecast" : ""} ${immediate ? "immediate-forecast" : ""}`}
      aria-labelledby="forecast-heading"
    >
      <div className="forecast-topline">
        <h2 id="forecast-heading">Timed forecast</h2>
        <span>
          <Clock3 size={12} /> Latest window
        </span>
      </div>
      <p className="forecast-question">
        Based on the latest 20 seconds, when is sustained hypotension most
        likely to begin?
      </p>
      <div className="forecast-main">
        <div className="risk-symbol">
          {none ? <ShieldCheck size={24} /> : <TriangleAlert size={24} />}
        </div>
        <div>
          <h3>
            {none ? (
              <>
                None within <strong>15</strong> minutes
              </>
            ) : (
              <>
                Within <strong>{horizon}</strong> minutes
              </>
            )}
          </h3>
        </div>
      </div>
      <div className="horizon-segments" aria-label="Forecast horizons">
        {horizonKeys.map((key, index) => (
          <div
            key={key}
            className={`horizon-segment ${forecast.horizon === key ? "horizon-selected" : ""}`}
            aria-current={forecast.horizon === key ? "true" : undefined}
          >
            <span>{horizonSegments[index]}</span>
            {forecast.horizon === key && <div className="segment-marker" />}
          </div>
        ))}
      </div>
      <div className="horizon-axis-label">
        <span>Forecast horizon · minutes</span>
        <span>Categorical output</span>
      </div>
      <ForecastChangeIndicator forecast={forecast} previous={previous} />
      <span className="sr-only" role="status" aria-live="polite">
        {horizonLabels[forecast.horizon]}
      </span>
    </section>
  );
}

export function ReasoningPanel({
  forecast,
  expanded = false,
}: {
  forecast: ForecastResponse;
  expanded?: boolean;
}) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  useEffect(() => {
    if (reduced) return;
    const ctx = gsap.context(() => {
      gsap
        .timeline()
        .fromTo(
          ".reasoning-block",
          { opacity: 0.3, y: 5 },
          {
            opacity: 1,
            y: 0,
            stagger: 0.085,
            duration: 0.24,
            ease: "power2.out",
          },
        );
    }, ref);
    return () => ctx.revert();
  }, [forecast.requestId, reduced]);
  return (
    <section
      className={`reasoning-panel ${expanded ? "reasoning-expanded" : ""}`}
      ref={ref}
      aria-labelledby="reasoning-heading"
    >
      <div className="reasoning-heading">
        <h2 id="reasoning-heading">
          <Layers size={17} />
          {expanded ? "Model reasoning" : "What supports the forecast"}
        </h2>
        <span>
          {forecast.provider === "baseline"
            ? "Baseline output"
            : forecast.provider === "fixture"
              ? "Fixture output"
              : "Generated explanation"}
        </span>
      </div>
      <div className="reasoning-block">
        <h3>{expanded ? "INTERPRET" : "Observed"}</h3>
        <p>{forecast.interpret}</p>
      </div>
      {expanded && (
        <div className="reasoning-block">
          <h3>ANTICIPATE</h3>
          <p>{forecast.anticipate}</p>
        </div>
      )}
      <div className="reasoning-block">
        <h3>{expanded ? "ACT" : "Reassess"}</h3>
        <p>{forecast.act}</p>
      </div>
      {expanded ? (
        <div className="reasoning-block answer-block">
          <h3>ANSWER</h3>
          <p>{horizonLabels[forecast.horizon]}</p>
        </div>
      ) : (
        <button
          className="reasoning-link text-button"
          onClick={() => useReplayStore.getState().setView("analysis")}
        >
          Open model analysis <ArrowRight size={15} />
        </button>
      )}
    </section>
  );
}

export function EvidenceDrawer({
  forecast,
  window,
}: {
  forecast: ForecastResponse;
  window: TelemetryWindow;
}) {
  const open = useReplayStore((s) => s.evidenceOpen);
  const setOpen = useReplayStore((s) => s.setEvidenceOpen);
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  useEffect(() => {
    if (!open || reduced) return;
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".evidence-details",
        { y: -4, opacity: 0.3 },
        { y: 0, opacity: 1, duration: 0.24 },
      );
    }, ref);
    return () => ctx.revert();
  }, [open, reduced]);
  return (
    <div className="evidence-drawer" ref={ref}>
      <Collapsible.Root open={open} onOpenChange={setOpen}>
        <Collapsible.Trigger className="evidence-trigger">
          <span>
            <CircleAlert size={14} /> Why this forecast?
          </span>
          <ChevronDown size={15} className={open ? "rotate-chevron" : ""} />
        </Collapsible.Trigger>
        <Collapsible.Content className="evidence-details">
          <dl>
            <div>
              <dt>Source sample span</dt>
              <dd>
                {window.timestamps[0].slice(11, 19)}–
                {window.timestamps[9].slice(11, 19)} UTC
              </dd>
            </div>
            <div>
              <dt>Window / interval</dt>
              <dd>
                20 s / {window.sampleIntervalSeconds} s · samples at +2…+20 s
              </dd>
            </div>
            <div>
              <dt>Observation coverage</dt>
              <dd>
                {(forecast.inputQuality.observedFraction * 100).toFixed(0)}%
              </dd>
            </div>
            <div>
              <dt>Missing signals</dt>
              <dd>
                {forecast.inputQuality.missingSignals.join(", ") || "None"}
              </dd>
            </div>
            <div>
              <dt>Provider / model</dt>
              <dd>
                {forecast.provider} / {forecast.modelId}
              </dd>
            </div>
            <div>
              <dt>Checkpoint</dt>
              <dd>{forecast.checkpointId || "No task-trained checkpoint"}</dd>
            </div>
            <div>
              <dt>Generated</dt>
              <dd>
                {forecast.generatedAt.replace("T", " ").replace("Z", " UTC")}
              </dd>
            </div>
            <div>
              <dt>Sample / split</dt>
              <dd>
                {forecast.sampleId} / {forecast.datasetSplit}
              </dd>
            </div>
            <div>
              <dt>Replay cutoff</dt>
              <dd>{formatTime(window.cutoffSeconds)}</dd>
            </div>
          </dl>
          <h3>Input parameters used</h3>
          <p>
            All {window.signals.length} parameters below were supplied in their
            source order, with explicit observation masks.
          </p>
          <h3>Measured changes · first to last observed</h3>
          <ul className="measured-changes">
            {window.signals.map((signal) => {
              const values = physicalValues(signal).filter(
                (v): v is number => v !== null,
              );
              const delta = (values.at(-1) ?? 0) - (values[0] ?? 0);
              return (
                <li key={signal.channel}>
                  <span>{signal.label}</span>
                  <strong>
                    {values.length > 1
                      ? `${delta > 0 ? "+" : ""}${delta.toFixed(2)} ${signal.normalization?.physicalUnit ?? signal.unit}`
                      : "Insufficient observations"}
                  </strong>
                </li>
              );
            })}
          </ul>
          <p>
            Target: MAP below 65 mmHg for at least 60 seconds. Research labels
            are not clinician-adjudicated. Generated rationale is not causal
            evidence.
          </p>
        </Collapsible.Content>
      </Collapsible.Root>
    </div>
  );
}
export function InferenceErrorState({
  message,
  retry,
}: {
  message: string;
  retry: () => void;
}) {
  return (
    <div className="inference-error" role="alert">
      <CircleAlert size={20} />
      <h3>Model unavailable</h3>
      <p>Model unavailable — telemetry replay remains active</p>
      {message !== "Model unavailable — telemetry replay remains active" && (
        <p>{message}</p>
      )}
      <button onClick={retry} className="text-button">
        Retry analysis <ArrowRight size={13} />
      </button>
    </div>
  );
}
export function LoadingAnalysisState() {
  return (
    <section className="analysis-loading" role="status" aria-busy="true">
      <div className="loading-line" />
      <h3>Reading the observed window</h3>
      <p>Checking inputs and preparing the timed forecast…</p>
      <div className="skeleton" />
      <div className="skeleton short" />
    </section>
  );
}
