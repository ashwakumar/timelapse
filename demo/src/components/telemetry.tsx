"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";
import { scaleLinear } from "d3-scale";
import { area, line } from "d3-shape";
import * as Popover from "@radix-ui/react-popover";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Minus,
  Check,
  ChevronDown,
  CircleHelp,
  Pause,
  Play,
  RotateCcw,
  Settings2,
  SkipBack,
  SkipForward,
  Sparkles,
} from "lucide-react";
import type { SignalSeries, TelemetryWindow } from "@/lib/contracts";
import { physicalValues } from "@/lib/adapters";
import { useReplayStore } from "@/lib/replay-store";
import { formatTime } from "@/lib/utils";
import { Button } from "./ui/button";

const shorthand: Record<string, string> = {
  map: "Mean arterial pressure",
  remifentanil_effect_site: "Remifentanil · effect-site",
  respiratory_rate_co2: "Respiratory rate · CO₂",
  remifentanil_infusion_rate: "Remifentanil · infusion",
  bis_signal_quality_index: "BIS · signal quality",
  inspired_co2: "Inspired CO₂",
  etco2: "End-tidal CO₂",
  bis_emg: "BIS · EMG",
  bis: "Bispectral index",
  bis_suppression_ratio: "BIS · suppression ratio",
};

export const formatSignalValue = (
  value: number | null | undefined,
  channel: string,
) =>
  value == null
    ? "—"
    : new Intl.NumberFormat("en", {
        minimumFractionDigits: [
          "remifentanil_effect_site",
          "remifentanil_infusion_rate",
          "inspired_co2",
        ].includes(channel)
          ? 2
          : 0,
        maximumFractionDigits: 3,
      }).format(value);

export function SignalSelector({ signals }: { signals: SignalSeries[] }) {
  const selected = useReplayStore((s) => s.selectedSignals);
  const toggle = useReplayStore((s) => s.toggleSignal);
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <Button variant="ghost" size="sm">
          <Settings2 size={14} /> Signals{" "}
          <span className="count-tag">{selected.length}</span>
          <ChevronDown size={12} />
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className="signal-popover" sideOffset={8} align="end">
          <h3>Displayed signals</h3>
          <p>All {signals.length} available parameters remain model inputs.</p>
          {signals.map((signal) => (
            <label key={signal.channel} className="signal-option">
              <input
                type="checkbox"
                checked={selected.includes(signal.channel)}
                onChange={() => toggle(signal.channel)}
                disabled={
                  selected.length === 1 && selected[0] === signal.channel
                }
              />
              <span>
                {signal.label}
                <small>
                  {signal.normalization?.physicalUnit ?? signal.unit}
                </small>
              </span>
            </label>
          ))}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

export const TelemetryChart = memo(function TelemetryChart({
  signal,
  cutoff,
  visibleCount,
  crosshair,
  setCrosshair,
}: {
  signal: SignalSeries;
  cutoff: number;
  visibleCount: number;
  crosshair: number | null;
  setCrosshair: (value: number | null) => void;
}) {
  const isMap = signal.channel === "map";
  const [width, setWidth] = useState(660);
  const svgRef = useRef<SVGSVGElement>(null);
  const height = isMap ? 100 : 49;
  const values = useMemo(() => physicalValues(signal), [signal]);
  const unit = signal.normalization?.physicalUnit ?? signal.unit;
  useEffect(() => {
    const node = svgRef.current;
    if (!node) return;
    const resize = new ResizeObserver((entries) => {
      const measured = entries[0]?.contentRect.width;
      if (measured) setWidth(measured);
    });
    resize.observe(node);
    return () => resize.disconnect();
  }, []);
  const chart = useMemo(() => {
    const observedValues = values.filter((v): v is number => v !== null);
    const numeric = observedValues.length ? observedValues : [0, 1];
    const min = Math.min(...numeric, isMap ? 65 : Infinity);
    const max = Math.max(...numeric);
    const pad = Math.max(
      (max - min) * 0.45,
      isMap ? 5 : Math.max(max * 0.035, 0.05),
    );
    const y = scaleLinear()
      .domain([min - pad, max + pad])
      .range([height - 9, 9]);
    const x = scaleLinear()
      .domain([0, 9])
      .range([10, width - 36]);
    const samples = values.map((value, index) => ({ value, index }));
    const shown = samples.map((p) => ({
      ...p,
      value: p.index < visibleCount ? p.value : null,
    }));
    return {
      y,
      x,
      samples: shown,
      path:
        line<(typeof shown)[number]>()
          .defined((d) => d.value !== null)
          .x((d) => x(d.index))
          .y((d) => y(d.value!))(shown) ?? "",
      fill:
        area<(typeof shown)[number]>()
          .defined((d) => d.value !== null)
          .x((d) => x(d.index))
          .y0(height)
          .y1((d) => y(d.value!))(shown) ?? "",
      ticks: isMap ? y.ticks(3) : min === max ? [min] : [min, max],
    };
  }, [values, height, visibleCount, isMap, width]);
  const index =
    crosshair !== null && crosshair < visibleCount
      ? crosshair
      : visibleCount - 1;
  const value = visibleCount > 0 ? values[index] : null;
  const first = values
    .slice(0, visibleCount)
    .find((v): v is number => v !== null);
  const change = value != null && first !== undefined ? value - first : null;
  const ChangeIcon =
    change && change > 0
      ? ArrowUpRight
      : change && change < 0
        ? ArrowDownRight
        : Minus;
  return (
    <div className={`telemetry-row ${isMap ? "map-row" : ""}`}>
      <div className="signal-info">
        <div className="signal-label">
          <span className={`signal-dot ${isMap ? "amber" : ""}`} />
          {isMap ? "MAP" : shorthand[signal.channel]}
        </div>
        <div
          className="signal-number"
          aria-label={`${signal.label}: ${formatSignalValue(value, signal.channel)} ${unit}`}
        >
          <span>{formatSignalValue(value, signal.channel)}</span>
          <small>{unit}</small>
        </div>
        <div className="signal-description">{shorthand[signal.channel]}</div>
        {isMap && (
          <span className="map-change">
            <ChangeIcon size={13} />{" "}
            {change === null
              ? "Not observed"
              : `${change > 0 ? "+" : ""}${formatSignalValue(change, signal.channel)} mmHg`}{" "}
            <span>in this window</span>
          </span>
        )}
      </div>
      <svg
        ref={svgRef}
        className="signal-chart"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        tabIndex={0}
        aria-label={`${signal.label}, ${unit}, ${visibleCount} of ten source samples. Use left and right arrows to inspect. ${values
          .slice(0, visibleCount)
          .map((v) => (v === null ? "missing" : v))
          .join(", ")}`}
        onPointerMove={(event) => {
          if (!visibleCount) return;
          const box = event.currentTarget.getBoundingClientRect();
          setCrosshair(
            Math.min(
              visibleCount - 1,
              Math.max(
                0,
                Math.round(
                  ((((event.clientX - box.left) / box.width) * width - 10) /
                    (width - 46)) *
                    9,
                ),
              ),
            ),
          );
        }}
        onPointerLeave={() => setCrosshair(null)}
        onFocus={() => setCrosshair(visibleCount ? visibleCount - 1 : null)}
        onBlur={() => setCrosshair(null)}
        onKeyDown={(event) => {
          if (
            visibleCount &&
            (event.key === "ArrowRight" || event.key === "ArrowLeft")
          ) {
            event.preventDefault();
            setCrosshair(
              Math.max(
                0,
                Math.min(
                  visibleCount - 1,
                  index + (event.key === "ArrowRight" ? 1 : -1),
                ),
              ),
            );
          }
        }}
      >
        <title>{signal.label} — observed samples, no imputation</title>
        {[0, 3, 6, 9].map((i) => (
          <line
            key={`v${i}`}
            x1={chart.x(i)}
            x2={chart.x(i)}
            y1={0}
            y2={height}
            className="chart-grid"
          />
        ))}
        {chart.ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={0}
              x2={width - 30}
              y1={chart.y(tick)}
              y2={chart.y(tick)}
              className="chart-grid"
            />
            <text
              x={width - 2}
              y={chart.y(tick) + 3}
              textAnchor="end"
              className="chart-y-label"
            >
              {tick < 1 ? tick.toFixed(2) : tick < 10 ? tick.toFixed(1) : tick}
            </text>
          </g>
        ))}
        {isMap && (
          <>
            <path d={chart.fill} fill="var(--amber)" opacity=".045" />
            <line
              x1={0}
              x2={width - 30}
              y1={chart.y(65)}
              y2={chart.y(65)}
              className="threshold-line"
            />
            <text x={15} y={chart.y(65) - 6} className="threshold-label">
              65 · hypotension threshold
            </text>
          </>
        )}
        <path
          d={chart.path}
          fill="none"
          stroke={isMap ? "var(--amber)" : "var(--cyan)"}
          strokeWidth={isMap ? 2.3 : 1.65}
          vectorEffect="non-scaling-stroke"
          strokeLinejoin="miter"
        />
        {chart.samples.map(({ value: sample, index: i }) =>
          sample !== null ? (
            <circle
              key={i}
              cx={chart.x(i)}
              cy={chart.y(sample)}
              r={i === visibleCount - 1 ? 3.3 : 2}
              fill={isMap ? "var(--amber)" : "var(--cyan)"}
            />
          ) : i < visibleCount ? (
            <g key={i}>
              <line
                x1={chart.x(i) - 3}
                x2={chart.x(i) + 3}
                y1={height / 2 - 3}
                y2={height / 2 + 3}
                stroke="var(--muted)"
              />
              <line
                x1={chart.x(i) - 3}
                x2={chart.x(i) + 3}
                y1={height / 2 + 3}
                y2={height / 2 - 3}
                stroke="var(--muted)"
              />
            </g>
          ) : null,
        )}
        {crosshair !== null && (
          <line
            x1={chart.x(crosshair)}
            x2={chart.x(crosshair)}
            y1={0}
            y2={height}
            stroke="var(--text)"
            strokeDasharray="3 3"
            opacity=".65"
          />
        )}
      </svg>
      <span className="sr-only">
        {index >= 0
          ? `Source ${formatTime(cutoff - 18 + index * 2)}. ${signal.observed[index] ? "Observed" : "Missing"} sample.`
          : "Waiting for the first source sample at +2 seconds."}
      </span>
    </div>
  );
});

export function ReplayControls({
  onPrevious,
  onNext,
}: {
  onPrevious: () => void;
  onNext: () => void;
}) {
  const { playing, position, speed, play, pause, restart, scrub, setSpeed } =
    useReplayStore();
  return (
    <div className="replay-controls">
      <div className="transport">
        <Button
          variant="ghost"
          size="icon"
          aria-label="Previous case"
          onClick={onPrevious}
        >
          <SkipBack size={15} />
        </Button>
        <Button
          className="play-button"
          variant="secondary"
          size="icon"
          aria-label={playing ? "Pause replay" : "Play replay"}
          onClick={playing ? pause : play}
        >
          {playing ? (
            <Pause size={16} />
          ) : (
            <Play size={16} fill="currentColor" />
          )}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Next case"
          onClick={onNext}
        >
          <SkipForward size={15} />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Restart replay"
          onClick={restart}
        >
          <RotateCcw size={14} />
        </Button>
      </div>
      <div className="scrubber">
        <input
          aria-label="Replay timeline"
          type="range"
          min="0"
          max="20"
          step="0.1"
          value={position}
          onChange={(e) => scrub(Number(e.target.value))}
          style={
            { "--progress": `${(position / 20) * 100}%` } as React.CSSProperties
          }
        />
        <div className="scrubber-labels">
          <span>{position.toFixed(1).padStart(4, "0")} s</span>
          <span>20 s window</span>
        </div>
      </div>
      <select
        className="speed-select"
        aria-label="Replay speed"
        value={speed}
        onChange={(e) => setSpeed(Number(e.target.value))}
      >
        {[0.5, 1, 2, 4].map((value) => (
          <option key={value} value={value}>
            {value}×
          </option>
        ))}
      </select>
      <span className="replay-state">
        <span className={`status-dot ${playing ? "cyan" : ""}`} />
        {playing ? "Replaying" : position >= 20 ? "At cutoff" : "Paused"}
      </span>
    </div>
  );
}

export function TelemetryWorkspace({
  window,
  analyzing,
  onAnalyze,
  onPrevious,
  onNext,
}: {
  window: TelemetryWindow;
  analyzing: boolean;
  onAnalyze: () => void;
  onPrevious: () => void;
  onNext: () => void;
}) {
  const position = useReplayStore((s) => s.position);
  const selected = useReplayStore((s) => s.selectedSignals);
  const [crosshair, setCrosshair] = useState<number | null>(null);
  const visibleCount = Math.min(10, Math.max(0, Math.floor(position / 2)));
  const signals = window.signals.filter((signal) =>
    selected.includes(signal.channel),
  );
  const observed = window.signals.reduce(
    (sum, signal) => sum + signal.observed.filter(Boolean).length,
    0,
  );
  const total = window.signals.length * window.timestamps.length;
  const cursor =
    crosshair !== null && crosshair < visibleCount ? crosshair : null;
  return (
    <section
      className="telemetry-workspace panel"
      aria-labelledby="telemetry-heading"
    >
      <div className="panel-toolbar">
        <div>
          <h2 id="telemetry-heading">
            <Activity size={16} /> Live telemetry
          </h2>
          <p className="telemetry-subtitle">
            20-second window · 10 samples · 2 s interval
          </p>
        </div>
        <div className="toolbar-actions">
          <SignalSelector signals={window.signals} />
          <Button
            size="sm"
            onClick={onAnalyze}
            disabled={analyzing || position < 20}
          >
            <Sparkles size={14} />
            {analyzing ? "Analyzing…" : "Analyze window"}
          </Button>
        </div>
      </div>
      <div
        className={`chart-stack ${signals.length > 5 ? "many-signals" : ""}`}
      >
        {signals.map((signal) => (
          <TelemetryChart
            key={signal.channel}
            signal={signal}
            cutoff={window.cutoffSeconds}
            visibleCount={visibleCount}
            crosshair={cursor}
            setCrosshair={setCrosshair}
          />
        ))}
      </div>
      <div className="time-axis">
        <span>Case time</span>
        <div>
          {[0, 3, 6, 9].map((i) => (
            <span
              key={i}
              style={{ left: `calc(10px + (100% - 46px) * ${i / 9})` }}
            >
              {formatTime(window.cutoffSeconds - 18 + i * 2).slice(3)}
            </span>
          ))}
        </div>
      </div>
      <div className="telemetry-quality">
        <span>
          <Check size={12} /> {((observed / total) * 100).toFixed(0)}% observed{" "}
          <span className="subtle-dot">·</span> {total - observed} missing of{" "}
          {total}
        </span>
        <span>
          {cursor !== null ? (
            <>
              Source {window.timestamps[cursor].slice(11, 19)} UTC ·{" "}
              {signals.every((s) => s.observed[cursor])
                ? "Observed"
                : "Missing values at cursor"}
            </>
          ) : (
            <>
              <CircleHelp size={12} /> Hover or focus a trace to inspect
            </>
          )}
        </span>
      </div>
      <ReplayControls onPrevious={onPrevious} onNext={onNext} />
    </section>
  );
}
