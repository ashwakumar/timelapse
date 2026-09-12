"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import {
  Activity,
  ArrowRight,
  AudioLines,
  BookOpen,
  ChartNoAxesCombined,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Command,
  FlaskConical,
  Hourglass,
  Info,
  Layers,
  Play,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import type {
  ForecastHorizon as Horizon,
  ForecastResponse,
  ScenarioMetadata,
  TelemetryWindow,
} from "@/lib/contracts";
import { ForecastResponseSchema } from "@/lib/contracts";
import {
  BaselineInferenceProvider,
  FixtureInferenceProvider,
  OpenTSLMInferenceProvider,
} from "@/lib/providers";
import {
  createTelemetryProvider,
  usesHeldOutTelemetry,
} from "@/lib/configuration";
import { useReducedMotion, useReplayClock } from "@/lib/motion";
import { useReplayStore, type View } from "@/lib/replay-store";
import { formatTime } from "@/lib/utils";
import { Button } from "./ui/button";
import {
  EvidenceDrawer,
  ForecastHorizon,
  InferenceErrorState,
  LoadingAnalysisState,
  ModelStatusBadge,
  ReasoningPanel,
} from "./forecast";
import {
  EvaluationPanel,
  LimitationsPanel,
  ResearchDisclaimer,
} from "./evaluation";
import { TelemetryWorkspace } from "./telemetry";
import { ProblemToProofStrip } from "./workflow";

const ThreeTemporalRibbon = dynamic(() => import("./three-temporal-ribbon"), {
  ssr: false,
  loading: () => <div className="ribbon-fallback" />,
});
const telemetry = createTelemetryProvider();
const fixtureInference = new FixtureInferenceProvider();
const baselineInference = new BaselineInferenceProvider();
const backendInference = new OpenTSLMInferenceProvider();
const navigation: { id: View; label: string; icon: typeof Activity }[] = [
  { id: "replay", label: "Live Replay", icon: Activity },
  { id: "analysis", label: "Model Analysis", icon: Layers },
  { id: "evaluation", label: "Evaluation", icon: ChartNoAxesCombined },
  { id: "evidence", label: "Evidence & Limitations", icon: BookOpen },
];
const guide = [
  {
    title: "A moment of warning can matter.",
    body: "The task: anticipate new sustained MAP below 65 mmHg for at least 60 seconds. Explore the research path below.",
    action: "Show the input",
  },
  {
    title: "The observed input.",
    body: "This is a labeled VitalDB-style fixture: 10 source samples per parameter, observation masks, and a 20-second history.",
    action: "Start replay",
  },
  {
    title: "Watch the measured history.",
    body: "Replay advances only through supplied samples. Missing observations remain gaps. Continue to pause at the analysis cutoff.",
    action: "Pause at cutoff",
  },
  {
    title: "The decision point.",
    body: "The complete 20-second input is available. The model has no access to the future event label.",
    action: "Analyze the window",
  },
  {
    title: "A timed warning.",
    body: "The selected horizon is a categorical forecast. It is not a calibrated probability.",
    action: "Read the explanation",
  },
  {
    title: "Four structured sections.",
    body: "INTERPRET and ANTICIPATE describe observed relationships. ACT asks for signal verification and haemodynamic reassessment.",
    action: "Inspect the evidence",
  },
  {
    title: "Make the forecast inspectable.",
    body: "Review exact inputs, measured changes, missingness, sample identity, provider and checkpoint provenance.",
    action: "Reveal the outcome",
  },
  {
    title: "Now reveal the future.",
    body: "Only Evaluation mode exposes the true event horizon and onset. The current outcomes are fixtures.",
    action: "Compare the baseline",
  },
  {
    title: "Same input. Different approach.",
    body: "Compare the baseline with the prerecorded OpenTSLM-format output. An explanation does not prove a better prediction.",
    action: "Show the data contract",
  },
  {
    title: "TimeNet makes the research reusable.",
    body: "Signals, units, masks and patient-disjoint splits share one contract. A real held-out export and split manifest are still required.",
    action: "Read the limitations",
  },
  {
    title: "Know the boundaries.",
    body: "Research prototype. No prospective validation, no causal evidence, no medication or treatment recommendations.",
    action: "Finish demo",
  },
];

type ProviderChoice = "fixture" | "baseline" | "backend";
export function ClinicalHeader({
  window,
  forecast,
  provider,
  onProvider,
  unavailable,
}: {
  window: TelemetryWindow | null;
  forecast: ForecastResponse | null;
  provider: ProviderChoice;
  onProvider: (provider: ProviderChoice) => void;
  unavailable: boolean;
}) {
  const position = useReplayStore((s) => s.position);
  return (
    <header className="clinical-header">
      <Link className="brand" href="/" aria-label="TimeLapse home">
        <Hourglass size={28} strokeWidth={1.6} />
        <div>
          <span className="wordmark">TimeLapse</span>
          <span className="brand-subtitle">
            Intraoperative Hypotension Forecast
          </span>
        </div>
      </Link>
      <div className="header-context">
        <span className="header-case">{window?.caseId ?? "Loading case"}</span>
        <span className="header-time">
          <Clock3 size={16} />
          {window
            ? formatTime(window.cutoffSeconds - 20 + position)
            : "00:00:00"}
        </span>
      </div>
      <div className="header-status">
        <ModelStatusBadge
          forecast={forecast}
          unavailable={unavailable}
          selectedProvider={provider}
        />
        <select
          className="provider-select"
          aria-label="Inference provider"
          value={provider}
          onChange={(e) => onProvider(e.target.value as ProviderChoice)}
        >
          <option value="fixture" disabled={usesHeldOutTelemetry}>
            Fixture provider
          </option>
          <option value="baseline" disabled={usesHeldOutTelemetry}>
            Baseline provider
          </option>
          <option value="backend">OpenTSLM backend</option>
        </select>
        <span className="research-badge">
          <FlaskConical size={14} /> Research Prototype
        </span>
      </div>
    </header>
  );
}
export function PatientContextStrip({ window }: { window: TelemetryWindow }) {
  const context = window.staticContext;
  const items = [
    ["Age band", context.ageBand],
    ["ASA class", context.asaClass],
    ["Procedure", context.procedureCategory],
    ["Case phase", context.casePhase],
  ];
  return (
    <section
      className="patient-strip"
      aria-label="Static de-identified patient context"
    >
      <span className="patient-label">
        <ShieldCheck size={16} />
        De-identified research data
      </span>
      <dl className="patient-fields">
        {items.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <span className="signal-quality-badge">
        <span className="status-dot cyan" />
        {context.signalQualityStatus === "adequate"
          ? "Signal quality adequate"
          : "Review signal gaps"}
      </span>
    </section>
  );
}
export function FixtureModeBanner({
  window,
}: {
  window: TelemetryWindow | null;
}) {
  return (
    <div className="fixture-banner">
      <Info size={12} />
      <span>
        {window?.source === "vitaldb_held_out_export"
          ? "Held-out VitalDB export · observed input only"
          : usesHeldOutTelemetry
            ? "Loading held-out VitalDB export"
            : "Fixture Replay — awaiting held-out VitalDB export"}
      </span>
      <span className="source-label">
        DATA SOURCE{" "}
        <strong>
          {window?.source === "vitaldb_held_out_export"
            ? "VitalDB export"
            : "VitalDB-style fixture"}
        </strong>
        <span className="subtle-dot">·</span> SPLIT{" "}
        <strong>
          {window?.datasetSplit.replaceAll("_", " ") ?? "Loading"}
        </strong>
      </span>
    </div>
  );
}

export function AppShell() {
  useReplayClock();
  const {
    view,
    setView,
    position,
    playing,
    guideStep,
    setGuideStep,
    setEvidenceOpen,
  } = useReplayStore();
  const [cases, setCases] = useState<ScenarioMetadata[]>([]);
  const [caseIndex, setCaseIndex] = useState(0);
  const [windowData, setWindowData] = useState<TelemetryWindow | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [previous, setPrevious] = useState<Horizon | null>(null);
  const [provider, setProvider] = useState<ProviderChoice>(
    usesHeldOutTelemetry || process.env.NEXT_PUBLIC_DEMO_MODE === "backend"
      ? "backend"
      : process.env.NEXT_PUBLIC_DEMO_MODE === "baseline"
        ? "baseline"
        : "fixture",
  );
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [caseError, setCaseError] = useState<string | null>(null);
  const [ribbonReady, setRibbonReady] = useState(false);
  const [help, setHelp] = useState(false);
  const [workflowStage, setWorkflowStage] = useState<number | null>(null);
  const requestVersion = useRef(0);
  const shellRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  useEffect(() => {
    let active = true;
    telemetry
      .listCases()
      .then((data) => {
        if (active) setCases(data);
      })
      .catch((error) => {
        if (active)
          setCaseError(
            error instanceof Error
              ? error.message
              : "Unable to load the case index",
          );
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    if (!cases[caseIndex]) return;
    let active = true;
    const requestId = ++requestVersion.current;
    telemetry
      .loadCase(cases[caseIndex].sampleId)
      .then(async (window) => {
        if (!active) return;
        setWindowData(window);
        setError(null);
        setCaseError(null);
        setAnalyzing(false);
        setPrevious(null);
        useReplayStore.setState({
          position: 20,
          playing: false,
          selectedSignals: window.signals
            .slice(0, 5)
            .map((signal) => signal.channel),
          evidenceOpen: false,
        });
        if (provider === "fixture" || provider === "baseline") {
          const response = await (
            provider === "fixture" ? fixtureInference : baselineInference
          ).forecast({ requestId: `initial-${requestId}`, window });
          if (active && requestVersion.current === requestId)
            setForecast(ForecastResponseSchema.parse(response));
        } else setForecast(null);
      })
      .catch((error) => {
        if (active)
          setCaseError(
            error instanceof Error ? error.message : "Unable to load this case",
          );
      });
    return () => {
      active = false;
    };
  }, [caseIndex, provider, cases]);
  useEffect(() => {
    if (reduced) return;
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".view-content",
        { opacity: 0.6, y: 9, filter: "blur(1.5px)" },
        {
          opacity: 1,
          y: 0,
          filter: "blur(0px)",
          duration: 0.36,
          ease: "power3.out",
        },
      );
    }, shellRef);
    return () => ctx.revert();
  }, [view, windowData?.sampleId, reduced]);
  useEffect(() => {
    if (reduced) return;
    const id = requestAnimationFrame(() => {
      if (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4)
        return;
      if (window.matchMedia("(max-width: 1023px)").matches) return;
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2", {
        failIfMajorPerformanceCaveat: true,
      });
      if (gl) {
        const debug = gl.getExtension("WEBGL_debug_renderer_info");
        const renderer = debug
          ? String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL))
          : "";
        const software = /swiftshader|software|llvmpipe|softpipe/i.test(
          renderer,
        );
        gl.getExtension("WEBGL_lose_context")?.loseContext();
        if (debug && !software) setRibbonReady(true);
      }
    });
    return () => cancelAnimationFrame(id);
  }, [reduced]);
  const analyze = useCallback(async () => {
    if (!windowData || useReplayStore.getState().position < 20) return;
    const version = ++requestVersion.current;
    setAnalyzing(true);
    setError(null);
    useReplayStore.getState().pause();
    try {
      const selectedProvider =
        provider === "backend"
          ? backendInference
          : provider === "baseline"
            ? baselineInference
            : fixtureInference;
      const [response] = await Promise.all([
        selectedProvider.forecast({
          requestId: `analysis-${version}`,
          window: windowData,
        }),
        new Promise((resolve) => setTimeout(resolve, reduced ? 0 : 580)),
      ]);
      if (version !== requestVersion.current) return;
      const valid = ForecastResponseSchema.parse(response);
      setPrevious(forecast?.horizon ?? null);
      setForecast(valid);
    } catch (error) {
      if (version === requestVersion.current)
        setError(
          error instanceof Error
            ? error.message
            : "Could not generate a forecast.",
        );
    } finally {
      if (version === requestVersion.current) setAnalyzing(false);
    }
  }, [windowData, provider, reduced, forecast]);
  const changeCase = (index: number) => {
    if (!cases.length) return;
    const next = (index + cases.length) % cases.length;
    if (next === caseIndex) return;
    requestVersion.current++;
    setWindowData(null);
    setForecast(null);
    setError(null);
    setAnalyzing(false);
    setCaseIndex(next);
  };
  const changeProvider = (next: ProviderChoice) => {
    requestVersion.current++;
    setProvider(next);
    setForecast(null);
    setError(null);
  };
  const advanceGuide = () => {
    const next = (guideStep ?? -1) + 1;
    if (next >= guide.length) {
      setGuideStep(null);
      setWorkflowStage(null);
      return;
    }
    setGuideStep(next);
    if (next <= 4) setView("replay");
    if (next === 0) setWorkflowStage(0);
    if (next === 1) setWorkflowStage(null);
    if (next === 2) useReplayStore.getState().play();
    if (next === 3) useReplayStore.getState().scrub(20);
    if (next === 4) void analyze();
    if (next === 5) setView("analysis");
    if (next === 6) setEvidenceOpen(true);
    if (next === 7 || next === 8) setView("evaluation");
    if (next === 9) {
      setView("replay");
      setWorkflowStage(2);
    }
    if (next === 10) {
      setView("evidence");
      setWorkflowStage(null);
    }
  };
  const activeForecast =
    position >= 20 && !analyzing && !error ? forecast : null;
  const forecastColumn = (
    <aside className="forecast-column" aria-label="Forecast and reasoning">
      {analyzing ? (
        <LoadingAnalysisState />
      ) : error ? (
        <InferenceErrorState message={error} retry={() => void analyze()} />
      ) : activeForecast ? (
        <>
          <ForecastHorizon forecast={activeForecast} previous={previous} />
          <div className="reasoning-surface panel">
            <ReasoningPanel forecast={activeForecast} />
            <EvidenceDrawer forecast={activeForecast} window={windowData!} />
          </div>
        </>
      ) : (
        <section className="awaiting-forecast panel">
          <Clock3 size={27} />
          <h2>
            {position < 20
              ? "Following the observed history"
              : "Ready for analysis"}
          </h2>
          <p>
            {position < 20
              ? "The forecast becomes available at the 20-second decision cutoff. Continue replay or scrub to the end."
              : "Analyze the complete telemetry window to request a timed forecast."}
          </p>
          {position >= 20 && (
            <Button onClick={() => void analyze()}>
              <Sparkles size={14} /> Analyze window
            </Button>
          )}
        </section>
      )}
    </aside>
  );
  return (
    <div className="app-shell" ref={shellRef}>
      <a href="#main-workspace" className="skip-link">
        Skip to workspace
      </a>
      <ClinicalHeader
        window={windowData}
        forecast={forecast}
        provider={provider}
        onProvider={changeProvider}
        unavailable={Boolean(error)}
      />
      <nav className="main-navigation" aria-label="Workspace views">
        <div className="nav-tabs" role="tablist">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                id={`tab-${item.id}`}
                role="tab"
                tabIndex={view === item.id ? 0 : -1}
                onKeyDown={(event) => {
                  const index = navigation.findIndex(
                    (tab) => tab.id === item.id,
                  );
                  const next =
                    event.key === "ArrowRight"
                      ? (index + 1) % navigation.length
                      : event.key === "ArrowLeft"
                        ? (index + navigation.length - 1) % navigation.length
                        : event.key === "Home"
                          ? 0
                          : event.key === "End"
                            ? navigation.length - 1
                            : null;
                  if (next !== null) {
                    event.preventDefault();
                    setView(navigation[next].id);
                    setWorkflowStage(null);
                    document
                      .getElementById(`tab-${navigation[next].id}`)
                      ?.focus();
                  }
                }}
                aria-selected={view === item.id}
                aria-controls="workspace-tabpanel"
                onClick={() => {
                  setView(item.id);
                  setWorkflowStage(null);
                }}
                className={view === item.id ? "active" : ""}
              >
                <Icon size={17} />
                {item.label}
              </button>
            );
          })}
        </div>
        <Button
          size="sm"
          variant="secondary"
          onClick={() =>
            guideStep === null ? advanceGuide() : setGuideStep(null)
          }
        >
          <Play size={14} />
          {guideStep === null ? "Guided Demo" : "Exit guided demo"}
        </Button>
      </nav>
      {windowData && <PatientContextStrip window={windowData} />}
      <main tabIndex={-1} id="main-workspace" className="main-workspace">
        <div className="workspace-heading">
          <div>
            <h1>
              {view === "evaluation"
                ? "Evaluation"
                : view === "evidence"
                  ? "Evidence & limitations"
                  : view === "analysis"
                    ? "Model analysis"
                    : "Review the observed window"}
            </h1>
            <p>
              {view === "replay"
                ? "Replay the signals, then analyze at the cutoff."
                : view === "analysis"
                  ? "One input window. One timed forecast. Inspect every step."
                  : view === "evaluation"
                    ? "Reveal the outcome and compare both approaches."
                    : "What this research can—and cannot—tell us."}
            </p>
          </div>
          <label className="scenario-control">
            <span>Demonstration scenario</span>
            <select
              aria-label="Demonstration scenario"
              value={caseIndex}
              onChange={(e) => changeCase(Number(e.target.value))}
            >
              {cases.map((scenario, index) => (
                <option key={scenario.sampleId} value={index}>
                  {scenario.title}
                </option>
              ))}
            </select>
          </label>
        </div>
        <FixtureModeBanner window={windowData} />
        {guideStep !== null && (
          <div className="guided-panel" role="status">
            <span className="guide-counter">
              {guideStep + 1}
              <span> / 11</span>
            </span>
            <div>
              <h2>{guide[guideStep].title}</h2>
              <p>
                {guideStep === 1 && windowData?.source !== "fixture"
                  ? `This held-out VitalDB window supplies ${windowData?.signals.length ?? 0} parameters, ten source samples each, and explicit observation masks.`
                  : guideStep === 7 && windowData?.source !== "fixture"
                    ? "Evaluation alone exposes labels and eventual onset, loaded from the versioned evaluation file."
                    : guide[guideStep].body}
              </p>
            </div>
            <Button
              size="sm"
              variant="secondary"
              onClick={advanceGuide}
              disabled={analyzing}
            >
              {guide[guideStep].action}
              <ArrowRight size={14} />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              aria-label="Close guided demo"
              onClick={() => {
                setGuideStep(null);
                setWorkflowStage(null);
              }}
            >
              <X size={17} />
            </Button>
          </div>
        )}
        <div
          className="view-content"
          id="workspace-tabpanel"
          role="tabpanel"
          aria-labelledby={`tab-${view}`}
        >
          {caseError ? (
            <div className="panel case-error" role="alert">
              <h2>Case unavailable</h2>
              <p>{caseError}</p>
              <Button
                onClick={() =>
                  cases.length > 1
                    ? changeCase(caseIndex + 1)
                    : window.location.reload()
                }
              >
                Try loading again
              </Button>
            </div>
          ) : !windowData ? (
            <div className="workspace-loading" role="status">
              <AudioLines size={30} />
              <h2>Preparing the observed window</h2>
              <p>Loading telemetry and observation masks…</p>
              <div className="skeleton" />
            </div>
          ) : view === "evaluation" ? (
            <EvaluationPanel key={windowData.sampleId} window={windowData} />
          ) : view === "evidence" ? (
            <LimitationsPanel isFixture={windowData.source === "fixture"} />
          ) : view === "analysis" ? (
            <div className="analysis-layout">
              <div className="analysis-input panel">
                <div className="panel-toolbar">
                  <h2>
                    <Layers size={18} /> How the model reads the window
                  </h2>
                </div>
                <div className="model-architecture">
                  <span>
                    <Activity size={23} />
                    <strong>Telemetry</strong>
                    <small>20 s · {windowData.signals.length} parameters</small>
                  </span>
                  <ChevronRight size={18} />
                  <span>
                    <Layers size={23} />
                    <strong>Encoder</strong>
                    <small>Time-series features</small>
                  </span>
                  <ChevronRight size={18} />
                  <span>
                    <Command size={23} />
                    <strong>Projector</strong>
                    <small>Aligned embeddings</small>
                  </span>
                  <ChevronRight size={18} />
                  <span>
                    <Sparkles size={23} />
                    <strong>Llama + LoRA</strong>
                    <small>
                      {forecast?.provider === "opentslm_task_trained"
                        ? "Adapter loaded"
                        : "Task tuning pending"}
                    </small>
                  </span>
                </div>
                <div className="analysis-model-note">
                  <Info size={16} />
                  <span>
                    {forecast?.provider === "opentslm_pretrained"
                      ? "Generic OpenTSLM checkpoint — task fine-tuning pending"
                      : forecast?.provider === "opentslm_task_trained"
                        ? `Task-trained adapter ${forecast.checkpointId} · model revision ${forecast.modelId}`
                        : forecast?.provider === "baseline"
                          ? "Architecture overview. The baseline is prerecorded and has no generated explanation."
                          : forecast?.provider === "fixture"
                            ? "Architecture overview. The current output is a deterministic fixture."
                            : "Architecture overview. Analyze a window to verify the loaded model."}
                  </span>
                </div>
                {activeForecast ? (
                  <>
                    <ReasoningPanel forecast={activeForecast} expanded />
                    <EvidenceDrawer
                      forecast={activeForecast}
                      window={windowData}
                    />
                  </>
                ) : (
                  <div className="analysis-empty">
                    <p>
                      Reach the replay cutoff, then analyze the window to reveal
                      the explanation.
                    </p>
                  </div>
                )}
              </div>
              <div>
                {analyzing ? (
                  <LoadingAnalysisState />
                ) : error ? (
                  <InferenceErrorState
                    message={error}
                    retry={() => void analyze()}
                  />
                ) : activeForecast ? (
                  <ForecastHorizon
                    forecast={activeForecast}
                    previous={previous}
                  />
                ) : null}
                <div className="analysis-provenance">
                  <h3>
                    <ShieldCheck size={17} /> Model provenance
                  </h3>
                  <dl>
                    <div>
                      <dt>Provider</dt>
                      <dd>{forecast?.provider ?? provider}</dd>
                    </div>
                    <div>
                      <dt>Checkpoint</dt>
                      <dd>
                        {forecast?.checkpointId ?? "No task-trained checkpoint"}
                      </dd>
                    </div>
                    <div>
                      <dt>Model revision</dt>
                      <dd>{forecast?.modelId ?? "Not loaded"}</dd>
                    </div>
                    <div>
                      <dt>Evaluation</dt>
                      <dd>Inspect version in Evaluation</dd>
                    </div>
                  </dl>
                  <Button
                    onClick={() => void analyze()}
                    disabled={analyzing || position < 20}
                  >
                    <Sparkles size={16} /> Analyze window
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="replay-layout">
              <TelemetryWorkspace
                key={windowData.sampleId}
                window={windowData}
                analyzing={analyzing}
                onAnalyze={() => void analyze()}
                onPrevious={() => changeCase(caseIndex - 1)}
                onNext={() => changeCase(caseIndex + 1)}
              />
              {forecastColumn}
            </div>
          )}
        </div>
        <details
          className="research-story"
          open={workflowStage !== null ? true : undefined}
        >
          <summary>
            How TimeLapse works{" "}
            <span>Problem → data → forecast → evidence</span>
            <ChevronDown size={17} />
          </summary>
          <ProblemToProofStrip
            key={workflowStage ?? "free"}
            activeStage={workflowStage}
            onClose={() => setWorkflowStage(null)}
            isFixture={windowData?.source !== "vitaldb_held_out_export"}
          />
        </details>
      </main>
      <footer className="app-footer">
        <ResearchDisclaimer />
        <span className="temporal-preview">
          {ribbonReady && !reduced ? (
            <ThreeTemporalRibbon
              playing={playing}
              urgent={Boolean(
                forecast && forecast.horizon !== "none_within_15",
              )}
            />
          ) : (
            <span className="ribbon-fallback" />
          )}
        </span>
        <button
          className="footer-help"
          aria-expanded={help}
          onClick={() => setHelp(!help)}
        >
          <CircleHelp size={15} /> About this workspace
        </button>
      </footer>
      {help && (
        <div className="help-panel panel">
          <div>
            <h3>Research, open to inspection.</h3>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Close workspace help"
              onClick={() => setHelp(false)}
            >
              <X size={17} />
            </Button>
          </div>
          <p>
            Inspect the observed samples, analyze at the cutoff, then open
            Evaluation to reveal the eventual outcome. All navigation and
            playback controls support keyboard input.
          </p>
          <p>
            Version 1.0 ·{" "}
            {windowData?.source === "fixture"
              ? "VitalDB-style fixtures"
              : "Held-out VitalDB telemetry"}
            . Inspect model and evaluation provenance before interpreting
            results.
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setView("evidence");
              setHelp(false);
            }}
          >
            Read limitations <ArrowRight size={15} />
          </Button>
        </div>
      )}
    </div>
  );
}
