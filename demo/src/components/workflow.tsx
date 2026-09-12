"use client";

import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import {
  ArrowRight,
  ArrowUpRight,
  Box,
  Check,
  CircleDot,
  Database,
  FlaskConical,
  GitBranch,
  Layers3,
  X,
} from "lucide-react";
import { useReducedMotion } from "@/lib/motion";

const stages = [
  {
    name: "Problem",
    label: "Earlier awareness",
    icon: CircleDot,
    title: "A short window to see what is coming.",
    body: "Unexpected sustained intraoperative hypotension gives the anaesthetist limited time to respond. TimeLapse forecasts the onset of a new event from a recent telemetry window.",
  },
  {
    name: "Data",
    label: "VitalDB telemetry",
    icon: Database,
    title: "Observed signals. Explicit missingness.",
    body: "Open VitalDB surgical telemetry is windowed into 20-second histories and future event labels. Ten source samples per parameter are separated by two seconds. Current demonstration inputs are deterministic fixtures.",
  },
  {
    name: "TimeNet",
    label: "One data contract",
    icon: GitBranch,
    title: "A reusable research connector.",
    body: "TimeNet standardizes signals, units, metadata, observation masks, annotations and tasks. Patient-disjoint splits keep the same patient out of training and held-out evaluation. Real export and split manifest verification are pending in this demo.",
  },
  {
    name: "Train",
    label: "OpenTSLM + LoRA",
    icon: Layers3,
    title: "From time series to structured language.",
    body: "OpenTSLM connects a time-series encoder to a projection layer and a Llama language model. Task-specific LoRA fine-tuning is required before a checkpoint can be labeled Task-Trained. Fixture output here is prerecorded.",
  },
  {
    name: "Evaluate",
    label: "Held-out comparison",
    icon: FlaskConical,
    title: "Compare on the same patients.",
    body: "The baseline and time-series language model must use identical held-out patients and inputs. Evaluation reports episode sensitivity, lead time, false warnings, macro F1, balanced accuracy and a five-class confusion matrix. Final results are pending.",
  },
  {
    name: "Demonstrate",
    label: "Evidence to decision",
    icon: Box,
    title: "Replay. Forecast. Reveal.",
    body: "Inspect the observed input, generate a timed warning, examine the four-part explanation, and deliberately reveal the eventual outcome in Evaluation mode. All fixture content stays labeled throughout.",
  },
];
export function ProblemToProofStrip({
  activeStage,
  onStage,
  onClose,
  isFixture = true,
}: {
  activeStage?: number | null;
  onStage?: (index: number) => void;
  onClose?: () => void;
  isFixture?: boolean;
}) {
  const [openStage, setOpenStage] = useState<number | null>(null);
  const reduced = useReducedMotion();
  const ref = useRef<HTMLElement>(null);
  const selected = activeStage ?? openStage;
  useEffect(() => {
    if (reduced) return;
    gsap.registerPlugin(ScrollTrigger);
    const context = gsap.context(() => {
      gsap.fromTo(
        ".workflow-connector",
        { scaleX: 0.7, transformOrigin: "left" },
        {
          scaleX: 1,
          duration: 0.45,
          ease: "power2.out",
          scrollTrigger: {
            trigger: ref.current,
            start: "top bottom",
            once: true,
          },
        },
      );
    }, ref);
    return () => context.revert();
  }, [reduced]);
  return (
    <section
      className="workflow-strip"
      ref={ref}
      aria-label="Problem-to-proof research workflow"
    >
      <div className="workflow-heading">
        <span>From problem to proof</span>
        <span>
          THE RESEARCH PATH <ArrowUpRight size={12} />
        </span>
      </div>
      <div className="workflow-stages">
        {stages.map((stage, index) => {
          const Icon = stage.icon;
          return (
            <div className="workflow-item" key={stage.name}>
              <button
                aria-expanded={selected === index}
                className={selected === index ? "selected" : ""}
                onClick={() => {
                  const next = selected === index ? null : index;
                  setOpenStage(next);
                  if (onStage && next !== null) onStage(next);
                }}
              >
                <Icon size={17} />
                <span>
                  <strong>{stage.name}</strong>
                  <small>{stage.label}</small>
                </span>
                {index === 0 && <Check size={11} className="workflow-check" />}
              </button>
              {index < stages.length - 1 && (
                <ArrowRight size={13} className="workflow-connector" />
              )}
            </div>
          );
        })}
      </div>
      {selected != null && (
        <div className="workflow-detail">
          <div>
            <strong>{stages[selected].title}</strong>
            <p>
              {isFixture
                ? stages[selected].body
                : stages[selected].body
                    .replace(
                      "Current demonstration inputs are deterministic fixtures.",
                      "Current inputs come from the configured held-out VitalDB export.",
                    )
                    .replace(
                      "Real export and split manifest verification are pending in this demo.",
                      "An input export alone does not verify the split manifest.",
                    )
                    .replace(
                      "Fixture output here is prerecorded.",
                      "The active model state and checkpoint must be verified from each response.",
                    )
                    .replace(
                      "Final results are pending.",
                      "Results and their status come from the configured versioned evaluation file.",
                    )}
            </p>
          </div>
          <button
            className="button button-ghost button-icon"
            aria-label="Close workflow details"
            onClick={() => {
              setOpenStage(null);
              onClose?.();
            }}
          >
            <X size={16} />
          </button>
        </div>
      )}
    </section>
  );
}
