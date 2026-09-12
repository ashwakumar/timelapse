# TimeLapse

<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
User-specified Next.js 16 App Router, TypeScript strict, React, Tailwind, shadcn/ui primitives, GSAP, React Three Fiber, D3, Zustand, Zod, Lucide, Vitest, Testing Library and Playwright. Isolate all application work in `demo/`.

## Users
Anaesthetists reviewing recent intraoperative telemetry; hackathon judges following the research demonstration at a 1440×900 presentation viewport.

## Product Purpose
Forecast the tightest horizon containing a new sustained hypotension event: MAP below 65 mmHg for at least 60 seconds, from ten source samples in a 20-second window. Show observed evidence and a signal-quality reassessment prompt without prescribing treatment.

## Operating Context
User-controlled replay, analysis, evidence inspection, evaluation reveal and limitations. Ground truth remains outside live inputs. Static de-identified context is separated from time-varying signals.

## Capabilities and Constraints
Five exact forecast categories; INTERPRET, ANTICIPATE, ACT, ANSWER response structure. Four repeatable demonstration scenarios. No causal claims, invented probabilities, medication advice, identifying data or generic pretrained model mislabeled as task-trained. Existing Python connector and training code must remain untouched.

## Brand Commitments
TimeLapse. The user's latest explicit instruction supersedes the original dark aesthetic: rebuild the entire frontend with a white/light background, cleaner composition, larger readable type, understandable flows and restrained polished animations. Geist remains the interface and measurement typeface. Teal marks observed telemetry and actions; amber marks categorical warning; red is reserved for immediate risk. No stock imagery. Progressive disclosure replaces dense secondary chrome.

## Evidence on Hand
Python connector and training implementation exists under `timelapse/`. No verified held-out frontend export or final evaluation metrics supplied. Frontend fixtures must remain visibly labeled. Generic upstream checkpoint files are not evidence of completed task fine-tuning.

## Product Principles
Separate observation, forecast and ground truth. Preserve provenance. Keep replay usable without inference. Provide no treatment prescriptions. Expose research limitations.

## Accessibility & Inclusion
WCAG AA, keyboard controls, reduced motion, accessible chart summaries and risk communicated in text and icons as well as color. Desktop, landscape tablet and portrait tablet support.
