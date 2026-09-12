---
name: TimeLapse
description: A readable research workspace for timed hypotension forecasts.
colors:
  primary: "#086b62"
  telemetry: "#167970"
  background: "#f5f7f8"
  surface: "#ffffff"
  ink: "#1b2d33"
  secondary-ink: "#445c65"
  muted-ink: "#62747c"
  rule: "#d9e3e6"
  warning: "#9e601a"
  warning-surface: "#fff8ed"
  immediate: "#ab4139"
  verified: "#33724c"
typography:
  headline:
    fontFamily: "Geist Sans, sans-serif"
    fontSize: "26px"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "-0.03em"
  body:
    fontFamily: "Geist Sans, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.55
  title:
    fontFamily: "Geist Sans, sans-serif"
    fontSize: "16px"
    fontWeight: 600
  measurement:
    fontFamily: "Geist Mono, monospace"
    fontSize: "25px"
    fontWeight: 450
    lineHeight: 1.15
    letterSpacing: "-0.025em"
  label:
    fontFamily: "Geist Sans, sans-serif"
    fontSize: "12px"
    fontWeight: 500
rounded:
  control: "7px"
  surface: "14px"
spacing:
  compact: "8px"
  control: "12px"
  group: "20px"
  inset: "24px"
  desktop: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
    padding: "10px 15px"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "10px 15px"
  panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.surface}"
  forecast:
    backgroundColor: "{colors.warning-surface}"
    textColor: "{colors.warning}"
    rounded: "{rounded.surface}"
---

# Design System: TimeLapse

## Overview

**Creative North Star: "The Scientific Worksheet"**

White working surfaces, clear dark type and quiet measurement traces make the research task easy to follow. A timed forecast is the decision output; evidence stays close without competing with it. The user explicitly replaced the original dark direction with this light, readable system.

**Key Characteristics:**

- Readable prose and explicit actions.
- Continuous synchronized telemetry.
- Progressive disclosure of secondary evidence.
- Restrained, interruptible motion.

## Colors

Teal identifies observed telemetry and the primary action. Amber identifies a categorical warning, with red reserved for the shortest horizon. Dark ink and muted ink maintain the hierarchy on white surfaces. Green is reserved for verified evaluation correctness; a no-event forecast uses teal.

**The State Meaning Rule.** Color must agree with the visible text and icon; it is never the only representation of risk.

## Typography

Geist Sans carries interface text and reading copy. Geist Mono is used for measurements, timestamps and structured output. Reading copy is 14–15px in the main flow and expands to 15px in Model Analysis. Major forecast text is 26–34px; MAP is 46px. Measurement ticks and metadata may be 10–12px, but the primary explanation is never miniature text.

**The Reading Rule.** Simplify content and disclose secondary detail before reducing prose size.

## Layout

The desktop workspace uses a 32px outer inset, a dominant telemetry column and a narrower forecast column separated by 20px. Below 1024px, the forecast precedes the telemetry and reasoning follows it. Phone layouts stack controls and preserve horizontal safety. Research narrative and expanded evidence may scroll; the default desktop replay and analysis prioritize one complete readable viewport.

## Elevation & Depth

Working surfaces are flat with fine single borders. Shadows are reserved for temporary popovers and help surfaces. The signal grid is a measurement aid, not decorative background texture.

## Shapes

Controls use a 7px radius and working surfaces a 14px radius. Small status shapes communicate state alongside text. Avoid nested visual containers that fragment a continuous signal view.

## Components

Primary actions use deep teal, white text, visible hover feedback and a 3px focus outline. Tabs use one teal underline with keyboard arrow navigation. Signal selectors use native checkboxes inside a focus-managed popover. Charts retain physical source values and missing-data gaps; their crosshairs support both pointer and keyboard use.

View and scenario changes use a 360ms `power3.out` reveal, resolving a small blur and vertical offset. Forecast changes use 300ms and evidence/reasoning use 240ms. Animation always begins from legible content and is removed under reduced motion. The optional temporal visualization is lazy, nonessential and disabled for reduced-motion or lower-power contexts.

## Do's and Don'ts

- Do keep the exact forecast category and fixture provenance explicit.
- Do prioritize measured telemetry, forecast and reassessment.
- Do disclose the full structured explanation and research limitations.
- Don't hide essential meaning behind hover alone.
- Don't use dense miniature prose to fit a viewport.
- Don't use decorative clinical signals, stock imagery or continuous flashing.
