import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import fixtures from "../src/fixtures/scenarios.v1.json";

const reviewDirectory = resolve(process.cwd(), "../.impeccable/review");
const analyzeButton = (page: import("@playwright/test").Page) =>
  page
    .locator(".telemetry-workspace")
    .getByRole("button", { name: "Analyze window", exact: true });

test("complete replay, analysis, evidence, evaluation and limitations flow", async ({
  page,
}) => {
  const errors: string[] = [];
  const evaluationRequests: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "warning" || message.type() === "error")
      errors.push(message.text());
  });
  page.on("request", (request) => {
    if (request.url().includes("/evaluation/"))
      evaluationRequests.push(request.url());
  });
  await page.goto("/");
  await expect(
    page.getByLabel("Demonstration scenario").locator("option"),
  ).toHaveCount(4);
  await expect(analyzeButton(page)).toBeVisible();
  await expect(
    page.getByText("Fixture Replay — awaiting held-out VitalDB export"),
  ).toBeVisible();
  expect(evaluationRequests).toHaveLength(0);
  await expect(page.getByText("Ground truth", { exact: false })).toHaveCount(0);
  await page
    .getByLabel("Demonstration scenario")
    .selectOption({ label: "Stable no-event window" });
  await expect(page.locator(".forecast-main h3")).toHaveText(
    "None within 15 minutes",
  );
  await page
    .getByLabel("Demonstration scenario")
    .selectOption({ label: "Early warning before declining pressure" });
  await expect(page.locator(".forecast-main h3")).toHaveText(
    "Within 5 minutes",
  );
  await page.getByLabel("Replay speed").selectOption("4");
  await page.getByRole("button", { name: "Play replay", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Pause replay" }),
  ).toBeVisible();
  await expect(analyzeButton(page)).toBeDisabled();
  await expect(analyzeButton(page)).toBeEnabled({ timeout: 8000 });
  await analyzeButton(page).click();
  await expect(
    page.getByRole("heading", { name: "Reading the observed window" }),
  ).toBeVisible();
  await expect(page.locator(".forecast-main h3")).toHaveText(
    "Within 5 minutes",
  );
  await page.getByRole("button", { name: "Why this forecast?" }).click();
  await expect(
    page.getByText("Source sample span", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Generated rationale is not causal evidence.", {
      exact: false,
    }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Evaluation", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Prediction meets outcome" }),
  ).toBeVisible();
  await expect(page.locator(".ground-truth")).toContainText(
    "Hypotension within 5 minutes",
  );
  await expect(page.locator(".onset-marker")).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Baseline", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "OpenTSLM", exact: true }),
  ).toBeVisible();
  expect(evaluationRequests.length).toBeGreaterThan(0);
  await page.getByRole("tab", { name: "Live Replay" }).click();
  await expect(page.locator(".ground-truth, .onset-marker")).toHaveCount(0);
  await page.getByRole("tab", { name: "Evidence & Limitations" }).click();
  await expect(
    page.getByRole("heading", { name: "Read before interpreting a forecast" }),
  ).toBeVisible();
  await expect(
    page.getByText("Physical device identities are unavailable", {
      exact: false,
    }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("backend failure preserves replay and permits recovery", async ({
  page,
}) => {
  await page.route("**/api/inference", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: '{"error":"model_unavailable"}',
    }),
  );
  await page.goto("/");
  await expect(analyzeButton(page)).toBeVisible();
  await page.getByLabel("Inference provider").selectOption("backend");
  await analyzeButton(page).click();
  await expect(
    page.getByText("Model unavailable — telemetry replay remains active"),
  ).toBeVisible();
  await expect(page.getByLabel("Replay timeline")).toBeEnabled();
  await page.getByRole("button", { name: "Play replay", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Pause replay" }),
  ).toBeVisible();
  await page.getByLabel("Inference provider").selectOption("fixture");
  await expect(page.locator(".forecast-main h3")).toHaveText(
    "Within 5 minutes",
  );
});

test("keyboard tabs, signals and linked crosshair work without hover", async ({
  page,
}) => {
  await page.goto("/");
  await expect(analyzeButton(page)).toBeVisible();
  const replayTab = page.getByRole("tab", { name: "Live Replay" });
  await replayTab.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Model Analysis" })).toBeFocused();
  await expect(
    page.getByRole("tab", { name: "Model Analysis" }),
  ).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Home");
  await expect(replayTab).toBeFocused();
  await page.getByRole("button", { name: "Signals 5" }).click();
  const checkbox = page.getByRole("checkbox", { name: "EtCO2 mmHg" });
  await checkbox.focus();
  await page.keyboard.press("Space");
  await expect(checkbox).toBeChecked();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: "Signals 6" })).toBeFocused();
  const map = page.getByRole("img", { name: /^MAP,/ });
  await map.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(
    page.getByText("Source 08:00:16 UTC", { exact: false }),
  ).toBeVisible();
  await expect(page.getByLabel("MAP: 74 mmHg", { exact: true })).toBeVisible();
});

test("rejects stale and malformed backend output without exposing future labels", async ({
  page,
}) => {
  let heldRoute: import("@playwright/test").Route | undefined;
  const requests: unknown[] = [];
  await page.route("**/api/inference", (route) => {
    requests.push(route.request().postDataJSON());
    heldRoute = route;
  });
  await page.goto("/");
  await expect(analyzeButton(page)).toBeVisible();
  await page.getByLabel("Inference provider").selectOption("backend");
  await analyzeButton(page).click();
  await expect.poll(() => Boolean(heldRoute)).toBe(true);
  await page
    .getByLabel("Demonstration scenario")
    .selectOption({ label: "Stable no-event window" });
  const input = requests[0] as {
    requestId: string;
    window: { sampleId: string };
  };
  expect(JSON.stringify(input)).not.toMatch(/groundTruth|actualOnset/);
  const original = fixtures.scenarios[0].forecasts[0];
  await heldRoute!.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      ...original,
      requestId: input.requestId,
      provider: "opentslm_pretrained",
    }),
  });
  await expect(page.locator(".forecast-main")).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Ready for analysis" }),
  ).toBeVisible();
  await page.unroute("**/api/inference");
  await page.route("**/api/inference", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ...original, horizon: "within_20" }),
    }),
  );
  await analyzeButton(page).click();
  await expect(
    page.getByRole("heading", { name: "Model unavailable", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".forecast-main")).toHaveCount(0);
});

for (const [name, viewport] of [
  ["desktop", { width: 1440, height: 900 }],
  ["tablet", { width: 1024, height: 768 }],
  ["portrait", { width: 768, height: 1024 }],
] as const) {
  test(`readable ${name} viewport without clipping`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");
    await expect(analyzeButton(page)).toBeVisible();
    await expect(page.locator(".forecast-main h3")).toHaveText(
      "Within 5 minutes",
    );
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    expect(
      await page
        .locator(".reasoning-block p")
        .first()
        .evaluate((el) => parseFloat(getComputedStyle(el).fontSize)),
    ).toBeGreaterThanOrEqual(14);
    mkdirSync(reviewDirectory, { recursive: true });
    await page.screenshot({
      path: resolve(reviewDirectory, `${name}.png`),
      fullPage: true,
    });
    if (name === "desktop") {
      expect(
        await page.evaluate(() => document.documentElement.scrollHeight),
      ).toBeLessThanOrEqual(900);
      for (const [view, label] of [
        ["analysis", "Model Analysis"],
        ["evaluation", "Evaluation"],
        ["evidence", "Evidence & Limitations"],
      ] as const) {
        await page.getByRole("tab", { name: label, exact: true }).click();
        await expect(
          page.getByRole("tab", { name: label, exact: true }),
        ).toHaveCSS("border-bottom-color", "rgb(22, 121, 112)");
        if (view === "evaluation")
          await expect(
            page.getByRole("heading", { name: "Prediction meets outcome" }),
          ).toBeVisible();
        await page.screenshot({
          path: resolve(reviewDirectory, `${view}.png`),
          fullPage: true,
        });
        if (view === "analysis")
          expect(
            await page.evaluate(() => document.documentElement.scrollHeight),
          ).toBeLessThanOrEqual(900);
      }
    }
  });
}

test("reduced motion disables transitions and the WebGL scene", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(analyzeButton(page)).toBeVisible();
  await expect(page.locator("canvas")).toHaveCount(0);
  expect(
    await page
      .locator(".evidence-trigger")
      .evaluate((el) => getComputedStyle(el).transitionDuration),
  ).toBe("1e-05s");
  await analyzeButton(page).click();
  await expect(page.locator(".forecast-main h3")).toHaveText(
    "Within 5 minutes",
  );
  await expect(page.locator(".reasoning-block").first()).toHaveCSS(
    "opacity",
    "1",
  );
});

test("guided demo advances predictably through the complete story", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(analyzeButton(page)).toBeVisible();
  await page.getByRole("button", { name: "Guided Demo", exact: true }).click();
  for (const label of [
    "Show the input",
    "Start replay",
    "Pause at cutoff",
    "Analyze the window",
    "Read the explanation",
    "Inspect the evidence",
    "Reveal the outcome",
    "Compare the baseline",
    "Show the data contract",
    "Read the limitations",
    "Finish demo",
  ]) {
    await page
      .locator(".guided-panel")
      .getByRole("button", { name: label, exact: true })
      .click();
  }
  await expect(page.locator(".guided-panel")).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Read before interpreting a forecast" }),
  ).toBeVisible();
});
