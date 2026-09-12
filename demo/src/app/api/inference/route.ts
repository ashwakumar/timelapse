import { NextResponse } from "next/server";
import {
  ForecastResponseSchema,
  InferenceRequestSchema,
} from "../../../lib/contracts";

const INFERENCE_TIMEOUT_MS = 12_000;

function unavailable(message = "OpenTSLM backend is not configured.") {
  return NextResponse.json(
    { error: "model_unavailable", message },
    { status: 503, headers: { "cache-control": "no-store" } },
  );
}

export async function POST(request: Request) {
  const apiUrl = process.env.OPENTSLM_API_URL;
  const apiToken = process.env.OPENTSLM_API_TOKEN;
  if (!apiUrl) return unavailable();

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const parsedRequest = InferenceRequestSchema.safeParse(payload);
  if (!parsedRequest.success) {
    return NextResponse.json(
      {
        error: "invalid_inference_request",
        issues: parsedRequest.error.flatten(),
      },
      { status: 400 },
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), INFERENCE_TIMEOUT_MS);
  try {
    const upstream = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json",
        ...(apiToken ? { authorization: `Bearer ${apiToken}` } : {}),
      },
      body: JSON.stringify(parsedRequest.data),
      signal: controller.signal,
      cache: "no-store",
    });

    if (!upstream.ok) {
      return unavailable(`OpenTSLM backend returned HTTP ${upstream.status}.`);
    }

    const parsedResponse = ForecastResponseSchema.safeParse(
      await upstream.json(),
    );
    if (!parsedResponse.success) {
      return NextResponse.json(
        { error: "invalid_inference_response" },
        { status: 502, headers: { "cache-control": "no-store" } },
      );
    }

    const forecast = parsedResponse.data;
    const input = parsedRequest.data;
    if (!forecast.provider.startsWith("opentslm_")) {
      return NextResponse.json(
        { error: "invalid_inference_response" },
        { status: 502, headers: { "cache-control": "no-store" } },
      );
    }
    if (
      forecast.requestId !== input.requestId ||
      forecast.sampleId !== input.window.sampleId ||
      forecast.caseId !== input.window.caseId ||
      forecast.cutoffSeconds !== input.window.cutoffSeconds ||
      forecast.datasetSplit !== input.window.datasetSplit
    ) {
      return NextResponse.json(
        { error: "stale_inference_response" },
        { status: 409, headers: { "cache-control": "no-store" } },
      );
    }

    return NextResponse.json(forecast, {
      headers: { "cache-control": "no-store" },
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      return unavailable("OpenTSLM inference timed out.");
    }
    return unavailable("OpenTSLM backend could not be reached.");
  } finally {
    clearTimeout(timeout);
  }
}
