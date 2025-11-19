import { Meal, PredictRequest, PredictResponse } from "./types";

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function predict(meal: PredictRequest): Promise<PredictResponse> {
  const res = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(meal),
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error("Prediction failed");
  }

  return res.json();
}

export async function fetchMenu(): Promise<Meal[]> {
  const res = await fetch(`${BACKEND_URL}/menu`, {
    method: "GET",
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error("Failed to load menu");
  }

  return res.json();
}
