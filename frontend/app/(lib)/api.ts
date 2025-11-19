import {
  Meal,
  ModelVariant,
  PredictRequest,
  PredictResponse,
} from "./types";

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export const MODEL_LABELS: Record<ModelVariant, string> = {
  sbert_fusion_mlp: "MLP Fusion Ensemble",
  oracle_knn_embeddings: "Oracle SBERT KNN",
};

const MODEL_PATHS: Record<ModelVariant, string[]> = {
  sbert_fusion_mlp: ["sbert_fusion_mlp", "fusion"],
  oracle_knn_embeddings: ["oracle_knn_embeddings", "oracle-knn"],
};

async function postJson(url: string, payload: PredictRequest & { model_name?: string }) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
}

export async function predict(
  meal: PredictRequest,
  model: ModelVariant = "sbert_fusion_mlp"
): Promise<PredictResponse> {
  const payload = { ...meal, model_name: model };
  const variantPaths = MODEL_PATHS[model] ?? [model];
  const candidates = [
    ...variantPaths.map((slug) => `${BACKEND_URL}/predict/${slug}`),
    `${BACKEND_URL}/predict?model=${model}`,
    `${BACKEND_URL}/predict`,
  ];

  let lastError: Error | null = null;
  for (const url of candidates) {
    try {
      const res = await postJson(url, payload);
      if (!res.ok) {
        if (res.status === 404 || res.status === 405) {
          continue;
        }
        throw new Error(`Prediction failed (${res.status})`);
      }
      const data = (await res.json()) as PredictResponse;
      return { ...data, model };
    } catch (err) {
      lastError =
        err instanceof Error ? err : new Error("Prediction request failed");
    }
  }

  throw lastError ?? new Error("Prediction failed");
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
