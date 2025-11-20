"use client";

import { useEffect, useMemo, useState } from "react";
import { MODEL_LABELS, fetchMenu, predict } from "../(lib)/api";
import type {
  Meal,
  ModelVariant,
  PredictResponse,
  SatisfactionLabel,
} from "../(lib)/types";

const LABEL_TEXT: Record<SatisfactionLabel, string> = {
  0: "Dislike",
  1: "Neutral",
  2: "Like",
};

const LABEL_COLOR_BG: Record<SatisfactionLabel, string> = {
  0: "bg-red-50",
  1: "bg-gray-50",
  2: "bg-green-50",
};

const MODEL_OPTIONS: ModelVariant[] = [
  "sbert_fusion_mlp",
  "oracle_knn_embeddings",
];

type MealWithPrediction = Meal & {
  fusionPrediction?: PredictResponse;
  oraclePrediction?: PredictResponse;
};

export default function DashboardPage() {
  const [meals, setMeals] = useState<MealWithPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [showVegan, setShowVegan] = useState(false);
  const [showVegetarian, setShowVegetarian] = useState(false);
  const [threshold, setThreshold] = useState(0.0);
  const [activeModel, setActiveModel] =
    useState<ModelVariant>("sbert_fusion_mlp");

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const menu = await fetchMenu();

        const withPreds: MealWithPrediction[] = await Promise.all(
          menu.map(async (meal) => {
            const payload = {
              name: meal.name,
              ingredients: meal.ingredients ?? "",
              allergens: meal.allergens ?? "",
              station: meal.station ?? "",
              diet_key: meal.diet_key,
              calories: meal.calories ?? undefined,
              protein: meal.protein ?? undefined,
              fat: meal.fat ?? undefined,
              sugar: meal.sugar ?? undefined,
              sodium: meal.sodium ?? undefined,
              carbohydrates: meal.carbohydrates ?? undefined,
              fiber: meal.fiber ?? undefined,
              iron: meal.iron ?? undefined,
              calcium: meal.calcium ?? undefined,
              potassium: meal.potassium ?? undefined,
              meal_time: meal.meal_time ?? undefined,
              is_vegan: meal.is_vegan,
              is_vegetarian: meal.is_vegetarian,
              is_mindful: meal.is_mindful,
            };

            const [fusionPrediction, oraclePrediction] = await Promise.allSettled([
              predict(payload, "sbert_fusion_mlp"),
              predict(payload, "oracle_knn_embeddings"),
            ]);

            return {
              ...meal,
              fusionPrediction:
                fusionPrediction.status === "fulfilled"
                  ? fusionPrediction.value
                  : undefined,
              oraclePrediction:
                oraclePrediction.status === "fulfilled"
                  ? oraclePrediction.value
                  : undefined,
            };
          })
        );

        setMeals(withPreds);
      } catch (err) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Failed to load dashboard");
        }
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  const filteredMeals = useMemo(() => {
    return meals.filter((m) => {
      if (showVegan && !m.is_vegan) return false;
      if (showVegetarian && !m.is_vegetarian) return false;
      const activePrediction =
        activeModel === "oracle_knn_embeddings"
          ? m.oraclePrediction
          : m.fusionPrediction;
      if (threshold > 0) {
        if (!activePrediction || activePrediction.label !== 2) return false;
        if (activePrediction.probability < threshold) return false;
      }
      return true;
    });
  }, [meals, showVegan, showVegetarian, threshold, activeModel]);

  const topFavorites = useMemo(() => {
    const picks = meals
      .map((meal) => {
        const pred =
          activeModel === "oracle_knn_embeddings"
            ? meal.oraclePrediction
            : meal.fusionPrediction;
        return { meal, pred };
      })
      .filter((entry) => entry.pred?.label === 2)
      .sort(
        (a, b) => (b.pred?.probability ?? 0) - (a.pred?.probability ?? 0)
      )
      .slice(0, 5);
    return picks;
  }, [meals, activeModel]);

  return (
    <main className="min-h-screen px-6 pb-6 pt-16">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div className="flex flex-col gap-3">
            <h1 className="text-3xl font-semibold mb-1">
              Interactive Menu Dashboard
            </h1>
            <p className="text-sm text-slate-600 max-w-2xl">
              Browse the current menu, see predicted satisfaction for each item,
              and filter for vegan/vegetarian or high-confidence favorites.
            </p>
            <div className="flex flex-wrap gap-2 text-xs text-slate-600">
              <span className="uppercase tracking-wide">Model view:</span>
              {MODEL_OPTIONS.map((variant) => (
                <button
                  key={variant}
                  className={`px-3 py-1 rounded-full border ${
                    activeModel === variant
                      ? "bg-black text-white border-black"
                      : "border-slate-300 bg-white"
                  }`}
                  onClick={() => setActiveModel(variant)}
                  type="button"
                >
                  {MODEL_LABELS[variant]}
                </button>
              ))}
            </div>
          </div>
        </header>

        <section className="grid md:grid-cols-[2fr,1fr] gap-6 items-start">
          <div className="space-y-4">
            <div className="flex flex-wrap gap-4 items-center mb-2">
              <label className="inline-flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={showVegan}
                  onChange={(e) => setShowVegan(e.target.checked)}
                />
                Vegan only
              </label>
              <label className="inline-flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={showVegetarian}
                  onChange={(e) => setShowVegetarian(e.target.checked)}
                />
                Vegetarian only
              </label>
              <div className="flex items-center gap-2 text-sm">
                <span>Min &quot;Like&quot; confidence:</span>
                <select
                  className="border rounded px-2 py-1 text-xs"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                >
                  <option value={0}>Any</option>
                  <option value={0.5}>50%</option>
                  <option value={0.7}>70%</option>
                  <option value={0.85}>85%</option>
                </select>
              </div>
            </div>

            {loading && (
              <p className="text-sm text-slate-600">Loading menu...</p>
            )}
            {error && (
              <p className="text-sm text-red-600">Error: {error}</p>
            )}

            <div className="grid gap-3">
              {filteredMeals.map((meal) => {
                const activePrediction =
                  activeModel === "oracle_knn_embeddings"
                    ? meal.oraclePrediction
                    : meal.fusionPrediction;
                const label = activePrediction?.label ?? 1;
                const score =
                  activePrediction && activePrediction.label === 2
                    ? Math.round(activePrediction.probability * 100)
                    : activePrediction && activePrediction.label === 1
                    ? Math.round(activePrediction.probability * 100)
                    : activePrediction
                    ? Math.round((1 - activePrediction.probability) * 100)
                    : null;

                return (
                  <div
                    key={meal.id}
                    className={`border rounded-lg p-3 flex flex-col md:flex-row md:items-center md:justify-between gap-3 ${LABEL_COLOR_BG[label as SatisfactionLabel]}`}
                  >
                    <div>
                      <div className="font-medium text-sm">{meal.name}</div>
                      <div className="text-xs text-slate-600 mt-1">
                        {meal.station && <span>{meal.station} · </span>}
                        {meal.calories != null && (
                          <span>{meal.calories} kcal · </span>
                        )}
                        <span>
                          {meal.protein ?? "?"}g protein,{" "}
                          {meal.fat ?? "?"}g fat, {meal.sugar ?? "?"}g sugar
                        </span>
                      </div>
                      <div className="text-xs text-slate-500 mt-1">
                        {meal.diet_key}
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      {activePrediction ? (
                        <>
                          <div className="text-right">
                            <div className="text-xs uppercase tracking-wide text-slate-500">
                              Predicted
                            </div>
                            <div className="font-semibold">
                              {LABEL_TEXT[label as SatisfactionLabel]}
                            </div>
                            {score != null && (
                              <div className="text-xs text-slate-600">
                                Score: {score}%
                              </div>
                            )}
                          </div>
                          <div className="flex flex-col text-xs text-slate-600">
                            <ModelBadge
                              label="Fusion"
                              prediction={meal.fusionPrediction}
                            />
                            <ModelBadge
                              label="Oracle KNN"
                              prediction={meal.oraclePrediction}
                            />
                          </div>
                        </>
                      ) : (
                        <div className="text-xs text-slate-500">
                          Prediction unavailable
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {!loading && !error && filteredMeals.length === 0 && (
                <p className="text-sm text-slate-600">
                  No meals match the current filters.
                </p>
              )}
            </div>
          </div>

          <aside className="space-y-3">
            <div className="border rounded-lg p-4 bg-white shadow-sm">
              <h2 className="font-semibold mb-2 text-sm">
                Top 5 predicted favorites
              </h2>
              {topFavorites.length === 0 && (
                <p className="text-xs text-slate-600">
                  Predictions still loading or no favorites yet.
                </p>
              )}
              <ol className="space-y-2 text-sm list-decimal list-inside">
                {topFavorites.map(({ meal, pred }) => (
                  <li key={meal.id}>
                    <div className="flex justify-between gap-2">
                      <span>{meal.name}</span>
                      <span className="text-xs text-slate-600">
                        {Math.round((pred?.probability ?? 0) * 100)}% like
                      </span>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </aside>
        </section>
      </div>
    </main>
  );
}

type ModelBadgeProps = {
  label: string;
  prediction?: PredictResponse;
};

function ModelBadge({ label, prediction }: ModelBadgeProps) {
  if (!prediction) {
    return (
      <span className="text-slate-400">
        {label}: <span className="font-mono">—</span>
      </span>
    );
  }

  const pct = Math.round(prediction.probability * 100);
  const labelText = LABEL_TEXT[prediction.label];

  return (
    <span className="text-slate-600">
      {label}: {labelText} ({pct}%)
    </span>
  );
}
