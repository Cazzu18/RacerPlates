"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchMenu, predict } from "../(lib)/api";
import type { Meal, PredictResponse, SatisfactionLabel } from "../(lib)/types";

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

type MealWithPrediction = Meal & {
  prediction?: PredictResponse;
};

export default function DashboardPage() {
  const [meals, setMeals] = useState<MealWithPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [showVegan, setShowVegan] = useState(false);
  const [showVegetarian, setShowVegetarian] = useState(false);
  const [threshold, setThreshold] = useState(0.0);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const menu = await fetchMenu();

        const withPreds: MealWithPrediction[] = await Promise.all(
          menu.map(async (meal) => {
            try {
              const prediction = await predict({
                name: meal.name,
                ingredients: "", // not in /menu payload currently
                calories: meal.calories ?? undefined,
                protein: meal.protein ?? undefined,
                fat: meal.fat ?? undefined,
                sugar: meal.sugar ?? undefined,
                sodium: meal.sodium ?? undefined,
                carbohydrates: meal.carbohydrates ?? undefined,
                fiber: meal.fiber ?? undefined,
                diet_key: meal.diet_key,
                meal_time: meal.meal_time ?? undefined,
                is_vegan: meal.is_vegan,
                is_vegetarian: meal.is_vegetarian,
                is_mindful: meal.is_mindful,
              });
              return { ...meal, prediction };
            } catch {
              return { ...meal };
            }
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
      if (threshold > 0 && m.prediction) {
        if (m.prediction.label !== 2) return false;
        if (m.prediction.probability < threshold) return false;
      }
      return true;
    });
  }, [meals, showVegan, showVegetarian, threshold]);

  const topFavorites = useMemo(() => {
    return [...meals]
      .filter((m) => m.prediction?.label === 2)
      .sort(
        (a, b) =>
          (b.prediction?.probability ?? 0) - (a.prediction?.probability ?? 0)
      )
      .slice(0, 5);
  }, [meals]);

  return (
    <main className="min-h-screen p-6 bg-slate-50">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold mb-1">
              Interactive Menu Dashboard
            </h1>
            <p className="text-sm text-slate-600 max-w-2xl">
              Browse the current menu, see predicted satisfaction for each item,
              and filter for vegan/vegetarian or high-confidence favorites.
            </p>
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
                const pred = meal.prediction;
                const label = pred?.label ?? 1;
                const score =
                  pred && pred.label === 2
                    ? Math.round(pred.probability * 100)
                    : pred && pred.label === 1
                    ? Math.round(pred.probability * 100)
                    : pred
                    ? Math.round((1 - pred.probability) * 100)
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
                      {pred ? (
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
                {topFavorites.map((meal) => (
                  <li key={meal.id}>
                    <div className="flex justify-between gap-2">
                      <span>{meal.name}</span>
                      <span className="text-xs text-slate-600">
                        {Math.round(
                          (meal.prediction?.probability ?? 0) * 100
                        )}
                        % like
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
