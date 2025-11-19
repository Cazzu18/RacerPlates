"use client";

import { useState } from "react";
import { predict } from "./(lib)/api";
import type { PredictResponse, SatisfactionLabel } from "./(lib)/types";

const LABEL_TEXT: Record<SatisfactionLabel, string> = {
  0: "Dislike",
  1: "Neutral",
  2: "Like",
};

const LABEL_COLOR: Record<SatisfactionLabel, string> = {
  0: "bg-red-100 text-red-800 border-red-300",
  1: "bg-gray-100 text-gray-800 border-gray-300",
  2: "bg-green-100 text-green-800 border-green-300",
};

export default function Home() {
  const [form, setForm] = useState<{
    name: string;
    ingredients: string;
    calories: number | "";
    protein: number | "";
    fat: number | "";
    sugar: number | "";
    sodium: number | "";
    diet_key: string;
    meal_time: string;
    is_vegan: boolean;
    is_vegetarian: boolean;
    is_mindful: boolean;
  }>({
    name: "Cheese Pizza",
    ingredients: "dough, tomato sauce, mozzarella cheese",
    calories: 320,
    protein: 14,
    fat: 12,
    sugar: 4,
    sodium: 600,
    diet_key: "standard",
    meal_time: "dinner",
    is_vegan: false,
    is_vegetarian: true,
    is_mindful: false,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    setResult(null);

    try {
      const payload = {
        name: form.name,
        ingredients: form.ingredients,
        calories: Number(form.calories) || 0,
        protein: Number(form.protein) || 0,
        fat: Number(form.fat) || 0,
        sugar: Number(form.sugar) || 0,
        sodium: Number(form.sodium) || 0,
        diet_key: form.diet_key,
        meal_time: form.meal_time,
        is_vegan: form.is_vegan,
        is_vegetarian: form.is_vegetarian,
        is_mindful: form.is_mindful,
      };

      const prediction: PredictResponse = await predict(payload);
      setResult(prediction);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Prediction failed");
      }
    } finally {
      setLoading(false);
    }
  }

  const classes = result?.classes ?? [0, 1, 2];
  const proba = result?.proba_per_class ?? [];

  return (
    <main className="min-h-screen p-6 flex flex-col items-center bg-slate-50">
      <div className="w-full max-w-5xl grid md:grid-cols-2 gap-8">
        <section>
          <h1 className="text-3xl font-semibold mb-2">
            Meal Satisfaction Predictor
          </h1>
          <p className="text-sm text-slate-600 mb-6">
            Enter a meal and see whether students are likely to dislike,
            feel neutral, or like it—based on your trained model.
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Meal name</label>
              <input
                className="w-full border rounded px-3 py-2 text-sm"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">
                Ingredients
              </label>
              <textarea
                className="w-full border rounded px-3 py-2 text-sm min-h-[60px]"
                value={form.ingredients}
                onChange={(e) =>
                  setForm({ ...form, ingredients: e.target.value })
                }
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium mb-1">
                  Calories
                </label>
                <input
                  type="number"
                  className="w-full border rounded px-2 py-1 text-sm"
                  value={form.calories}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      calories: e.target.value === "" ? "" : Number(e.target.value),
                    })
                  }
                />
              </div>
              <div>
                <label className="block text-xs font-medium mb-1">Protein</label>
                <input
                  type="number"
                  className="w-full border rounded px-2 py-1 text-sm"
                  value={form.protein}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      protein: e.target.value === "" ? "" : Number(e.target.value),
                    })
                  }
                />
              </div>
              <div>
                <label className="block text-xs font-medium mb-1">Fat</label>
                <input
                  type="number"
                  className="w-full border rounded px-2 py-1 text-sm"
                  value={form.fat}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      fat: e.target.value === "" ? "" : Number(e.target.value),
                    })
                  }
                />
              </div>
              <div>
                <label className="block text-xs font-medium mb-1">Sugar</label>
                <input
                  type="number"
                  className="w-full border rounded px-2 py-1 text-sm"
                  value={form.sugar}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      sugar: e.target.value === "" ? "" : Number(e.target.value),
                    })
                  }
                />
              </div>
              <div>
                <label className="block text-xs font-medium mb-1">Sodium</label>
                <input
                  type="number"
                  className="w-full border rounded px-2 py-1 text-sm"
                  value={form.sodium}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      sodium: e.target.value === "" ? "" : Number(e.target.value),
                    })
                  }
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium mb-1">
                  Diet labels (comma separated)
                </label>
                <input
                  className="w-full border rounded px-2 py-1 text-sm"
                  placeholder="vegan, mindful"
                  value={form.diet_key}
                  onChange={(e) =>
                    setForm({ ...form, diet_key: e.target.value })
                  }
                />
              </div>
              <div>
                <label className="block text-xs font-medium mb-1">
                  Meal time
                </label>
                <select
                  className="w-full border rounded px-2 py-1 text-sm"
                  value={form.meal_time}
                  onChange={(e) =>
                    setForm({ ...form, meal_time: e.target.value })
                  }
                >
                  <option value="">Select...</option>
                  <option value="breakfast">Breakfast</option>
                  <option value="lunch">Lunch</option>
                  <option value="dinner">Dinner</option>
                </select>
              </div>
            </div>

            <div className="flex items-center gap-4 text-sm">
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={form.is_vegan}
                  onChange={(e) =>
                    setForm({ ...form, is_vegan: e.target.checked })
                  }
                />
                Vegan
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={form.is_vegetarian}
                  onChange={(e) =>
                    setForm({ ...form, is_vegetarian: e.target.checked })
                  }
                />
                Vegetarian
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={form.is_mindful}
                  onChange={(e) =>
                    setForm({ ...form, is_mindful: e.target.checked })
                  }
                />
                Mindful
              </label>
            </div>

            <button
              type="submit"
              className="mt-2 inline-flex items-center justify-center px-4 py-2 rounded bg-black text-white text-sm disabled:opacity-60"
              disabled={loading}
            >
              {loading ? "Predicting..." : "Predict"}
            </button>
            {error && <p className="text-sm text-red-600 mt-2">{error}</p>}
          </form>
        </section>

        <section className="space-y-4">
          <h2 className="text-lg font-semibold mb-2">Prediction</h2>
          {!result && (
            <p className="text-sm text-slate-600">
              Fill in the form and click Predict to see the model&apos;s
              estimate.
            </p>
          )}

          {result && (
            <div className="space-y-4">
              <div
                className={`border rounded-lg p-4 ${LABEL_COLOR[result.label]}`}
              >
                <div className="text-xs font-semibold uppercase tracking-wide mb-1">
                  Predicted satisfaction
                </div>
                <div className="text-2xl font-bold mb-1">
                  {LABEL_TEXT[result.label]}
                </div>
                <div className="text-sm opacity-80">
                  Confidence: {(result.probability * 100).toFixed(1)}%
                </div>
              </div>

              {proba.length === 3 && (
                <div className="border rounded-lg p-4">
                  <div className="text-xs font-semibold uppercase tracking-wide mb-2">
                    Class probabilities
                  </div>
                  <div className="space-y-2 text-sm">
                    {classes.map((c, idx) => {
                      const pct = proba[idx] * 100;
                      const label = LABEL_TEXT[c as SatisfactionLabel];
                      const barColor =
                        c === 2
                          ? "bg-green-500"
                          : c === 1
                          ? "bg-gray-400"
                          : "bg-red-500";
                      return (
                        <div key={c}>
                          <div className="flex justify-between mb-1">
                            <span>{label}</span>
                            <span>{pct.toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                            <div
                              className={`${barColor} h-full`}
                              style={{ width: `${Math.max(pct, 2)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
