export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function predict(meal: {
    name: string;
    ingredients?: string;
    allergens?: string;
    station?: string;
    diet_key?: string;
    meal_time?: string;
}) {
    const res = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(meal),
        cache: "no-store",
    });

    if (!res.ok) throw new Error("Prediction failed");
    return res.json();
}
