export type SatisfactionLabel = 0 | 1 | 2;

export type ModelVariant = "sbert_fusion_mlp" | "oracle_knn_embeddings";

export interface PredictRequest {
  menu_item_id?: number | null;
  name: string;
  ingredients?: string;
  allergens?: string;
  station?: string;
  diet_key?: string;
  calories?: number | null;
  fat?: number | null;
  cholesterol?: number | null;
  sodium?: number | null;
  carbohydrates?: number | null;
  fiber?: number | null;
  sugar?: number | null;
  protein?: number | null;
  iron?: number | null;
  calcium?: number | null;
  potassium?: number | null;
  meal_time?: string | null;
  is_vegan?: boolean;
  is_vegetarian?: boolean;
  is_mindful?: boolean;
  is_plant_based?: boolean;
  is_standard?: boolean;
}

export interface PredictResponse {
  model?: ModelVariant | string;
  label: SatisfactionLabel;
  probability: number;
  proba_per_class?: number[];
  classes?: number[];
}

export interface Meal {
  id: number;
  menu_item_id: number | null;
  name: string;
  ingredients?: string;
  allergens: string;
  station: string;
  diet_key: string;
  calories: number | null;
  fat: number | null;
  cholesterol: number | null;
  sodium: number | null;
  carbohydrates: number | null;
  fiber: number | null;
  sugar: number | null;
  protein: number | null;
  iron: number | null;
  calcium: number | null;
  potassium: number | null;
  meal_time: string | null;
  is_vegan: boolean;
  is_vegetarian: boolean;
  is_mindful: boolean;
}
