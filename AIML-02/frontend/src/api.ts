export type Prediction = {
  label: string;
  probability: number;
  positive_at_threshold: boolean;
};

export type ClassificationResponse = {
  model: string;
  model_artifact: string;
  threshold: number;
  predictions: Prediction[];
  limitation: string;
};

export type GradCamResponse = {
  target_label: string;
  target_probability: number;
  predicted_label: string;
  predicted_probability: number;
  heatmap_shape: number[];
  overlay_base64: string;
  limitation: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  const body = await response.json();
  if (!response.ok) {
    const message = body.detail?.message ?? "The request failed.";
    throw new Error(message);
  }
  return body as T;
}

export function classify(file: File): Promise<ClassificationResponse> {
  return request("/predict/classification", {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });
}

export function generateGradCam(file: File, label: string): Promise<GradCamResponse> {
  return request(`/explainability/grad-cam?label=${encodeURIComponent(label)}`, {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });
}

export function submitReview(reference: string, action: "approve" | "reject", modelArtifact: string) {
  return request("/reviews", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reference, action, model_artifact: modelArtifact }),
  });
}
