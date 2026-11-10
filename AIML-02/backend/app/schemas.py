"""API response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelHealthResponse(BaseModel):
    status: str
    classification_checkpoint: str
    classification_model_available: bool


class ClassificationPrediction(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    positive_at_threshold: bool


class ClassificationResponse(BaseModel):
    model: str
    model_artifact: str
    threshold: float
    predictions: list[ClassificationPrediction]
    limitation: str


class GradCamResponse(BaseModel):
    model: str
    model_artifact: str
    target_label: str
    target_probability: float = Field(ge=0.0, le=1.0)
    predicted_label: str
    predicted_probability: float = Field(ge=0.0, le=1.0)
    heatmap_shape: list[int]
    overlay_base64: str
    limitation: str


class DicomUploadResponse(BaseModel):
    orthanc_id: str | None
    resource_type: str
    metadata: dict[str, str]


class ReviewRequest(BaseModel):
    reference: str
    action: str
    model_artifact: str


class ReviewResponse(BaseModel):
    id: str
    reference: str
    action: str
    model_artifact: str
    created_at: str
