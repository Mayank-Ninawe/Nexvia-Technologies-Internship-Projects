"""Minimal FastAPI application for educational classification inference."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import PROJECT_ROOT, classification_checkpoint_path, max_upload_size_bytes
from backend.app.db import create_review, list_reviews
from backend.app.schemas import (
    ClassificationResponse,
    DicomUploadResponse,
    GradCamResponse,
    HealthResponse,
    ModelHealthResponse,
    ReviewRequest,
    ReviewResponse,
)
from backend.app.services.classification import (
    ClassificationInputError,
    ClassificationModelUnavailable,
    ClassificationService,
)
from backend.app.services.dicom import DicomInputError, OrthancService, OrthancUnavailable
from backend.app.services.explainability import ExplainabilityInputError, ExplainabilityService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Assisted Medical Imaging API",
    description="Educational image-classification API; not for clinical use.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
classification_service = ClassificationService(classification_checkpoint_path())
explainability_service = ExplainabilityService(
    config_path=PROJECT_ROOT / "ml/classification/config.yaml",
    checkpoint_path=classification_checkpoint_path(),
)
orthanc_service = OrthancService()


async def read_limited_body(request: Request, limit: int) -> bytes:
    """Read an HTTP body without allowing it to exceed the configured limit."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_content_length", "message": "Invalid Content-Length."},
            ) from error
        if declared_length > limit:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "upload_too_large",
                    "message": "Uploaded file exceeds the size limit.",
                },
            )

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "upload_too_large",
                    "message": "Uploaded file exceeds the size limit.",
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="medical-imaging-api")


@app.get("/health/models", response_model=ModelHealthResponse)
def model_health() -> ModelHealthResponse:
    available = classification_service.is_available()
    return ModelHealthResponse(
        status="ok" if available else "degraded",
        classification_checkpoint=classification_service.checkpoint_path.as_posix(),
        classification_model_available=available,
    )


@app.post("/predict/classification", response_model=ClassificationResponse)
async def predict_classification(request: Request) -> ClassificationResponse:
    payload = await read_limited_body(request, max_upload_size_bytes())
    try:
        predictions = classification_service.predict(payload)
    except ClassificationInputError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_image", "message": str(error)},
        ) from error
    except ClassificationModelUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": str(error)},
        ) from error
    except (KeyError, OSError, RuntimeError) as error:
        logger.exception("Classification inference failed.")
        raise HTTPException(
            status_code=503,
            detail={"code": "inference_unavailable", "message": "Classification inference failed."},
        ) from error

    return ClassificationResponse(
        model="densenet121",
        model_artifact=classification_service.checkpoint_path.name,
        threshold=classification_service.threshold,
        predictions=predictions,
        limitation=(
            "Educational model output only; this is not a diagnosis or clinical advice."
        ),
    )


@app.post("/explainability/grad-cam", response_model=GradCamResponse)
async def grad_cam(request: Request, label: str) -> GradCamResponse:
    payload = await read_limited_body(request, max_upload_size_bytes())
    try:
        result = explainability_service.generate(payload, label)
    except ExplainabilityInputError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_grad_cam_request", "message": str(error)},
        ) from error
    except (OSError, RuntimeError) as error:
        logger.exception("Grad-CAM generation failed.")
        raise HTTPException(
            status_code=503,
            detail={"code": "explainability_unavailable", "message": "Grad-CAM is unavailable."},
        ) from error

    return GradCamResponse(
        model="densenet121",
        model_artifact=classification_service.checkpoint_path.name,
        target_label=str(result["target_label"]),
        target_probability=float(result["target_probability"]),
        predicted_label=str(result["predicted_label"]),
        predicted_probability=float(result["predicted_probability"]),
        heatmap_shape=list(result["heatmap_shape"]),
        overlay_base64=str(result["overlay_base64"]),
        limitation=str(result["interpretation_limitation"]),
    )


@app.post("/dicom/instances", response_model=DicomUploadResponse)
async def upload_dicom(request: Request) -> DicomUploadResponse:
    payload = await read_limited_body(request, max_upload_size_bytes())
    try:
        result = orthanc_service.upload_instance(payload)
    except DicomInputError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_dicom", "message": str(error)},
        ) from error
    except OrthancUnavailable as error:
        logger.exception("Orthanc upload failed.")
        raise HTTPException(
            status_code=503,
            detail={"code": "orthanc_unavailable", "message": str(error)},
        ) from error
    return DicomUploadResponse(**result)


@app.get("/dicom/studies", response_model=list[str])
def list_dicom_studies() -> list[str]:
    try:
        return orthanc_service.list_studies()
    except OrthancUnavailable as error:
        logger.exception("Orthanc study listing failed.")
        raise HTTPException(
            status_code=503,
            detail={"code": "orthanc_unavailable", "message": str(error)},
        ) from error


@app.get("/dicom/instances/{instance_id}")
def retrieve_dicom(instance_id: str) -> Response:
    try:
        payload = orthanc_service.retrieve_instance(instance_id)
    except DicomInputError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_dicom_reference", "message": str(error)},
        ) from error
    except OrthancUnavailable as error:
        logger.exception("Orthanc instance retrieval failed.")
        raise HTTPException(
            status_code=503,
            detail={"code": "orthanc_unavailable", "message": str(error)},
        ) from error
    return Response(content=payload, media_type="application/dicom")


@app.post("/reviews", response_model=ReviewResponse, status_code=201)
def add_review(request: ReviewRequest) -> ReviewResponse:
    try:
        record = create_review(
            reference=request.reference,
            action=request.action,
            model_artifact=request.model_artifact,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_review", "message": str(error)},
        ) from error
    return ReviewResponse(**record)


@app.get("/reviews/{reference}", response_model=list[ReviewResponse])
def get_reviews(reference: str) -> list[ReviewResponse]:
    if not reference.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_review_reference", "message": "Reference is required."},
        )
    return [ReviewResponse(**record) for record in list_reviews(reference)]
