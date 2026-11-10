"""Validated local Orthanc integration boundary."""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import pydicom
import requests


class DicomInputError(ValueError):
    """Raised when an uploaded DICOM object is invalid or unsafe."""


class OrthancUnavailable(RuntimeError):
    """Raised when the configured Orthanc service cannot be reached."""


def validate_dicom(payload: bytes) -> dict[str, str]:
    """Parse a DICOM object and return only navigation-safe identifiers."""
    if not payload:
        raise DicomInputError("Uploaded DICOM file is empty.")
    try:
        dataset = pydicom.dcmread(BytesIO(payload), force=False)
    except (OSError, pydicom.errors.InvalidDicomError) as error:
        raise DicomInputError("Uploaded file is not a valid DICOM object.") from error

    required = ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID")
    missing = [field for field in required if not getattr(dataset, field, None)]
    if missing:
        raise DicomInputError(f"DICOM object is missing required identifiers: {missing}.")

    return {
        "study_instance_uid": str(dataset.StudyInstanceUID),
        "series_instance_uid": str(dataset.SeriesInstanceUID),
        "sop_instance_uid": str(dataset.SOPInstanceUID),
        "modality": str(getattr(dataset, "Modality", "UNKNOWN")),
    }


class OrthancService:
    """Forward validated DICOM requests to a local authenticated Orthanc."""

    def __init__(self) -> None:
        self.base_url = os.getenv("ORTHANC_URL", "http://localhost:8042").rstrip("/")
        self.username = os.getenv("ORTHANC_USERNAME", "orthanc")
        self.password = os.getenv("ORTHANC_PASSWORD", "orthanc-dev-password")
        self.timeout_seconds = 10

    def upload_instance(self, payload: bytes) -> dict[str, Any]:
        metadata = validate_dicom(payload)
        try:
            response = requests.post(
                f"{self.base_url}/instances",
                data=payload,
                auth=(self.username, self.password),
                headers={"Content-Type": "application/dicom"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as error:
            raise OrthancUnavailable("Orthanc instance upload failed.") from error

        return {
            "orthanc_id": body.get("ID"),
            "resource_type": body.get("Type", "Instance"),
            "metadata": metadata,
        }

    def list_studies(self) -> list[str]:
        try:
            response = requests.get(
                f"{self.base_url}/studies",
                auth=(self.username, self.password),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as error:
            raise OrthancUnavailable("Orthanc study listing failed.") from error

        if not isinstance(body, list) or not all(isinstance(item, str) for item in body):
            raise OrthancUnavailable("Orthanc returned an invalid study list.")
        return body

    def retrieve_instance(self, instance_id: str) -> bytes:
        if not instance_id or "/" in instance_id or "\\" in instance_id:
            raise DicomInputError("A valid Orthanc instance identifier is required.")
        try:
            response = requests.get(
                f"{self.base_url}/instances/{instance_id}/file",
                auth=(self.username, self.password),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise OrthancUnavailable("Orthanc instance retrieval failed.") from error
        if not response.content:
            raise OrthancUnavailable("Orthanc returned an empty DICOM instance.")
        try:
            validate_dicom(response.content)
        except DicomInputError as error:
            raise OrthancUnavailable("Orthanc returned invalid DICOM content.") from error
        return response.content
