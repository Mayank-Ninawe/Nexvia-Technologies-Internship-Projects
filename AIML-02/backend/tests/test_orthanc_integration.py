"""Live Orthanc integration tests, enabled explicitly for a running local service."""

import os
from pathlib import Path

import pytest

from backend.app.services.dicom import DicomInputError, OrthancService

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ORTHANC_INTEGRATION") != "1",
    reason="Set RUN_ORTHANC_INTEGRATION=1 to run against local Orthanc.",
)


def test_orthanc_upload_list_retrieve_and_validate() -> None:
    service = OrthancService()
    payload = (Path(__file__).parent / "fixtures" / "sample.dcm").read_bytes()

    uploaded = service.upload_instance(payload)
    instance_id = uploaded["orthanc_id"]
    assert instance_id
    assert uploaded["metadata"]["modality"] == "OT"
    assert service.list_studies()

    retrieved = service.retrieve_instance(str(instance_id))
    assert retrieved == payload


def test_orthanc_rejects_malformed_dicom() -> None:
    with pytest.raises(DicomInputError) as error:
        OrthancService().upload_instance(b"not a DICOM object")
    assert "valid DICOM" in str(error.value)
