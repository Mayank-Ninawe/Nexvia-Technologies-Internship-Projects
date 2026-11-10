# AI-Assisted Medical Imaging

<p align="center">
  <strong>Explainable chest X-ray classification with a local FastAPI + React workspace</strong>
  <br />
  <sub>Plus experimental MONAI spleen segmentation and Orthanc/DICOM integration.</sub>
</p>

<p align="center">
  <a href="https://github.com/Mayank-Ninawe/ai-assisted-medical-imaging/actions/workflows/python-ci.yml">
    <img src="https://github.com/Mayank-Ninawe/ai-assisted-medical-imaging/actions/workflows/python-ci.yml/badge.svg" alt="Python CI" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%20%E2%80%93%203.11-3776AB?logo=python&logoColor=white" alt="Python 3.10 to 3.11" />
  <img src="https://img.shields.io/badge/React-TypeScript-149ECA?logo=react&logoColor=white" alt="React and TypeScript" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker Compose" />
  <img src="https://img.shields.io/badge/Use-Educational%20Only-F59E0B" alt="Educational use only" />
</p>

> **Important:** This is an educational prototype. It is **not** a medical
> device and must not be used for diagnosis, treatment, triage, or patient-care
> decisions.

## What this project does

This internship project demonstrates a local medical-imaging workflow:

1. Upload a chest X-ray image.
2. Run a 14-label DenseNet-121 classifier.
3. Inspect label-specific scores and Grad-CAM overlays.
4. Record reviewer feedback in SQLite.
5. Upload, list, and retrieve DICOM instances through Orthanc.

It also contains a separate MONAI-based 3D spleen-segmentation research
workflow. Segmentation is experimental, has poor recorded performance, and is
not integrated into the current classification workspace.

## Capability snapshot

| Area | Status | Boundary |
| --- | :---: | --- |
| Chest X-ray classification | **Available** | Educational scores; no clinical validation |
| Grad-CAM | **Available** | Attribution is not proof of clinical reasoning |
| Reviewer feedback | **Available** | Persisted locally; does not retrain the model |
| FastAPI backend | **Available** | No authentication or authorization |
| React frontend | **Available** | No DICOM viewer or segmentation visualization |
| Orthanc/DICOM | **Local demo** | API integration, not a complete PACS product |
| Spleen segmentation | **Experimental** | Poor measured Dice; not a product feature |
| Docker Compose | **Local demo** | Not production-ready |
| DVC | **Metadata only** | No operational pipeline or remote |

## Architecture

```text
┌──────────────────────┐       HTTP        ┌─────────────────────────┐
│ React + TypeScript   │ ────────────────► │ FastAPI backend         │
│ Classification UI   │ ◄──────────────── │                         │
└──────────────────────┘                  │ ├─ DenseNet-121         │
                                          │ ├─ Grad-CAM             │
                                          │ ├─ SQLite reviews        │
                                          │ └─ Orthanc REST client    │
                                          └───────────┬─────────────┘
                                                      │
                                          ┌───────────▼─────────────┐
                                          │ Orthanc                  │
                                          │ DICOM storage + index     │
                                          └──────────────────────────┘

Separate research workflow:
MSD Task09 Spleen CT → MONAI preprocessing → 3D U-Net → evaluation reports
```

## Classification results

Recorded evaluation artifact: `reports/classification_test_metrics.json`

| Metric | Recorded value |
| --- | ---: |
| Dataset | NIH ChestX-ray14 |
| Test images | 25,596 |
| Labels | 14 |
| Macro AUROC | 0.796025 |
| Macro average precision | 0.256642 |
| Decision threshold | 0.5 |

These are historical recorded results, not a new evaluation performed during
clone or installation.

### Interpretation limits

- Macro AUROC is not classification accuracy.
- Several labels have zero recall and zero F1 at threshold `0.5`.
- Thresholds have not been shown to be calibrated or clinically appropriate.
- External validation, confidence intervals, and clinical validation are not
  established.
- Public dataset labels may be incomplete or uncertain.
- Model scores must not be interpreted as validated individual disease risk.

## Experimental segmentation

The repository includes MONAI preprocessing, a 3D U-Net workflow, training and
evaluation scripts, notebooks, checkpoints, and historical reports.

Recorded artifacts include:

- Current checkpoint audit: validation Dice approximately `0.0533`.
- Historical held-out checkpoint: validation Dice approximately `0.1099`.
- Historical six-volume report: mean Dice approximately `0.1477`.

These values belong to different runs and configurations. They are not directly
comparable without reconciling their splits, parameters, and provenance.

Segmentation remains unsuitable as a reliable product feature because of poor
overlap scores and reported over-segmentation. It is not exposed in the current
application UI.

## Quick start

### Requirements

- Python 3.10 or 3.11
- Node.js and npm
- Docker with Docker Compose for the local container stack
- A compatible classifier checkpoint at
  `models/classification/best_classifier.pt`

Large datasets and trained model files are not provided through normal Git
history. No automated model download or DVC remote is configured.

### Install Python dependencies

From the repository root:

```bash
python -m venv .venv
```

Windows:

```cmd
.venv\Scripts\activate.bat
python -m pip install -e ".[dev,backend,ml]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,backend,ml]"
```

### Run the backend

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

| Resource | URL |
| --- | --- |
| API documentation | <http://localhost:8000/docs> |
| Service health | <http://localhost:8000/health> |
| Model availability | <http://localhost:8000/health/models> |

### Run the frontend

```bash
cd frontend
npm ci
npm test
npm run build
npm run dev
```

The frontend defaults to `http://localhost:8000`. Set
`VITE_API_BASE_URL` to use another backend URL.

### Run the local Docker stack

With the checkpoint available:

```bash
docker compose up -d --build
docker compose ps
```

| Service | Address |
| --- | --- |
| Web workspace | <http://localhost:8080> |
| Backend API | <http://localhost:8000/docs> |
| Orthanc HTTP interface | <http://localhost:8042> |
| Orthanc DICOM port | `4242` |

The stack mounts `models/` read-only and uses named volumes for backend runtime
data and Orthanc storage.

> The checked-in Orthanc configuration contains development credentials. Review
> it before starting services and never expose this stack to an untrusted
> network.

Useful commands:

```bash
# Rebuild only the frontend
docker compose up -d --build --no-deps frontend

# Inspect backend logs
docker compose logs --tail 80 backend

# Stop services without deleting named volumes
docker compose down
```

## API overview

| Method | Path | Purpose |
| :---: | --- | --- |
| `GET` | `/health` | Service health |
| `GET` | `/health/models` | Classifier checkpoint availability |
| `POST` | `/predict/classification` | Classify raw image bytes |
| `POST` | `/explainability/grad-cam?label=Effusion` | Generate a label-specific overlay |
| `POST` | `/reviews` | Save reviewer feedback |
| `GET` | `/reviews/{reference}` | Retrieve reviews |
| `POST` | `/dicom/instances` | Forward DICOM bytes to Orthanc |
| `GET` | `/dicom/studies` | List Orthanc studies |
| `GET` | `/dicom/instances/{instance_id}` | Retrieve a DICOM instance |

Classification, Grad-CAM, and DICOM uploads use raw request bodies rather than
multipart form uploads. See `/docs` for generated request and response schemas.

## Testing and quality checks

### Python

```bash
python -m ruff check .
python -m pytest -q
```

### Frontend

```bash
cd frontend
npm ci
npm test
npm run build
npm audit
```

### Optional Orthanc integration tests

Start Orthanc first, then:

Windows CMD:

```cmd
set RUN_ORTHANC_INTEGRATION=1
python -m pytest backend/tests/test_orthanc_integration.py -v
```

Linux/macOS:

```bash
RUN_ORTHANC_INTEGRATION=1 python -m pytest -q
```

These tests upload the repository's DICOM fixture to the configured Orthanc
instance. Use a dedicated development instance.

## Project layout

```text
backend/                  FastAPI application and API tests
frontend/                 React/TypeScript workspace
ml/classification/        Classification workflow and configuration
ml/explainability/        Grad-CAM implementation
ml/segmentation/          Experimental MONAI segmentation workflow
ml/notebooks/             Segmentation research notebooks
tests/                    ML, data, and helper tests
dicom/orthanc/             Local Orthanc configuration
scripts/                  Supporting DICOM and segmentation utilities
models/                   Separately provisioned model artifacts
reports/                  Generated evaluation artifacts
.github/workflows/        CI configuration
```

## Known gaps

- No authentication, authorization, or production access controls.
- No production TLS, secret management, backups, or monitoring.
- Development credentials remain in local deployment configuration.
- No integrated DICOM viewer or segmentation UI.
- Poor segmentation performance.
- No validated per-label thresholds or calibration.
- No external or clinical model validation.
- Incomplete experiment provenance.
- No operational DVC pipeline or remote.
- No automated browser E2E or Docker-build CI.
- Reviewer references use uploaded filenames rather than immutable inference IDs.
- Python dependency ranges do not fully lock the environment.

## Responsible use and privacy

- Use public or appropriately de-identified images only.
- Do not upload private patient data to this demonstration.
- Do not commit datasets, patient identifiers, credentials, or access tokens.
- A reviewer approval is not clinical approval.
- Grad-CAM does not establish causal or clinical reasoning.
- Model scores must not guide care decisions.

## License and dataset terms

Review the repository's code license and each dataset's terms separately. A
code license does not grant permission to redistribute third-party datasets or
trained artifacts.

Dataset access, provenance, attribution, and redistribution permissions must be
documented before sharing data or models.
