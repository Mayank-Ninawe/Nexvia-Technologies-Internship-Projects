# Project Requirements and Acceptance Criteria

> **For educational purposes only, not for clinical use.**

## 1. Purpose and status

This document converts the internship project brief into requirements that can be implemented and tested.

At the time of writing, these requirements are planning targets. A requirement is not complete merely because it appears here. Completion must be supported by code, tests, and documented evidence.

## 2. Project objective

Build an educational web application that demonstrates:

- A chest-radiograph classification workflow.
- A separate 3D medical-image segmentation workflow.
- DICOM study handling through Orthanc where feasible.
- A backend API and browser-based viewer.
- Human review actions recorded persistently.
- Testing, reproducibility, privacy, limitations, and deployment documentation.

The application is not intended for clinical use or to provide medical advice.

## 3. Requirement status vocabulary

Use these labels in progress reports:

- **PLANNED** — not implemented.
- **IN PROGRESS** — implementation has started but acceptance checks are incomplete.
- **IMPLEMENTED** — code exists, but test evidence may be incomplete.
- **VERIFIED** — defined acceptance checks passed in the documented environment.
- **PROTOTYPE** — works only for a documented subset or constrained demonstration.
- **BLOCKED** — cannot proceed until a named dependency or constraint is resolved.
- **DEFERRED** — intentionally excluded from the current scope, with a reason.

Never label a feature VERIFIED without recording evidence. Do not invent metrics, model outputs, test results, or deployment status.

## 4. Functional requirements

### FR-01: Project setup and reproducibility

**Requirement:** A developer can follow documented setup instructions to prepare the project.

**Acceptance criteria:**

- Required software and tested versions are documented.
- Installation and startup commands are documented for the supported environment.
- Configuration uses environment variables or documented config files.
- Secrets and local environment files are excluded from Git.
- A fresh setup is tested before the project is declared complete.

**Evidence:** Setup checklist, command output, and clean-install notes.

### FR-02: Dataset access and organization

**Requirement:** The selected public dataset is accessed and organized according to its terms.

**Acceptance criteria:**

- Dataset source, version, access steps, and terms are documented.
- Data is not committed to ordinary Git history.
- A manifest identifies expected files and labels without unnecessarily exposing identifiers.
- Dataset availability and readability are checked before training.
- Dataset download size and required disk space are measured or confirmed from the source.

**Evidence:** Dataset documentation, validation output, and storage estimate.

### FR-03: Classification data preparation

**Requirement:** Chest-radiograph inputs are prepared reproducibly for classification.

**Acceptance criteria:**

- Supported image formats and unsupported cases are documented.
- Preprocessing defines grayscale handling, resizing, normalization, and training augmentation.
- Patient-level splits are used when patient identifiers are available.
- No patient appears in more than one split.
- Missing and uncertain labels follow a documented policy.

**Evidence:** Preprocessing tests, split-integrity test, and label-policy documentation.

### FR-04: Classification training and inference

**Requirement:** A real transfer-learning classifier can be trained and used for inference, if feasible with available data and compute.

**Acceptance criteria:**

- The selected model and configuration are documented.
- Training uses only the training split for gradient updates.
- Validation data is used for model selection without mixing with the final test set.
- Inference returns mapped class labels and finite scores.
- Missing or incompatible model artifacts produce a clear error.
- Checkpoints, configuration, and training logs are saved outside ordinary Git history unless a suitable artifact store is configured.

**Evidence:** Real training log, saved configuration/artifact, inference test, and evaluation report. If compute prevents training, mark the requirement BLOCKED or PROTOTYPE; do not replace results with fabricated output.

### FR-05: Classification evaluation

**Requirement:** Model performance is reported with suitable metrics and split information.

**Acceptance criteria:**

- Report per-label AUROC and average precision when the data supports their calculation.
- Report suitable thresholded measures such as precision, recall, F1, and confusion information where appropriate.
- State class counts, split policy, label handling, thresholds, and limitations.
- Do not claim clinical validity or generalization beyond the evaluated data.

**Evidence:** Reproducible evaluation command, metric outputs, and report.

### FR-06: Grad-CAM explainability

**Requirement:** Generate a Grad-CAM visualization for supported classification predictions.

**Acceptance criteria:**

- The supported model and target layer are documented.
- Heatmap dimensions and overlay alignment are tested.
- Unsupported models, missing gradients, and invalid inputs fail safely.
- The viewer states that Grad-CAM is a model visualization, not proof that a prediction is correct or clinically meaningful.

**Evidence:** Explainability tests and a reviewed sample visualization generated from actual model inference.

### FR-07: 3D segmentation

**Requirement:** Train or run a MONAI-based 3D U-Net on one selected MSD task, subject to feasibility.

**Acceptance criteria:**

- The MSD task, modality, labels, access terms, and data scope are documented.
- Preprocessing handles volume orientation, spacing, dimensions, and image-mask alignment.
- Spatial metadata is retained or correctly accounted for when displaying/exporting results.
- Empty masks and incompatible input shapes are handled safely.
- Dice and any other reported metrics are computed from actual outputs and documented splits.

**Evidence:** Data alignment tests, model shape tests, actual run configuration, and measured evaluation report. If full training is infeasible, document a bounded pilot or mark the full requirement BLOCKED/DEFERRED.

### FR-08: DICOM and Orthanc

**Requirement:** Support a documented local DICOM workflow using Orthanc where feasible.

**Acceptance criteria:**

- DICOM files are validated before processing.
- The application can demonstrate upload and retrieval of studies, series, instances, and selected metadata through Orthanc.
- Malformed and unsupported DICOM files return understandable errors.
- The UI does not display unnecessary identifiers.
- Logs avoid unnecessary sensitive metadata.
- Orthanc is not exposed publicly without an explicit security design.

**Evidence:** Orthanc connectivity check, API tests, and manual multi-study/series verification.

### FR-09: Backend API

**Requirement:** Provide API endpoints for health, supported uploads, study access where enabled, inference, and review/audit workflows.

**Acceptance criteria:**

- Request and response schemas are documented.
- Invalid requests and unsupported files return structured, understandable errors.
- Upload size and input type limits are enforced.
- Inference invokes the actual selected pipeline when its artifacts are available.
- Health checks distinguish service availability from model-artifact availability.

**Evidence:** Automated API tests and generated API documentation review.

### FR-10: Frontend viewer and review flow

**Requirement:** Provide a React and TypeScript interface for the supported workflow.

**Acceptance criteria:**

- Dashboard or system-status view is available.
- Upload includes loading, success, unsupported, and error states.
- Supported images or DICOM studies can be viewed with relevant metadata.
- Predictions and confidence scores are shown as model outputs, not diagnoses.
- Grad-CAM and segmentation outputs are shown only when real outputs are available.
- Approve/reject review controls call the backend and provide user feedback.
- Audit history is accessible.
- Controls have accessible labels and the layout works at documented screen sizes.
- The education-only disclaimer is visible in the application.

**Evidence:** Frontend tests and a manual checklist covering navigation, workflows, accessibility, and responsive behavior.

### FR-11: Persistent audit records

**Requirement:** Human review decisions and relevant events persist across application restarts.

**Acceptance criteria:**

- Approve and reject actions create persistent records.
- The record includes only the fields needed for audit and demonstration.
- Database changes are versioned or documented.
- Restarting the service does not erase the audit history.
- Failed requests do not create misleading successful review events.

**Evidence:** Integration test showing record creation and retrieval after restart.

### FR-12: Docker and deployment

**Requirement:** Package the supported application for reproducible local execution and deploy the feasible components.

**Acceptance criteria:**

- Container configuration documents services and required variables.
- A clean start and health check are tested.
- Secrets and sensitive datasets are not baked into images.
- Deployment target, limitations, storage behavior, and service configuration are documented.
- Hosted demo uses no real patient data.
- Any service that remains local or unhosted is clearly identified.

**Evidence:** Build/start logs, smoke test, deployment URL if actually deployed, and limitations note.

## 5. Quality requirements

### QR-01: Safe input handling

Reject empty, corrupt, unsupported, oversized, or malformed inputs without crashing or exposing sensitive internals.

### QR-02: Privacy and security

Do not commit credentials, patient identifiers, private medical images, or secrets. Minimize metadata in logs and UI. Review upload, path handling, service exposure, and configuration before deployment.

### QR-03: Reproducibility

Record the code revision, configuration, dataset version or manifest, split policy, relevant random seeds, software versions, and actual run outputs for experiments.

### QR-04: Maintainability

Keep UI, API, ML workflows, DICOM integration, and data access separated into understandable components. Use typed interfaces where appropriate.

### QR-05: Honest reporting

Distinguish verified features, prototypes, blocked items, skipped tests, and unverified behavior. Never claim clinical validation.

## 6. Minimum test categories

The final test plan must consider:

- File uploads: valid, invalid, corrupt, empty, oversized, grayscale, unusual dimensions, and unsupported extensions.
- DICOM: invalid/missing metadata, multiple studies and series, multiple slices, orientation/spacing, and privacy-sensitive metadata.
- Classification: missing model, incompatible shapes, class mapping, non-finite scores, inference failure, split leakage, and reproducibility.
- Segmentation: volume-mask alignment, output dimensions, empty masks, spatial metadata, and metric edge cases.
- Grad-CAM: model/layer compatibility, dimensions, overlay alignment, and unsupported cases.
- Backend/database: invalid requests, errors, timeouts, persistence, restart, and access controls where applicable.
- Frontend: upload flows, navigation, review actions, visualization, accessibility, responsive layout, and loading/error states.
- Integration/security: API/Orthanc connectivity, path traversal, unsafe uploads, oversized requests, secrets, and sensitive-data leakage.
- Deployment: clean build/start, health checks, persistent storage expectations, and a full demo smoke test.

Each test outcome must be recorded as **PASS**, **FAIL**, **SKIPPED**, or **NOT VERIFIED**, with notes.

## 7. Out of scope

Unless explicitly approved and separately reviewed, this project does not include:

- Clinical diagnosis, treatment recommendations, or patient-care decisions.
- Claims of regulatory approval, clinical validity, or medical-device readiness.
- Public processing of real patient data.
- Support for every DICOM transfer syntax, modality, or image format.
- Guaranteed availability or performance of free hosted GPU services.
- Fabricated or placeholder model results represented as real evaluation.

## 8. Feasibility decisions still required

Before locking dataset/model and training scope, record:

- Available local GPU, VRAM, RAM, and disk.
- Whether hosted GPU access is available and under what limits.
- Dataset registration/access and permitted use.
- Download, extracted-data, cache, and artifact storage estimates.
- Deadline and expected demonstration requirements.
- Whether the hosted application will use only synthetic or public educational inputs.

## 9. Completion rule

A requirement is complete only when:

1. Its implementation exists.
2. Its acceptance checks have been run.
3. Results and environment are documented.
4. Known limitations are stated.
5. The work is committed to the repository.

A failed or unavailable check must be reported, not hidden.