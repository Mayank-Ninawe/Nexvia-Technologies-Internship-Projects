# Model Card: Chest X-ray Classification

## Model summary

| Property | Value |
|---|---|
| Task | Multi-label chest X-ray classification |
| Architecture | DenseNet-121 |
| Output labels | 14 NIH ChestX-ray14 findings |
| Checkpoint | `models/classification/best_classifier.pt` |
| Application | Educational local demonstration |
| Intended users | Developers, students, and project reviewers |
| Clinical use | Not supported |

This model produces scores for multiple findings independently. It does not
provide a diagnosis, a clinically validated risk estimate, or a declaration
that an image is normal.

## Intended use

Supported educational uses:

- Demonstrating image preprocessing and multi-label inference.
- Examining recorded classification metrics.
- Generating label-specific Grad-CAM overlays.
- Demonstrating a web-based reviewer-feedback workflow.

## Out-of-scope use

Do not use the model for:

- Diagnosis, screening, triage, or treatment decisions.
- Ruling out disease.
- Autonomous report generation.
- Estimating an individual's validated disease probability.
- Evaluating modalities other than supported chest radiograph images.
- Processing identifiable patient information in the current demo stack.

The application has no production authentication or authorization.

## Dataset and split policy

The current classification configuration identifies NIH ChestX-ray14.

Configured data inputs:

```text
data/raw/nih_chestxray14/Data_Entry_2017_v2020.csv
data/raw/nih_chestxray14/train_val_list.txt
data/raw/nih_chestxray14/test_list.txt
data/manifests/nih_chestxray14_manifest.csv
```

Configured split policy:

- Reserve the official test list.
- Create a patient-level validation split from the train/validation pool.
- Validation fraction: 0.1.
- Random seed: 42.
- No stratification configured.

These values describe the current configuration. The supplied evaluation
report does not include a manifest hash or an immutable training
configuration snapshot, so complete historical run lineage is not established.

Dataset access and redistribution terms must be checked separately.
Dataset images are not included in normal Git history.

## Labels

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural_Thickening
13. Pneumonia
14. Pneumothorax

"No Finding" is not a fifteenth output class.

Absence of a high score must not be interpreted as evidence that an image
is normal.

## Preprocessing and inference

The inspected API service:

1. Loads a supported image.
2. Applies EXIF orientation handling.
3. Converts the image to grayscale.
4. Resizes it to 224 x 224 pixels.
5. Normalizes with mean 0.5 and standard deviation 0.5.
6. Runs the classification model.
7. Applies sigmoid to each output logit.

The service uses CUDA when available, otherwise CPU.

The configured default decision threshold is 0.5. This threshold has not
been demonstrated to be calibrated or appropriate for clinical decisions.

Scores displayed as percentages are model outputs, not validated disease
probabilities. They are not mutually exclusive.

## Training information

Current configuration:

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Data loader workers | 2 |
| Mixed precision | Enabled |
| Early-stopping patience | 2 |

The earlier checkpoint audit recorded:

- Epoch: 2.
- Validation loss: 0.1337518056134286.

A complete immutable record of initialization, dataset contents, dependency
versions, source revision, and training configuration is not available for
this historical checkpoint.

Current source configuration must not be assumed to reconstruct the
historical training run exactly.

## Recorded evaluation

Source artifact:

```text
reports/classification_test_metrics.json
```

Checkpoint referenced by the report:

```text
models/classification/best_classifier.pt
```

| Metric/property | Value |
|---|---|
| Test images | 25,596 |
| Macro AUROC | 0.796025 |
| Macro average precision | 0.256642 |
| Decision threshold | 0.5 |
| Recorded device | CUDA |

These are stored evaluation results, not a newly rerun evaluation.

Macro AUROC is not accuracy. The report does not establish clinical
sensitivity, safety, or external generalization.

### Per-label results

Recall and F1 below use threshold 0.5.

| Label | Positive images | AUROC | Average precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Atelectasis | 3,279 | 0.765292 | 0.334797 | 0.045746 | 0.085203 |
| Cardiomegaly | 1,069 | 0.869861 | 0.314172 | 0.052385 | 0.095726 |
| Consolidation | 1,815 | 0.728199 | 0.141672 | 0.000000 | 0.000000 |
| Edema | 925 | 0.836901 | 0.152259 | 0.000000 | 0.000000 |
| Effusion | 4,658 | 0.824184 | 0.507512 | 0.265994 | 0.371570 |
| Emphysema | 1,093 | 0.907631 | 0.416368 | 0.309241 | 0.397881 |
| Fibrosis | 435 | 0.797915 | 0.080754 | 0.000000 | 0.000000 |
| Hernia | 86 | 0.874921 | 0.192889 | 0.000000 | 0.000000 |
| Infiltration | 6,112 | 0.695361 | 0.392582 | 0.187664 | 0.265264 |
| Mass | 1,748 | 0.803380 | 0.297045 | 0.145309 | 0.227395 |
| Nodule | 1,623 | 0.742581 | 0.218279 | 0.067776 | 0.118280 |
| Pleural_Thickening | 1,143 | 0.759384 | 0.127140 | 0.016623 | 0.031799 |
| Pneumonia | 555 | 0.700899 | 0.048883 | 0.000000 | 0.000000 |
| Pneumothorax | 2,665 | 0.837844 | 0.368631 | 0.097936 | 0.165190 |

Positive-image counts overlap because this is a multi-label dataset.

### Important interpretation

At threshold 0.5, five labels have zero recorded recall and F1:

- Consolidation
- Edema
- Fibrosis
- Hernia
- Pneumonia

Other labels also show limited recall. Ranking performance therefore must
not be presented as evidence of reliable binary decisions at the default
threshold.

Calibration, validation-set threshold selection, confidence intervals, and
external evaluation remain future work. Thresholds must not be selected
using the held-out test set.

## Grad-CAM

The application generates target-label Grad-CAM overlays and displays them
beside the uploaded image.

An overlay:

- Is dependent on the model, target label, and preprocessing.
- Does not establish that the prediction is correct.
- Does not prove causal or clinically meaningful reasoning.
- Has not been validated as lesion localization.
- Must not be used as clinical evidence.

Visually plausible heatmaps are not an acceptance criterion for model
accuracy.

## Reviewer feedback

The application records approve/reject feedback with an image reference
and model artifact name.

Feedback:

- Is not clinical approval.
- Does not change the model score.
- Does not automatically retrain the model.
- Currently uses the uploaded filename as its reference.

Filenames are not immutable image identifiers. Unique inference identifiers,
image hashes, reviewer identity, and access control remain gaps.

## Verification boundaries

Local verification on 2026-11-08 included:

- Real-checkpoint classification API inference.
- Grad-CAM API generation.
- Browser classification and overlay display.
- Reviewer-feedback submission.
- A total Python suite result of 75 passed with live Orthanc enabled.
- A checkpoint-absent API result of 12 passed and 3 explicitly skipped.

These tests verify software behavior, not diagnostic validity.

CI does not provision the trained checkpoint. Real-model tests skip when
the required artifact is absent; a green CI run does not establish model
performance.

## Limitations and risks

- Incomplete historical training/evaluation provenance.
- Very low recall for several labels at the default threshold.
- No established probability calibration.
- No external or clinical validation.
- No confidence intervals in the supplied report.
- No established subgroup fairness analysis.
- Potential dataset, acquisition, label, and population biases.
- Resizing may discard image detail.
- Model responses to non-chest images are not reliably rejected.
- Scores and heatmaps may encourage overconfidence.
- The current stack is a local educational deployment, not a secure
  clinical platform.

## Artifact availability and reproducibility

Model files and dataset images are not distributed through normal Git
history. A fresh clone does not provide a working trained classifier by
itself.

The project currently has no operational DVC remote or complete
artifact-retrieval mechanism.

Before claiming reproducible evaluation, record:

1. Checkpoint checksum and trusted distribution location.
2. Dataset manifest checksum and image inventory.
3. Patient-split verification.
4. Exact source revision and configuration snapshot.
5. Python dependency versions.
6. Evaluation command and complete output.

## Planned improvements

1. Establish immutable checkpoint-to-report provenance.
2. Add validation-set threshold selection and calibration analysis.
3. Report uncertainty intervals and subgroup results where data permits.
4. Add external evaluation when an appropriate dataset is available.
5. Add image-content validation and robust out-of-distribution handling.
6. Introduce immutable inference references and controlled reviewer access.
7. Validate attribution behavior beyond visual inspection.

## License and ownership

Review repository licensing, checkpoint ownership, and dataset terms
separately before redistribution.

No clinical approval, regulatory clearance, or medical-device certification
is claimed.