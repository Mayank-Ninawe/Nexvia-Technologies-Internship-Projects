# Dataset Access, Terms, and Citation

Status: Phase 12 complete — NIH ChestX-ray14 selected for Workflow A.

## Educational-use notice

This project is for educational purposes only, not for clinical use.
It does not provide clinically validated diagnoses or treatment advice.

## Final Phase 12 decision

Workflow A classification dataset: **NIH ChestX-ray14 / CXR8**.

Reasons:

- The official NIH Box collection is publicly accessible.
- Its usage requirements are straightforward: link to the original download
  site, acknowledge the NIH Clinical Center, and cite the associated CVPR 2017
  paper.
- The metadata supports patient-level train/validation/test separation.
- It avoids CheXpert’s stricter redistribution and hosted-demo restrictions.

The full NIH archive, extracted images, derived features, and trained model
checkpoints must remain outside ordinary Git history.

## Workflow A — NIH ChestX-ray14 / CXR8
### Measured local split counts

After building the local manifest from the official NIH files:

- Train/validation images: 86,524
- Held-out test images: 25,596
- Total images: 112,120
- Extracted PNG images verified locally: 112,120

These are measured counts from the local manifest, not estimates.

### Official source

- Official NIH Box collection:
  https://nihcc.app.box.com/v/ChestXray-NIHCC
- Official NIH announcement:
  https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-research
- Google Cloud Healthcare mirror/documentation:
  https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest

### Dataset contents

- Approximately 112,120 frontal-view chest X-ray images.
- 14 disease labels derived from radiology reports using natural-language
  processing.
- Metadata includes image filename, finding labels, follow-up number, patient
  ID, patient age, patient gender, view position, and image dimensions.
- The official release includes training/validation and test image-index files.

### Important label limitation

The labels were automatically extracted from free-text radiology reports.
They are research labels, not guaranteed expert-verified clinical diagnoses.
The project must clearly document this limitation and must not describe model
outputs as clinical diagnoses.

### Usage and attribution requirements

When using NIH ChestX-ray14, this project will:

1. Link to the official NIH Box download site.
2. Acknowledge the NIH Clinical Center.
3. Cite the CVPR 2017 paper associated with the dataset release.
4. Avoid presenting model predictions as clinically validated diagnoses.
5. Avoid committing the full dataset, bulk patient metadata, or large derived
   artifacts to Git.

### Citation

Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale
Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and
Localization of Common Thorax Diseases. In: 2017 IEEE Conference on Computer
Vision and Pattern Recognition (CVPR); 2017. p. 3462–3471.

DOI: https://doi.org/10.1109/CVPR.2017.369

### Local storage policy

- Raw archive location:
  `data/raw/nih_chestxray14/`
- Extracted dataset location:
  `data/raw/nih_chestxray14/extracted/`
- Dataset manifest location:
  `data/manifests/nih_chestxray14_manifest.csv`
- Processed arrays or cached tensors:
  `data/processed/`
- Model checkpoints:
  `models/`

None of these directories should contain files committed to Git, except small
non-sensitive sample files that are explicitly approved for the repository.

## Alternative considered — CheXpert Small

CheXpert was considered but not selected for this internship implementation.

- Official source:
  https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- Citation DOI:
  https://doi.org/10.71718/y7pj-4v93
- Access requires review and acceptance of the applicable Stanford dataset
  agreement.
- Reported access terms include non-commercial research use, citation, no
  redistribution/republication, and no patient identification.
- Publicly hosting CheXpert images or derivatives would likely conflict with
  those restrictions.

## Workflow B — MSD Task09 Spleen

### Official source

- Medical Segmentation Decathlon:
  http://medicaldecathlon.com/
- Task09 Spleen is the selected segmentation pilot.
- Reported contents: 61 3D CT volumes, comprising 41 training and 20 testing
  volumes.
- Reported archive size: approximately 1.6 GB.
- Reported license: CC BY-SA 4.0.

### Citation

Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon.
Nature Communications. 2022;13:4128.

DOI: https://doi.org/10.1038/s41467-022-30695-9

### Local storage policy

- Raw archive location:
  `data/raw/msd_task09_spleen/`
- Extracted dataset location:
  `data/raw/msd_task09_spleen/extracted/`
- Preprocessed volumes and cached arrays:
  `data/processed/`
- Segmentation checkpoints:
  `models/`

## Privacy and repository rules

- Never commit `.env`, credentials, API keys, database files, or private logs.
- Never commit the full NIH or MSD datasets.
- Never commit trained model checkpoints unless explicitly and deliberately
  approved.
- Never upload real patient data to a public hosted demonstration.
- Use only small, non-sensitive, clearly labeled synthetic samples in the
  public frontend if visual examples are required.