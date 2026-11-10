# Nexvia Technologies Internship Projects

This repository contains three machine-learning and software-engineering
projects completed during the Nexvia Technologies internship.

## Projects

| Project | Description | Documentation |
|---|---|---|
| [AIML-01](./AIML-01/) | SentinelGraph: real-time financial fraud detection with XGBoost, FastAPI, Kafka, PostgreSQL, and React. | [README](./AIML-01/README.md) |
| [AIML-02](./AIML-02/) | Explainable chest X-ray classification with DenseNet-121, Grad-CAM, FastAPI, React, and an experimental MONAI segmentation workflow. | [README](./AIML-02/README.md) |
| [Large-Scale Recommendation Engine](./large-scale-recommendation-engine/) | MovieLens-based recommendation system with item-based collaborative filtering, FastAPI, and React. | [README](./large-scale-recommendation-engine/README.md) |

Each project has its own detailed README covering its architecture, setup,
datasets, evaluation results, API endpoints, and limitations.

## Repository structure

```text
.
├── AIML-01/
├── AIML-02/
└── large-scale-recommendation-engine/
```

## General notes

- Read the project-specific README before running a project.
- Python virtual environments, dependency caches, local secrets, datasets,
  trained model files, and generated artifacts are intentionally excluded from
  version control.
- Copy each project's `.env.example` to `.env` and set local values as needed.
- Dataset and model availability varies by project; see the relevant README for
  download and preparation instructions.

## License and responsible use

These projects are intended for educational and portfolio use. The medical
imaging project is not a medical device and must not be used for diagnosis or
patient-care decisions. Dataset licenses and attribution requirements remain
the responsibility of anyone downloading or using the datasets.
