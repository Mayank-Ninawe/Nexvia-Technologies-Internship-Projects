# Large-Scale Personalized Recommendation Engine

A movie recommendation system built using the MovieLens 25M dataset, Python, item-based collaborative filtering, FastAPI, and React.

The project explores popularity-based recommendation, item-similarity retrieval, offline ranking evaluation, REST API development, and an interactive web application.

<p align="center">
  <a href="https://large-scale-recommendation-engine.vercel.app/"><strong>Live Demo</strong></a> ·
  <a href="https://recommendation-engine-api.onrender.com/docs"><strong>API Documentation</strong></a> ·
  <a href="https://github.com/Mayank-Ninawe/large-scale-recommendation-engine"><strong>GitHub Repository</strong></a>
</p>

## Table of Contents

- [Overview](#overview)
- [Live Deployment](#live-deployment)
- [Current Features](#current-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Evaluation Results](#evaluation-results)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Prepare the Dataset](#prepare-the-dataset)
- [Build Recommendation Artifacts](#build-recommendation-artifacts)
- [Run the Backend](#run-the-backend)
- [Run the Frontend](#run-the-frontend)
- [Run Tests](#run-tests)
- [API Endpoints](#api-endpoints)
- [Current Limitations and Future Work](#current-limitations-and-future-work)
- [License and Dataset Attribution](#license-and-dataset-attribution)

## Overview

The goal of this project is to explore how recommendation systems can use historical user ratings and item-to-item similarities to suggest relevant movies.

The project includes a popularity-based baseline and an item-based collaborative filtering implementation, with offline ranking evaluation comparing their recommendation quality.

The deployed application provides three recommendation experiences:

1. Popular movie recommendations.
2. Similar-movie recommendations based on a selected movie.
3. Precomputed personalized demo recommendations for selected users.

The application uses a React frontend connected to a FastAPI backend.

## Live Deployment

The application is deployed using Vercel and Render.

| Component | Platform | URL |
|---|---|---|
| Frontend | Vercel | https://large-scale-recommendation-engine.vercel.app/ |
| Backend API | Render | https://recommendation-engine-api.onrender.com/ |
| Interactive API Documentation | FastAPI Swagger UI | https://recommendation-engine-api.onrender.com/docs |
| API Health Check | Render | https://recommendation-engine-api.onrender.com/health |

**Deployment architecture:**

- React frontend hosted on Vercel.
- FastAPI backend hosted on Render.
- Vercel rewrites `/api/*` requests to the Render backend.
- Recommendation artifacts are loaded by the backend.

Note: Render's free web service may sleep after inactivity. The first request after a period of inactivity can take longer.

## Current Features

- MovieLens 25M dataset auditing and validation.
- Chronological training, validation, and test split.
- Popularity-based recommendation baseline with Bayesian smoothing.
- Item-based collaborative filtering using positive user ratings in the offline pipeline.
- Top-K similar-movie artifact generation.
- Offline Recall@10 and NDCG@10 evaluation.
- FastAPI endpoints for popular, similar-movie, and personalized demo recommendations.
- React and Vite frontend for interactive movie recommendations.
- Precomputed personalized demo recommendations for user IDs 1, 2, and 3.
- Automated API tests using pytest.
- Live frontend and backend deployment.

## Architecture

```text
                    MovieLens 25M
                         |
                         v
               Dataset Audit and
                Temporal Split
                         |
             +-----------+-----------+
             |                       |
             v                       v
      Popularity Baseline     Item Similarity Model
             |                       |
             v                       v
      Popularity Ranking       Top-K Similar-Movie
             |                    Artifact
             |                       |
             +-----------+-----------+
                         |
                         v
                  Recommendation
                     Artifacts
                         |
                         v
                   FastAPI Backend
                         |
                         v
                  Vercel API Rewrite
                         |
                         v
                  React + Vite UI
```

The offline item-based collaborative filtering pipeline uses historical user ratings to calculate item similarities and generate recommendation artifacts.

The deployed API serves recommendations from these precomputed artifacts. Users
1–3 use included precomputed profiles, while other valid MovieLens users are
served dynamically from the local ratings database when it is available.

## Dataset

This project uses the MovieLens 25M ratings dataset.

Dataset audit results:

| Metric | Value |
|---|---:|
| Total ratings | 25,000,095 |
| Unique users | 162,541 |
| Unique rated movies | 59,047 |
| Total movies in metadata | 62,423 |
| Missing rating values | 0 |
| Duplicate movie IDs in movie metadata | 0 |

The processed training dataset contains 20,062,533 interactions. The split is chronological per user to reduce temporal leakage during evaluation.

| Split | Interactions |
|---|---:|
| Training | 20,062,533 |
| Validation | 2,506,501 |
| Test | 2,431,061 |

A rating of 4.0 or higher is treated as a relevant interaction for the reported ranking evaluation.

Download MovieLens 25M from the official source:

https://grouplens.org/datasets/movielens/25m/

Extract the dataset so that the raw files are available under `data/raw/ml-25m/`.

Do not commit the full dataset to Git.

## Evaluation Results

The following results come from the offline item-similarity evaluation report. Both methods use the same evaluated user cohort and exclude items seen in training.

| Metric | Popularity Baseline | Item-Based Collaborative Filtering |
|---|---:|---:|
| Evaluated users | 150,626 | 150,626 |
| Recall@10 | 0.024811 | 0.081949 |
| NDCG@10 | 0.020305 | 0.070637 |

### Interpretation

- Recall@10 improves by 0.057138 in absolute terms.
- NDCG@10 improves by 0.050332 in absolute terms.
- Item-based collaborative filtering achieves approximately 3.30 times the baseline Recall@10 on this evaluation cohort.

These are offline ranking metrics, not online A/B test results. They do not establish production performance or business impact.

A separate popularity evaluation report uses a different protocol and cohort. Its results should not be directly compared with the table above.

## Technology Stack

### Currently implemented

- Python
- Pandas and NumPy
- FastAPI
- Uvicorn
- React
- Vite
- pytest
- Git and GitHub
- Vercel
- Render

### Planned or not yet verified as implemented

- TensorFlow Recommenders and Two-Tower retrieval
- FAISS vector search
- Deep & Cross Network (DCN) ranking
- PySpark distributed processing
- Redis caching
- Docker-based deployment
- Airflow orchestration
- NVIDIA Triton inference serving

These technologies are part of the broader project direction, not claims about the current working implementation.

## Project Structure

```text
large-scale-recommendation-engine/
├── backend/
│   └── main.py
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── docs/
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── vite.config.js
│   └── vercel.json
├── infra/
├── ml/
│   ├── evaluation/
│   │   ├── evaluate_item_similarity.py
│   │   ├── evaluate_popularity.py
│   │   └── evaluate_popularity_v2.py
│   ├── retrieval/
│   │   ├── item_similarity.py
│   │   └── popularity_baseline.py
│   └── artifacts/
│       ├── item_similarity_topk.csv
│       ├── popularity_ranking.csv
│       └── demo_recommendations.json
├── scripts/
│   ├── audit_dataset.py
│   ├── prepare_dataset.py
│   └── validate_split.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

Generated data and model artifacts may be excluded from Git depending on their size. The deployed backend requires the three recommendation artifacts listed above to be available.

## Setup and Installation

### Prerequisites

- Python 3.10
- Node.js and npm
- Git
- MovieLens 25M dataset

Run commands from the repository root in PowerShell.

### 1. Clone the repository

```powershell
git clone https://github.com/Mayank-Ninawe/large-scale-recommendation-engine.git

cd large-scale-recommendation-engine
```

### 2. Create and activate a virtual environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, invoke the environment's Python executable directly instead.

### 3. Install dependencies

For application dependencies:

```powershell
python -m pip install -r requirements.txt
```

For development and testing:

```powershell
python -m pip install -r requirements-dev.txt
```

### 4. Install frontend dependencies

```powershell
npm --prefix .\frontend install
```

## Prepare the Dataset

Run the dataset audit:

```powershell
python .\scripts\audit_dataset.py
```

Prepare chronological train, validation, and test files:

```powershell
python .\scripts\prepare_dataset.py
```

Validate the generated split:

```powershell
python .\scripts\validate_split.py
```

The processed training file is expected at:

`data/processed/train.csv.gz`

Use the generated audit and validation reports to verify dataset integrity.

## Build Recommendation Artifacts

Run these commands from the repository root after preparing the dataset.

### 1. Build the popularity baseline

```powershell
python .\ml\retrieval\popularity_baseline.py
```

### 2. Build item-similarity recommendations

```powershell
python .\ml\retrieval\item_similarity.py
```

The backend expects these artifacts:

- `ml/artifacts/popularity_ranking.csv`
- `ml/artifacts/item_similarity_topk.csv`
- `ml/artifacts/demo_recommendations.json`

The curated artifacts are included in the repository. The raw MovieLens
download, processed training files, and `data/processed/ratings.db` remain
local-only because of their size and dataset terms. Generate the local ratings
database and processed files before using dynamic recommendations for arbitrary
users.

### 3. Evaluate the models

```powershell
python .\ml\evaluation\evaluate_popularity.py
python .\ml\evaluation\evaluate_popularity_v2.py
python .\ml\evaluation\evaluate_item_similarity.py
```

Evaluation reports are written under `data/reports/` by the respective scripts.

## Run the Backend

Ensure the three recommendation artifacts and the local ratings database exist.

Start FastAPI from the repository root:

```powershell
python -m uvicorn backend.main:app --reload --port 8001
```

The local API is available at:

- Base URL: http://127.0.0.1:8001/
- Interactive API documentation: http://127.0.0.1:8001/docs
- Health check: http://127.0.0.1:8001/health

Keep this terminal running while using the frontend.

## Run the Frontend

Open a second PowerShell terminal at the repository root:

```powershell
npm --prefix .\frontend run dev
```

Open the local URL printed by Vite, usually:

http://localhost:5173

The Vite development server proxies `/api` requests to the local FastAPI server on port 8001.

To create a production build:

```powershell
npm --prefix .\frontend run build
```

## Run Tests

Install development dependencies, then run:

```powershell
python -m pytest tests\test_api.py -v
```

The API test suite covers home and health endpoints, recommendation responses,
input limits, known and unknown IDs, demo profiles, and dynamic
recommendations for non-demo users.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API home and status |
| GET | `/health` | Health check and artifact loading status |
| GET | `/recommendations/popular?limit=10` | Popular movie recommendations |
| GET | `/recommendations/movie/{movie_id}?limit=10` | Similar movies |
| GET | `/recommendations/user/{user_id}?limit=10` | Personalized recommendations |

The optional `limit` parameter defaults to `10` and accepts values from `1` to `50`.

### Example requests

```text
GET https://recommendation-engine-api.onrender.com/health

GET https://recommendation-engine-api.onrender.com/recommendations/popular?limit=10

GET https://recommendation-engine-api.onrender.com/recommendations/movie/318?limit=10

GET https://recommendation-engine-api.onrender.com/recommendations/user/1?limit=10
```

### Recommendation behavior

- **Popular movies:** Ranked using the precomputed popularity baseline. Results include `rating_count`, `average_rating`, and `weighted_score`.
- **Similar movies:** Returns related movies ordered by item similarity. Results include `similarity`.
- **Personalized recommendations:** Users `1`, `2`, and `3` use included precomputed profiles. Other valid users use movies rated `4.0` or higher and item similarities weighted by their ratings. Results include `personalized_score` and `liked_movies_used`.
- **Seen-item exclusion:** Dynamic recommendations exclude movies already rated by the user.
- **Unknown IDs:** Unknown movies and users return HTTP `404`.
- **Invalid limits:** Values outside `1` to `50` are rejected by API validation.

### Example response fields

A popular recommendation includes fields such as:

```json
{
  "movieId": 318,
  "title": "The Shawshank Redemption (1994)",
  "genres": "Crime|Drama",
  "rating_count": 75362,
  "average_rating": 4.418,
  "weighted_score": 4.418
}
```

A personalized recommendation includes fields such as:

```json
{
  "movieId": 4226,
  "title": "Memento (2000)",
  "genres": "Mystery|Thriller",
  "personalized_score": 4.1816
}
```

Similar-movie responses include a `similarity` score. Exact scores depend on the generated recommendation artifacts.

Open the interactive API documentation to explore the available endpoints:

https://recommendation-engine-api.onrender.com/docs

## Current Limitations and Future Work

The current implementation is a working recommendation-system baseline with a deployed web application. It is not yet the complete neural two-stage architecture originally envisioned.

Current limitations:

1. The local ratings database is required for dynamic recommendations and is not committed because of its size.
2. The API serves generated recommendation artifacts rather than running a continuously updated training or inference pipeline.
3. The deployed implementation does not yet include the planned Two-Tower retrieval model, FAISS vector index, or deep-learning ranking model.

Potential next steps:

1. Implement dynamic user-based recommendation generation.
2. Add a Two-Tower retrieval model and evaluate it against item-based retrieval.
3. Integrate FAISS for approximate nearest-neighbor retrieval and benchmark it.
4. Add a learned ranking model such as DCN and compare it using the same evaluation protocol.
5. Add Redis caching and measure its effect on latency.
6. Add reproducible Docker deployment and workflow orchestration.
7. Expand monitoring, API observability, and performance testing.

Advanced components should be described as completed only after their implementation and results have been verified.

## License and Dataset Attribution

MovieLens data is provided by GroupLens Research. Refer to the official dataset page for applicable terms and attribution:

https://grouplens.org/datasets/movielens/25m/
