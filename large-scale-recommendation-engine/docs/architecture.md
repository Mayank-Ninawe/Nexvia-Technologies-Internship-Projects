# Recommendation Engine — Architecture

## 1. Project Overview

This project implements a movie recommendation system using the
MovieLens 25M dataset. The current verified implementation includes
a popularity baseline, item-based collaborative filtering, a FastAPI
backend, and a React frontend.

## 2. Dataset

Dataset: MovieLens 25M

Verified dataset audit:
- Ratings: 25,000,095
- Unique users: 162,541
- Unique rated movies: 59,047
- Movie metadata entries: 62,423
- Missing rating values reported by audit: 0
- Duplicate movie IDs reported by audit: 0

The project includes processed temporal data splits. The evaluation
uses ratings >= 4.0 as the relevance threshold.

## 3. Current System Components

### Recommendation models
- Popularity-based recommendation baseline.
- Item-based collaborative filtering using precomputed movie
  similarity relationships.

### Backend
FastAPI exposes endpoints for health checks, popular movies,
similar movies, and personalized recommendations.

Verified endpoint tests:
`python -m pytest tests\test_api.py -q`

Previously observed result: 12 tests passed.

### Frontend
The React application includes:
- Popular Movies
- Personalized Recommendations
- Similar Movies

The production build was previously verified with `npm run build`.

## 4. API Endpoints

- `GET /`
- `GET /health`
- `GET /recommendations/popular?limit=10`
- `GET /recommendations/movie/{movie_id}?limit=10`
- `GET /recommendations/user/{user_id}?limit=10`

The `limit` parameter defaults to 10 and supports values from 1 to 50.

Consult the root README and the running API's `/docs` endpoint for
the latest request and response details.

## 5. Offline Evaluation

The item-similarity evaluation report uses:
- Candidate catalog size: 11,464
- Relevance threshold: rating >= 4.0
- K: 10
- Evaluated users: 150,626
- Seen training items excluded from recommendations
- Both methods evaluated on the same user cohort

| Metric | Popularity | Item-based CF | Absolute difference |
|---|---:|---:|---:|
| Recall@10 | 0.024811 | 0.081949 | +0.057138 |
| NDCG@10 | 0.020305 | 0.070637 | +0.050332 |

These values are from `data/reports/item_similarity_evaluation.txt`.
The results support item-based collaborative filtering outperforming
the popularity baseline under this evaluation protocol.

Do not directly combine these numbers with results from the separate
popularity evaluation reports, which use different evaluation cohorts
or protocols.

## 6. Recommendation Behaviour

The current API implementation supports personalized recommendations
based on positive user rating history, exclusion of already-seen items,
and popularity fallback when suitable personalized candidates are
unavailable. Verify the current README and API tests before changing
these behaviours.

## 7. Configuration

`.env.example` documents application, dataset, Redis connection, and
logging settings.

The presence of Redis settings does not establish that a Redis cache
is implemented or running.

## 8. Components Requiring Further Verification

The following components were not verified as implemented or runnable
during the current repository audit:
- Docker/Compose infrastructure
- Airflow DAGs
- Triton model-serving files
- Redis caching integration
- TensorFlow Recommenders Two-Tower retrieval
- FAISS retrieval index
- DCN ranking model

Treat these as pending until implementation and execution evidence
is available.

## 9. Reproduction Checks

From the repository root:

1. Activate the project's Python virtual environment.
2. Install the project's documented dependencies.
3. Run the API tests:
   `python -m pytest tests\test_api.py -q`
4. Build the frontend from its directory:
   `npm run build`
5. Start the backend using the instructions in the root README.
6. Start the frontend using its documented development command.
7. Verify the API and frontend manually.

Use the current README for exact setup commands and required data files.

## 10. Limitations

- Offline ranking metrics do not guarantee real-world user satisfaction.
- Results depend on the dataset, candidate catalog, relevance threshold,
  temporal split, and evaluation protocol.
- Advanced retrieval, ranking, orchestration, caching, and serving
  components must not be described as completed without verification.
