
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"

TEST_FILE = DATA_DIR / "processed" / "test.csv.gz"
RANKING_FILE = ARTIFACT_DIR / "popularity_ranking.csv"
REPORT_FILE = DATA_DIR / "reports" / "popularity_evaluation.txt"

K = 10
RELEVANT_RATING = 4.0
CHUNK_SIZE = 500_000


def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance[:k], dtype=float)
    if relevance.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(relevance / discounts))


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    if not RANKING_FILE.exists():
        raise FileNotFoundError(
            f"Popularity ranking not found: {RANKING_FILE}. "
            "Run ml/retrieval/popularity_baseline.py first."
        )

    ranking = pd.read_csv(RANKING_FILE)

    # This baseline recommends the same global top-K movies to everyone.
    top_movies = ranking["movieId"].head(K).tolist()
    top_movie_set = set(top_movies)

    # Keep only test users with at least one relevant test rating.
    relevant_by_user = {}

    print("Reading test ratings in chunks...")

    for chunk in pd.read_csv(
        TEST_FILE,
        usecols=["userId", "movieId", "rating"],
        chunksize=CHUNK_SIZE,
    ):
        relevant = chunk[chunk["rating"] >= RELEVANT_RATING]

        for user_id, group in relevant.groupby("userId"):
            relevant_by_user.setdefault(int(user_id), set()).update(
                group["movieId"].astype(int).tolist()
            )

    # Compute ranking metrics for users with relevant test items.
    recalls = []
    ndcgs = []

    for relevant_items in relevant_by_user.values():
        hits = [1 if movie_id in relevant_items else 0 for movie_id in top_movies]

        recall = sum(hits) / len(relevant_items)

        dcg = dcg_at_k(hits, K)
        ideal_hits = min(len(relevant_items), K)
        ideal_dcg = dcg_at_k([1] * ideal_hits, K)
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        recalls.append(recall)
        ndcgs.append(ndcg)

    if not recalls:
        raise ValueError(
            "No test users have ratings at or above the relevance threshold."
        )

    mean_recall = float(np.mean(recalls))
    mean_ndcg = float(np.mean(ndcgs))

    print("\n--- POPULARITY BASELINE EVALUATION ---")
    print(f"Relevance threshold: rating >= {RELEVANT_RATING}")
    print(f"K: {K}")
    print(f"Evaluated users: {len(recalls):,}")
    print(f"Recall@{K}: {mean_recall:.6f}")
    print(f"NDCG@{K}:   {mean_ndcg:.6f}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("Popularity Baseline Evaluation\n")
        report.write("=" * 35 + "\n")
        report.write(f"Relevance threshold: rating >= {RELEVANT_RATING}\n")
        report.write(f"K: {K}\n")
        report.write(f"Evaluated users: {len(recalls):,}\n")
        report.write(f"Recall@{K}: {mean_recall:.6f}\n")
        report.write(f"NDCG@{K}: {mean_ndcg:.6f}\n")

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
