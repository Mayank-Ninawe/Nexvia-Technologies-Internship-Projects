
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"

TRAIN_FILE = DATA_DIR / "processed" / "train.csv.gz"
TEST_FILE = DATA_DIR / "processed" / "test.csv.gz"
RANKING_FILE = ARTIFACT_DIR / "popularity_ranking.csv"
REPORT_FILE = DATA_DIR / "reports" / "popularity_evaluation_v2.txt"

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
    for path in [TRAIN_FILE, TEST_FILE, RANKING_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading training history...")
    train = pd.read_csv(
        TRAIN_FILE,
        usecols=["userId", "movieId"],
    )

    # Known catalog: movies with training interactions.
    known_movies = set(train["movieId"].unique())

    # User history: movies each user has already rated in training.
    seen_by_user = (
        train.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    print("Loading popularity ranking...")
    ranking = pd.read_csv(RANKING_FILE)

    # Restrict recommendations to movies present in the training catalog.
    ranking = ranking[ranking["movieId"].isin(known_movies)]

    # Use enough ranked movies to skip each user's seen items.
    ranked_movie_ids = ranking["movieId"].astype(int).tolist()

    print("Loading relevant test interactions...")
    relevant_by_user = {}

    for chunk in pd.read_csv(
        TEST_FILE,
        usecols=["userId", "movieId", "rating"],
        chunksize=CHUNK_SIZE,
    ):
        relevant = chunk[
            (chunk["rating"] >= RELEVANT_RATING)
            & (chunk["movieId"].isin(known_movies))
        ]

        for user_id, group in relevant.groupby("userId"):
            relevant_by_user.setdefault(int(user_id), set()).update(
                group["movieId"].astype(int).tolist()
            )

    recalls = []
    ndcgs = []
    evaluated_users = 0

    print("Evaluating recommendations...")

    for user_id, relevant_items in relevant_by_user.items():
        seen_items = seen_by_user.get(user_id, set())

        # Relevant items must be eligible for recommendation:
        # known from training and not already seen by the user.
        eligible_relevant = relevant_items - seen_items

        if not eligible_relevant:
            continue

        # Filter the global ranking to exclude the user's history.
        recommendations = []
        for movie_id in ranked_movie_ids:
            if movie_id not in seen_items:
                recommendations.append(movie_id)

            if len(recommendations) == K:
                break

        if not recommendations:
            continue

        hits = [
            1 if movie_id in eligible_relevant else 0
            for movie_id in recommendations
        ]

        recall = sum(hits) / len(eligible_relevant)

        dcg = dcg_at_k(hits, K)
        ideal_dcg = dcg_at_k(
            [1] * min(len(eligible_relevant), K),
            K,
        )
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

        recalls.append(recall)
        ndcgs.append(ndcg)
        evaluated_users += 1

    if not recalls:
        raise ValueError(
            "No eligible users found. Check the split, ranking, "
            "and relevance threshold."
        )

    mean_recall = float(np.mean(recalls))
    mean_ndcg = float(np.mean(ndcgs))

    print("\n--- IMPROVED POPULARITY EVALUATION ---")
    print(f"Relevance threshold: rating >= {RELEVANT_RATING}")
    print(f"K: {K}")
    print(f"Training catalog movies: {len(known_movies):,}")
    print(f"Evaluated users: {evaluated_users:,}")
    print(f"Recall@{K}: {mean_recall:.6f}")
    print(f"NDCG@{K}:   {mean_ndcg:.6f}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as report:
        report.write("Improved Popularity Baseline Evaluation\n")
        report.write("=" * 42 + "\n")
        report.write("Protocol: training-catalog, unseen relevant test items\n")
        report.write(f"Relevance threshold: rating >= {RELEVANT_RATING}\n")
        report.write(f"K: {K}\n")
        report.write(f"Training catalog movies: {len(known_movies):,}\n")
        report.write(f"Evaluated users: {evaluated_users:,}\n")
        report.write(f"Recall@{K}: {mean_recall:.6f}\n")
        report.write(f"NDCG@{K}: {mean_ndcg:.6f}\n")

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
