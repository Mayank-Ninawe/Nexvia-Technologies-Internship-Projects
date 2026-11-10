
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
ARTIFACTS = ROOT / "ml" / "artifacts"

TRAIN_FILE = DATA / "processed" / "train.csv.gz"
TEST_FILE = DATA / "processed" / "test.csv.gz"
RANKING_FILE = ARTIFACTS / "popularity_ranking.csv"
SIMILARITY_FILE = ARTIFACTS / "item_similarity_topk.csv"
REPORT_FILE = DATA / "reports" / "item_similarity_evaluation.txt"

K = 10
RELEVANT_RATING = 4.0
CHUNK_SIZE = 500_000


# --------------------------------------------------
# 2. METRICS
# --------------------------------------------------

def dcg_at_k(relevance, k=K):
    """Calculate Discounted Cumulative Gain at K."""
    relevance = np.asarray(relevance[:k], dtype=np.float64)

    if len(relevance) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance / discounts))


def calculate_metrics(recommendations, relevant_items):
    """Calculate Recall@K and NDCG@K."""

    if not relevant_items:
        return 0.0, 0.0

    hits = [
        1 if movie_index in relevant_items else 0
        for movie_index in recommendations[:K]
    ]

    recall = sum(hits) / len(relevant_items)

    ideal_hits = min(len(relevant_items), K)
    ideal_dcg = dcg_at_k([1] * ideal_hits)

    ndcg = dcg_at_k(hits) / ideal_dcg if ideal_dcg else 0.0

    return recall, ndcg


# --------------------------------------------------
# 3. MAIN EVALUATION
# --------------------------------------------------

def main():

    # Verify required input files.
    required_files = [
        TRAIN_FILE,
        TEST_FILE,
        RANKING_FILE,
        SIMILARITY_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------
    # A. LOAD CANDIDATE CATALOG
    # --------------------------------------------------

    print("Loading candidate catalog...")

    ranking = pd.read_csv(RANKING_FILE)

    movie_ids = ranking["movieId"].astype(int).tolist()

    movie_to_index = {
        movie_id: index
        for index, movie_id in enumerate(movie_ids)
    }

    candidate_ids = set(movie_ids)
    num_movies = len(movie_ids)

    print(f"Candidate movies: {num_movies:,}")

    # --------------------------------------------------
    # B. LOAD ITEM SIMILARITIES
    # --------------------------------------------------

    print("Loading item similarity relationships...")

    similarities = pd.read_csv(SIMILARITY_FILE)

    required_similarity_columns = {
        "movieId",
        "similarMovieId",
        "similarity",
    }

    missing_columns = (
        required_similarity_columns - set(similarities.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Similarity file is missing columns: {missing_columns}"
        )

    # Each source movie has a list of similar movies and scores.
    neighbors = [[] for _ in movie_ids]

    for row in similarities.itertuples(index=False):

        source = movie_to_index.get(int(row.movieId))
        target = movie_to_index.get(int(row.similarMovieId))

        if source is not None and target is not None:
            neighbors[source].append(
                (target, float(row.similarity))
            )

    neighbor_indices = []
    neighbor_weights = []

    for movie_neighbors in neighbors:

        neighbor_indices.append(
            np.asarray(
                [item[0] for item in movie_neighbors],
                dtype=np.int32,
            )
        )

        neighbor_weights.append(
            np.asarray(
                [item[1] for item in movie_neighbors],
                dtype=np.float32,
            )
        )

    del similarities

    print("Item similarity relationships loaded.")

    # --------------------------------------------------
    # C. BUILD USER TRAINING HISTORIES
    # --------------------------------------------------

    print("Building user histories from training data...")

    seen_by_user = {}
    positive_history_by_user = {}

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=["userId", "movieId", "rating"],
        chunksize=CHUNK_SIZE,
    ):

        chunk = chunk[chunk["movieId"].isin(candidate_ids)]

        if chunk.empty:
            continue

        chunk = chunk.copy()

        chunk["movie_index"] = (
            chunk["movieId"]
            .map(movie_to_index)
            .astype(np.int32)
        )

        for user_id, group in chunk.groupby(
            "userId",
            sort=False,
        ):

            user_id = int(user_id)

            # All training movies must be excluded from recommendations.
            seen_by_user.setdefault(
                user_id, set()
            ).update(
                group["movie_index"].tolist()
            )

            # Positive ratings form the user's preference history.
            positive = group.loc[
                group["rating"] >= RELEVANT_RATING,
                "movie_index",
            ].tolist()

            if positive:
                positive_history_by_user.setdefault(
                    user_id, []
                ).extend(positive)

    print(
        f"Users with training history: {len(seen_by_user):,}"
    )

    print(
        "Users with positive history: "
        f"{len(positive_history_by_user):,}"
    )

    # --------------------------------------------------
    # D. LOAD RELEVANT TEST INTERACTIONS
    # --------------------------------------------------

    print("Loading relevant test interactions...")

    relevant_by_user = {}

    for chunk in pd.read_csv(
        TEST_FILE,
        usecols=["userId", "movieId", "rating"],
        chunksize=CHUNK_SIZE,
    ):

        relevant = chunk.loc[
            (chunk["rating"] >= RELEVANT_RATING)
            & chunk["movieId"].isin(candidate_ids)
        ].copy()

        if relevant.empty:
            continue

        relevant["movie_index"] = (
            relevant["movieId"]
            .map(movie_to_index)
            .astype(np.int32)
        )

        for user_id, group in relevant.groupby(
            "userId",
            sort=False,
        ):

            relevant_by_user.setdefault(
                int(user_id), set()
            ).update(
                group["movie_index"].tolist()
            )

    print(
        f"Users with relevant test interactions: "
        f"{len(relevant_by_user):,}"
    )

    # --------------------------------------------------
    # E. PREPARE POPULARITY BASELINE
    # --------------------------------------------------

    popularity_indices = np.arange(
        num_movies,
        dtype=np.int32,
    )

    # --------------------------------------------------
    # F. EVALUATE BOTH METHODS
    # --------------------------------------------------

    print("Evaluating both methods on the same users...")

    recalls_item = []
    ndcgs_item = []

    recalls_pop = []
    ndcgs_pop = []

    users_without_positive_history = 0
    users_without_eligible_relevant_items = 0

    total_users = len(relevant_by_user)

    for iteration, (user_id, relevant_items) in enumerate(
        relevant_by_user.items(),
        start=1,
    ):

        if iteration % 10_000 == 0:
            print(
                f"Processed {iteration:,}/{total_users:,} users..."
            )

        seen = seen_by_user.get(user_id, set())

        # Test items already seen in training are not valid targets.
        eligible_relevant = relevant_items - seen

        if not eligible_relevant:
            users_without_eligible_relevant_items += 1
            continue

        positive_history = positive_history_by_user.get(
            user_id, []
        )

        if not positive_history:
            users_without_positive_history += 1
            continue

        # --------------------------------------------------
        # ITEM-BASED COLLABORATIVE FILTERING
        # --------------------------------------------------

        # Accumulate neighbor scores from positively rated movies.
        index_parts = []
        score_parts = []

        for movie_index in positive_history:

            indices = neighbor_indices[movie_index]
            weights = neighbor_weights[movie_index]

            if len(indices):
                index_parts.append(indices)
                score_parts.append(weights)

        if index_parts:

            all_indices = np.concatenate(index_parts)
            all_weights = np.concatenate(score_parts)

            scores = np.bincount(
                all_indices,
                weights=all_weights,
                minlength=num_movies,
            )

        else:
            scores = np.zeros(
                num_movies,
                dtype=np.float64,
            )

        # Never recommend items that exist in training history.
        if seen:
            seen_array = np.fromiter(
                seen,
                dtype=np.int32,
                count=len(seen),
            )

            scores[seen_array] = -np.inf

        # Select only candidates with positive recommendation scores.
        valid_indices = np.flatnonzero(
            np.isfinite(scores) & (scores > 0)
        )

        # Partial sorting is faster than sorting the full catalog.
        if len(valid_indices) > K:

            top_positions = np.argpartition(
                scores[valid_indices],
                -K,
            )[-K:]

            top_indices = valid_indices[top_positions]

            # Sort the selected Top-K candidates by descending score.
            top_indices = top_indices[
                np.argsort(
                    -scores[top_indices],
                    kind="stable",
                )
            ]

        else:

            top_indices = valid_indices[
                np.argsort(
                    -scores[valid_indices],
                    kind="stable",
                )
            ]

        item_recommendations = (
            top_indices.astype(int).tolist()
        )

        # --------------------------------------------------
        # POPULARITY BASELINE
        # --------------------------------------------------

        # Ranking file is already ordered from highest to lowest score.
        pop_recommendations = []

        for index in popularity_indices:

            index = int(index)

            if index not in seen:
                pop_recommendations.append(index)

            if len(pop_recommendations) == K:
                break

        # --------------------------------------------------
        # CALCULATE METRICS
        # --------------------------------------------------

        item_recall, item_ndcg = calculate_metrics(
            item_recommendations,
            eligible_relevant,
        )

        pop_recall, pop_ndcg = calculate_metrics(
            pop_recommendations,
            eligible_relevant,
        )

        recalls_item.append(item_recall)
        ndcgs_item.append(item_ndcg)

        recalls_pop.append(pop_recall)
        ndcgs_pop.append(pop_ndcg)

    # --------------------------------------------------
    # G. AGGREGATE RESULTS
    # --------------------------------------------------

    if not recalls_item:
        raise ValueError(
            "No eligible users with positive candidate history "
            "were found. Check the dataset and evaluation filters."
        )

    item_recall = float(np.mean(recalls_item))
    item_ndcg = float(np.mean(ndcgs_item))

    pop_recall = float(np.mean(recalls_pop))
    pop_ndcg = float(np.mean(ndcgs_pop))

    evaluated_users = len(recalls_item)

    recall_difference = item_recall - pop_recall
    ndcg_difference = item_ndcg - pop_ndcg

    # --------------------------------------------------
    # H. DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 55)

    print(f"Relevant rating threshold: >= {RELEVANT_RATING}")
    print(f"K: {K}")
    print(f"Candidate catalog size: {num_movies:,}")
    print(f"Evaluated users: {evaluated_users:,}")

    print(
        "Users skipped (no positive training history): "
        f"{users_without_positive_history:,}"
    )

    print(
        "Users skipped (no eligible relevant test items): "
        f"{users_without_eligible_relevant_items:,}"
    )

    print("\nPopularity baseline:")
    print(f"Recall@{K}: {pop_recall:.6f}")
    print(f"NDCG@{K}:   {pop_ndcg:.6f}")

    print("\nItem-based collaborative filtering:")
    print(f"Recall@{K}: {item_recall:.6f}")
    print(f"NDCG@{K}:   {item_ndcg:.6f}")

    print("\nDifference (item similarity - popularity):")
    print(f"Recall@{K}: {recall_difference:+.6f}")
    print(f"NDCG@{K}:   {ndcg_difference:+.6f}")

    # --------------------------------------------------
    # I. SAVE REPORT
    # --------------------------------------------------

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "Item-Based Collaborative Filtering Evaluation\n"
        )

        report.write("=" * 48 + "\n")
        report.write(f"Candidate catalog size: {num_movies:,}\n")
        report.write(
            f"Relevant test rating: >= {RELEVANT_RATING}\n"
        )

        report.write(
            "Seen training items are excluded from recommendations\n"
        )

        report.write(
            "Both methods use the same evaluated user cohort\n"
        )

        report.write(f"K: {K}\n")
        report.write(f"Evaluated users: {evaluated_users:,}\n")

        report.write(
            "Users skipped (no positive training history): "
            f"{users_without_positive_history:,}\n"
        )

        report.write(
            "Users skipped (no eligible relevant test items): "
            f"{users_without_eligible_relevant_items:,}\n\n"
        )

        report.write("Popularity baseline:\n")
        report.write(f"Recall@{K}: {pop_recall:.6f}\n")
        report.write(f"NDCG@{K}: {pop_ndcg:.6f}\n\n")

        report.write("Item-based collaborative filtering:\n")
        report.write(f"Recall@{K}: {item_recall:.6f}\n")
        report.write(f"NDCG@{K}: {item_ndcg:.6f}\n\n")

        report.write(
            "Difference (item similarity - popularity):\n"
        )

        report.write(
            f"Recall@{K}: {recall_difference:+.6f}\n"
        )

        report.write(
            f"NDCG@{K}: {ndcg_difference:+.6f}\n"
        )

    print(f"\nReport saved to: {REPORT_FILE}")
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
