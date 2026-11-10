
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw" / "ml-25m"
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"

TRAIN_FILE = PROCESSED_DIR / "train.csv.gz"
MOVIES_FILE = RAW_DIR / "movies.csv"

# A movie must have at least this many training ratings
# to qualify for the main popularity ranking.
MIN_RATINGS = 50

# Number of recommendations to generate by default.
TOP_K = 10


def build_popularity_ranking():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_FILE}. "
            "Run scripts/prepare_dataset.py first."
        )

    if not MOVIES_FILE.exists():
        raise FileNotFoundError(
            f"Movie metadata not found: {MOVIES_FILE}"
        )

    print("Loading training ratings...")
    ratings = pd.read_csv(
        TRAIN_FILE,
        usecols=["userId", "movieId", "rating"],
    )

    print("Calculating movie statistics...")
    movie_stats = (
        ratings.groupby("movieId")["rating"]
        .agg(rating_count="count", average_rating="mean")
        .reset_index()
    )

    # Bayesian weighted rating:
    # Movies with fewer ratings are pulled toward the global mean.
    global_mean = ratings["rating"].mean()
    minimum_count = MIN_RATINGS

    movie_stats["weighted_score"] = (
        (movie_stats["rating_count"] / (
            movie_stats["rating_count"] + minimum_count
        )) * movie_stats["average_rating"]
        +
        (minimum_count / (
            movie_stats["rating_count"] + minimum_count
        )) * global_mean
    )

    # Keep movies with enough evidence for the main ranking.
    eligible_movies = movie_stats[
        movie_stats["rating_count"] >= MIN_RATINGS
    ].copy()

    movies = pd.read_csv(MOVIES_FILE)

    result = eligible_movies.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left",
        validate="one_to_one",
    )

    result = result.sort_values(
        ["weighted_score", "rating_count", "movieId"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    result.insert(0, "rank", range(1, len(result) + 1))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ARTIFACT_DIR / "popularity_ranking.csv"

    result.to_csv(output_file, index=False)

    print("\n--- BASELINE SUMMARY ---")
    print(f"Global average rating: {global_mean:.3f}")
    print(f"Minimum ratings required: {MIN_RATINGS}")
    print(f"Eligible movies: {len(result):,}")
    print(f"Saved ranking: {output_file}")

    print(f"\n--- TOP {TOP_K} MOVIES ---")
    display_columns = [
        "rank",
        "title",
        "genres",
        "rating_count",
        "average_rating",
        "weighted_score",
    ]
    print(result[display_columns].head(TOP_K).to_string(index=False))

    return result


if __name__ == "__main__":
    build_popularity_ranking()
