
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv.gz"
RANKING_FILE = PROJECT_ROOT / "ml" / "artifacts" / "popularity_ranking.csv"
OUTPUT_FILE = PROJECT_ROOT / "ml" / "artifacts" / "item_similarity_topk.csv"

POSITIVE_RATING = 4.0
TOP_K = 20
CHUNK_SIZE = 500_000


def main():
    for path in [TRAIN_FILE, RANKING_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading candidate movie catalog...")
    ranking = pd.read_csv(RANKING_FILE)

    movie_ids = ranking["movieId"].astype(int).drop_duplicates().tolist()
    movie_to_index = {
        movie_id: index for index, movie_id in enumerate(movie_ids)
    }

    print(f"Candidate movies: {len(movie_ids):,}")

    print("Building positive user-movie interactions...")

    row_parts = []
    col_parts = []
    max_user_id = 0
    total_interactions = 0

    # Use only positive ratings from the training split.
    # Rows represent users; columns represent candidate movies.
    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=["userId", "movieId", "rating"],
        chunksize=CHUNK_SIZE,
    ):
        positive = chunk[
            (chunk["rating"] >= POSITIVE_RATING)
            & chunk["movieId"].isin(movie_to_index)
        ]

        if positive.empty:
            continue

        users = positive["userId"].to_numpy(dtype=np.int32)
        movies = (
            positive["movieId"]
            .map(movie_to_index)
            .to_numpy(dtype=np.int32)
        )

        row_parts.append(users)
        col_parts.append(movies)

        max_user_id = max(max_user_id, int(users.max()))
        total_interactions += len(positive)

    if not row_parts:
        raise ValueError(
            "No positive training interactions found for candidate movies."
        )

    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)

    # Binary interaction matrix: 1 means the user rated the movie >= 4.
    interactions = csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (rows, cols),
        ),
        shape=(max_user_id + 1, len(movie_ids)),
        dtype=np.float32,
    )

    interactions.sum_duplicates()
    interactions.data[:] = 1.0

    # Each row now represents a movie and each column a user.
    item_user_matrix = interactions.T.tocsr()

    print(f"Positive rating rows processed: {total_interactions:,}")
    print(f"Users represented: {max_user_id:,}")
    print(f"Sparse matrix shape: {item_user_matrix.shape}")
    print(f"Non-zero interactions: {item_user_matrix.nnz:,}")

    del rows, cols, row_parts, col_parts, interactions

    neighbor_count = min(TOP_K + 1, len(movie_ids))

    print("Fitting item similarity model...")
    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=neighbor_count,
        n_jobs=-1,
    )
    model.fit(item_user_matrix)

    print("Finding nearest movies in batches...")
    output_parts = []
    batch_size = 256

    for start in range(0, len(movie_ids), batch_size):
        end = min(start + batch_size, len(movie_ids))
        distances, indices = model.kneighbors(
            item_user_matrix[start:end],
            return_distance=True,
        )

        for local_index, (row_distances, row_indices) in enumerate(
            zip(distances, indices)
        ):
            source_index = start + local_index
            source_movie_id = movie_ids[source_index]

            saved_neighbors = 0

            for distance, neighbor_index in zip(
                row_distances, row_indices
            ):
                if neighbor_index == source_index:
                    continue

                similarity = float(1.0 - distance)

                # Ignore zero or negative similarity.
                if similarity <= 0:
                    continue

                output_parts.append(
                    (
                        source_movie_id,
                        movie_ids[int(neighbor_index)],
                        similarity,
                        saved_neighbors + 1,
                    )
                )

                saved_neighbors += 1

                if saved_neighbors >= TOP_K:
                    break

        print(f"Processed movies: {end:,}/{len(movie_ids):,}")

    print("Saving similar-movie results...")
    result = pd.DataFrame(
        output_parts,
        columns=[
            "movieId",
            "similarMovieId",
            "similarity",
            "rank",
        ],
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n--- ITEM SIMILARITY MODEL COMPLETE ---")
    print(f"Movies processed: {len(movie_ids):,}")
    print(f"Similarity relationships saved: {len(result):,}")
    print(f"Output file: {OUTPUT_FILE}")

    if not result.empty:
        print("\nSample results:")
        print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
