
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-25m"

print("=" * 55)
print("MOVIELENS 25M — DATASET AUDIT")
print("=" * 55)

# Check required dataset files
required_files = ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]

for filename in required_files:
    file_path = DATA_DIR / filename
    print(f"{filename}: {'FOUND' if file_path.exists() else 'MISSING'}")

if not (DATA_DIR / "ratings.csv").exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_DIR}")

# Inspect movie metadata
movies = pd.read_csv(DATA_DIR / "movies.csv")

print("\n--- MOVIE DATA ---")
print(f"Total movies: {len(movies):,}")
print(f"Missing movie IDs: {movies['movieId'].isna().sum():,}")
print(f"Missing titles: {movies['title'].isna().sum():,}")
print(f"Duplicate movie IDs: {movies['movieId'].duplicated().sum():,}")

# Process ratings in chunks to limit memory usage
ratings_file = DATA_DIR / "ratings.csv"
total_rows = 0
missing_values = {"userId": 0, "movieId": 0, "rating": 0, "timestamp": 0}
unique_users = set()
unique_movies = set()
rating_counts = {}
min_timestamp = None
max_timestamp = None

print("\n--- SCANNING RATINGS ---")

for chunk in pd.read_csv(
    ratings_file,
    usecols=["userId", "movieId", "rating", "timestamp"],
    chunksize=1_000_000,
):
    total_rows += len(chunk)

    for column in missing_values:
        missing_values[column] += int(chunk[column].isna().sum())

    unique_users.update(chunk["userId"].dropna().unique())
    unique_movies.update(chunk["movieId"].dropna().unique())

    counts = chunk["rating"].value_counts()
    for rating, count in counts.items():
        rating_counts[float(rating)] = (
            rating_counts.get(float(rating), 0) + int(count)
        )

    chunk_min = int(chunk["timestamp"].min())
    chunk_max = int(chunk["timestamp"].max())

    min_timestamp = (
        chunk_min if min_timestamp is None else min(min_timestamp, chunk_min)
    )
    max_timestamp = (
        chunk_max if max_timestamp is None else max(max_timestamp, chunk_max)
    )

print("\n--- RATINGS DATA ---")
print(f"Total ratings: {total_rows:,}")
print(f"Unique users: {len(unique_users):,}")
print(f"Unique rated movies: {len(unique_movies):,}")

print("\nMissing values:")
for column, count in missing_values.items():
    print(f"  {column}: {count:,}")

print("\nRating distribution:")
for rating, count in sorted(rating_counts.items()):
    print(f"  {rating}: {count:,}")

print("\nTimestamp range (Unix seconds):")
print(f"  Earliest: {min_timestamp}")
print(f"  Latest:   {max_timestamp}")

print("\n--- AUDIT COMPLETE ---")

# Save the audit summary
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

summary = {
    "total_ratings": total_rows,
    "unique_users": len(unique_users),
    "unique_rated_movies": len(unique_movies),
    "total_movies": len(movies),
    "missing_rating_values": sum(missing_values.values()),
    "duplicate_movie_ids": int(movies["movieId"].duplicated().sum()),
}

report_path = REPORT_DIR / "dataset_audit.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("MovieLens 25M Dataset Audit\n")
    report.write("=" * 35 + "\n")
    for key, value in summary.items():
        report.write(f"{key}: {value:,}\n")

print(f"\nAudit report saved to: {report_path}")
