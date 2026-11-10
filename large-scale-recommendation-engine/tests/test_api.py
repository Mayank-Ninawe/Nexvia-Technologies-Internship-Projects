
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["similarity_artifact_loaded"] is True
    assert data["movie_catalog_loaded"] is True
    assert data["demo_recommendations_loaded"] is True


def test_popular_recommendations_default_limit():
    response = client.get("/recommendations/popular")

    assert response.status_code == 200
    data = response.json()
    assert data["recommendation_type"] == "popularity"
    assert 1 <= data["recommendation_count"] <= 10
    assert len(data["recommendations"]) == data["recommendation_count"]


def test_popular_recommendations_custom_limit():
    response = client.get("/recommendations/popular?limit=5")

    assert response.status_code == 200
    data = response.json()
    assert data["recommendation_count"] == 5


def test_popular_recommendations_rejects_invalid_limit():
    for limit in (0, 51):
        response = client.get(f"/recommendations/popular?limit={limit}")
        assert response.status_code == 422


def test_similar_movies_for_known_catalog_movie():
    response = client.get("/recommendations/movie/318")

    assert response.status_code == 200
    data = response.json()
    assert data["input_movie"]["movieId"] == 318
    assert data["recommendation_count"] == len(data["recommendations"])

    for movie in data["recommendations"]:
        assert "movieId" in movie
        assert "title" in movie
        assert "similarity" in movie
        assert movie["movieId"] != 318


def test_similar_movies_rejects_invalid_limit():
    response = client.get("/recommendations/movie/318?limit=0")

    assert response.status_code == 422


def test_unknown_movie_returns_404():
    response = client.get("/recommendations/movie/999999999")

    assert response.status_code == 404


def test_personalized_recommendations_for_known_user():
    response = client.get("/recommendations/user/1?limit=10")

    assert response.status_code == 200

    data = response.json()

    assert data["userId"] == 1
    assert data["recommendation_type"] == "precomputed_demo_personalized"
    assert data["liked_movies_used"] > 0
    assert data["recommendation_count"] == len(data["recommendations"])

    recommended_ids = {
        movie["movieId"] for movie in data["recommendations"]
    }

    assert len(recommended_ids) == len(data["recommendations"])


def test_personalized_recommendations_exclude_rated_movies():
    response = client.get("/recommendations/user/1?limit=10")

    assert response.status_code == 200

    data = response.json()
    recommendations = data["recommendations"]

    movie_ids = [movie["movieId"] for movie in recommendations]

    assert len(movie_ids) == len(set(movie_ids))
    assert data["recommendation_count"] == len(recommendations)
    assert data["recommendation_type"] == "precomputed_demo_personalized"


def test_personalized_recommendations_for_non_demo_user():
    response = client.get("/recommendations/user/4?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert data["userId"] == 4
    assert data["recommendation_type"] == "item_similarity_personalized"
    assert data["liked_movies_used"] > 0
    assert data["recommendation_count"] == len(data["recommendations"])


def test_unknown_user_returns_404():
    response = client.get("/recommendations/user/999999999")

    assert response.status_code == 404


def test_personalized_recommendations_rejects_invalid_limit():
    response = client.get("/recommendations/user/1?limit=0")

    assert response.status_code == 422
