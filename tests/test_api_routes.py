"""Tests for API routes"""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestAPIRoutes:
    """Test cases for API routes"""

    def test_forecast_product_success(self):
        """Test successful forecast request"""
        request_data = {
            "product_id": 1,
            "horizon_days": 7,
            "regressors": [],
            "series": [
                {"ds": f"2024-01-{i:02d}", "y": float(i * 10)}
                for i in range(1, 30)
            ],
        }
        response = client.post("/api/v1/forecast/product", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == 1
        assert len(data["forecast"]) == 7
        assert all(fp["yhat"] >= 0 for fp in data["forecast"])

    def test_forecast_product_insufficient_data(self):
        """Test forecast request with insufficient data"""
        request_data = {
            "product_id": 1,
            "horizon_days": 7,
            "regressors": [],
            "series": [
                {"ds": "2024-01-01", "y": 10.0},
                {"ds": "2024-01-02", "y": 20.0},
            ],
        }
        response = client.post("/api/v1/forecast/product", json=request_data)
        assert response.status_code == 400

    def test_forecast_product_with_regressors(self):
        """Test forecast request with regressors"""
        request_data = {
            "product_id": 1,
            "horizon_days": 7,
            "regressors": ["promo_any_flag"],
            "series": [
                {
                    "ds": f"2024-01-{i:02d}",
                    "y": float(i * 10),
                    "promo_any_flag": 1 if i % 7 == 0 else 0,
                }
                for i in range(1, 30)
            ],
        }
        response = client.post("/api/v1/forecast/product", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == 1
        assert len(data["forecast"]) == 7

    def test_forecast_product_empty_series(self):
        """Test forecast request with empty series"""
        request_data = {
            "product_id": 1,
            "horizon_days": 7,
            "regressors": [],
            "series": [],
        }
        response = client.post("/api/v1/forecast/product", json=request_data)
        assert response.status_code == 400

    def test_forecast_product_invalid_horizon(self):
        """Test forecast request with invalid horizon_days"""
        request_data = {
            "product_id": 1,
            "horizon_days": 500,  # > 365
            "regressors": [],
            "series": [
                {"ds": f"2024-01-{i:02d}", "y": float(i * 10)}
                for i in range(1, 30)
            ],
        }
        response = client.post("/api/v1/forecast/product", json=request_data)
        assert response.status_code == 422  # Validation error


