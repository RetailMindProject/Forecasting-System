"""Pytest configuration and fixtures"""

import pytest
from datetime import date

from app.models.schemas import TimePoint, ForecastRequest


@pytest.fixture
def sample_time_series():
    """Sample time series data for testing"""
    return [
        TimePoint(
            ds=date(2024, 1, i),
            y=float(i * 10),
            promo_any_flag=1 if i % 7 == 0 else 0,
            avg_discount_pct=5.0 if i % 7 == 0 else 0.0,
        )
        for i in range(1, 30)
    ]


@pytest.fixture
def sample_forecast_request(sample_time_series):
    """Sample forecast request for testing"""
    return ForecastRequest(
        product_id=1,
        horizon_days=7,
        regressors=["promo_any_flag", "avg_discount_pct"],
        series=sample_time_series,
    )


@pytest.fixture
def minimal_time_series():
    """Minimal time series data (just enough for testing)"""
    return [
        TimePoint(ds=date(2024, 1, i), y=float(i * 10))
        for i in range(1, 10)
    ]


