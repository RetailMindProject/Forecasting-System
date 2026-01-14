"""Tests for DataCleaner"""

from datetime import date, timedelta

import pandas as pd
import pytest

from app.data_processing.cleaner import DataCleaner


class TestDataCleaner:
    """Test cases for DataCleaner"""

    def test_clean_sorts_by_date(self):
        """Test that data is sorted by date"""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "ds": [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)],
            "y": [10.0, 20.0, 30.0],
        })
        result = cleaner.clean(df, [])
        assert result["ds"].iloc[0] == pd.Timestamp(date(2024, 1, 1))
        assert result["ds"].iloc[-1] == pd.Timestamp(date(2024, 1, 3))

    def test_clean_removes_nan_y(self):
        """Test that rows with NaN y are removed"""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "ds": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "y": [10.0, None, 30.0],
        })
        result = cleaner.clean(df, [])
        assert len(result) == 2
        assert result["y"].notna().all()

    def test_clean_aggregates_duplicate_dates(self):
        """Test that duplicate dates are aggregated"""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "ds": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
            ],
            "y": [10.0, 20.0, 30.0],
        })
        result = cleaner.clean(df, [])
        assert len(result) == 2
        assert result[result["ds"] == pd.Timestamp(date(2024, 1, 1))]["y"].iloc[0] == 30.0

    def test_clean_aggregates_promo_flag(self):
        """Test that promo_any_flag uses max aggregation"""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "ds": [date(2024, 1, 1), date(2024, 1, 1)],
            "y": [10.0, 20.0],
            "promo_any_flag": [0, 1],
        })
        result = cleaner.clean(df, ["promo_any_flag"])
        assert len(result) == 1
        assert result["promo_any_flag"].iloc[0] == 1

    def test_clean_aggregates_discount_mean(self):
        """Test that avg_discount_pct uses mean aggregation"""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "ds": [date(2024, 1, 1), date(2024, 1, 1)],
            "y": [10.0, 20.0],
            "avg_discount_pct": [10.0, 20.0],
        })
        result = cleaner.clean(df, ["avg_discount_pct"])
        assert len(result) == 1
        assert result["avg_discount_pct"].iloc[0] == 15.0

    def test_clean_raises_error_insufficient_data(self):
        """Test that cleaner raises error for insufficient data"""
        cleaner = DataCleaner(min_data_points=5)
        df = pd.DataFrame({
            "ds": [date(2024, 1, 1), date(2024, 1, 2)],
            "y": [10.0, 20.0],
        })
        with pytest.raises(ValueError, match="Not enough cleaned data"):
            cleaner.clean(df, [])

    def test_clean_handles_empty_regressors(self):
        """Test that cleaner works with no regressors"""
        cleaner = DataCleaner()
        df = pd.DataFrame({
            "ds": [date(2024, 1, i) for i in range(1, 6)],
            "y": [float(i * 10) for i in range(1, 6)],
        })
        result = cleaner.clean(df, [])
        assert len(result) == 5


