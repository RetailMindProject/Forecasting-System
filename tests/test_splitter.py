"""Tests for DataSplitter"""

from datetime import date, timedelta

import pandas as pd
import pytest

from app.data_processing.splitter import DataSplitter


class TestDataSplitter:
    """Test cases for DataSplitter"""

    def test_temporal_split(self):
        """Test temporal splitting"""
        splitter = DataSplitter(test_size=0.2)
        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(50)],
            "y": [float(i * 10) for i in range(50)],
        })

        train_df, test_df = splitter.split(df, strategy="temporal")

        assert len(train_df) > len(test_df)
        assert len(train_df) + len(test_df) == len(df)
        assert train_df["ds"].max() <= test_df["ds"].min()

    def test_random_split(self):
        """Test random splitting"""
        splitter = DataSplitter(test_size=0.2)
        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(50)],
            "y": [float(i * 10) for i in range(50)],
        })

        train_df, test_df = splitter.split(df, strategy="random")

        assert len(train_df) > len(test_df)
        assert len(train_df) + len(test_df) == len(df)

    def test_split_raises_error_insufficient_data(self):
        """Test that split raises error for insufficient data"""
        splitter = DataSplitter(test_size=0.2, min_train_size=10)
        df = pd.DataFrame({
            "ds": [date(2024, 1, i) for i in range(1, 6)],
            "y": [float(i * 10) for i in range(1, 6)],
        })

        with pytest.raises(ValueError):
            splitter.split(df)

    def test_split_invalid_test_size(self):
        """Test that split raises error for invalid test_size"""
        with pytest.raises(ValueError):
            DataSplitter(test_size=1.5)

    def test_split_ensures_min_train_size(self):
        """Test that split ensures minimum train size"""
        splitter = DataSplitter(test_size=0.9, min_train_size=10)
        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(50)],
            "y": [float(i * 10) for i in range(50)],
        })

        train_df, test_df = splitter.split(df)

        assert len(train_df) >= splitter.min_train_size


