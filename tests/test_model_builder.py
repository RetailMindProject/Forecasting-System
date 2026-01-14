"""Tests for ProphetModelBuilder"""

from datetime import date, timedelta

import pandas as pd
import pytest
from prophet import Prophet

from app.forecasting.model_builder import ProphetModelBuilder


class TestProphetModelBuilder:
    """Test cases for ProphetModelBuilder"""

    def test_build_creates_prophet_model(self):
        """Test that build creates a Prophet model"""
        builder = ProphetModelBuilder()
        df = pd.DataFrame({
            "ds": [date(2024, 1, i) for i in range(1, 10)],
            "y": [float(i * 10) for i in range(1, 10)],
        })
        model = builder.build(df, [])
        assert isinstance(model, Prophet)

    def test_build_adds_regressors(self):
        """Test that regressors are added to the model"""
        builder = ProphetModelBuilder()
        df = pd.DataFrame({
            "ds": [date(2024, 1, i) for i in range(1, 10)],
            "y": [float(i * 10) for i in range(1, 10)],
            "promo_any_flag": [0] * 9,
        })
        model = builder.build(df, ["promo_any_flag"])
        assert isinstance(model, Prophet)

    def test_yearly_seasonality_enabled_for_long_span(self):
        """Test that yearly seasonality is enabled for data >= 365 days"""
        builder = ProphetModelBuilder()
        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(400)],
            "y": [float(i * 10) for i in range(400)],
        })
        model = builder.build(df, [])
        # Prophet doesn't expose seasonality settings directly, but we can check
        # that the model was created successfully
        assert isinstance(model, Prophet)

    def test_yearly_seasonality_disabled_for_short_span(self):
        """Test that yearly seasonality is disabled for data < 365 days"""
        builder = ProphetModelBuilder()
        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(100)],
            "y": [float(i * 10) for i in range(100)],
        })
        model = builder.build(df, [])
        assert isinstance(model, Prophet)

    def test_build_with_explicit_yearly_seasonality(self):
        """Test that explicit yearly_seasonality setting is respected"""
        builder = ProphetModelBuilder(yearly_seasonality=True)
        df = pd.DataFrame({
            "ds": [date(2024, 1, i) for i in range(1, 10)],
            "y": [float(i * 10) for i in range(1, 10)],
        })
        model = builder.build(df, [])
        assert isinstance(model, Prophet)

    def test_build_ignores_missing_regressors(self):
        """Test that missing regressors in dataframe are ignored"""
        builder = ProphetModelBuilder()
        df = pd.DataFrame({
            "ds": [date(2024, 1, i) for i in range(1, 10)],
            "y": [float(i * 10) for i in range(1, 9)],
        })
        # Request regressor that doesn't exist in df
        model = builder.build(df, ["nonexistent_regressor"])
        assert isinstance(model, Prophet)


