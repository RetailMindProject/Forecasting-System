"""Tests for MetricsCalculator"""

import pandas as pd
import pytest

from app.evaluation.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Test cases for MetricsCalculator"""

    def test_calculate_regression_metrics(self):
        """Test regression metrics calculation"""
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = pd.Series([12.0, 18.0, 32.0, 38.0, 52.0])

        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r2" in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0

    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation"""
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = pd.Series([12.0, 18.0, 32.0, 38.0, 52.0])

        metrics = MetricsCalculator.calculate_classification_metrics(
            y_true, y_pred, threshold=25.0
        )

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "auc" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["confusion_matrix"] is not None

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics"""
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = pd.Series([12.0, 18.0, 32.0, 38.0, 52.0])

        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)

        # Regression metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics

        # Classification metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "auc" in metrics
        assert "confusion_matrix" in metrics

    def test_metrics_with_nan_values(self):
        """Test metrics calculation with NaN values"""
        y_true = pd.Series([10.0, None, 30.0, 40.0, 50.0])
        y_pred = pd.Series([12.0, 18.0, None, 38.0, 52.0])

        metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred)

        # Should handle NaN gracefully
        assert "mae" in metrics

    def test_confusion_matrix_structure(self):
        """Test confusion matrix structure"""
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = pd.Series([12.0, 18.0, 32.0, 38.0, 52.0])

        metrics = MetricsCalculator.calculate_classification_metrics(
            y_true, y_pred, threshold=25.0
        )

        cm = metrics["confusion_matrix"]
        assert "tn" in cm
        assert "fp" in cm
        assert "fn" in cm
        assert "tp" in cm


