"""Tests for ModelEvaluator"""

from datetime import date, timedelta

import pandas as pd
import pytest

from app.evaluation.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test cases for ModelEvaluator"""

    def test_evaluate_basic(self):
        """Test basic evaluation"""
        evaluator = ModelEvaluator()

        # إنشاء بيانات تجريبية
        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(60)],
            "y": [float(10 + i * 2 + (i % 7) * 5) for i in range(60)],
        })

        results = evaluator.evaluate(df, regressors=[], test_size=0.2)

        assert "train_size" in results
        assert "test_size" in results
        assert "metrics" in results
        assert results["train_size"] > 0
        assert results["test_size"] > 0

        metrics = results["metrics"]
        assert "mae" in metrics
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics

    def test_evaluate_with_regressors(self):
        """Test evaluation with regressors"""
        evaluator = ModelEvaluator()

        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(60)],
            "y": [float(10 + i * 2) for i in range(60)],
            "promo_any_flag": [1 if i % 7 == 0 else 0 for i in range(60)],
        })

        results = evaluator.evaluate(
            df, regressors=["promo_any_flag"], test_size=0.2
        )

        assert "metrics" in results
        assert results["metrics"]["mae"] is not None

    def test_evaluate_raises_error_insufficient_data(self):
        """Test that evaluation raises error for insufficient data"""
        evaluator = ModelEvaluator()

        df = pd.DataFrame({
            "ds": [date(2024, 1, i) for i in range(1, 6)],
            "y": [float(i * 10) for i in range(1, 6)],
        })

        with pytest.raises(ValueError):
            evaluator.evaluate(df, regressors=[], test_size=0.2)

    def test_cross_validate_basic(self):
        """Test basic cross-validation"""
        evaluator = ModelEvaluator()

        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(100)],
            "y": [float(10 + i * 2 + (i % 7) * 5) for i in range(100)],
        })

        results = evaluator.cross_validate(df, regressors=[], n_splits=3)

        assert "n_splits" in results
        assert "fold_metrics" in results
        assert "average_metrics" in results
        assert results["n_splits"] > 0
        assert len(results["fold_metrics"]) == results["n_splits"]

    def test_evaluate_with_classification_threshold(self):
        """Test evaluation with custom classification threshold"""
        evaluator = ModelEvaluator()

        start_date = date(2024, 1, 1)
        df = pd.DataFrame({
            "ds": [start_date + timedelta(days=i) for i in range(60)],
            "y": [float(10 + i * 2) for i in range(60)],
        })

        results = evaluator.evaluate(
            df,
            regressors=[],
            test_size=0.2,
            classification_threshold=30.0,
        )

        assert "metrics" in results
        assert results["metrics"]["threshold"] == 30.0


