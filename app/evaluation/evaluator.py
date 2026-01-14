"""Model evaluator for forecasting models"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from app.data_processing.cleaner import DataCleaner
from app.data_processing.splitter import DataSplitter
from app.evaluation.metrics import MetricsCalculator
from app.forecasting.model_builder import ProphetModelBuilder


class ModelEvaluator:
    """
    تقييم نماذج Prophet باستخدام train/test split والمقاييس.
    يتبع Single Responsibility Principle.
    """

    def __init__(
        self,
        data_cleaner: DataCleaner = None,
        model_builder: ProphetModelBuilder = None,
        data_splitter: DataSplitter = None,
        metrics_calculator: MetricsCalculator = None,
    ):
        """
        Initialize the model evaluator.

        Args:
            data_cleaner: Data cleaner instance
            model_builder: Model builder instance
            data_splitter: Data splitter instance
            metrics_calculator: Metrics calculator instance
        """
        self.data_cleaner = data_cleaner or DataCleaner()
        self.model_builder = model_builder or ProphetModelBuilder()
        self.data_splitter = data_splitter or DataSplitter()
        self.metrics_calculator = metrics_calculator or MetricsCalculator()

    def evaluate(
        self,
        df: pd.DataFrame,
        regressors: List[str],
        test_size: float = 0.2,
        classification_threshold: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        تقييم النموذج باستخدام train/test split.

        Args:
            df: DataFrame يحتوي على البيانات
            regressors: قائمة المتغيرات المساعدة
            test_size: نسبة البيانات للاختبار
            classification_threshold: عتبة التصنيف (اختياري)

        Returns:
            Dictionary containing evaluation results and metrics
        """
        # تنظيف البيانات
        df_clean = self.data_cleaner.clean(df, regressors)

        # تقسيم البيانات
        splitter = DataSplitter(test_size=test_size)
        train_df, test_df = splitter.split(df_clean, strategy="temporal")

        # بناء النموذج
        model = self.model_builder.build(train_df, regressors)

        # تدريب النموذج
        train_cols = ["ds", "y"] + [
            c for c in regressors if c in train_df.columns
        ]
        model.fit(train_df[train_cols])

        # التنبؤ على بيانات الاختبار
        # إنشاء future dataframe يشمل بيانات الاختبار
        future = model.make_future_dataframe(periods=len(test_df))

        # إضافة المتغيرات المساعدة
        for reg in regressors:
            if reg in train_df.columns and reg not in future.columns:
                # استخدام القيم من البيانات الأصلية إذا كانت متوفرة
                if reg in df_clean.columns:
                    # Merge مع البيانات الأصلية
                    future = future.merge(
                        df_clean[["ds", reg]], on="ds", how="left"
                    )
                    future[reg] = future[reg].fillna(0)
                else:
                    future[reg] = 0

        # التنبؤ
        forecast_df = model.predict(future)

        # استخراج التنبؤات لبيانات الاختبار فقط
        test_forecast = forecast_df[forecast_df["ds"].isin(test_df["ds"])].copy()
        test_forecast = test_forecast.sort_values("ds")
        test_df_sorted = test_df.sort_values("ds")

        # التأكد من تطابق التواريخ
        test_forecast = test_forecast[test_forecast["ds"].isin(test_df_sorted["ds"])]
        test_df_sorted = test_df_sorted[test_df_sorted["ds"].isin(test_forecast["ds"])]

        # حساب المقاييس
        y_true = test_df_sorted["y"]
        y_pred = test_forecast["yhat"].clip(lower=0)  # Clip negative values

        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, classification_threshold
        )

        return {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "metrics": metrics,
            "predictions": {
                "dates": test_df_sorted["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "actual": y_true.tolist(),
                "predicted": y_pred.tolist(),
            },
        }

    def cross_validate(
        self,
        df: pd.DataFrame,
        regressors: List[str],
        n_splits: int = 5,
        classification_threshold: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Cross-validation للسلاسل الزمنية (Time Series Cross-Validation).

        Args:
            df: DataFrame يحتوي على البيانات
            regressors: قائمة المتغيرات المساعدة
            n_splits: عدد التقسيمات
            classification_threshold: عتبة التصنيف (اختياري)

        Returns:
            Dictionary containing cross-validation results
        """
        # تنظيف البيانات
        df_clean = self.data_cleaner.clean(df, regressors)

        if len(df_clean) < n_splits * 5:
            raise ValueError(
                f"Not enough data for {n_splits} splits. "
                f"Need at least {n_splits * 5} points, got {len(df_clean)}"
            )

        all_metrics = []

        # Time series cross-validation
        for i in range(n_splits):
            # زيادة حجم بيانات التدريب تدريجياً
            train_size = int(len(df_clean) * (i + 1) / (n_splits + 1))
            test_size = int(len(df_clean) / (n_splits + 1))

            if train_size < 10 or test_size < 1:
                continue

            train_df = df_clean.iloc[:train_size].copy()
            test_df = df_clean.iloc[train_size : train_size + test_size].copy()

            if len(test_df) == 0:
                continue

            # بناء وتدريب النموذج
            model = self.model_builder.build(train_df, regressors)
            train_cols = ["ds", "y"] + [
                c for c in regressors if c in train_df.columns
            ]
            model.fit(train_df[train_cols])

            # التنبؤ
            future = model.make_future_dataframe(periods=len(test_df))
            for reg in regressors:
                if reg in train_df.columns and reg not in future.columns:
                    if reg in df_clean.columns:
                        future = future.merge(
                            df_clean[["ds", reg]], on="ds", how="left"
                        )
                        future[reg] = future[reg].fillna(0)
                    else:
                        future[reg] = 0

            forecast_df = model.predict(future)
            test_forecast = forecast_df[
                forecast_df["ds"].isin(test_df["ds"])
            ].copy()
            test_forecast = test_forecast.sort_values("ds")
            test_df_sorted = test_df.sort_values("ds")

            test_forecast = test_forecast[
                test_forecast["ds"].isin(test_df_sorted["ds"])
            ]
            test_df_sorted = test_df_sorted[
                test_df_sorted["ds"].isin(test_forecast["ds"])
            ]

            if len(test_df_sorted) == 0:
                continue

            y_true = test_df_sorted["y"]
            y_pred = test_forecast["yhat"].clip(lower=0)

            metrics = self.metrics_calculator.calculate_all_metrics(
                y_true, y_pred, classification_threshold
            )
            metrics["fold"] = i + 1
            all_metrics.append(metrics)

        if not all_metrics:
            raise ValueError("No valid folds for cross-validation")

        # حساب المتوسطات
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key == "fold" or key == "confusion_matrix":
                continue
            values = [m[key] for m in all_metrics if m[key] is not None]
            if values:
                avg_metrics[key] = float(np.mean(values))

        return {
            "n_splits": len(all_metrics),
            "fold_metrics": all_metrics,
            "average_metrics": avg_metrics,
        }

