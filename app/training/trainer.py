"""
Module للتدريب والاختبار المباشر
يمكن استيراده واستخدامه في أي مكان
"""

from typing import Dict, List, Optional

import pandas as pd
from prophet import Prophet

from app.data_processing.cleaner import DataCleaner
from app.data_processing.splitter import DataSplitter
from app.evaluation.metrics import MetricsCalculator
from app.forecasting.model_builder import ProphetModelBuilder


class ModelTrainer:
    """
    كلاس للتدريب والاختبار المباشر للنماذج.
    يمكن استخدامه بدون API.
    """

    def __init__(
        self,
        data_cleaner: DataCleaner = None,
        model_builder: ProphetModelBuilder = None,
        data_splitter: DataSplitter = None,
        metrics_calculator: MetricsCalculator = None,
    ):
        """
        Initialize the trainer.

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

    def train(
        self,
        df: pd.DataFrame,
        regressors: List[str] = None,
    ) -> Prophet:
        """
        تدريب النموذج على البيانات.

        Args:
            df: DataFrame يحتوي على البيانات
            regressors: قائمة المتغيرات المساعدة

        Returns:
            نموذج Prophet مدرب
        """
        if regressors is None:
            regressors = []

        # تنظيف البيانات
        df_clean = self.data_cleaner.clean(df, regressors)

        # بناء النموذج
        model = self.model_builder.build(df_clean, regressors)

        # تدريب النموذج
        train_cols = ["ds", "y"] + [
            c for c in regressors if c in df_clean.columns
        ]
        model.fit(df_clean[train_cols])

        return model

    def train_test(
        self,
        df: pd.DataFrame,
        regressors: List[str] = None,
        test_size: float = 0.2,
        classification_threshold: Optional[float] = None,
    ) -> Dict:
        """
        تدريب واختبار النموذج مع حساب المقاييس.

        Args:
            df: DataFrame يحتوي على البيانات
            regressors: قائمة المتغيرات المساعدة
            test_size: نسبة البيانات للاختبار
            classification_threshold: عتبة التصنيف (اختياري)

        Returns:
            Dictionary يحتوي على النموذج المدرب والنتائج والمقاييس
        """
        if regressors is None:
            regressors = []

        # تنظيف البيانات
        df_clean = self.data_cleaner.clean(df, regressors)

        # تقسيم البيانات
        splitter = DataSplitter(test_size=test_size)
        train_df, test_df = splitter.split(df_clean, strategy="temporal")

        # بناء وتدريب النموذج
        model = self.model_builder.build(train_df, regressors)
        train_cols = ["ds", "y"] + [
            c for c in regressors if c in train_df.columns
        ]
        model.fit(train_df[train_cols])

        # التنبؤ على بيانات الاختبار
        future = model.make_future_dataframe(periods=len(test_df))

        # إضافة regressors
        for reg in regressors:
            if reg in train_df.columns and reg not in future.columns:
                if reg in df_clean.columns:
                    future = future.merge(df_clean[["ds", reg]], on="ds", how="left")
                    future[reg] = future[reg].fillna(0)
                else:
                    future[reg] = 0

        # التنبؤ
        forecast_df = model.predict(future)

        # استخراج التنبؤات لبيانات الاختبار
        test_forecast = forecast_df[forecast_df["ds"].isin(test_df["ds"])].copy()
        test_forecast = test_forecast.sort_values("ds")
        test_df_sorted = test_df.sort_values("ds")

        test_forecast = test_forecast[
            test_forecast["ds"].isin(test_df_sorted["ds"])
        ]
        test_df_sorted = test_df_sorted[
            test_df_sorted["ds"].isin(test_forecast["ds"])
        ]

        # حساب المقاييس
        y_true = test_df_sorted["y"]
        y_pred = test_forecast["yhat"].clip(lower=0)

        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, classification_threshold
        )

        return {
            "model": model,
            "train_df": train_df,
            "test_df": test_df_sorted,
            "predictions": {
                "dates": test_df_sorted["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "actual": y_true.tolist(),
                "predicted": y_pred.tolist(),
            },
            "metrics": metrics,
        }

    def save_model(self, model: Prophet, file_path: str):
        """
        حفظ النموذج المدرب.

        Args:
            model: نموذج Prophet مدرب
            file_path: مسار الملف
        """
        import pickle

        with open(file_path, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, file_path: str) -> Prophet:
        """
        تحميل نموذج مدرب.

        Args:
            file_path: مسار الملف

        Returns:
            نموذج Prophet مدرب
        """
        import pickle

        with open(file_path, "rb") as f:
            return pickle.load(f)


