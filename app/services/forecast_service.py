"""Forecast service - orchestrates forecasting workflow"""

from typing import List

import pandas as pd
from prophet import Prophet

from app.data_processing.cleaner import DataCleaner
from app.forecasting.model_builder import ProphetModelBuilder
from app.models.schemas import ForecastPoint, ForecastRequest, ForecastResponse


class ForecastService:
    """
    خدمة التنبؤ التي تنسق بين مكونات النظام.
    يتبع Single Responsibility Principle و Dependency Inversion Principle.
    """

    def __init__(
        self,
        data_cleaner: DataCleaner = None,
        model_builder: ProphetModelBuilder = None,
    ):
        """
        Initialize the forecast service with dependencies.

        Args:
            data_cleaner: Data cleaner instance (injected dependency)
            model_builder: Model builder instance (injected dependency)
        """
        self.data_cleaner = data_cleaner or DataCleaner()
        self.model_builder = model_builder or ProphetModelBuilder()

    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        """
        تنفيذ عملية التنبؤ الكاملة.

        Args:
            request: طلب التنبؤ

        Returns:
            استجابة التنبؤ

        Raises:
            ValueError: إذا كانت البيانات غير كافية
        """
        # التحقق من البيانات الأولية
        if not request.series or len(request.series) < 5:
            raise ValueError(
                "Not enough data points to build a reliable forecast (min 5)."
            )

        # تحويل البيانات إلى DataFrame
        df = pd.DataFrame([tp.dict() for tp in request.series])

        # التحقق من الأعمدة المطلوبة
        required_cols = {"ds", "y"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # تنظيف البيانات
        df_clean = self.data_cleaner.clean(df, request.regressors)

        # بناء النموذج
        model = self.model_builder.build(df_clean, request.regressors)

        # تدريب النموذج
        train_cols = ["ds", "y"] + [
            c for c in request.regressors if c in df_clean.columns
        ]
        model.fit(df_clean[train_cols])

        # إنشاء future dataframe
        future = model.make_future_dataframe(periods=request.horizon_days)

        # إضافة المتغيرات المساعدة للمستقبل (قيم افتراضية = 0)
        for reg in request.regressors:
            if reg in df_clean.columns and reg not in future.columns:
                future[reg] = 0

        # التنبؤ
        forecast_df = model.predict(future)

        # استخراج الأيام المستقبلية فقط
        forecast_points = self._extract_forecast_points(
            forecast_df, request.horizon_days
        )

        return ForecastResponse(
            product_id=request.product_id, forecast=forecast_points
        )

    def _extract_forecast_points(
        self, forecast_df: pd.DataFrame, horizon_days: int
    ) -> List[ForecastPoint]:
        """
        استخراج نقاط التنبؤ من DataFrame الناتج.

        Args:
            forecast_df: DataFrame يحتوي على التنبؤات
            horizon_days: عدد الأيام المستقبلية

        Returns:
            قائمة بنقاط التنبؤ
        """
        # أخذ آخر horizon_days صفوف فقط
        tail_df = forecast_df.tail(horizon_days)

        forecast_points: List[ForecastPoint] = []
        for _, row in tail_df.iterrows():
            # ضمان عدم وجود قيم سالبة (clip at 0)
            yhat = max(0.0, float(row["yhat"]))
            yhat_lower = None
            yhat_upper = None

            if "yhat_lower" in row and pd.notna(row["yhat_lower"]):
                yhat_lower = max(0.0, float(row["yhat_lower"]))
            if "yhat_upper" in row and pd.notna(row["yhat_upper"]):
                yhat_upper = max(0.0, float(row["yhat_upper"]))

            forecast_points.append(
                ForecastPoint(
                    ds=row["ds"].date(),
                    yhat=yhat,
                    yhat_lower=yhat_lower,
                    yhat_upper=yhat_upper,
                )
            )

        return forecast_points


