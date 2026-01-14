"""Prophet model builder with explicit configuration"""

from typing import List

import pandas as pd
from prophet import Prophet


class ProphetModelBuilder:
    """
    مسؤول عن بناء وتكوين نموذج Prophet.
    يتبع Single Responsibility Principle.
    """

    def __init__(
        self,
        yearly_seasonality: bool = None,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
    ):
        """
        Initialize the model builder with configuration.

        Args:
            yearly_seasonality: Enable yearly seasonality (None = auto-detect)
            weekly_seasonality: Enable weekly seasonality
            daily_seasonality: Enable daily seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Strength of seasonality
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale

    def build(self, df: pd.DataFrame, regressors: List[str]) -> Prophet:
        """
        بناء نموذج Prophet مع الإعدادات المحددة.

        Args:
            df: DataFrame نظيف يحتوي على البيانات
            regressors: قائمة بأسماء المتغيرات المساعدة

        Returns:
            نموذج Prophet جاهز للتدريب
        """
        # تحديد الموسمية السنوية تلقائياً إذا لم يتم تحديدها
        yearly_seasonality = self.yearly_seasonality
        if yearly_seasonality is None:
            yearly_seasonality = self._should_enable_yearly_seasonality(df)

        # إنشاء النموذج مع الإعدادات
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )

        # إضافة المتغيرات المساعدة
        for reg in regressors:
            if reg in df.columns:
                model.add_regressor(reg)

        return model

    def _should_enable_yearly_seasonality(self, df: pd.DataFrame) -> bool:
        """
        تحديد ما إذا كان يجب تفعيل الموسمية السنوية بناءً على مدى البيانات.

        Args:
            df: DataFrame يحتوي على البيانات

        Returns:
            True إذا كان المدى الزمني >= 365 يوم
        """
        if df.empty or "ds" not in df.columns:
            return False

        span_days = (df["ds"].max() - df["ds"].min()).days
        return span_days >= 365


