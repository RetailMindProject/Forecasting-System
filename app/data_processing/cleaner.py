"""Data cleaning and preprocessing for time series"""

from typing import Dict, List

import pandas as pd


class DataCleaner:
    """
    مسؤول عن تنظيف ومعالجة البيانات الزمنية قبل التدريب.
    يتبع Single Responsibility Principle.
    """

    def __init__(self, min_data_points: int = 5):
        """
        Initialize the data cleaner.

        Args:
            min_data_points: Minimum number of data points required after cleaning
        """
        self.min_data_points = min_data_points

    def clean(self, df: pd.DataFrame, regressors: List[str]) -> pd.DataFrame:
        """
        تنظيف ومعالجة السلسلة الزمنية.

        Args:
            df: DataFrame يحتوي على البيانات الخام
            regressors: قائمة بأسماء المتغيرات المساعدة

        Returns:
            DataFrame نظيف ومجهز للتدريب

        Raises:
            ValueError: إذا لم يكن هناك بيانات كافية بعد التنظيف
        """
        df = df.copy()

        # تحويل التاريخ إلى datetime
        df["ds"] = pd.to_datetime(df["ds"])

        # ترتيب حسب التاريخ
        df = df.sort_values("ds")

        # حذف الصفوف التي تحتوي على y = null/NaN
        df = df.dropna(subset=["y"])

        # تجميع القيم عند تكرار التاريخ
        df = self._aggregate_duplicates(df, regressors)

        # التحقق من وجود بيانات كافية
        if len(df) < self.min_data_points:
            raise ValueError(
                f"Not enough cleaned data points to build a reliable forecast "
                f"(min {self.min_data_points} after cleaning, got {len(df)})."
            )

        return df

    def _aggregate_duplicates(
        self, df: pd.DataFrame, regressors: List[str]
    ) -> pd.DataFrame:
        """
        تجميع البيانات عند وجود تواريخ مكررة.

        Args:
            df: DataFrame المراد تجميعه
            regressors: قائمة المتغيرات المساعدة

        Returns:
            DataFrame مع تواريخ فريدة
        """
        # إنشاء خريطة التجميع
        agg_map: Dict[str, str] = {"y": "sum"}

        for reg in regressors:
            if reg in df.columns:
                agg_map[reg] = self._get_aggregation_method(reg)

        # تجميع البيانات
        df_grouped = df.groupby("ds", as_index=False).agg(agg_map)

        return df_grouped

    def _get_aggregation_method(self, regressor_name: str) -> str:
        """
        تحديد طريقة التجميع المناسبة للمتغير المساعد.

        Args:
            regressor_name: اسم المتغير المساعد

        Returns:
            طريقة التجميع ('sum', 'mean', 'max', etc.)
        """
        # Flags: نستخدم max (أي وجود في اليوم يكفي)
        if regressor_name == "promo_any_flag":
            return "max"

        # Percentages/Averages: نستخدم mean
        if regressor_name == "avg_discount_pct":
            return "mean"

        # افتراضي: mean لبقية المتغيرات
        return "mean"


