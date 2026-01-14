"""Data splitting utilities for time series"""

from typing import Tuple

import pandas as pd


class DataSplitter:
    """
    مسؤول عن تقسيم البيانات الزمنية إلى train/test.
    يتبع Single Responsibility Principle.
    """

    def __init__(self, test_size: float = 0.2, min_train_size: int = 10):
        """
        Initialize the data splitter.

        Args:
            test_size: نسبة البيانات للاختبار (0.0 - 1.0)
            min_train_size: الحد الأدنى لحجم بيانات التدريب
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0")
        self.test_size = test_size
        self.min_train_size = min_train_size

    def split(
        self, df: pd.DataFrame, strategy: str = "temporal"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        تقسيم البيانات إلى train/test.

        Args:
            df: DataFrame مرتب حسب التاريخ
            strategy: استراتيجية التقسيم ('temporal' أو 'random')
                    'temporal': تقسيم زمني (الأقدم للتدريب، الأحدث للاختبار)
                    'random': تقسيم عشوائي

        Returns:
            Tuple of (train_df, test_df)

        Raises:
            ValueError: إذا كانت البيانات غير كافية
        """
        if len(df) < self.min_train_size + 1:
            raise ValueError(
                f"Not enough data for splitting. "
                f"Need at least {self.min_train_size + 1} points, got {len(df)}"
            )

        # التأكد من أن البيانات مرتبة حسب التاريخ
        if "ds" not in df.columns:
            raise ValueError("DataFrame must contain 'ds' column")

        df_sorted = df.sort_values("ds").copy()

        if strategy == "temporal":
            return self._temporal_split(df_sorted)
        elif strategy == "random":
            return self._random_split(df_sorted)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _temporal_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        تقسيم زمني: الأقدم للتدريب، الأحدث للاختبار.

        Args:
            df: DataFrame مرتب حسب التاريخ

        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * (1 - self.test_size))

        # التأكد من وجود بيانات كافية للتدريب
        if split_idx < self.min_train_size:
            split_idx = self.min_train_size

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        return train_df, test_df

    def _random_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        تقسيم عشوائي (غير موصى به للسلاسل الزمنية).

        Args:
            df: DataFrame

        Returns:
            Tuple of (train_df, test_df)
        """
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * (1 - self.test_size))

        if split_idx < self.min_train_size:
            split_idx = self.min_train_size

        train_df = df_shuffled.iloc[:split_idx].copy()
        test_df = df_shuffled.iloc[split_idx:].copy()

        # إعادة الترتيب حسب التاريخ
        train_df = train_df.sort_values("ds")
        test_df = test_df.sort_values("ds")

        return train_df, test_df


