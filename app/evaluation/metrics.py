"""Metrics calculation for forecasting models"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class MetricsCalculator:
    """
    حساب مقاييس التقييم للتنبؤ والتصنيف.
    يتبع Single Responsibility Principle.
    """

    @staticmethod
    def calculate_regression_metrics(
        y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, float]:
        """
        حساب مقاييس التنبؤ (Regression Metrics).

        Args:
            y_true: القيم الحقيقية
            y_pred: القيم المتوقعة

        Returns:
            Dictionary containing MAE, MSE, RMSE, MAPE, R²
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # إزالة القيم المفقودة
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {
                "mae": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "r2": np.nan,
            }

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true - y_pred))

        # MSE (Mean Squared Error)
        mse = np.mean((y_true - y_pred) ** 2)

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)

        # MAPE (Mean Absolute Percentage Error)
        # تجنب القسمة على صفر
        mask_nonzero = y_true != 0
        if mask_nonzero.sum() > 0:
            mape = np.mean(
                np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])
            ) * 100
        else:
            mape = np.nan

        # R² (R-squared)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape) if not np.isnan(mape) else None,
            "r2": float(r2) if not np.isnan(r2) else None,
        }

    @staticmethod
    def calculate_classification_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        حساب مقاييس التصنيف (Classification Metrics).

        يحول التنبؤ إلى تصنيف بناءً على threshold:
        - إذا كان threshold محدد: 1 إذا y_pred >= threshold، 0 وإلا
        - إذا لم يكن threshold محدد: 1 إذا y_pred > median(y_true)، 0 وإلا

        Args:
            y_true: القيم الحقيقية
            y_pred: القيم المتوقعة
            threshold: عتبة التصنيف (اختياري)

        Returns:
            Dictionary containing accuracy, precision, recall, AUC, confusion_matrix
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # إزالة القيم المفقودة
        mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
        y_true_arr = y_true_arr[mask]
        y_pred_arr = y_pred_arr[mask]

        if len(y_true_arr) == 0:
            return {
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "auc": np.nan,
                "confusion_matrix": None,
            }

        # تحديد threshold
        if threshold is None:
            threshold = np.median(y_true_arr)

        # تحويل إلى تصنيف ثنائي
        y_true_binary = (y_true_arr >= threshold).astype(int)
        y_pred_binary = (y_pred_arr >= threshold).astype(int)

        # Accuracy
        accuracy = accuracy_score(y_true_binary, y_pred_binary)

        # Precision
        try:
            precision = precision_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
        except Exception:
            precision = 0.0

        # Recall
        try:
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        except Exception:
            recall = 0.0

        # AUC (Area Under Curve)
        try:
            # استخدام y_pred_arr كـ probabilities
            # تطبيع القيم بين 0 و 1
            y_pred_normalized = (y_pred_arr - y_pred_arr.min()) / (
                y_pred_arr.max() - y_pred_arr.min() + 1e-10
            )
            auc_score = roc_auc_score(y_true_binary, y_pred_normalized)
        except Exception:
            auc_score = np.nan

        # Confusion Matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        cm_dict = {
            "tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            "fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            "fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            "tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        }

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc_score) if not np.isnan(auc_score) else None,
            "confusion_matrix": cm_dict,
            "threshold": float(threshold),
        }

    @staticmethod
    def calculate_all_metrics(
        y_true: pd.Series,
        y_pred: pd.Series,
        threshold: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        حساب جميع المقاييس (التنبؤ والتصنيف).

        Args:
            y_true: القيم الحقيقية
            y_pred: القيم المتوقعة
            threshold: عتبة التصنيف (اختياري)

        Returns:
            Dictionary containing all metrics
        """
        regression_metrics = MetricsCalculator.calculate_regression_metrics(
            y_true, y_pred
        )
        classification_metrics = MetricsCalculator.calculate_classification_metrics(
            y_true, y_pred, threshold
        )

        return {
            **regression_metrics,
            **classification_metrics,
        }


