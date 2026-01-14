"""Pydantic models for API request/response schemas"""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class TimePoint(BaseModel):
    """نقطة زمنية واحدة في السلسلة الزمنية"""
    ds: date  # تاريخ اليوم
    y: float  # الكمية المباعة (target)

    # Regressors اختيارية حالياً
    promo_any_flag: Optional[int] = Field(default=None)
    avg_discount_pct: Optional[float] = Field(default=None)


class ForecastRequest(BaseModel):
    """طلب التنبؤ"""
    product_id: int
    horizon_days: int = Field(default=30, ge=1, le=365)
    regressors: List[str] = Field(
        default_factory=list,
        description="Names of regressors columns to use, e.g. ['promo_any_flag', 'avg_discount_pct']"
    )
    series: List[TimePoint]


class ForecastPoint(BaseModel):
    """نقطة تنبؤ واحدة"""
    ds: date
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None


class ForecastResponse(BaseModel):
    """استجابة التنبؤ"""
    product_id: int
    forecast: List[ForecastPoint]


class EvaluationRequest(BaseModel):
    """طلب تقييم النموذج"""
    product_id: int
    regressors: List[str] = Field(
        default_factory=list,
        description="Names of regressors columns to use"
    )
    series: List[TimePoint]
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    classification_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for binary classification (optional)"
    )


class MetricsResult(BaseModel):
    """نتائج المقاييس"""
    # Regression metrics
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    auc: Optional[float] = None
    confusion_matrix: Optional[dict] = None
    threshold: Optional[float] = None


class EvaluationResponse(BaseModel):
    """استجابة التقييم"""
    product_id: int
    train_size: int
    test_size: int
    metrics: MetricsResult
    predictions: Optional[dict] = None

