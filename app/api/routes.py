"""API routes for forecasting endpoints"""

from fastapi import APIRouter, HTTPException

from app.evaluation.evaluator import ModelEvaluator
from app.models.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    ForecastRequest,
    ForecastResponse,
    MetricsResult,
)
from app.services.forecast_service import ForecastService

router = APIRouter(prefix="/api/v1", tags=["forecast"])


@router.post("/forecast/product", response_model=ForecastResponse)
def forecast_product(request: ForecastRequest) -> ForecastResponse:
    """
    تنبؤ بالطلب على منتج معين.

    Args:
        request: طلب التنبؤ

    Returns:
        استجابة التنبؤ

    Raises:
        HTTPException: في حالة وجود خطأ في البيانات أو المعالجة
    """
    try:
        service = ForecastService()
        return service.forecast(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )


@router.post("/evaluate/product", response_model=EvaluationResponse)
def evaluate_product(request: EvaluationRequest) -> EvaluationResponse:
    """
    تقييم نموذج التنبؤ باستخدام train/test split والمقاييس.

    Args:
        request: طلب التقييم

    Returns:
        استجابة التقييم مع جميع المقاييس

    Raises:
        HTTPException: في حالة وجود خطأ في البيانات أو المعالجة
    """
    try:
        import pandas as pd

        # تحويل البيانات إلى DataFrame
        df = pd.DataFrame([tp.dict() for tp in request.series])

        # التحقق من الأعمدة المطلوبة
        required_cols = {"ds", "y"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # تقييم النموذج
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(
            df=df,
            regressors=request.regressors,
            test_size=request.test_size,
            classification_threshold=request.classification_threshold,
        )

        # تحويل المقاييس إلى MetricsResult
        metrics_dict = results["metrics"]
        metrics_result = MetricsResult(**metrics_dict)

        return EvaluationResponse(
            product_id=request.product_id,
            train_size=results["train_size"],
            test_size=results["test_size"],
            metrics=metrics_result,
            predictions=results.get("predictions"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )

