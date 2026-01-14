# Model Evaluation Guide

## نظرة عامة

تم إضافة نظام تقييم شامل للنماذج يتضمن:
- تقسيم البيانات إلى train/test
- حساب مقاييس التنبؤ (Regression Metrics)
- حساب مقاييس التصنيف (Classification Metrics)
- Cross-validation للسلاسل الزمنية

## المقاييس المتاحة

### مقاييس التنبؤ (Regression Metrics)

1. **MAE (Mean Absolute Error)**: متوسط الخطأ المطلق
2. **MSE (Mean Squared Error)**: متوسط مربع الخطأ
3. **RMSE (Root Mean Squared Error)**: الجذر التربيعي لمتوسط مربع الخطأ
4. **MAPE (Mean Absolute Percentage Error)**: متوسط الخطأ المطلق النسبي
5. **R² (R-squared)**: معامل التحديد

### مقاييس التصنيف (Classification Metrics)

1. **Accuracy**: دقة التصنيف
2. **Precision**: الدقة (الموجبة)
3. **Recall**: الاستدعاء (الحساسية)
4. **AUC (Area Under Curve)**: المساحة تحت منحنى ROC
5. **Confusion Matrix**: مصفوفة الارتباك (TN, FP, FN, TP)

> **ملاحظة**: مقاييس التصنيف تحول التنبؤ إلى تصنيف ثنائي بناءً على threshold:
> - إذا كان `y_pred >= threshold` → 1 (زيادة الطلب)
> - إذا كان `y_pred < threshold` → 0 (انخفاض الطلب)

## الاستخدام

### 1. عبر API

```python
import requests

url = "http://localhost:8000/api/v1/evaluate/product"
data = {
    "product_id": 1,
    "regressors": ["promo_any_flag", "avg_discount_pct"],
    "test_size": 0.2,
    "classification_threshold": 30.0,  # اختياري
    "series": [
        {
            "ds": "2024-01-01",
            "y": 100.0,
            "promo_any_flag": 0,
            "avg_discount_pct": 0.0
        },
        # ... المزيد من البيانات
    ]
}

response = requests.post(url, json=data)
results = response.json()

print(f"MAE: {results['metrics']['mae']}")
print(f"RMSE: {results['metrics']['rmse']}")
print(f"Accuracy: {results['metrics']['accuracy']}")
print(f"AUC: {results['metrics']['auc']}")
print(f"Confusion Matrix: {results['metrics']['confusion_matrix']}")
```

### 2. برمجياً

```python
from app.evaluation.evaluator import ModelEvaluator
import pandas as pd

# تحضير البيانات
df = pd.DataFrame({
    "ds": [...],
    "y": [...],
    "promo_any_flag": [...],
})

# التقييم
evaluator = ModelEvaluator()
results = evaluator.evaluate(
    df=df,
    regressors=["promo_any_flag"],
    test_size=0.2,
    classification_threshold=30.0
)

print(f"Train size: {results['train_size']}")
print(f"Test size: {results['test_size']}")
print(f"Metrics: {results['metrics']}")
```

### 3. Cross-Validation

```python
from app.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.cross_validate(
    df=df,
    regressors=["promo_any_flag"],
    n_splits=5,
    classification_threshold=30.0
)

print(f"Number of folds: {results['n_splits']}")
print(f"Average metrics: {results['average_metrics']}")
print(f"Fold metrics: {results['fold_metrics']}")
```

## تفسير النتائج

### مقاييس التنبؤ

- **MAE/RMSE**: كلما كانت أقل، كان النموذج أفضل
- **MAPE**: كلما كانت أقل، كان النموذج أفضل (بالنسبة المئوية)
- **R²**: كلما كان أقرب إلى 1، كان النموذج أفضل

### مقاييس التصنيف

- **Accuracy**: نسبة التوقعات الصحيحة (0-1)
- **Precision**: نسبة التوقعات الإيجابية الصحيحة
- **Recall**: نسبة الحالات الإيجابية المكتشفة
- **AUC**: كلما كان أقرب إلى 1، كان النموذج أفضل في التمييز
- **Confusion Matrix**:
  - TN (True Negative): توقعات صحيحة للقيم المنخفضة
  - FP (False Positive): توقعات خاطئة للقيم المرتفعة
  - FN (False Negative): توقعات خاطئة للقيم المنخفضة
  - TP (True Positive): توقعات صحيحة للقيم المرتفعة

## أمثلة على الاستخدام

### مثال 1: تقييم بسيط

```python
results = evaluator.evaluate(df, regressors=[], test_size=0.2)
```

### مثال 2: تقييم مع regressors

```python
results = evaluator.evaluate(
    df,
    regressors=["promo_any_flag", "avg_discount_pct"],
    test_size=0.2
)
```

### مثال 3: تقييم مع threshold مخصص

```python
results = evaluator.evaluate(
    df,
    regressors=[],
    test_size=0.2,
    classification_threshold=50.0  # قيم أعلى من 50 = زيادة
)
```

## ملاحظات مهمة

1. **تقسيم البيانات**: يتم استخدام تقسيم زمني (temporal split) افتراضياً، حيث يتم استخدام البيانات الأقدم للتدريب والأحدث للاختبار.

2. **الحد الأدنى للبيانات**: يحتاج التقييم إلى بيانات كافية (على الأقل 10 نقاط للتدريب).

3. **Threshold للتصنيف**: إذا لم يتم تحديد threshold، سيتم استخدام الوسيط (median) للقيم الحقيقية.

4. **القيم السالبة**: يتم تقليم القيم السالبة في التنبؤات إلى 0.

5. **Cross-Validation**: Time series cross-validation يزيد حجم بيانات التدريب تدريجياً في كل fold.


