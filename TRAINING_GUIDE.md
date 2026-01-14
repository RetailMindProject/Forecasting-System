# دليل التدريب والاختبار المباشر

هذا الدليل يشرح كيفية تدريب واختبار النموذج مباشرة بدون استخدام API.

## الطرق المتاحة

### 1. استخدام Script مباشر (`train_and_test.py`)

#### تشغيل مع بيانات تجريبية:

```bash
python train_and_test.py --sample-data
```

#### تشغيل مع بيانات من ملف CSV:

```bash
python train_and_test.py --data data.csv --regressors promo_any_flag avg_discount_pct --test-size 0.2
```

#### جميع الخيارات المتاحة:

```bash
python train_and_test.py --help
```

**الخيارات:**
- `--data`: مسار ملف CSV
- `--regressors`: قائمة المتغيرات المساعدة
- `--test-size`: نسبة البيانات للاختبار (افتراضي: 0.2)
- `--threshold`: عتبة التصنيف (اختياري)
- `--save-model`: حفظ النموذج المدرب
- `--model-path`: مسار حفظ النموذج
- `--sample-data`: استخدام بيانات تجريبية
- `--sample-days`: عدد الأيام في البيانات التجريبية

**مثال كامل:**

```bash
python train_and_test.py \
    --data sales_data.csv \
    --regressors promo_any_flag avg_discount_pct \
    --test-size 0.2 \
    --threshold 50.0 \
    --save-model \
    --model-path my_model.pkl
```

### 2. استخدام Module (`ModelTrainer`)

#### مثال بسيط:

```python
from app.training.trainer import ModelTrainer
import pandas as pd

# تحميل البيانات
df = pd.read_csv("data.csv")
df["ds"] = pd.to_datetime(df["ds"])

# إنشاء trainer
trainer = ModelTrainer()

# تدريب واختبار
results = trainer.train_test(
    df=df,
    regressors=["promo_any_flag", "avg_discount_pct"],
    test_size=0.2,
    classification_threshold=50.0,
)

# عرض النتائج
print(f"MAE: {results['metrics']['mae']}")
print(f"RMSE: {results['metrics']['rmse']}")
print(f"Accuracy: {results['metrics']['accuracy']}")
```

#### تدريب فقط (بدون اختبار):

```python
from app.training.trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.train(df, regressors=["promo_any_flag"])

# حفظ النموذج
trainer.save_model(model, "trained_model.pkl")
```

#### تحميل نموذج مدرب:

```python
from app.training.trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.load_model("trained_model.pkl")

# استخدام النموذج للتنبؤ
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

### 3. استخدام المثال (`example_train_test.py`)

```bash
python example_train_test.py
```

## تنسيق ملف CSV

الملف يجب أن يحتوي على الأعمدة التالية:

```csv
ds,y,promo_any_flag,avg_discount_pct
2024-01-01,100.0,0,0.0
2024-01-02,120.0,1,5.0
2024-01-03,110.0,0,0.0
...
```

- **ds**: التاريخ بصيغة YYYY-MM-DD
- **y**: القيمة المستهدفة (الطلب/المبيعات)
- **promo_any_flag**: (اختياري) 0 أو 1
- **avg_discount_pct**: (اختياري) نسبة الخصم

## المقاييس المُرجعة

### مقاييس التنبؤ (Regression):
- **MAE**: متوسط الخطأ المطلق
- **MSE**: متوسط مربع الخطأ
- **RMSE**: الجذر التربيعي لمتوسط مربع الخطأ
- **MAPE**: متوسط الخطأ المطلق النسبي (%)
- **R²**: معامل التحديد

### مقاييس التصنيف (Classification):
- **Accuracy**: دقة التصنيف
- **Precision**: الدقة
- **Recall**: الاستدعاء
- **AUC**: المساحة تحت منحنى ROC
- **Confusion Matrix**: مصفوفة الارتباك (TN, FP, FN, TP)

## أمثلة متقدمة

### مثال 1: تدريب مع cross-validation

```python
from app.evaluation.evaluator import ModelEvaluator
import pandas as pd

df = pd.read_csv("data.csv")
df["ds"] = pd.to_datetime(df["ds"])

evaluator = ModelEvaluator()
results = evaluator.cross_validate(
    df=df,
    regressors=["promo_any_flag"],
    n_splits=5,
    classification_threshold=50.0,
)

print(f"Average MAE: {results['average_metrics']['mae']}")
print(f"Average Accuracy: {results['average_metrics']['accuracy']}")
```

### مثال 2: مقارنة نماذج مختلفة

```python
from app.training.trainer import ModelTrainer
from app.forecasting.model_builder import ProphetModelBuilder

# نموذج 1: إعدادات افتراضية
trainer1 = ModelTrainer()
results1 = trainer1.train_test(df, regressors=[], test_size=0.2)

# نموذج 2: إعدادات مخصصة
builder2 = ProphetModelBuilder(
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=15.0,
)
trainer2 = ModelTrainer(model_builder=builder2)
results2 = trainer2.train_test(df, regressors=[], test_size=0.2)

# مقارنة
print(f"Model 1 RMSE: {results1['metrics']['rmse']}")
print(f"Model 2 RMSE: {results2['metrics']['rmse']}")
```

## حفظ واستخدام النماذج

### حفظ النموذج:

```python
from app.training.trainer import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_test(df, regressors=[])

# حفظ النموذج
trainer.save_model(results['model'], 'my_model.pkl')
```

### تحميل واستخدام النموذج:

```python
from app.training.trainer import ModelTrainer
from prophet import Prophet

trainer = ModelTrainer()

# تحميل النموذج
model = trainer.load_model('my_model.pkl')

# التنبؤ للمستقبل
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# عرض التنبؤات
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
```

## نصائح

1. **حجم البيانات**: تأكد من وجود بيانات كافية (على الأقل 20-30 نقطة)
2. **تقسيم البيانات**: استخدم test_size بين 0.1 و 0.3
3. **Regressors**: أضف regressors فقط إذا كانت متوفرة في بيانات المستقبل
4. **Threshold**: اختر threshold مناسب بناءً على توزيع بياناتك
5. **حفظ النماذج**: احفظ النماذج المدربة لإعادة استخدامها لاحقاً

## استكشاف الأخطاء

### خطأ: "Not enough data"
- تأكد من وجود بيانات كافية (على الأقل 10-15 نقطة)

### خطأ: "Missing required columns"
- تأكد من وجود عمودي `ds` و `y` في البيانات

### خطأ: "Not enough cleaned data"
- قد تكون هناك قيم مفقودة كثيرة، تحقق من البيانات


