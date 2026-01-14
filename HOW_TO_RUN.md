# دليل تشغيل النظام

هذا الدليل يوضح كيفية تشغيل نظام التنبؤ بطرق مختلفة.

## الطريقة 1: تشغيل API Server (للخدمة)

### الخطوة 1: تفعيل البيئة الافتراضية
```bash
# إذا كان لديك virtual environment
myvenv\Scripts\activate

# أو إذا كان اسمه venv
venv\Scripts\activate
```

### الخطوة 2: تشغيل الخادم
```bash
uvicorn main:app --reload
```

### الخطوة 3: الوصول للخدمة
- **API**: `http://localhost:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### مثال على استخدام API:
```bash
# استخدام curl
curl -X POST "http://localhost:8000/api/v1/forecast/product" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": 1,
    "horizon_days": 30,
    "regressors": [],
    "series": [
      {"ds": "2024-01-01", "y": 100.0},
      {"ds": "2024-01-02", "y": 120.0}
    ]
  }'
```

---

## الطريقة 2: التدريب والاختبار المباشر (بدون API)

### استخدام بيانات تجريبية:
```bash
python train_and_test.py --sample-data
```

### استخدام بيانات من ملف CSV:
```bash
python train_and_test.py --data your_data.csv --regressors promo_any_flag avg_discount_pct --test-size 0.2
```

### مع حفظ النموذج:
```bash
python train_and_test.py --data your_data.csv --save-model --model-path my_model.pkl
```

### جميع الخيارات:
```bash
python train_and_test.py --help
```

---

## الطريقة 3: استخدام Python Script مباشرة

### مثال بسيط:
```python
from app.training.trainer import ModelTrainer
import pandas as pd

# تحميل البيانات
df = pd.read_csv("data.csv")
df["ds"] = pd.to_datetime(df["ds"])

# تدريب واختبار
trainer = ModelTrainer()
results = trainer.train_test(
    df=df,
    regressors=["promo_any_flag"],
    test_size=0.2
)

# عرض النتائج
print(f"MAE: {results['metrics']['mae']}")
print(f"RMSE: {results['metrics']['rmse']}")
```

### تشغيل المثال الجاهز:
```bash
python example_train_test.py
```

---

## متطلبات التشغيل

### 1. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

### 2. التأكد من تثبيت Prophet:
```bash
pip install prophet
```

### 3. إذا واجهت مشاكل مع Prophet:
```bash
# على Windows قد تحتاج:
pip install prophet --no-cache-dir
```

---

## تنسيق ملف CSV (إذا كنت تستخدم بيانات من ملف)

الملف يجب أن يحتوي على:

```csv
ds,y,promo_any_flag,avg_discount_pct
2024-01-01,100.0,0,0.0
2024-01-02,120.0,1,5.0
2024-01-03,110.0,0,0.0
```

- **ds**: التاريخ بصيغة YYYY-MM-DD
- **y**: القيمة المستهدفة (الطلب/المبيعات)
- **promo_any_flag**: (اختياري) 0 أو 1
- **avg_discount_pct**: (اختياري) نسبة الخصم

---

## حل المشاكل الشائعة

### مشكلة: "ModuleNotFoundError: No module named 'prophet'"
**الحل:**
```bash
pip install prophet
```

### مشكلة: "No module named 'fastapi'"
**الحل:**
```bash
pip install -r requirements.txt
```

### مشكلة: "Not enough data"
**الحل:** تأكد من وجود بيانات كافية (على الأقل 20-30 نقطة)

### مشكلة: "Port already in use"
**الحل:** استخدم منفذ آخر:
```bash
uvicorn main:app --reload --port 8001
```

---

## أمثلة سريعة

### 1. تشغيل API وتجربته:
```bash
# Terminal 1: تشغيل الخادم
uvicorn main:app --reload

# Terminal 2: تجربة API
curl http://localhost:8000/docs
```

### 2. تدريب سريع:
```bash
python train_and_test.py --sample-data
```

### 3. تدريب مع بيانات حقيقية:
```bash
python train_and_test.py --data sales_data.csv --regressors promo_any_flag --test-size 0.2 --save-model
```

---

## ملاحظات مهمة

1. **للتدريب والاختبار**: استخدم `train_and_test.py`
2. **للخدمة API**: استخدم `uvicorn main:app --reload`
3. **للاختبارات**: استخدم `pytest`
4. **للتنبؤ فقط**: استخدم API endpoint `/api/v1/forecast/product`
5. **لتقييم النموذج**: استخدم API endpoint `/api/v1/evaluate/product` أو `train_and_test.py`

---

## الخطوات السريعة (Quick Start)

```bash
# 1. تفعيل البيئة
myvenv\Scripts\activate

# 2. تشغيل التدريب والاختبار
python train_and_test.py --sample-data

# أو تشغيل API
uvicorn main:app --reload
```
