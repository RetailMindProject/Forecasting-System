# دليل البدء السريع 🚀

## الطريقة الأسهل: التدريب والاختبار المباشر

### خطوة واحدة فقط:
```bash
python train_and_test.py --sample-data
```

هذا سيقوم بـ:
- ✅ إنشاء بيانات تجريبية
- ✅ تدريب النموذج
- ✅ اختبار النموذج
- ✅ عرض جميع المقاييس
- ✅ حفظ النتائج في `evaluation_results.json`

---

## تشغيل API Server

### الخطوات:
```bash
# 1. تفعيل البيئة الافتراضية
myvenv\Scripts\activate

# 2. تشغيل الخادم
uvicorn main:app --reload
```

### الوصول:
- 📍 API: http://localhost:8000
- 📖 Swagger UI: http://localhost:8000/docs
- 📚 ReDoc: http://localhost:8000/redoc

---

## استخدام بياناتك الخاصة

### من ملف CSV:
```bash
python train_and_test.py --data your_file.csv --regressors promo_any_flag avg_discount_pct
```

### تنسيق الملف:
```csv
ds,y,promo_any_flag,avg_discount_pct
2024-01-01,100.0,0,0.0
2024-01-02,120.0,1,5.0
```

---

## الخيارات المتاحة

```bash
# عرض جميع الخيارات
python train_and_test.py --help

# أمثلة:
python train_and_test.py --sample-data --test-size 0.3
python train_and_test.py --data data.csv --save-model --model-path model.pkl
python train_and_test.py --data data.csv --threshold 50.0
```

---

## ملاحظات

- ✅ **للتدريب والاختبار**: استخدم `train_and_test.py`
- ✅ **للخدمة API**: استخدم `uvicorn main:app --reload`
- ✅ **للأمثلة**: استخدم `python example_train_test.py`

---

## مساعدة

إذا واجهت أي مشاكل، راجع:
- 📄 `HOW_TO_RUN.md` - دليل شامل
- 📄 `TRAINING_GUIDE.md` - دليل التدريب
- 📄 `README.md` - معلومات عامة
