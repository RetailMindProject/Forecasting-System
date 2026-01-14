# Architecture Documentation

## SOLID Principles Implementation

### 1. Single Responsibility Principle (SRP)

كل كلاس له مسؤولية واحدة واضحة:

- **`DataCleaner`**: تنظيف ومعالجة البيانات فقط
- **`ProphetModelBuilder`**: بناء نماذج Prophet فقط
- **`ForecastService`**: تنسيق عملية التنبؤ
- **`ForecastRequest/ForecastResponse`**: نماذج البيانات فقط
- **API Routes**: معالجة HTTP requests/responses فقط

### 2. Open/Closed Principle

النظام قابل للتوسع بدون تعديل:

- يمكن إضافة أنواع جديدة من cleaners بإنشاء كلاس جديد يرث من `DataCleaner`
- يمكن إضافة أنواع جديدة من model builders بإنشاء كلاس جديد يرث من `ProphetModelBuilder`
- `ForecastService` يمكن أن يقبل أي implementation من cleaners و builders

### 3. Liskov Substitution Principle

يمكن استبدال implementations:

- أي `DataCleaner` يمكن استبداله بآخر
- أي `ProphetModelBuilder` يمكن استبداله بآخر
- `ForecastService` يعمل مع أي implementation

### 4. Interface Segregation Principle

الواجهات صغيرة ومحددة:

- `DataCleaner` لديه واجهة بسيطة: `clean()`
- `ProphetModelBuilder` لديه واجهة بسيطة: `build()`
- `ForecastService` لديه واجهة بسيطة: `forecast()`

### 5. Dependency Inversion Principle

الاعتماد على abstractions وليس implementations:

- `ForecastService` يعتمد على `DataCleaner` و `ProphetModelBuilder` كـ dependencies
- يمكن حقن implementations مختلفة (مثلاً للاختبار)
- الكود عالي المستوى لا يعتمد على تفاصيل التنفيذ

## Data Flow

```
API Request (ForecastRequest)
    ↓
API Route Handler
    ↓
ForecastService.forecast()
    ↓
DataCleaner.clean() → Cleaned DataFrame
    ↓
ProphetModelBuilder.build() → Prophet Model
    ↓
Model.fit() → Trained Model
    ↓
Model.predict() → Forecast DataFrame
    ↓
Extract Forecast Points → ForecastResponse
    ↓
API Response
```

## Testing Strategy

### Unit Tests
- **`test_data_cleaner.py`**: اختبار منطق تنظيف البيانات
- **`test_model_builder.py`**: اختبار بناء النماذج
- **`test_forecast_service.py`**: اختبار منطق التنبؤ

### Integration Tests
- **`test_api_routes.py`**: اختبار API endpoints كاملة

### Test Fixtures
- **`conftest.py`**: Fixtures مشتركة للاختبارات

## Extension Points

### Adding New Data Cleaners

```python
class AdvancedDataCleaner(DataCleaner):
    def clean(self, df, regressors):
        # Custom cleaning logic
        df = super().clean(df, regressors)
        # Additional processing
        return df
```

### Adding New Model Builders

```python
class CustomProphetBuilder(ProphetModelBuilder):
    def build(self, df, regressors):
        # Custom model configuration
        model = super().build(df, regressors)
        # Additional setup
        return model
```

### Using Custom Components

```python
service = ForecastService(
    data_cleaner=AdvancedDataCleaner(),
    model_builder=CustomProphetBuilder()
)
```


