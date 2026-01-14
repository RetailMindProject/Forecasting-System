"""
مثال على استخدام ModelTrainer للتدريب والاختبار
"""

from datetime import date, timedelta

import pandas as pd

from app.training.trainer import ModelTrainer


def create_sample_data(n_days: int = 100):
    """إنشاء بيانات تجريبية"""
    start_date = date(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    y_values = []
    for i in range(n_days):
        base = 50.0
        trend = i * 0.5
        weekly = 10 * (i % 7 == 0)
        noise = (i % 3) * 2
        y_values.append(base + trend + weekly + noise)
    
    df = pd.DataFrame({
        "ds": dates,
        "y": y_values,
        "promo_any_flag": [1 if i % 14 == 0 else 0 for i in range(n_days)],
        "avg_discount_pct": [5.0 if i % 14 == 0 else 0.0 for i in range(n_days)],
    })
    
    return df


def main():
    """مثال على الاستخدام"""
    
    # إنشاء بيانات تجريبية
    print("إنشاء بيانات تجريبية...")
    df = create_sample_data(100)
    print(f"تم إنشاء {len(df)} صف من البيانات")
    
    # إنشاء trainer
    trainer = ModelTrainer()
    
    # تدريب واختبار
    print("\nبدء التدريب والاختبار...")
    results = trainer.train_test(
        df=df,
        regressors=["promo_any_flag", "avg_discount_pct"],
        test_size=0.2,
        classification_threshold=60.0,
    )
    
    # عرض النتائج
    print("\n" + "=" * 60)
    print("النتائج")
    print("=" * 60)
    
    print(f"\nحجم بيانات التدريب: {len(results['train_df'])}")
    print(f"حجم بيانات الاختبار: {len(results['test_df'])}")
    
    metrics = results["metrics"]
    
    print("\n📊 مقاييس التنبؤ:")
    print(f"   MAE:  {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   R²:   {metrics['r2']:.4f if metrics['r2'] else 'N/A'}")
    
    print("\n🎯 مقاييس التصنيف:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   AUC:       {metrics['auc']:.4f if metrics['auc'] else 'N/A'}")
    
    print("\n📋 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   TN: {cm['tn']}, FP: {cm['fp']}")
    print(f"   FN: {cm['fn']}, TP: {cm['tp']}")
    
    # حفظ النموذج (اختياري)
    # trainer.save_model(results['model'], 'my_trained_model.pkl')
    
    print("\n✓ تم بنجاح!")


if __name__ == "__main__":
    main()


