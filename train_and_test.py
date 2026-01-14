"""
Script للتدريب والاختبار المباشر للنموذج
يمكن تشغيله مباشرة بدون API
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from app.data_processing.cleaner import DataCleaner
from app.data_processing.splitter import DataSplitter
from app.evaluation.evaluator import ModelEvaluator
from app.evaluation.metrics import MetricsCalculator
from app.forecasting.model_builder import ProphetModelBuilder


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    تحميل البيانات من ملف CSV.
    
    الملف يجب أن يحتوي على أعمدة:
    - ds: التاريخ (YYYY-MM-DD)
    - y: القيمة المستهدفة
    - (اختياري) regressors: promo_any_flag, avg_discount_pct, etc.
    """
    df = pd.read_csv(file_path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def create_sample_data(n_days: int = 100) -> pd.DataFrame:
    """
    إنشاء بيانات تجريبية للاختبار.
    """
    start_date = date(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # بيانات مع موسمية أسبوعية
    y_values = []
    for i in range(n_days):
        base = 50.0
        trend = i * 0.5
        weekly = 10 * (i % 7 == 0)  # زيادة كل يوم أحد
        noise = (i % 3) * 2
        y_values.append(base + trend + weekly + noise)
    
    df = pd.DataFrame({
        "ds": dates,
        "y": y_values,
        "promo_any_flag": [1 if i % 14 == 0 else 0 for i in range(n_days)],
        "avg_discount_pct": [5.0 if i % 14 == 0 else 0.0 for i in range(n_days)],
    })
    
    return df


def evaluate_model_performance(metrics_train: dict, metrics_test: dict) -> dict:
    """
    Evaluate and compare train vs test performance.
    
    Returns:
        Dictionary with evaluation and comparison
    """
    evaluation = {
        "overfitting_check": {},
        "performance_assessment": {},
        "recommendations": []
    }
    
    # Check for overfitting (train should be similar to test)
    if metrics_train['mae'] and metrics_test['mae']:
        mae_diff = abs(metrics_train['mae'] - metrics_test['mae']) / metrics_test['mae'] * 100
        evaluation["overfitting_check"]["mae_difference_pct"] = mae_diff
        
        if mae_diff > 20:
            evaluation["recommendations"].append("⚠️  Potential overfitting: Large gap between train and test MAE")
        elif mae_diff < 5:
            evaluation["recommendations"].append("✓ Good generalization: Train and test MAE are very close")
    
    if metrics_train['rmse'] and metrics_test['rmse']:
        rmse_diff = abs(metrics_train['rmse'] - metrics_test['rmse']) / metrics_test['rmse'] * 100
        evaluation["overfitting_check"]["rmse_difference_pct"] = rmse_diff
    
    # Performance assessment
    if metrics_test['r2']:
        r2 = metrics_test['r2']
        if r2 >= 0.9:
            evaluation["performance_assessment"]["r2_rating"] = "Excellent"
        elif r2 >= 0.7:
            evaluation["performance_assessment"]["r2_rating"] = "Good"
        elif r2 >= 0.5:
            evaluation["performance_assessment"]["r2_rating"] = "Moderate"
        else:
            evaluation["performance_assessment"]["r2_rating"] = "Poor"
            evaluation["recommendations"].append("⚠️  Low R²: Model explains less than 50% of variance")
    
    if metrics_test['mape']:
        mape = metrics_test['mape']
        if mape < 10:
            evaluation["performance_assessment"]["mape_rating"] = "Excellent"
        elif mape < 20:
            evaluation["performance_assessment"]["mape_rating"] = "Good"
        elif mape < 30:
            evaluation["performance_assessment"]["mape_rating"] = "Moderate"
        else:
            evaluation["performance_assessment"]["mape_rating"] = "Poor"
            evaluation["recommendations"].append("⚠️  High MAPE: Prediction errors are significant")
    
    if metrics_test['accuracy']:
        acc = metrics_test['accuracy']
        if acc >= 0.9:
            evaluation["performance_assessment"]["accuracy_rating"] = "Excellent"
        elif acc >= 0.7:
            evaluation["performance_assessment"]["accuracy_rating"] = "Good"
        elif acc >= 0.5:
            evaluation["performance_assessment"]["accuracy_rating"] = "Moderate"
        else:
            evaluation["performance_assessment"]["accuracy_rating"] = "Poor"
            evaluation["recommendations"].append("⚠️  Low accuracy: Model classification performance is weak")
    
    if metrics_test['auc']:
        auc = metrics_test['auc']
        if auc >= 0.9:
            evaluation["performance_assessment"]["auc_rating"] = "Excellent"
        elif auc >= 0.7:
            evaluation["performance_assessment"]["auc_rating"] = "Good"
        else:
            evaluation["performance_assessment"]["auc_rating"] = "Moderate"
    
    # Overall assessment
    good_metrics = sum([
        metrics_test.get('r2', 0) >= 0.7,
        metrics_test.get('mape', 100) < 20,
        metrics_test.get('accuracy', 0) >= 0.7,
    ])
    
    if good_metrics >= 2:
        evaluation["performance_assessment"]["overall"] = "Good"
    elif good_metrics >= 1:
        evaluation["performance_assessment"]["overall"] = "Moderate"
    else:
        evaluation["performance_assessment"]["overall"] = "Needs Improvement"
        evaluation["recommendations"].append("💡 Consider: Adding more features, increasing data, or tuning hyperparameters")
    
    return evaluation


def train_and_test(
    df: pd.DataFrame,
    regressors: list = None,
    test_size: float = 0.2,
    classification_threshold: Optional[float] = None,
    save_model: bool = False,
    model_path: str = "trained_model.json",
) -> dict:
    """
    Train and test the model with comprehensive evaluation.
    
    Args:
        df: DataFrame containing the data
        regressors: List of regressor column names
        test_size: Proportion of data for testing (0.0 - 1.0)
        classification_threshold: Classification threshold (optional)
        save_model: Save the trained model
        model_path: Path to save the model
        
    Returns:
        Dictionary containing results and metrics
    """
    if regressors is None:
        regressors = []
    
    print("=" * 70)
    print("Starting Training and Testing Process")
    print("=" * 70)
    
    # Clean data
    print("\n[1/6] Cleaning data...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df, regressors)
    print(f"   ✓ Data cleaned: {len(df)} → {len(df_clean)} rows")
    
    # Split data
    print(f"\n[2/6] Splitting data (test_size={test_size})...")
    splitter = DataSplitter(test_size=test_size)
    train_df, test_df = splitter.split(df_clean, strategy="temporal")
    print(f"   ✓ Training data: {len(train_df)} rows")
    print(f"   ✓ Test data: {len(test_df)} rows")
    
    # Build model
    print("\n[3/6] Building model...")
    model_builder = ProphetModelBuilder()
    model = model_builder.build(train_df, regressors)
    print(f"   ✓ Model built with {len(regressors)} regressors")
    
    # Train model
    print("\n[4/6] Training model...")
    train_cols = ["ds", "y"] + [c for c in regressors if c in train_df.columns]
    model.fit(train_df[train_cols])
    print("   ✓ Model trained successfully")
    
    # Save model (if requested)
    if save_model:
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"   ✓ Model saved to: {model_path}")
    
    # Predict on training data
    print("\n[5/6] Evaluating on training data...")
    train_forecast = model.predict(train_df[["ds"]])
    train_y_true = train_df["y"]
    train_y_pred = train_forecast["yhat"].clip(lower=0)
    
    metrics_calc = MetricsCalculator()
    train_metrics = metrics_calc.calculate_all_metrics(
        train_y_true, train_y_pred, classification_threshold
    )
    print("   ✓ Training metrics calculated")
    
    # Predict on test data
    print("\n[6/6] Evaluating on test data...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=len(test_df))
    
    # Add regressors
    for reg in regressors:
        if reg in train_df.columns and reg not in future.columns:
            if reg in df_clean.columns:
                future = future.merge(df_clean[["ds", reg]], on="ds", how="left")
                future[reg] = future[reg].fillna(0)
            else:
                future[reg] = 0
    
    # Predict
    forecast_df = model.predict(future)
    
    # Extract predictions for test data only
    test_forecast = forecast_df[forecast_df["ds"].isin(test_df["ds"])].copy()
    test_forecast = test_forecast.sort_values("ds")
    test_df_sorted = test_df.sort_values("ds")
    
    # Ensure date matching
    test_forecast = test_forecast[test_forecast["ds"].isin(test_df_sorted["ds"])]
    test_df_sorted = test_df_sorted[test_df_sorted["ds"].isin(test_forecast["ds"])]
    
    # Calculate metrics
    test_y_true = test_df_sorted["y"]
    test_y_pred = test_forecast["yhat"].clip(lower=0)
    
    test_metrics = metrics_calc.calculate_all_metrics(
        test_y_true, test_y_pred, classification_threshold
    )
    
    print("   ✓ Test metrics calculated")
    
    # Evaluate performance
    evaluation = evaluate_model_performance(train_metrics, test_metrics)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS AND METRICS")
    print("=" * 70)
    
    # Training metrics
    print("\n" + "-" * 70)
    print("TRAINING SET METRICS")
    print("-" * 70)
    
    print("\n📊 Regression Metrics (Training):")
    print(f"   MAE  (Mean Absolute Error):      {train_metrics['mae']:.4f}")
    print(f"   MSE  (Mean Squared Error):       {train_metrics['mse']:.4f}")
    print(f"   RMSE (Root Mean Squared Error):  {train_metrics['rmse']:.4f}")
    if train_metrics['mape']:
        print(f"   MAPE (Mean Absolute % Error):    {train_metrics['mape']:.2f}%")
    if train_metrics['r2']:
        print(f"   R²   (R-squared):                {train_metrics['r2']:.4f}")
    
    print("\n🎯 Classification Metrics (Training):")
    print(f"   Accuracy:                        {train_metrics['accuracy']:.4f}")
    print(f"   Precision:                       {train_metrics['precision']:.4f}")
    print(f"   Recall:                          {train_metrics['recall']:.4f}")
    if train_metrics['auc']:
        print(f"   AUC:                             {train_metrics['auc']:.4f}")
    
    # Test metrics
    print("\n" + "-" * 70)
    print("TEST SET METRICS")
    print("-" * 70)
    
    print("\n📊 Regression Metrics (Test):")
    print(f"   MAE  (Mean Absolute Error):      {test_metrics['mae']:.4f}")
    print(f"   MSE  (Mean Squared Error):       {test_metrics['mse']:.4f}")
    print(f"   RMSE (Root Mean Squared Error):  {test_metrics['rmse']:.4f}")
    if test_metrics['mape']:
        print(f"   MAPE (Mean Absolute % Error):    {test_metrics['mape']:.2f}%")
    if test_metrics['r2']:
        print(f"   R²   (R-squared):                {test_metrics['r2']:.4f}")
    
    print("\n🎯 Classification Metrics (Test):")
    print(f"   Accuracy:                        {test_metrics['accuracy']:.4f}")
    print(f"   Precision:                       {test_metrics['precision']:.4f}")
    print(f"   Recall:                          {test_metrics['recall']:.4f}")
    if test_metrics['auc']:
        print(f"   AUC:                             {test_metrics['auc']:.4f}")
    
    print("\n📋 Confusion Matrix (Test):")
    cm = test_metrics['confusion_matrix']
    print(f"   True Negatives (TN):  {cm['tn']}")
    print(f"   False Positives (FP): {cm['fp']}")
    print(f"   False Negatives (FN): {cm['fn']}")
    print(f"   True Positives (TP):  {cm['tp']}")
    if 'threshold' in test_metrics:
        print(f"\n   Classification Threshold: {test_metrics['threshold']:.2f}")
    
    # Comparison
    print("\n" + "-" * 70)
    print("TRAIN vs TEST COMPARISON")
    print("-" * 70)
    
    if train_metrics['mae'] and test_metrics['mae']:
        mae_diff = ((train_metrics['mae'] - test_metrics['mae']) / test_metrics['mae']) * 100
        print(f"\n   MAE Difference:  {mae_diff:+.2f}% (Train vs Test)")
        if abs(mae_diff) > 20:
            print("   ⚠️  Warning: Significant difference indicates potential overfitting")
        else:
            print("   ✓ Good: Similar performance on train and test sets")
    
    if train_metrics['rmse'] and test_metrics['rmse']:
        rmse_diff = ((train_metrics['rmse'] - test_metrics['rmse']) / test_metrics['rmse']) * 100
        print(f"   RMSE Difference: {rmse_diff:+.2f}% (Train vs Test)")
    
    if train_metrics['r2'] and test_metrics['r2']:
        r2_diff = train_metrics['r2'] - test_metrics['r2']
        print(f"   R² Difference:   {r2_diff:+.4f} (Train - Test)")
        if r2_diff > 0.1:
            print("   ⚠️  Warning: Large R² gap suggests overfitting")
    
    # Evaluation
    print("\n" + "-" * 70)
    print("MODEL PERFORMANCE EVALUATION")
    print("-" * 70)
    
    if evaluation['performance_assessment'].get('overall'):
        print(f"\n   Overall Assessment: {evaluation['performance_assessment']['overall']}")
    
    if evaluation['performance_assessment'].get('r2_rating'):
        print(f"   R² Rating: {evaluation['performance_assessment']['r2_rating']}")
    
    if evaluation['performance_assessment'].get('mape_rating'):
        print(f"   MAPE Rating: {evaluation['performance_assessment']['mape_rating']}")
    
    if evaluation['performance_assessment'].get('accuracy_rating'):
        print(f"   Accuracy Rating: {evaluation['performance_assessment']['accuracy_rating']}")
    
    if evaluation['performance_assessment'].get('auc_rating'):
        print(f"   AUC Rating: {evaluation['performance_assessment']['auc_rating']}")
    
    if evaluation['recommendations']:
        print("\n   Recommendations:")
        for rec in evaluation['recommendations']:
            print(f"   {rec}")
    
    # Return results
    results = {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "evaluation": evaluation,
        "predictions": {
            "dates": test_df_sorted["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "actual": test_y_true.tolist(),
            "predicted": test_y_pred.tolist(),
        },
    }
    
    return results


def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(
        description="Train and test Prophet forecasting model"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to CSV file containing data (ds, y, regressors)",
    )
    parser.add_argument(
        "--regressors",
        nargs="+",
        default=[],
        help="List of regressor columns (e.g., promo_any_flag avg_discount_pct)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (optional)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_model.json",
        help="Path to save the model (default: trained_model.json)",
    )
    parser.add_argument(
        "--sample-data",
        action="store_true",
        help="Use sample data for testing",
    )
    parser.add_argument(
        "--sample-days",
        type=int,
        default=100,
        help="Number of days in sample data (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Load data
    if args.sample_data:
        print("Using sample data...")
        df = create_sample_data(args.sample_days)
    elif args.data:
        print(f"Loading data from: {args.data}")
        df = load_data_from_csv(args.data)
    else:
        print("⚠️  No data source specified!")
        print("Using sample data...")
        df = create_sample_data(100)
    
    # تدريب واختبار
    try:
        results = train_and_test(
            df=df,
            regressors=args.regressors,
            test_size=args.test_size,
            classification_threshold=args.threshold,
            save_model=args.save_model,
            model_path=args.model_path,
        )
        
        # Save results to JSON file
        output_file = "evaluation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            # Convert dates to strings for JSON
            results_json = json.dumps(results, default=str, indent=2, ensure_ascii=False)
            f.write(results_json)
        print(f"\n✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()

