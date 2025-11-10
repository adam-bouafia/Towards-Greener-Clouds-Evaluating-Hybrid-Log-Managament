"""
Train XGBoost routing model.

Trains a gradient boosting classifier to intelligently route logs
based on features extracted from log entries.
"""

import xgboost as xgb
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import TRAINED_MODELS_DIR, XGBOOST_PARAMS
from src.features import FeatureExtractor


def generate_training_data(
    baseline_results_path: Path
) -> Tuple[List[List[float]], List[str]]:
    """
    Generate training data from baseline experiment results.
    
    Strategy:
    1. Run baseline experiments with direct routers (all logs to each backend)
    2. Measure performance (latency, energy) for each backend
    3. Label each log with the best-performing backend
    
    Args:
        baseline_results_path: Path to CSV with baseline results
            Expected columns: LogID, Level, Component, LogSource, Content,
                             clickhouse_latency, minio_latency, best_backend
    
    Returns:
        Tuple of (features, labels)
    """
    # Read baseline results
    df = pd.read_csv(baseline_results_path)
    
    if "best_backend" not in df.columns:
        raise ValueError("baseline_results must have 'best_backend' column")
    
    # Fill any NaN values with defaults
    df['Level'] = df['Level'].fillna('INFO').astype(str)
    df['Component'] = df['Component'].fillna('unknown').astype(str)
    df['LogSource'] = df['LogSource'].fillna('unknown').astype(str)
    df['Content'] = df['Content'].fillna('').astype(str)
    df['EventTemplate'] = df['EventTemplate'].fillna('').astype(str)
    df['best_backend'] = df['best_backend'].astype(str)
    
    # Extract features
    extractor = FeatureExtractor()
    
    features = []
    labels = []
    
    for _, row in df.iterrows():
        log_entry = {
            "Level": row["Level"],
            "Component": row["Component"],
            "LogSource": row["LogSource"],
            "Content": row["Content"],
            "EventTemplate": row["EventTemplate"]
        }
        
        feature_vector = extractor.extract(log_entry)
        features.append(feature_vector)
        labels.append(row["best_backend"])
    
    print(f"✅ Generated {len(features)} training samples")
    return features, labels


def train_xgboost_router(
    baseline_results_path: Path,
    model_name: str = "xgboost_router",
    test_size: float = 0.2
) -> Dict:
    """
    Train XGBoost routing model.
    
    Args:
        baseline_results_path: Path to baseline results CSV
        model_name: Name for saved model files
        test_size: Fraction of data for testing (0.2 = 20%)
    
    Returns:
        Dictionary with training metrics
    """
    print(f"🎓 Training XGBoost router...")
    
    # Generate training data
    X, y = generate_training_data(baseline_results_path)
    
    # Check if we have variety in labels
    unique_labels = set(y)
    if len(unique_labels) < 2:
        raise ValueError(
            f"❌ Training data has only one class: {unique_labels}. "
            "Need both 'clickhouse' and 'minio' labels. "
            "Try increasing test data size or adjusting backend configurations."
        )
    
    # Encode labels (clickhouse=0, minio=1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    watchlist = [(dtrain, "train"), (dtest, "test")]
    
    model = xgb.train(
        params=XGBOOST_PARAMS,
        dtrain=dtrain,
        num_boost_round=100,
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # Evaluate
    y_pred = (model.predict(dtest) > 0.5).astype(int)
    accuracy = (y_pred == y_test).mean()
    
    print(f"✅ Test accuracy: {accuracy:.4f}")
    
    # Save model
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = TRAINED_MODELS_DIR / f"{model_name}.json"
    encoder_path = TRAINED_MODELS_DIR / f"{model_name}_encoders.pkl"
    
    model.save_model(str(model_path))
    
    with open(encoder_path, "wb") as f:
        pickle.dump({"label_encoder": label_encoder}, f)
    
    print(f"💾 Model saved to {model_path}")
    print(f"💾 Encoders saved to {encoder_path}")
    
    # Feature importance
    feature_names = FeatureExtractor().get_feature_names()
    importance = model.get_score(importance_type="weight")
    
    print("\n📊 Feature Importance:")
    for fname, fid in zip(feature_names, range(len(feature_names))):
        score = importance.get(f"f{fid}", 0)
        print(f"  {fname}: {score}")
    
    return {
        "accuracy": accuracy,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_path": str(model_path),
        "encoder_path": str(encoder_path)
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.training.train_xgboost <baseline_results.csv>")
        sys.exit(1)
    
    baseline_path = Path(sys.argv[1])
    
    if not baseline_path.exists():
        print(f"❌ File not found: {baseline_path}")
        sys.exit(1)
    
    results = train_xgboost_router(baseline_path)
    print(f"\n✅ Training complete: {results['accuracy']:.2%} accuracy")
