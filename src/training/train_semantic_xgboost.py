"""
Training script for Semantic XGBoost Router.

This script trains an XGBoost model using semantic features (DistilBERT embeddings)
to intelligently route logs to hot (ClickHouse) or cold (MinIO) storage.

Usage:
    python -m src.training.train_semantic_xgboost \
        --data data/Loghub-zenodo_Logs.csv \
        --output trained_models/xgboost_semantic_router \
        --test_split 0.2 \
        --semantic \
        --balance
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features import EnhancedFeatureExtractor
from src.config import TRAINED_MODELS_DIR


def load_log_data(filepath: str, limit: int = None) -> pd.DataFrame:
    """
    Load log data from CSV file.
    
    Expected columns: Level, Component, LogSource, Content, Timestamp
    """
    print(f"📂 Loading log data from {filepath}...")
    
    df = pd.read_csv(filepath, nrows=limit)
    print(f"   Loaded {len(df)} log entries")
    
    # Validate required columns
    required_cols = ['Level', 'Component', 'LogSource', 'Content']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def create_training_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create training labels (0=clickhouse, 1=minio) based on heuristics.
    
    Labeling strategy:
    - Security/auth logs → minio (cold archive)
    - Error/critical logs → clickhouse (hot debugging)
    - High-frequency components → clickhouse (analytics)
    - Default → clickhouse (hot)
    """
    print("🏷️  Creating training labels...")
    
    labels = []
    component_counts = df['Component'].value_counts()
    
    for idx, row in df.iterrows():
        level = str(row.get('Level', '')).lower()
        content = str(row.get('Content', '')).lower()
        component = str(row.get('Component', ''))
        
        # Security/auth logs → cold (1) - long-term archive
        if any(kw in content for kw in ['auth', 'login', 'password', 'ssh', 'security', 
                                        'unauthorized', 'permission', 'denied', 'failed password']):
            labels.append(1)
        # INFO logs (non-error) → cold (1) - can be archived
        elif level in ['info', 'notice', 'debug'] and 'error' not in content and 'fail' not in content:
            labels.append(1)
        # Errors/critical logs → hot (0) for debugging
        elif level in ['error', 'err', 'critical', 'crit', 'fatal', 'alert', 'emerg', 'warn', 'warning']:
            labels.append(0)
        # High-frequency components → hot (0) for analytics
        elif component_counts.get(component, 0) > len(df) * 0.01:  # >1% of logs
            labels.append(0)
        # Default → cold (1) - archive unless explicitly needed hot
        else:
            labels.append(1)
    
    labels = pd.Series(labels)
    
    # Report distribution
    cold_count = (labels == 1).sum()
    hot_count = (labels == 0).sum()
    print(f"   Hot storage (ClickHouse): {hot_count} ({hot_count/len(labels)*100:.1f}%)")
    print(f"   Cold storage (MinIO): {cold_count} ({cold_count/len(labels)*100:.1f}%)")
    
    return labels


def balance_dataset(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Balance dataset by undersampling majority class.
    """
    print("⚖️  Balancing dataset...")
    
    # Find minority class
    class_0_count = (y == 0).sum()
    class_1_count = (y == 1).sum()
    
    if class_0_count == class_1_count:
        print("   Dataset already balanced")
        return X, y
    
    minority_count = min(class_0_count, class_1_count)
    
    # Sample equal amounts from each class
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    sampled_0 = np.random.choice(class_0_indices, minority_count, replace=False)
    sampled_1 = np.random.choice(class_1_indices, minority_count, replace=False)
    
    balanced_indices = np.concatenate([sampled_0, sampled_1])
    np.random.shuffle(balanced_indices)
    
    print(f"   Balanced to {len(balanced_indices)} samples ({minority_count} per class)")
    
    return X[balanced_indices], y[balanced_indices]


def extract_features(df: pd.DataFrame, enable_semantic: bool = True) -> np.ndarray:
    """
    Extract features from log data using EnhancedFeatureExtractor.
    """
    feature_dim = 778 if enable_semantic else 10
    print(f"🔧 Extracting {feature_dim}-dimensional features from {len(df)} logs...")
    
    extractor = EnhancedFeatureExtractor(enable_semantic=enable_semantic)
    
    features_list = []
    for idx, row in df.iterrows():
        log_entry = row.to_dict()
        features = extractor.extract_features(log_entry)
        features_list.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(df)} logs...", end='\r')
    
    print(f"   Processed {len(df)}/{len(df)} logs - Done!        ")
    
    # Print extractor stats
    stats = extractor.get_stats()
    print(f"   Feature extractor stats: {stats}")
    
    return np.array(features_list)


def train_xgboost(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str
):
    """
    Train XGBoost model and save to disk.
    """
    print("🚀 Training XGBoost model...")
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    print("✅ Training complete!")
    
    # Evaluate
    print("\n📊 Evaluation on test set:")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   ┌─────────────┬──────────┬──────────┐")
    print(f"   │             │ Pred CH  │ Pred Min │")
    print(f"   ├─────────────┼──────────┼──────────┤")
    print(f"   │ Actual CH   │ {cm[0,0]:^8} │ {cm[0,1]:^8} │")
    print(f"   │ Actual Min  │ {cm[1,0]:^8} │ {cm[1,1]:^8} │")
    print(f"   └─────────────┴──────────┴──────────┘")
    
    # Feature importance (top 10)
    print(f"\n🔝 Top 10 Features by Importance:")
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. Feature {idx}: {importances[idx]:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    print(f"✅ Model saved!")
    
    # Save metadata
    metadata = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'feature_dim': X_train.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'params': params
    }
    
    metadata_path = output_path.replace('.json', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {metadata_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Semantic XGBoost Router')
    parser.add_argument('--data', required=True, help='Path to log CSV file')
    parser.add_argument('--output', default='xgboost_semantic_router', 
                       help='Output model path (without extension)')
    parser.add_argument('--test_split', type=float, default=0.2, 
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--semantic', action='store_true', 
                       help='Enable semantic features (DistilBERT)')
    parser.add_argument('--balance', action='store_true', 
                       help='Balance dataset by undersampling')
    parser.add_argument('--limit', type=int, default=None, 
                       help='Limit number of logs to process')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    
    args = parser.parse_args()
    
    if not DEPS_AVAILABLE:
        print("❌ Required dependencies not installed:")
        print("   pip install xgboost scikit-learn pandas numpy")
        return 1
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("🤖 SEMANTIC XGBOOST ROUTER TRAINING")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Semantic features: {args.semantic}")
    print(f"Balance dataset: {args.balance}")
    print(f"Test split: {args.test_split}")
    print("=" * 70 + "\n")
    
    # Load data
    df = load_log_data(args.data, limit=args.limit)
    
    # Create labels
    y = create_training_labels(df).values
    
    # Extract features
    X = extract_features(df, enable_semantic=args.semantic)
    
    # Balance if requested
    if args.balance:
        X, y = balance_dataset(X, y)
    
    # Train/test split
    print(f"\n📊 Splitting into train/test ({int((1-args.test_split)*100)}/{int(args.test_split*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed, stratify=y
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Prepare output path
    if not os.path.isabs(args.output):
        output_path = str(TRAINED_MODELS_DIR / f"{args.output}.json")
    else:
        output_path = args.output
    
    # Train model
    model = train_xgboost(X_train, y_train, X_test, y_test, output_path)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}")
    print("\nTo use this model:")
    print(f"  from src.routers import SemanticXGBoostRouter")
    print(f"  router = SemanticXGBoostRouter('{os.path.basename(args.output)}', enable_semantic={args.semantic})")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
