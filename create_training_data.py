"""
Create training dataset for XGBoost by combining baseline results.

This script:
1. Reads results from both baseline experiments (direct_clickhouse and direct_minio)
2. Compares their performance on the same logs
3. Labels each log with the "best" backend (lowest latency)
4. Creates a training CSV for XGBoost model
"""

import pandas as pd
from pathlib import Path
import sys


def create_training_data(
    clickhouse_csv: str = "results/direct_clickhouse_loghub.csv",
    minio_csv: str = "results/direct_minio_loghub.csv",
    output_csv: str = "results/training_data_loghub.csv"
):
    """
    Combine baseline results and label each log with best backend.
    
    Args:
        clickhouse_csv: Path to ClickHouse baseline results
        minio_csv: Path to MinIO baseline results
        output_csv: Path to save training data
    """
    print("=" * 80)
    print("📊 Creating Training Data for XGBoost Router")
    print("=" * 80)
    
    # Check files exist
    if not Path(clickhouse_csv).exists():
        print(f"❌ File not found: {clickhouse_csv}")
        print(f"   Run: python -m src --router direct_clickhouse --log_source loghub --limit 1000")
        sys.exit(1)
    
    if not Path(minio_csv).exists():
        print(f"❌ File not found: {minio_csv}")
        print(f"   Run: python -m src --router direct_minio --log_source loghub --limit 1000")
        sys.exit(1)
    
    # Read both baseline results
    print(f"\n📁 Reading {clickhouse_csv}...")
    clickhouse_df = pd.read_csv(clickhouse_csv)
    
    print(f"📁 Reading {minio_csv}...")
    minio_df = pd.read_csv(minio_csv)
    
    # Ensure same number of logs
    if len(clickhouse_df) != len(minio_df):
        print(f"❌ Baseline results have different lengths:")
        print(f"   ClickHouse: {len(clickhouse_df)} logs")
        print(f"   MinIO: {len(minio_df)} logs")
        print(f"\n   Both experiments must use same logs (same --limit and --log_source)")
        sys.exit(1)
    
    print(f"✅ Both files have {len(clickhouse_df)} logs")
    
    # Create combined dataset
    training_data = []
    
    print("\n🔄 Labeling logs with best backend...")
    
    for i in range(len(clickhouse_df)):
        ch_row = clickhouse_df.iloc[i]
        minio_row = minio_df.iloc[i]
        
        # Get latencies (backend write latency)
        ch_latency = ch_row.get('backend_write_latency_ms', 0.0)
        minio_latency = minio_row.get('backend_write_latency_ms', 0.0)
        
        # Get energy consumption
        ch_energy = ch_row.get('energy_cpu_pkg_j', 0.0)
        minio_energy = minio_row.get('energy_cpu_pkg_j', 0.0)
        
        # Extract log characteristics
        level = ch_row.get('level', 'INFO').upper()
        content = ch_row.get('content', '')
        content_lower = content.lower()
        
        # Error severity score (higher = more critical)
        severity_map = {'INFO': 0, 'DEBUG': 0, 'WARN': 1, 'WARNING': 1, 'ERROR': 2, 'CRITICAL': 3, 'FATAL': 3}
        severity = severity_map.get(level, 0)
        
        # Check for error indicators
        error_keywords = ['error', 'fail', 'denied', 'reject', 'timeout', 'exception', 'crash', 'fatal']
        has_errors = any(kw in content_lower for kw in error_keywords)
        
        # Check for security indicators
        security_keywords = ['ssh', 'login', 'password', 'auth', 'permission', 'access', 'security', 'credential']
        is_security = any(kw in content_lower for kw in security_keywords)
        
        # INTELLIGENT ROUTING DECISION - Multi-Objective Optimization
        # Rule-based routing with intelligent logic:
        
        # 1. High-priority logs (errors, security) → ClickHouse (hot storage for fast querying)
        if severity >= 2 or has_errors or is_security:
            best_backend = 'clickhouse'
            routing_reason = 'high_priority'
        
        # 2. If ClickHouse is much faster (>2x) and low energy cost → ClickHouse
        elif ch_latency > 0 and minio_latency > 0 and (minio_latency / ch_latency) > 2.0 and ch_energy <= minio_energy * 1.5:
            best_backend = 'clickhouse'
            routing_reason = 'performance'
        
        # 3. If MinIO is faster or similar latency with better energy efficiency → MinIO
        elif minio_latency <= ch_latency * 1.2 and minio_energy <= ch_energy:
            best_backend = 'minio'
            routing_reason = 'energy_efficient'
        
        # 4. Default: Use latency-based decision
        else:
            best_backend = 'clickhouse' if ch_latency <= minio_latency else 'minio'
            routing_reason = 'latency'
        
        # Create training sample with all required fields
        sample = {
            'LogID': i,
            'Level': ch_row.get('level', 'INFO'),
            'Component': ch_row.get('component', 'unknown'),
            'LogSource': ch_row.get('log_source', 'loghub'),
            'Content': content,
            'EventTemplate': ch_row.get('event_template', ''),
            'clickhouse_latency': ch_latency,
            'minio_latency': minio_latency,
            'clickhouse_energy': ch_energy,
            'minio_energy': minio_energy,
            'latency_diff': abs(ch_latency - minio_latency),
            'severity': severity,
            'has_errors': int(has_errors),
            'is_security': int(is_security),
            'best_backend': best_backend,
            'routing_reason': routing_reason
        }
        
        training_data.append(sample)
    
    # Create DataFrame
    training_df = pd.DataFrame(training_data)
    
    # Statistics
    total = len(training_df)
    clickhouse_count = (training_df['best_backend'] == 'clickhouse').sum()
    minio_count = (training_df['best_backend'] == 'minio').sum()
    
    print("\n" + "=" * 80)
    print("📊 Training Data Statistics")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Best backend = clickhouse: {clickhouse_count} ({clickhouse_count/total*100:.1f}%)")
    print(f"Best backend = minio: {minio_count} ({minio_count/total*100:.1f}%)")
    print(f"\nAvg latency difference: {training_df['latency_diff'].mean():.2f} ms")
    print(f"Max latency difference: {training_df['latency_diff'].max():.2f} ms")
    
    # Check if we have variety (need both classes for training)
    if clickhouse_count == 0 or minio_count == 0:
        print("\n⚠️  WARNING: Training data has only one class!")
        print("   This means one backend is ALWAYS better than the other.")
        print("   XGBoost training may fail or achieve poor results.")
        print("\n   Suggestions:")
        print("   - Use larger dataset (increase --limit)")
        print("   - Check if ClickHouse/MinIO are configured correctly")
        print("   - Consider using energy instead of latency for labeling")
    
    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Training data saved to {output_csv}")
    print(f"   Columns: {list(training_df.columns)}")
    print(f"   Ready for: python -m src.training.train_xgboost {output_csv}")
    print("=" * 80)
    
    return output_csv


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) == 1:
        # Use defaults
        create_training_data()
    elif len(sys.argv) == 4:
        # Custom paths
        create_training_data(
            clickhouse_csv=sys.argv[1],
            minio_csv=sys.argv[2],
            output_csv=sys.argv[3]
        )
    else:
        print("Usage:")
        print("  python create_training_data.py")
        print("  python create_training_data.py <clickhouse.csv> <minio.csv> <output.csv>")
        sys.exit(1)
