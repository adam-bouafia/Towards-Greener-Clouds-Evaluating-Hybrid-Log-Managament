"""
Log provider - Load and sample log datasets.

Reads Loghub and Synthetic datasets, supports various sampling modes.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Literal
from src.config import DATASETS


SamplingMode = Literal["head", "random", "balanced"]


class LogProvider:
    """
    Load and sample log datasets for experiments.
    
    Supports:
    - Loghub-zenodo: Real-world logs from various systems
    - Synthetic Datacenter: Artificially generated datacenter logs
    
    Sampling modes:
    - head: First N logs (fast, but may be biased)
    - random: Random N logs (good for general testing)
    - balanced: Stratified by level (ensures all severity levels represented)
    """
    
    def __init__(self, dataset_name: str):
        """
        Initialize log provider.
        
        Args:
            dataset_name: Either "loghub" or "synthetic"
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'loghub' or 'synthetic'")
        
        self.dataset_name = dataset_name
        self.dataset_path = Path(DATASETS[dataset_name])
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        print(f"✅ LogProvider initialized: {dataset_name} → {self.dataset_path.name}")
    
    def load_logs(
        self, 
        n_logs: int = None, 
        mode: SamplingMode = "head"
    ) -> List[Dict]:
        """
        Load and sample logs from dataset.
        
        Args:
            n_logs: Number of logs to load (None = all logs)
            mode: Sampling mode ("head", "random", "balanced")
        
        Returns:
            List of log dictionaries with standardized keys:
            - LogID: Unique identifier
            - Timestamp: Log timestamp
            - Level: Severity level (INFO, WARN, ERROR, CRITICAL)
            - Component: System component
            - LogSource: Source system/file
            - EventTemplate: Log template pattern
            - Content: Full log message
        """
        # Read CSV
        df = pd.read_csv(self.dataset_path)
        
        # If n_logs is None, use all logs
        total_logs = len(df)
        if n_logs is None:
            n_logs = total_logs
            print(f"📊 Loading ALL {total_logs} logs from dataset")
        elif n_logs > total_logs:
            print(f"⚠️  Requested {n_logs} logs but dataset has {total_logs}, using all")
            n_logs = total_logs
        
        # Sample based on mode
        if mode == "head":
            sampled_df = df.head(n_logs)
        elif mode == "random":
            sampled_df = df.sample(n=n_logs, random_state=42)
        elif mode == "balanced":
            # Stratified sampling by Level
            if "Level" in df.columns:
                sampled_df = df.groupby("Level", group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), n_logs // df["Level"].nunique() + 1), random_state=42)
                ).head(n_logs)
            else:
                # Fallback to random if no Level column
                sampled_df = df.sample(n=n_logs, random_state=42)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Convert to list of dicts
        logs = sampled_df.to_dict(orient="records")
        
        print(f"✅ Loaded {len(logs)} logs using '{mode}' sampling")
        return logs
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        df = pd.read_csv(self.dataset_path)
        
        info = {
            "name": self.dataset_name,
            "path": str(self.dataset_path),
            "total_logs": len(df),
            "columns": list(df.columns),
            "size_mb": self.dataset_path.stat().st_size / 1024 / 1024,
        }
        
        # Add level distribution if available
        if "Level" in df.columns:
            info["level_distribution"] = df["Level"].value_counts().to_dict()
        
        return info
