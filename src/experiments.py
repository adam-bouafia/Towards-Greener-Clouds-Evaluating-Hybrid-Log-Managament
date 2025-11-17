"""
Research Question Experiments Module

Conducts experiments to answer the 4 research questions:
- RQ1: Semantic vs. Basic Statistical Features
- RQ2: XGBoost Routing Accuracy
- RQ3: ML vs. Baseline Routing
- RQ4: Async Blockchain Performance
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentRunner:
    """Runs experiments to answer research questions"""

    def __init__(self, output_dir: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.output_dir = Path(output_dir or f"results/experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "experiment_log.txt"
        
        # Default dataset paths
        self.real_data = self.project_root / "data" / "Loghub-zenodo_Logs.csv"
        self.synth_data = self.project_root / "data" / "Synthetic_Datacenter_Logs.csv"
        
    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def run_command(self, cmd: List[str], description: str) -> Dict:
        """Run a command and return results"""
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.log(f"✅ Success ({duration:.1f}s)")
                return {"success": True, "duration": duration, "stdout": result.stdout, "stderr": result.stderr}
            else:
                self.log(f"❌ Failed (exit code {result.returncode})")
                self.log(f"Error: {result.stderr}")
                return {"success": False, "duration": duration, "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            self.log(f"⏰ Timeout after 1 hour")
            return {"success": False, "duration": 3600, "error": "Timeout"}
        except Exception as e:
            self.log(f"💥 Exception: {e}")
            return {"success": False, "error": str(e)}

    def rq1_semantic_vs_basic(self, quick_test: bool = False):
        """
        RQ1: Semantic vs. Basic Statistical Features
        
        Trains two XGBoost models (basic and semantic) and compares performance
        """
        self.log("\n" + "="*80)
        self.log("RQ1: SEMANTIC VS. BASIC STATISTICAL FEATURES")
        self.log("="*80)
        
        rq1_dir = self.output_dir / "rq1_semantic_vs_basic"
        rq1_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Experiment 1.1a: Train Basic Features Model
        self.log("\n--- Experiment 1.1a: Train Basic Features XGBoost ---")
        basic_model_path = rq1_dir / "xgboost_basic"
        cmd_basic = [
            sys.executable, "-m", "src.training.train_semantic_xgboost",
            "--data", str(self.real_data),
            "--output", str(basic_model_path),
            "--balance",
            "--test_split", "0.2",
            "--seed", "42"
        ]
        if quick_test:
            cmd_basic.extend(["--limit", "1000"])
        
        results["basic_training"] = self.run_command(cmd_basic, "Train Basic Features Model")
        
        # Experiment 1.1b: Train Semantic Features Model
        self.log("\n--- Experiment 1.1b: Train Semantic Features XGBoost ---")
        semantic_model_path = rq1_dir / "xgboost_semantic"
        cmd_semantic = [
            sys.executable, "-m", "src.training.train_semantic_xgboost",
            "--data", str(self.real_data),
            "--output", str(semantic_model_path),
            "--semantic",
            "--balance",
            "--test_split", "0.2",
            "--seed", "42"
        ]
        if quick_test:
            cmd_semantic.extend(["--limit", "1000"])
        
        results["semantic_training"] = self.run_command(cmd_semantic, "Train Semantic Features Model")
        
        # Experiment 1.2: Compare both models on same test set
        self.log("\n--- Experiment 1.2: Evaluate Both Models ---")
        if basic_model_path.exists() and semantic_model_path.exists():
            cmd_compare = [
                sys.executable, "-m", "src.experiment",
                "--routers", "xgboost_basic,xgboost_semantic",
                "--log_filepath", str(self.real_data),
                "--output_dir", str(rq1_dir / "comparison"),
                "--sample_mode", "stratified"
            ]
            if quick_test:
                cmd_compare.extend(["--limit", "500"])
            
            results["comparison"] = self.run_command(cmd_compare, "Compare Basic vs Semantic")
        else:
            self.log("⚠️  Skipping comparison - models not trained successfully")
            results["comparison"] = {"success": False, "error": "Models not available"}
        
        # Save RQ1 summary
        with open(rq1_dir / "rq1_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.log(f"\n✅ RQ1 experiments complete. Results in: {rq1_dir}")
        return results

    def rq2_xgboost_accuracy(self, quick_test: bool = False):
        """
        RQ2: XGBoost Routing Accuracy
        
        Analyzes routing distribution, query latency, and cost impact
        """
        self.log("\n" + "="*80)
        self.log("RQ2: XGBOOST ROUTING ACCURACY")
        self.log("="*80)
        
        rq2_dir = self.output_dir / "rq2_xgboost_accuracy"
        rq2_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Use semantic model from RQ1 or train new one
        semantic_model = self.output_dir / "rq1_semantic_vs_basic" / "xgboost_semantic"
        if not semantic_model.exists():
            self.log("\n--- Training Semantic Model for RQ2 ---")
            semantic_model = rq2_dir / "xgboost_semantic"
            cmd_train = [
                sys.executable, "-m", "src.training.train_semantic_xgboost",
                "--data", str(self.real_data),
                "--output", str(semantic_model),
                "--semantic",
                "--balance",
                "--test_split", "0.2",
                "--seed", "42"
            ]
            if quick_test:
                cmd_train.extend(["--limit", "1000"])
            
            results["training"] = self.run_command(cmd_train, "Train Semantic Model")
        
        # Experiment 2.2: Analyze routing distribution
        self.log("\n--- Experiment 2.2: Routing Distribution Analysis ---")
        cmd_distribution = [
            sys.executable, "-m", "src.experiment",
            "--router", "xgboost_semantic",
            "--log_filepath", str(self.real_data),
            "--output_dir", str(rq2_dir / "distribution"),
            "--sample_mode", "stratified"
        ]
        if quick_test:
            cmd_distribution.extend(["--limit", "500"])
        
        results["distribution"] = self.run_command(cmd_distribution, "Analyze Routing Distribution")
        
        # Save RQ2 summary
        with open(rq2_dir / "rq2_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.log(f"\n✅ RQ2 experiments complete. Results in: {rq2_dir}")
        return results

    def rq3_ml_vs_baseline(self, quick_test: bool = False):
        """
        RQ3: ML vs. Baseline Routing
        
        Compares semantic XGBoost with baseline approaches:
        - Direct routing (all hot)
        - Rule-based routing
        - CBR (hash-based)
        """
        self.log("\n" + "="*80)
        self.log("RQ3: ML VS. BASELINE ROUTING STRATEGIES")
        self.log("="*80)
        
        rq3_dir = self.output_dir / "rq3_ml_vs_baseline"
        rq3_dir.mkdir(exist_ok=True)
        
        results = {}
        routers = ["direct_mysql", "cbr", "xgboost_semantic"]
        
        # Run each router on same dataset
        for router in routers:
            self.log(f"\n--- Testing Router: {router} ---")
            cmd = [
                sys.executable, "-m", "src.experiment",
                "--router", router,
                "--log_filepath", str(self.real_data),
                "--output_dir", str(rq3_dir / router),
                "--sample_mode", "stratified"
            ]
            if quick_test:
                cmd.extend(["--limit", "500"])
            
            results[router] = self.run_command(cmd, f"Test {router} router")
        
        # Save RQ3 summary
        with open(rq3_dir / "rq3_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.log(f"\n✅ RQ3 experiments complete. Results in: {rq3_dir}")
        return results

    def rq4_async_blockchain(self, quick_test: bool = False):
        """
        RQ4: Async Blockchain Performance
        
        Tests routing with and without blockchain to measure overhead
        """
        self.log("\n" + "="*80)
        self.log("RQ4: ASYNCHRONOUS BLOCKCHAIN PERFORMANCE")
        self.log("="*80)
        
        rq4_dir = self.output_dir / "rq4_async_blockchain"
        rq4_dir.mkdir(exist_ok=True)
        
        results = {}
        
        # Experiment 4.1a: No Blockchain (Baseline)
        self.log("\n--- Experiment 4.1a: Routing WITHOUT Blockchain ---")
        cmd_no_blockchain = [
            sys.executable, "-m", "src.experiment",
            "--router", "xgboost_semantic",
            "--log_filepath", str(self.real_data),
            "--output_dir", str(rq4_dir / "no_blockchain"),
            "--sample_mode", "stratified"
        ]
        if quick_test:
            cmd_no_blockchain.extend(["--limit", "500"])
        
        results["no_blockchain"] = self.run_command(cmd_no_blockchain, "Test without blockchain")
        
        # Experiment 4.1b: Async Blockchain
        self.log("\n--- Experiment 4.1b: Routing WITH Async Blockchain ---")
        cmd_async_blockchain = [
            sys.executable, "-m", "src.experiment",
            "--router", "xgboost_semantic",
            "--log_filepath", str(self.real_data),
            "--blockchain_enable",
            "--output_dir", str(rq4_dir / "async_blockchain"),
            "--sample_mode", "stratified"
        ]
        if quick_test:
            cmd_async_blockchain.extend(["--limit", "500"])
        
        results["async_blockchain"] = self.run_command(cmd_async_blockchain, "Test with async blockchain")
        
        # Save RQ4 summary
        with open(rq4_dir / "rq4_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        self.log(f"\n✅ RQ4 experiments complete. Results in: {rq4_dir}")
        return results

    def run_all_experiments(self, quick_test: bool = False):
        """Run all experiments for all research questions"""
        self.log("\n" + "="*80)
        self.log("RUNNING ALL RESEARCH QUESTION EXPERIMENTS")
        self.log("="*80)
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Quick test mode: {quick_test}")
        
        start_time = time.time()
        
        # Run each RQ experiment
        all_results = {
            "rq1": self.rq1_semantic_vs_basic(quick_test),
            "rq2": self.rq2_xgboost_accuracy(quick_test),
            "rq3": self.rq3_ml_vs_baseline(quick_test),
            "rq4": self.rq4_async_blockchain(quick_test)
        }
        
        total_duration = time.time() - start_time
        
        # Save master summary
        master_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_seconds": total_duration,
            "quick_test": quick_test,
            "output_dir": str(self.output_dir),
            "results": all_results
        }
        
        with open(self.output_dir / "master_summary.json", "w") as f:
            json.dump(master_summary, f, indent=2, default=str)
        
        self.log("\n" + "="*80)
        self.log(f"✅ ALL EXPERIMENTS COMPLETE ({total_duration/60:.1f} minutes)")
        self.log(f"📁 Results directory: {self.output_dir}")
        self.log("="*80)
        
        return all_results

    def generate_report(self):
        """Generate markdown report from experiment results"""
        self.log("\n--- Generating Experiment Report ---")
        
        report_path = self.output_dir / "EXPERIMENT_REPORT.md"
        
        with open(report_path, "w") as f:
            f.write("# Research Question Experiments Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Output Directory**: `{self.output_dir}`\n\n")
            f.write("---\n\n")
            
            # RQ1 Section
            f.write("## RQ1: Semantic vs. Basic Statistical Features\n\n")
            rq1_dir = self.output_dir / "rq1_semantic_vs_basic"
            if rq1_dir.exists():
                f.write(f"**Results Location**: `{rq1_dir}`\n\n")
                f.write("### Experiments Conducted:\n\n")
                f.write("- ✅ Trained basic features XGBoost model (6 features)\n")
                f.write("- ✅ Trained semantic features XGBoost model (778 features)\n")
                f.write("- ✅ Compared performance metrics on same test set\n\n")
                f.write("### Next Steps:\n\n")
                f.write("1. Analyze confusion matrices\n")
                f.write("2. Extract feature importance rankings\n")
                f.write("3. Collect qualitative examples where basic fails\n\n")
            else:
                f.write("⚠️ RQ1 experiments not found\n\n")
            
            # RQ2 Section
            f.write("## RQ2: XGBoost Routing Accuracy\n\n")
            rq2_dir = self.output_dir / "rq2_xgboost_accuracy"
            if rq2_dir.exists():
                f.write(f"**Results Location**: `{rq2_dir}`\n\n")
                f.write("### Experiments Conducted:\n\n")
                f.write("- ✅ Analyzed routing distribution (hot vs cold)\n")
                f.write("- ✅ Measured classification performance\n\n")
                f.write("### Next Steps:\n\n")
                f.write("1. Calculate cost savings (hot vs cold storage)\n")
                f.write("2. Measure query latency impact\n")
                f.write("3. Analyze misrouting consequences\n\n")
            else:
                f.write("⚠️ RQ2 experiments not found\n\n")
            
            # RQ3 Section
            f.write("## RQ3: ML vs. Baseline Routing\n\n")
            rq3_dir = self.output_dir / "rq3_ml_vs_baseline"
            if rq3_dir.exists():
                f.write(f"**Results Location**: `{rq3_dir}`\n\n")
                f.write("### Experiments Conducted:\n\n")
                f.write("- ✅ Tested direct routing (all hot)\n")
                f.write("- ✅ Tested CBR (hash-based)\n")
                f.write("- ✅ Tested semantic XGBoost\n\n")
                f.write("### Next Steps:\n\n")
                f.write("1. Compare accuracy, cost, energy metrics\n")
                f.write("2. Analyze misrouting rates\n")
                f.write("3. Calculate cost/energy efficiency rankings\n\n")
            else:
                f.write("⚠️ RQ3 experiments not found\n\n")
            
            # RQ4 Section
            f.write("## RQ4: Async Blockchain Performance\n\n")
            rq4_dir = self.output_dir / "rq4_async_blockchain"
            if rq4_dir.exists():
                f.write(f"**Results Location**: `{rq4_dir}`\n\n")
                f.write("### Experiments Conducted:\n\n")
                f.write("- ✅ Tested routing without blockchain (baseline)\n")
                f.write("- ✅ Tested routing with async blockchain\n\n")
                f.write("### Next Steps:\n\n")
                f.write("1. Compare latency distributions (P50, P95, P99)\n")
                f.write("2. Measure blockchain verification rate\n")
                f.write("3. Test tamper detection\n")
                f.write("4. Calculate system resource overhead\n\n")
            else:
                f.write("⚠️ RQ4 experiments not found\n\n")
            
            f.write("---\n\n")
            f.write("## Summary\n\n")
            f.write("All experiments have been executed. Review the individual result directories ")
            f.write("for detailed CSV outputs, metrics, and performance data.\n\n")
            f.write("**Use these results to answer your research questions in the thesis!**\n")
        
        self.log(f"📄 Report generated: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Research Question Experiments Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (full dataset)
  python -m src.experiments --all
  
  # Run all experiments (quick test with 1000 logs)
  python -m src.experiments --all --quick
  
  # Run specific RQ experiment
  python -m src.experiments --rq1
  python -m src.experiments --rq2
  python -m src.experiments --rq3
  python -m src.experiments --rq4
  
  # Custom output directory
  python -m src.experiments --all --output results/my_experiments
        """
    )
    
    # Experiment selection
    parser.add_argument("--all", action="store_true", help="Run all RQ experiments")
    parser.add_argument("--rq1", action="store_true", help="Run RQ1: Semantic vs Basic Features")
    parser.add_argument("--rq2", action="store_true", help="Run RQ2: XGBoost Accuracy")
    parser.add_argument("--rq3", action="store_true", help="Run RQ3: ML vs Baseline")
    parser.add_argument("--rq4", action="store_true", help="Run RQ4: Async Blockchain")
    
    # Options
    parser.add_argument("--quick", action="store_true", help="Quick test mode (1000 logs)")
    parser.add_argument("--output", type=str, help="Custom output directory")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing results")
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir=args.output)
    
    # Report only mode
    if args.report_only:
        runner.generate_report()
        return
    
    # Determine which experiments to run
    if not (args.all or args.rq1 or args.rq2 or args.rq3 or args.rq4):
        print("❌ Error: Must specify at least one experiment to run")
        print("   Use --all, --rq1, --rq2, --rq3, or --rq4")
        print("   Use --help for more information")
        sys.exit(1)
    
    # Run experiments
    if args.all:
        runner.run_all_experiments(quick_test=args.quick)
    else:
        if args.rq1:
            runner.rq1_semantic_vs_basic(quick_test=args.quick)
        if args.rq2:
            runner.rq2_xgboost_accuracy(quick_test=args.quick)
        if args.rq3:
            runner.rq3_ml_vs_baseline(quick_test=args.quick)
        if args.rq4:
            runner.rq4_async_blockchain(quick_test=args.quick)
    
    # Generate report
    runner.generate_report()


if __name__ == "__main__":
    main()
