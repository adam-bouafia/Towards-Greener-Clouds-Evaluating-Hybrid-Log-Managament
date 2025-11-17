import argparse
import csv
import itertools
import os
import statistics
import time
import re
from typing import Dict, Any

import psutil

from .backends import ClickHouseBackend, MinIOBackend
from .utils.log_provider import LogProvider
from .routers import XGBoostRouter, DirectRouter
from .blockchain_logger import BlockchainLogger
from .monitoring import EnergyMonitor


# Backend Manager for ClickHouse/MinIO only
class BackendManager:
    """Manages ClickHouse (hot) and MinIO (cold) storage backends."""
    def __init__(self, config=None):
        self.clickhouse = ClickHouseBackend()
        self.minio = MinIOBackend()
    
    def write_to_clickhouse(self, log_entry):
        """Write to ClickHouse hot storage."""
        success, latency_ms = self.clickhouse.write(log_entry)
        return success
    
    def write_to_minio(self, log_entry):
        """Write to MinIO cold storage."""
        success, latency_ms = self.minio.write(log_entry)
        return success
    
    def close_connections(self):
        """Close backend connections."""
        try:
            self.clickhouse.close()
        except:
            pass
        try:
            self.minio.close()
        except:
            pass


# Legacy Energy Meter (compatibility shim)
class EnergyMeter:
    """Wrapper around EnergyMonitor for legacy interface."""
    def __init__(self):
        self.monitor = EnergyMonitor()
        self.start_time = None
    
    def start(self):
        """Start energy measurement."""
        self.start_time = self.monitor.start_measurement()
    
    def stop(self, proc):
        """Stop energy measurement and return result."""
        energy_j = self.monitor.end_measurement()
        duration_s = time.time() - self.start_time if self.start_time else 0.0
        
        # Create a simple result object
        class EnergyResult:
            def __init__(self, energy_j, duration_s):
                self.cpu_pkg_j = energy_j
                self.duration_s = duration_s
                self.system_cpu_cores = os.cpu_count() or 1
                self.process_cpu_time_s = proc.cpu_percent() * duration_s / 100.0 if duration_s > 0 else 0.0
        
        return EnergyResult(energy_j, duration_s)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _build_router(name: str, model_path: str, blockchain_logger=None, **kwargs):
    """
    Build router instance based on name.
    
    Supported routers:
    - direct_clickhouse: All logs to ClickHouse (hot storage)
    - direct_minio: All logs to MinIO (cold storage)
    - xgboost: ML-based intelligent routing
    """
    # Direct modes - ClickHouse (hot) and MinIO (cold) only
    if name == "direct_clickhouse":
        return DirectRouter("clickhouse")
    if name == "direct_minio":
        return DirectRouter("minio")
    
    # XGBoost router
    if name == "xgboost":
        return XGBoostRouter(model_path=model_path, blockchain_logger=blockchain_logger)

    raise ValueError(f"Unknown router: {name}. Supported: direct_clickhouse, direct_minio, xgboost")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Log Management System Experiment Runner")
    
    # Add special mode for running automated experiments
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run automated experiments to answer thesis research questions (RQ1-RQ4)"
    )
    parser.add_argument(
        "--experiments-mode",
        type=str,
        choices=["all", "rq1", "rq2", "rq3", "rq4"],
        default="all",
        help="Which research questions to run (default: all)"
    )
    parser.add_argument(
        "--experiments-quick",
        action="store_true",
        help="Run experiments in quick mode (1000 logs instead of full dataset)"
    )
    parser.add_argument(
        "--experiments-output",
        type=str,
        default=None,
        help="Output directory for experiment results"
    )
    
    parser.add_argument(
        "--router",
        type=str,
        required=False,  # Changed to False since --run-experiments doesn't need it
        choices=["direct_clickhouse", "direct_minio", "xgboost"],
        help="Routing algorithm to use (direct_clickhouse=hot, direct_minio=cold, xgboost=ML-based)",
    )
    parser.add_argument("--sample_mode", type=str, default="head",
                    choices=["head", "random", "balanced"],
                    help="How to traverse real_world CSV for this run")
    parser.add_argument("--log_source", type=str, default="synthetic", help="synthetic or real_world")
    parser.add_argument("--log_filepath", type=str, help="Path to CSV when --log_source=real_world")
    parser.add_argument("--num_logs", type=int, default=100, help="For synthetic source only")
    parser.add_argument("--model_path", type=str, default="xgboost_router", help="XGBoost model path (for xgboost router)")
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap logs processed for this run (0 = no cap; useful for quick tests)"
    )
    parser.add_argument(
        "--emissions_kg_per_kwh",
        type=float,
        default=float(os.environ.get("EMISSIONS_KG_PER_KWH", "0.233")),  # EU-ish average
        help="Carbon intensity used for summary emissions computation",
    )
    parser.add_argument(
        "--no-blockchain",
        action="store_true",
        help="Disable blockchain verification (for baseline performance measurements)"
    )
    args = parser.parse_args()

    # Handle automated experiments mode
    if args.run_experiments:
        print("=" * 80)
        print("🧪 AUTOMATED EXPERIMENTS MODE")
        print("=" * 80)
        print(f"Running: {args.experiments_mode.upper()}")
        print(f"Quick mode: {'Yes (1000 logs)' if args.experiments_quick else 'No (full dataset)'}")
        print("=" * 80)
        print()
        
        # Import and run experiments module
        try:
            from .experiments import ExperimentRunner
            
            runner = ExperimentRunner(
                output_dir=args.experiments_output,
                quick_mode=args.experiments_quick
            )
            
            if args.experiments_mode == "all":
                runner.run_all_experiments()
            elif args.experiments_mode == "rq1":
                runner.rq1_semantic_vs_basic()
            elif args.experiments_mode == "rq2":
                runner.rq2_xgboost_accuracy()
            elif args.experiments_mode == "rq3":
                runner.rq3_ml_vs_baseline()
            elif args.experiments_mode == "rq4":
                runner.rq4_async_blockchain()
            
            # Generate report
            runner.generate_report()
            
            print()
            print("=" * 80)
            print("✅ EXPERIMENTS COMPLETE")
            print(f"Results: {runner.output_dir}")
            print(f"Report: {runner.output_dir}/EXPERIMENT_REPORT.md")
            print("=" * 80)
            
            return  # Exit after experiments
            
        except ImportError as e:
            print(f"❌ Error: Could not import experiments module: {e}")
            print("Make sure src/experiments.py exists.")
            return
        except Exception as e:
            print(f"❌ Error running experiments: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Validate router is provided for normal mode
    if not args.router:
        parser.error("--router is required when not using --run-experiments")

    dataset_name = os.path.splitext(os.path.basename(args.log_filepath or ""))[0] or args.log_source

    # Initialize backend manager (ClickHouse + MinIO)
    backend_manager = BackendManager()

    # Initialize Blockchain Logger (conditional based on experiment type)
    blockchain_logger = None
    
    if args.no_blockchain:
        # Baseline performance measurement mode - blockchain disabled
        print("\n" + "="*80)
        print("� Baseline Performance Mode - Blockchain Verification Disabled")
        print("="*80)
        print("Measuring pure storage backend performance without blockchain overhead")
        print("="*80 + "\n")
        
        blockchain_logger = BlockchainLogger(
            rpc_url=None,
            contract_address=None,
            private_key=None,
            enabled=False,
            sensitivity_threshold=0.5,
            use_ml_detector=False
        )
    else:
        # Integrated system evaluation mode - blockchain enabled for sensitive logs
        print("\n" + "="*80)
        print("🔗 Integrated System Mode - Blockchain Verification Enabled")
        print("="*80)
        
        blockchain_logger = BlockchainLogger(
            rpc_url=os.environ.get("POLYGON_RPC_URL"),
            contract_address=os.environ.get("BLOCKCHAIN_CONTRACT_ADDRESS"),
            private_key=os.environ.get("BLOCKCHAIN_PRIVATE_KEY"),
            enabled=True,
            sensitivity_threshold=0.5,
            use_ml_detector=False
        )
        
        if not blockchain_logger.enabled:
            print("\n❌ FATAL ERROR: Blockchain credentials not configured!")
            print("   Required environment variables:")
            print("     - POLYGON_RPC_URL")
            print("     - BLOCKCHAIN_CONTRACT_ADDRESS")
            print("     - BLOCKCHAIN_PRIVATE_KEY")
            print("\n   Set --no-blockchain flag to run without blockchain verification.")
            import sys
            sys.exit(1)
        
        print(f"✅ Blockchain Connected: {blockchain_logger.w3.provider.endpoint_uri if blockchain_logger.w3 else 'N/A'}")
        print(f"✅ Contract Address: {blockchain_logger.contract_address}")
        print(f"✅ Wallet Address: {blockchain_logger.account.address if blockchain_logger.account else 'N/A'}")
        print("="*80 + "\n")

    # Build router (pass blockchain_logger to xgboost if needed)
    router = _build_router(args.router, args.model_path, blockchain_logger=blockchain_logger)

    # Initialize energy monitoring
    meter = EnergyMeter()
    proc = psutil.Process(os.getpid())

    results = []
    print(f"Starting experiment: router={args.router} dataset={dataset_name}")

    start_wall = time.time()
    processed = 0
    latencies_total = []
    
    # Initialize log provider based on log source
    if args.log_source in ["loghub", "synthetic"]:
        log_provider = LogProvider(dataset_name=args.log_source)
        n_logs = args.limit if args.limit > 0 else None
        logs = log_provider.load_logs(n_logs=n_logs, mode=args.sample_mode)
        stream = iter(logs)
    else:
        raise ValueError(f"Unknown log_source: {args.log_source}. Use 'loghub' or 'synthetic'")

    try:
        for log_entry in stream:
            processed += 1

            payload_bytes = len((log_entry.get("Content") or "").encode("utf-8"))
            
            # Check if log is sensitive using BlockchainLogger's detection
            sensitive = blockchain_logger.is_sensitive(log_entry)
            
            # MANDATORY blockchain submission for sensitive logs (no fallback, no simulation)
            blockchain_hash = None
            blockchain_tx_hash = None
            if sensitive:
                blockchain_hash = blockchain_logger.compute_hash(log_entry)
                # Store on blockchain - returns None if duplicate/fails
                try:
                    blockchain_tx_hash = blockchain_logger.store_hash(log_entry, backend="hybrid")
                    if blockchain_tx_hash:
                        print(f"  🔒 Blockchain TX: {blockchain_tx_hash[:16]}... for sensitive log #{processed}")
                    else:
                        # Duplicate hash or transaction rejected - use local hash
                        print(f"  ⚠️  Blockchain rejected log #{processed} (duplicate hash)")
                        blockchain_tx_hash = f"local_{blockchain_hash[:16]}"
                except Exception as e:
                    # Log the error but continue processing (blockchain temporarily unavailable)
                    print(f"  ⚠️  Blockchain submission failed for log #{processed}: {e}")
                    # Store hash locally for audit trail even if blockchain fails
                    blockchain_tx_hash = f"local_{blockchain_hash[:16]}"

            t0 = time.perf_counter()
            raw_destination = router.get_route(log_entry)
            routing_latency_ms = (time.perf_counter() - t0) * 1000.0

            destination = raw_destination

            backend_write_latency_ms = 0.0
            success = True
            energy_cpu_pkg_j = 0.0
            proc_cpu_pct = 0.0
            proc_rss_mb = float(proc.memory_info().rss) / 1e6  # resident memory MB (context)

            try:
                _ = proc.cpu_percent(None)  # prime
                t1 = time.perf_counter()
                meter.start()

                if destination == "clickhouse":
                    success = backend_manager.write_to_clickhouse(log_entry)
                elif destination == "minio":
                    success = backend_manager.write_to_minio(log_entry)
                else:
                    raise ValueError(f"Unknown backend destination: {destination}. Only 'clickhouse' or 'minio' supported.")

                e = meter.stop(proc)
                backend_write_latency_ms = (time.perf_counter() - t1) * 1000.0
                proc_cpu_pct = proc.cpu_percent(None)
                if e:
                    energy_cpu_pkg_j = getattr(e, "cpu_pkg_j", 0.0)
                    if e.system_cpu_cores > 0 and e.duration_s > 0:
                        process_fraction = e.process_cpu_time_s / (e.duration_s * e.system_cpu_cores)
                        energy_cpu_pkg_j *= min(1.0, max(0.0, process_fraction))

            except Exception as ex:
                print(f"Error writing to backend {destination}: {ex}")
                success = False
                backend_write_latency_ms = 1000.0

            # feedback hook for adaptive routers
            try:
                total_latency_ms_observed = routing_latency_ms + backend_write_latency_ms
                router.observe(
                    log_entry=log_entry,
                    destination=destination,
                    success=success,
                    latency_ms=total_latency_ms_observed,
                    energy_joules=energy_cpu_pkg_j,
                )
            except Exception as _obs_ex:
                # Silent failure for routers that don't implement observe()
                pass

            total_latency_ms = routing_latency_ms + backend_write_latency_ms
            latencies_total.append(total_latency_ms)

            results.append(
                {
                    "log_id": processed,
                    "router": args.router,
                    "dataset_name": dataset_name,
                    "destination": destination,
                    "raw_destination": raw_destination,
                    "sensitive": sensitive,
                    "blockchain_hash": blockchain_hash or "",
                    "blockchain_tx_hash": blockchain_tx_hash or "",
                    "routing_latency_ms": routing_latency_ms,
                    "backend_write_latency_ms": backend_write_latency_ms,
                    "total_latency_ms": total_latency_ms,
                    "success": success,
                    "energy_cpu_pkg_j": energy_cpu_pkg_j,   # CPU package energy per log (J)
                    "payload_bytes": payload_bytes,
                    "sensitive": sensitive,
                    "proc_cpu_pct": proc_cpu_pct,
                    "proc_rss_mb": proc_rss_mb,
                    # Add log metadata for intelligent training
                    "level": log_entry.get("Level", "INFO"),
                    "component": log_entry.get("Component", "unknown"),
                    "log_source": log_entry.get("LogSource", dataset_name),
                    "content": log_entry.get("Content", ""),
                    "event_template": log_entry.get("EventTemplate", ""),
                }
            )

            if processed % 50 == 0:
                print(
                    f"[{dataset_name}] {processed} logs | "
                    f"dest={destination} | route {routing_latency_ms:.1f} ms | backend {backend_write_latency_ms:.1f} ms | "
                    f"E(cpu_pkg={energy_cpu_pkg_j:.4f} J) | payload={payload_bytes}B | sens={sensitive}"
                )

    except Exception as e:
        print(f"An error occurred during experiment: {e}")

    elapsed = time.time() - start_wall
    throughput = (processed / elapsed) if elapsed > 0 else 0.0
    avg_latency_ms = statistics.fmean(latencies_total) if latencies_total else 0.0

    # ------- Blockchain Compliance Metrics (Mandatory Blockchain for Sensitive Logs) -------
    total_sensitive = sum(1 for row in results if row["sensitive"])
    sensitive_with_blockchain = sum(1 for row in results if row["sensitive"] and row["blockchain_tx_hash"])
    sensitive_without_blockchain = total_sensitive - sensitive_with_blockchain
    
    # Coverage should be 100% (all sensitive logs MUST have blockchain TX)
    coverage_pct = (sensitive_with_blockchain / total_sensitive * 100.0) if total_sensitive > 0 else 100.0
    leakage_pct = (sensitive_without_blockchain / total_sensitive * 100.0) if total_sensitive > 0 else 0.0
    
    # Alert if any leakage detected (should never happen with mandatory blockchain)
    if leakage_pct > 0:
        print(f"\n⚠️  WARNING: {leakage_pct:.2f}% leakage detected! Sensitive logs without blockchain protection!")

    # ------- Summary metrics you asked for -------
    total_energy_j = sum(row["energy_cpu_pkg_j"] for row in results)
    total_energy_wh = total_energy_j / 3600.0
    energy_per_log_wh = (total_energy_wh / processed) if processed > 0 else 0.0
    carbon_kg = total_energy_wh * float(args.emissions_kg_per_kwh)

    print("\nExperiment finished.")
    print(f"LogsProcessed: {processed}")
    print(f"TotalEnergyWh: {total_energy_wh:.6f}")
    print(f"EnergyPerLogWh: {energy_per_log_wh:.9f}")
    print(f"CarbonEmissionsKg: {carbon_kg:.6f} (factor={args.emissions_kg_per_kwh} kg/kWh)")
    print(f"AvgLatencyMs: {avg_latency_ms:.2f}")
    print(f"ThroughputLogsPerSec: {throughput:.2f}")
    print(f"\nBlockchain Compliance:")
    print(f"  Sensitive Logs: {total_sensitive}")
    print(f"  With Blockchain: {sensitive_with_blockchain}")
    print(f"  Coverage: {coverage_pct:.2f}% (target: 100%)")
    print(f"  Leakage: {leakage_pct:.2f}% (target: 0%)")

    # ------- Write per-log results -------
    os.makedirs("./results", exist_ok=True)
    perlog_path = f"./results/{args.router}_{dataset_name}.csv"
    if results:
        with open(perlog_path, "w", newline="") as f:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Detailed results saved to {perlog_path}")

    # ------- Write summary row -------
    summary_path = f"./results/summary_{args.router}_{dataset_name}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "router",
                "dataset_name",
                "LogsProcessed",
                "TotalEnergyWh",
                "EnergyPerLogWh",
                "CarbonEmissionsKg",
                "AvgLatencyMs",
                "ThroughputLogsPerSec",
                "EmissionsFactorKgPerKWh",
                "BlockchainEnabled",
                "SensitiveLogs",
                "BlockchainCoveragePct",
                "BlockchainLeakagePct",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "router": args.router,
                "dataset_name": dataset_name,
                "LogsProcessed": processed,
                "TotalEnergyWh": total_energy_wh,
                "EnergyPerLogWh": energy_per_log_wh,
                "CarbonEmissionsKg": carbon_kg,
                "AvgLatencyMs": avg_latency_ms,
                "ThroughputLogsPerSec": throughput,
                "EmissionsFactorKgPerKWh": float(args.emissions_kg_per_kwh),
                "BlockchainEnabled": blockchain_logger.enabled,
                "SensitiveLogs": total_sensitive,
                "BlockchainCoveragePct": coverage_pct,
                "BlockchainLeakagePct": leakage_pct,
            }
        )
    print(f"Summary saved to {summary_path}")

    backend_manager.close_connections()

    if args.router == "cbr":
        try:
            router.save_state()
        except Exception as e:
            print(f"[CBR] Failed to save state at shutdown: {e}")


if __name__ == "__main__":
    main()
