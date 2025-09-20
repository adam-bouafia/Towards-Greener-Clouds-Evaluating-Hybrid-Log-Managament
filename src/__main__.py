import argparse
import csv
import itertools
import os
import statistics
import time
import re

import psutil

from .backends import BackendManager
from .config import MYSQL_DATABASE, MYSQL_ROOT_PASSWORD
from .log_provider import LogProvider
from .metrics import EnergyMeter
from .routers import StaticRouter, QLearningRouter, A2CRouter, DirectRouter, CBRRouter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SENSITIVE_LEVELS = {"error", "crit", "alert", "emerg"}  # retained legacy heuristic (may be superseded)
DEFAULT_COMPLIANCE_PATTERNS = [
    "sessionid",  # session identifiers
    "token",       # access / job / security tokens
    "secret key",  # explicit key reference
    "permission denied",  # auth failures
    "login",       # authentication events
    "/home/",      # user home paths -> potential user identification
]


def build_sensitive_matcher(user_patterns: list[str] | None):
    # Merge user patterns with defaults, keeping order & uniqueness (case-insensitive comparison)
    pats: list[str] = []
    for src in [DEFAULT_COMPLIANCE_PATTERNS, user_patterns or []]:
        for p in src:
            q = p.strip()
            if not q:
                continue
            if q.lower() not in {x.lower() for x in pats}:
                pats.append(q)
    lowered = [p.lower() for p in pats]

    # Precompile simple substring matcher; regex not required unless patterns contain anchors.
    def _matcher(log: dict) -> bool:
        content_full = " ".join(
            [
                str(log.get("Content", "")),
                str(log.get("EventTemplate", "")),
                str(log.get("Component", "")),
            ]
        ).lower()
        lvl = str(log.get("Level", "")).lower()

        # Level-based legacy heuristic (kept for possible critical security/error escalations)
        if lvl in SENSITIVE_LEVELS:
            return True

        for pat in lowered:
            if pat in content_full:
                return True
        return False

    return _matcher, pats


def _normalize_model_base(p: str | None) -> str:
    """
    Accepts:
      - 'trained_models/a2c_NAME'
      - 'trained_models/a2c_NAME.zip'
      - accidental '...zip.zip'
    Returns a base path (no .zip). A2CRouter will resolve appropriately.
    """
    if not p:
        return "trained_models/a2c_log_router"
    s = str(p)
    s = s[:-4] if s.lower().endswith(".zip") else s
    s = s[:-4] if s.lower().endswith(".zip") else s  # strip .zip.zip -> .zip -> base
    return s


def _build_router(name: str, model_path: str, **kwargs):
    if name == "static":
        return StaticRouter()
    if name == "q_learning":
        return QLearningRouter()
    if name == "a2c":
        base = _normalize_model_base(model_path)
        if not os.path.isabs(base):
            base = os.path.join(PROJECT_ROOT, base)
        return A2CRouter(base)
    if name == "cbr":
        return CBRRouter(**kwargs)

    # direct modes:
    if name == "direct_mysql":
        return DirectRouter("mysql")
    if name == "direct_elk":
        return DirectRouter("elk")
    if name == "direct_ipfs":
        return DirectRouter("ipfs")

    raise ValueError(f"Unknown router: {name}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Log Management System Experiment Runner")
    parser.add_argument(
        "--router",
        type=str,
        required=True,
        choices=["static", "q_learning", "a2c", "cbr", "direct_mysql", "direct_elk", "direct_ipfs"],
        help="Routing algorithm to use",
    )
    parser.add_argument("--sample_mode", type=str, default="head",
                    choices=["head", "random", "balanced"],
                    help="How to traverse real_world CSV for this run")
    parser.add_argument("--log_source", type=str, default="synthetic", help="synthetic or real_world")
    parser.add_argument("--log_filepath", type=str, help="Path to CSV when --log_source=real_world")
    parser.add_argument("--num_logs", type=int, default=100, help="For synthetic source only")
    parser.add_argument("--model_path", type=str, default="trained_models/a2c_log_router", help="A2C model base path")
    parser.add_argument(
        "--limit", type=int, default=0, help="Cap logs processed for this run (0 = no cap; useful for quick tests)"
    )
    parser.add_argument(
        "--emissions_kg_per_kwh",
        type=float,
        default=float(os.environ.get("EMISSIONS_KG_PER_KWH", "0.233")),  # EU-ish average
        help="Carbon intensity used for summary emissions computation",
    )
    parser.add_argument("--cbr_num_buckets", type=int, default=24, help="CBR: number of hash buckets per attribute")
    parser.add_argument("--cbr_sample_prob", type=float, default=0.06, help="CBR: probability of sampling a log for stats")
    parser.add_argument("--cbr_warm_samples", type=int, default=150, help="CBR: samples required before attribute selection")
    parser.add_argument("--cbr_recompute_interval", type=int, default=200, help="CBR: decisions between attribute rescoring")
    parser.add_argument("--cbr_cost_metric", choices=["latency", "energy", "combined"], default="latency", help="CBR: cost metric")
    parser.add_argument("--cbr_json_dump_path", type=str, default=None, help="CBR: path to write periodic JSON diagnostic snapshot")
    parser.add_argument("--cbr_json_dump_interval", type=int, default=0, help="CBR: dump interval in decisions (0 disables)")
    parser.add_argument("--cbr_latency_weight", type=float, default=1.0, help="CBR: latency weight for combined cost")
    parser.add_argument("--cbr_energy_weight", type=float, default=1000.0, help="CBR: energy weight for combined cost")
    parser.add_argument("--cbr_json_dump_mode", choices=["overwrite", "append", "timestamp"], default="overwrite", help="CBR: JSON dump writing mode")
    parser.add_argument("--cbr_state_path", type=str, default=None, help="CBR: path to persist learned stats between runs")
    # --- Compliance (hard override) ---
    parser.add_argument("--compliance_enable", action="store_true", help="Enable hard compliance routing override (force sensitive logs to IPFS)")
    parser.add_argument(
        "--compliance_patterns",
        type=str,
        default=None,
        help="Comma-separated additional substring patterns to classify a log as sensitive (merged with defaults)",
    )
    args = parser.parse_args()

    dataset_name = os.path.splitext(os.path.basename(args.log_filepath or ""))[0] or args.log_source

    backend_manager = BackendManager(
        type("Config", (object,), {"MYSQL_ROOT_PASSWORD": MYSQL_ROOT_PASSWORD, "MYSQL_DATABASE": MYSQL_DATABASE})()
    )
    cbr_kwargs = dict(
        num_buckets=args.cbr_num_buckets,
        sample_prob=args.cbr_sample_prob,
        warm_samples=args.cbr_warm_samples,
        recompute_interval=args.cbr_recompute_interval,
        cost_metric=args.cbr_cost_metric,
        json_dump_path=args.cbr_json_dump_path,
        json_dump_interval=args.cbr_json_dump_interval,
        latency_weight=args.cbr_latency_weight,
        energy_weight=args.cbr_energy_weight,
        json_dump_mode=args.cbr_json_dump_mode,
        state_path=args.cbr_state_path,
    ) if args.router == "cbr" else {}

    router = _build_router(args.router, args.model_path, **cbr_kwargs)

    user_pattern_list = []
    if args.compliance_patterns:
        user_pattern_list = [p.strip() for p in args.compliance_patterns.split(',') if p.strip()]
    sensitive_fn, effective_patterns = build_sensitive_matcher(user_pattern_list)
    meter = EnergyMeter()
    proc = psutil.Process(os.getpid())

    results = []
    print(f"Starting experiment: router={args.router} dataset={dataset_name}")

    start_wall = time.time()
    processed = 0
    latencies_total = []
    log_provider = LogProvider(
        log_source_type=args.log_source,
        filepath=args.log_filepath,
        sample_mode=args.sample_mode
)
    # choose stream + optional cap
    stream = log_provider.get_log_stream(num_logs=args.num_logs)
    if args.limit and args.limit > 0:
        stream = itertools.islice(stream, args.limit)

    try:
        for log_entry in stream:
            processed += 1

            payload_bytes = len((log_entry.get("Content") or "").encode("utf-8"))
            sensitive = sensitive_fn(log_entry) if args.compliance_enable else False

            t0 = time.perf_counter()
            raw_destination = router.get_route(log_entry)
            routing_latency_ms = (time.perf_counter() - t0) * 1000.0

            compliance_forced = False
            # Hard override: enforce IPFS for sensitive logs when compliance enabled
            if args.compliance_enable and sensitive and raw_destination != "ipfs":
                destination = "ipfs"
                compliance_forced = True
            else:
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

                if destination == "mysql":
                    res = backend_manager.write_to_mysql(log_entry)
                    success = bool(res)
                elif destination == "elk":
                    res = backend_manager.write_to_elk(log_entry)
                    success = bool(res)
                elif destination == "ipfs":
                    cid = backend_manager.write_to_ipfs(log_entry)
                    success = bool(cid)
                else:
                    success = False

                e = meter.stop()
                backend_write_latency_ms = (time.perf_counter() - t1) * 1000.0
                proc_cpu_pct = proc.cpu_percent(None)
                if e:
                    energy_cpu_pkg_j = getattr(e, "cpu_pkg_j", 0.0)

            except Exception as ex:
                print(f"Error writing to backend {destination}: {ex}")
                success = False
                backend_write_latency_ms = 1000.0

            # feedback hook for adaptive routers
            try:
                router.observe(
                    log_entry=log_entry,
                    destination=destination,
                    success=success,
                    routing_latency_ms=routing_latency_ms,
                    backend_latency_ms=backend_write_latency_ms,
                    energy_cpu_pkg_j=energy_cpu_pkg_j,
                )
            except Exception as _obs_ex:
                print(f"[ROUTER_OBSERVE] {type(_obs_ex).__name__}: {_obs_ex}")

            total_latency_ms = routing_latency_ms + backend_write_latency_ms
            latencies_total.append(total_latency_ms)

            results.append(
                {
                    "log_id": processed,
                    "router": args.router,
                    "dataset_name": dataset_name,
                    "destination": destination,
                    "raw_destination": raw_destination,
                    "compliance_forced": compliance_forced,
                    "routing_latency_ms": routing_latency_ms,
                    "backend_write_latency_ms": backend_write_latency_ms,
                    "total_latency_ms": total_latency_ms,
                    "success": success,
                    "energy_cpu_pkg_j": energy_cpu_pkg_j,   # CPU package energy per log (J)
                    "payload_bytes": payload_bytes,
                    "sensitive": sensitive,
                    "proc_cpu_pct": proc_cpu_pct,
                    "proc_rss_mb": proc_rss_mb,
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
                "ComplianceEnabled",
                "CompliancePatterns",
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
                "ComplianceEnabled": args.compliance_enable,
                "CompliancePatterns": ";".join(effective_patterns) if args.compliance_enable else "",
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
