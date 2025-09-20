"""Unified experiment orchestrator for six routers.

Responsibilities:
1. Train offline RL agents (Q-learning, A2C) if artifacts missing or --force_retrain.
2. Run evaluation for selected routers over a single dataset.
3. Aggregate per-log CSVs into a combined summary with extra metrics (success rate, p95 latency, destination mix, sensitive fraction).
4. Emit run metadata JSON capturing parameters & statuses.

Usage (example):

python -m src.experiment \
  --log_filepath data/Loghub-zenodo_Logs.csv \
  --sample_mode head \
  --q_episodes 120 \
  --a2c_timesteps 120000 \
  --routers all \
  --cbr_cost_metric combined \
  --output_dir results/exp1

Routers included when --routers all:
  a2c,cbr,q_learning,direct_mysql,direct_elk,direct_ipfs

Note: static router intentionally excluded from default comparative set.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODELS = PROJECT_ROOT / "trained_models"
RESULTS_DIR = PROJECT_ROOT / "results"

REQUIRED_Q_ARTIFACT_SUFFIXES = ["_q_table.pkl", "_pca.pkl", "_binner.pkl", "_scaler.pkl", "_metadata.json"]
A2C_BASE_NAME = "a2c_log_router"

DEFAULT_ROUTERS = ["a2c", "cbr", "q_learning", "direct_mysql", "direct_elk", "direct_ipfs"]


def _have_qlearning(prefix: str) -> bool:
    base = TRAINED_MODELS / prefix
    # prefix like 'q_learning'
    for suf in REQUIRED_Q_ARTIFACT_SUFFIXES:
        if not (TRAINED_MODELS / f"{prefix}{suf}").exists():
            return False
    return True

def _have_a2c(base: str) -> bool:
    # base without .zip; expect base.zip + metadata
    model_zip = TRAINED_MODELS / f"{base}.zip"
    metadata = TRAINED_MODELS / f"{base}_metadata.json"
    return model_zip.exists() and metadata.exists()

def _run(cmd: List[str], env: Dict[str,str] | None = None) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env or os.environ.copy())
    if proc.returncode != 0:
        print(f"[WARN] Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.returncode

def _train_q_learning(args) -> bool:
    prefix = args.q_prefix
    if not args.force_retrain and _have_qlearning(prefix):
        print("[QLEARN] Artifacts present; skipping retrain.")
        return True
    cmd = [sys.executable, "-m", "src.train_qlearning", "--log_filepath", args.log_filepath,
           "--episodes", str(args.q_episodes), "--model_path_prefix", f"trained_models/{prefix}",
           "--sample_mode", args.sample_mode]
    if args.q_reward_history_csv:
        cmd += ["--reward_history_csv_path", args.q_reward_history_csv]
    if args.q_adaptive_events:
        cmd += ["--adaptive_events_path", args.q_adaptive_events]
    return _run(cmd) == 0

def _train_a2c(args) -> bool:
    base = args.a2c_base
    if not args.force_retrain and _have_a2c(base):
        print("[A2C] Artifacts present; skipping retrain.")
        return True
    cmd = [sys.executable, "-m", "src.train", "--timesteps", str(args.a2c_timesteps),
           "--model_path", f"trained_models/{base}", "--log_source", "real_world", "--log_filepath", args.log_filepath,
           "--sample_mode", args.sample_mode]
    if args.a2c_lr_linear_decay:
        cmd.append("--lr_linear_decay")
    if args.a2c_ent_anneal:
        cmd += ["--ent_anneal"]
        if args.a2c_ent_target is not None:
            cmd += ["--ent_target", str(args.a2c_ent_target)]
        if args.a2c_ent_steps is not None:
            cmd += ["--ent_anneal_steps", str(args.a2c_ent_steps)]
    if args.a2c_scaler_warmup > 0:
        cmd += ["--scaler_warmup", str(args.a2c_scaler_warmup)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    return _run(cmd) == 0

def _evaluate_router(router: str, args) -> Dict[str, Any]:
    dataset_name = Path(args.log_filepath).stem
    cmd = [sys.executable, "-m", "src", "--router", router, "--log_source", "real_world", "--log_filepath", args.log_filepath,
           "--sample_mode", args.sample_mode]
    if router == "a2c":
        cmd += ["--model_path", f"trained_models/{args.a2c_base}"]
    if router == "cbr":
        cmd += ["--cbr_cost_metric", args.cbr_cost_metric]
        if args.cbr_state_path:
            cmd += ["--cbr_state_path", args.cbr_state_path]
    # compliance pass-through
    if args.compliance_enable:
        cmd.append("--compliance_enable")
        if args.compliance_patterns:
            cmd += ["--compliance_patterns", args.compliance_patterns]
    rc = _run(cmd)
    perlog_csv = RESULTS_DIR / f"{router}_{dataset_name}.csv"
    summary_csv = RESULTS_DIR / f"summary_{router}_{dataset_name}.csv"
    status = "ok" if (rc == 0 and perlog_csv.exists() and summary_csv.exists()) else "failed"
    return {"router": router, "status": status, "perlog_csv": str(perlog_csv), "summary_csv": str(summary_csv)}

import math
import csv

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')

def _aggregate(perlog_paths: List[Path], dataset_name: str, output_dir: Path) -> Dict[str, Any]:
    rows = []
    for p in perlog_paths:
        if not p.exists():
            continue
        with open(p, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    if not rows:
        return {}
    import statistics
    from collections import Counter
    # Convert
    total_latency = [ _safe_float(r.get('total_latency_ms', 'nan')) for r in rows ]
    success_vals = [ 1.0 if str(r.get('success','')).lower() == 'true' else 0.0 for r in rows ]
    destinations = [ r.get('destination','') for r in rows ]
    sensitive_flags = [ str(r.get('sensitive','')).lower() == 'true' for r in rows ]
    # Compliance metrics (hard override expectation: all sensitive -> ipfs)
    sensitive_total = sum(1 for s in sensitive_flags if s)
    sensitive_to_ipfs = sum(1 for r,s in zip(rows, sensitive_flags) if s and r.get('destination','') == 'ipfs')
    sensitive_leakage = sensitive_total - sensitive_to_ipfs
    sensitive_coverage = (sensitive_to_ipfs / sensitive_total) if sensitive_total else 1.0
    leakage_rate = (sensitive_leakage / sensitive_total) if sensitive_total else 0.0
    sensitive_fraction = sum(1 for s in sensitive_flags if s) / len(rows) if rows else 0.0
    non_sensitive_total = len(rows) - sensitive_total
    non_sensitive_ipfs = sum(1 for r,s in zip(rows, sensitive_flags) if (not s) and r.get('destination','') == 'ipfs')
    non_sensitive_ipfs_fraction = (non_sensitive_ipfs / non_sensitive_total) if non_sensitive_total else 0.0
    energy_j = [ _safe_float(r.get('energy_cpu_pkg_j','nan')) for r in rows ]

    def _pctl(data, p):
        dd = [d for d in data if not math.isnan(d)]
        if not dd:
            return float('nan')
        k = (len(dd)-1)*p/100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return dd[int(k)]
        return dd[f] + (dd[c]-dd[f])*(k-f)

    p95 = _pctl(total_latency, 95)
    median = _pctl(total_latency, 50)
    mean_latency = (sum(d for d in total_latency if not math.isnan(d))/max(1,len([d for d in total_latency if not math.isnan(d)]))) if total_latency else 0.0
    success_rate = sum(success_vals)/len(success_vals) if success_vals else 0.0
    sens_frac = sensitive_fraction
    dest_counter = Counter(destinations)
    total = sum(dest_counter.values()) or 1

    dest_mix = {f"destination_mix_{k}": dest_counter.get(k,0)/total for k in ['mysql','elk','ipfs']}
    energy_per_log_j = sum(e for e in energy_j if not math.isnan(e))/len([e for e in energy_j if not math.isnan(e)]) if energy_j else 0.0

    combined = {
        "dataset_name": dataset_name,
        "rows": len(rows),
        "mean_total_latency_ms": mean_latency,
        "median_total_latency_ms": median,
        "p95_total_latency_ms": p95,
        "success_rate": success_rate,
        "sensitive_fraction": sens_frac,
        "energy_per_log_j": energy_per_log_j,
        **dest_mix,
        # compliance metrics
        "sensitive_total": sensitive_total,
        "sensitive_to_ipfs": sensitive_to_ipfs,
        "sensitive_coverage": sensitive_coverage,
        "sensitive_leakage": sensitive_leakage,
        "leakage_rate": leakage_rate,
        "non_sensitive_ipfs_fraction": non_sensitive_ipfs_fraction,
        "compliance_score": 1.0 if sensitive_leakage == 0 else 0.0,
    }
    return combined


def parse_args():
    ap = argparse.ArgumentParser(description="Unified experiment orchestrator for six routers")
    ap.add_argument("--log_filepath", required=True, help="Path to real-world log CSV")
    ap.add_argument("--sample_mode", choices=["head","random","balanced"], default="head")
    ap.add_argument("--routers", default="all", help="Comma list of routers or 'all'")
    ap.add_argument("--q_episodes", type=int, default=120)
    ap.add_argument("--q_prefix", type=str, default="q_learning")
    ap.add_argument("--q_reward_history_csv", type=str, default=None)
    ap.add_argument("--q_adaptive_events", type=str, default=None)
    ap.add_argument("--a2c_timesteps", type=int, default=120000)
    ap.add_argument("--a2c_base", type=str, default=A2C_BASE_NAME)
    ap.add_argument("--a2c_scaler_warmup", type=int, default=0)
    ap.add_argument("--a2c_lr_linear_decay", action="store_true")
    ap.add_argument("--a2c_ent_anneal", action="store_true")
    ap.add_argument("--a2c_ent_target", type=float, default=None)
    ap.add_argument("--a2c_ent_steps", type=int, default=None)
    ap.add_argument("--cbr_cost_metric", choices=["latency","energy","combined"], default="combined")
    ap.add_argument("--cbr_state_path", type=str, default=None)
    ap.add_argument("--force_retrain", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--write_markdown", action="store_true", help="Also write markdown table of combined summary")
    # compliance
    ap.add_argument("--compliance_enable", action="store_true", help="Enable hard compliance override during evaluation")
    ap.add_argument("--compliance_patterns", type=str, default=None, help="Additional comma-separated patterns (merged with defaults)")
    return ap.parse_args()


def main():
    args = parse_args()
    dataset_name = Path(args.log_filepath).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    status = {
        "dataset_name": dataset_name,
        "start_time": start,
        "routers_requested": args.routers,
        "force_retrain": args.force_retrain,
    }

    # Determine router list
    if args.routers == "all":
        routers = DEFAULT_ROUTERS
    else:
        routers = [r.strip() for r in args.routers.split(',') if r.strip()]
    status["routers"] = routers

    # Train artifacts if needed
    train_results = {}
    if "q_learning" in routers:
        train_results['q_learning'] = _train_q_learning(args)
    if "a2c" in routers:
        train_results['a2c'] = _train_a2c(args)

    status['training'] = train_results

    # Evaluate
    eval_meta = []
    perlog_paths = []
    for r in routers:
        er = _evaluate_router(r, args)
        eval_meta.append(er)
        if er['status'] == 'ok':
            perlog_paths.append(Path(er['perlog_csv']))
    status['evaluation'] = eval_meta

    # Aggregate
    agg = _aggregate(perlog_paths, dataset_name, out_dir)
    status['combined_summary'] = agg

    # Write combined CSV
    if agg:
        combined_csv = out_dir / f"combined_summary_{dataset_name}.csv"
        import csv
        with open(combined_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(agg.keys()))
            writer.writeheader()
            writer.writerow(agg)
        print(f"[AGG] Wrote combined summary to {combined_csv}")
        if args.write_markdown:
            md_path = out_dir / f"combined_summary_{dataset_name}.md"
            with open(md_path, 'w') as f:
                f.write(f"# Combined Summary ({dataset_name})\n\n")
                f.write('| ' + ' | '.join(agg.keys()) + ' |\n')
                f.write('| ' + ' | '.join(['---']*len(agg)) + ' |\n')
                f.write('| ' + ' | '.join(str(v) for v in agg.values()) + ' |\n')
            print(f"[AGG] Wrote markdown summary to {md_path}")

    # Metadata
    status['end_time'] = time.time()
    status['elapsed_sec'] = status['end_time'] - start
    meta_path = out_dir / f"experiment_metadata_{dataset_name}.json"
    with open(meta_path, 'w') as f:
        json.dump(status, f, indent=2)
    print(f"[META] Wrote experiment metadata to {meta_path}")

if __name__ == "__main__":
    main()
