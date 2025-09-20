"""Interactive menu for orchestrated hybrid log management experiments.

Provides quick selection of common experiment presets:
 1) Smoke run (fast)               – small episodes/timesteps, core adaptive routers
 2) Full run (long)                – default larger training horizon, all routers
 3) Compliance smoke run           – smoke plus hard compliance enforcement
 4) Train only (no evaluation)     – retrain RL artifacts then stop
 5) Custom (prompt for key params) – flexible execution

The menu ultimately shells out to:  python -m src.experiment ...

Non-interactive usage:  python -m src.menu --preset smoke --log_filepath data/Loghub-zenodo_Logs.csv
Optionally add: --compliance to enable compliance patterns.

This file intentionally avoids importing heavy ML libs; it only invokes subprocess commands.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "Loghub-zenodo_Logs.csv"

PRESETS: Dict[str, Dict[str, str | int | bool]] = {
    "smoke": {
        "q_episodes": 20,
        "a2c_timesteps": 20000,
        "routers": "q_learning,a2c,cbr",
        "sample_mode": "head",
        "cbr_cost_metric": "latency",
    },
    "full": {
        "q_episodes": 120,
        "a2c_timesteps": 120000,
        "routers": "all",
        "sample_mode": "head",
        "cbr_cost_metric": "combined",
    },
    "compliance_smoke": {
        "q_episodes": 20,
        "a2c_timesteps": 20000,
        "routers": "q_learning,a2c,cbr",
        "sample_mode": "head",
        "cbr_cost_metric": "latency",
        "compliance_enable": True,
    },
}


def build_experiment_cmd(dataset: Path, preset: Dict[str, str | int | bool], output_dir: Path, extra_patterns: str | None = None):
    cmd = [sys.executable, "-m", "src.experiment", "--log_filepath", str(dataset), "--output_dir", str(output_dir)]
    # Core numeric/string params
    cmd += ["--q_episodes", str(preset.get("q_episodes", 120))]
    cmd += ["--a2c_timesteps", str(preset.get("a2c_timesteps", 120000))]
    cmd += ["--routers", str(preset.get("routers", "all"))]
    cmd += ["--sample_mode", str(preset.get("sample_mode", "head"))]
    cmd += ["--cbr_cost_metric", str(preset.get("cbr_cost_metric", "combined"))]

    if preset.get("compliance_enable"):
        cmd.append("--compliance_enable")
        if extra_patterns:
            cmd += ["--compliance_patterns", extra_patterns]
    return cmd


def run_cmd(cmd):
    print("[MENU RUN] " + " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def ensure_dataset(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    return path


def interactive_loop(args):
    dataset = ensure_dataset(Path(args.log_filepath))
    while True:
        print("\n=== Hybrid Log Management Experiment Menu ===")
        print("Dataset:", dataset)
        print("Output dir:", args.output_dir)
        print(" 1) Smoke run")
        print(" 2) Full run")
        print(" 3) Compliance smoke run")
        print(" 4) Train only (retrain RL; skip evaluation of others)")
        print(" 5) Custom parameters")
        print(" q) Quit")
        choice = input("Select option: ").strip().lower()
        if choice == 'q':
            break
        if choice not in {'1','2','3','4','5'}:
            print("Invalid selection.")
            continue

        if choice == '4':
            # Train Q-learning
            q_cmd = [sys.executable, "-m", "src.train_qlearning", "--log_filepath", str(dataset), "--episodes", str(args.train_only_q_episodes), "--model_path_prefix", "trained_models/q_learning", "--sample_mode", args.sample_mode]
            a2c_cmd = [sys.executable, "-m", "src.train", "--timesteps", str(args.train_only_a2c_timesteps), "--model_path", "trained_models/a2c_log_router", "--log_source", "real_world", "--log_filepath", str(dataset), "--sample_mode", args.sample_mode]
            print("\n[TRAIN ONLY] Q-learning...")
            run_cmd(q_cmd)
            print("[TRAIN ONLY] A2C...")
            run_cmd(a2c_cmd)
            continue

        if choice in {'1','2','3'}:
            key = {'1':'smoke','2':'full','3':'compliance_smoke'}[choice]
            preset = PRESETS[key]
            cmd = build_experiment_cmd(dataset, preset, Path(args.output_dir), args.extra_patterns)
            run_cmd(cmd)
            continue

        # Custom
        try:
            q_eps = int(input(f"Q-learning episodes [{args.custom_q_episodes}]: ") or args.custom_q_episodes)
            a2c_steps = int(input(f"A2C timesteps [{args.custom_a2c_timesteps}]: ") or args.custom_a2c_timesteps)
            routers = input("Routers (comma list or 'all') [all]: ").strip() or 'all'
            cbr_metric = input("CBR cost metric (latency|energy|combined) [combined]: ").strip() or 'combined'
            compliance = input("Enable compliance? (y/N): ").strip().lower() == 'y'
            patterns = None
            if compliance:
                patterns = input("Additional compliance patterns (comma sep or blank): ").strip() or None
            preset = {
                'q_episodes': q_eps,
                'a2c_timesteps': a2c_steps,
                'routers': routers,
                'cbr_cost_metric': cbr_metric,
                'sample_mode': args.sample_mode,
                'compliance_enable': compliance,
            }
            cmd = build_experiment_cmd(dataset, preset, Path(args.output_dir), patterns)
            run_cmd(cmd)
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as ex:
            print(f"[ERROR] {ex}")


def parse_args():
    ap = argparse.ArgumentParser(description="Interactive experiment menu for hybrid log management")
    ap.add_argument("--log_filepath", type=str, default=str(DEFAULT_DATASET), help="Path to real-world log CSV")
    ap.add_argument("--output_dir", type=str, default="results/menu_runs", help="Directory for experiment outputs")
    ap.add_argument("--preset", choices=["smoke","full","compliance_smoke"], default=None, help="Run a preset non-interactively then exit")
    ap.add_argument("--compliance", action="store_true", help="Enable compliance for non-interactive preset (overrides preset default)")
    ap.add_argument("--extra_patterns", type=str, default=None, help="Extra compliance patterns (comma separated)")
    ap.add_argument("--sample_mode", choices=["head","random","balanced"], default="head")
    # Train-only fallback defaults
    ap.add_argument("--train_only_q_episodes", type=int, default=60)
    ap.add_argument("--train_only_a2c_timesteps", type=int, default=60000)
    # Custom defaults
    ap.add_argument("--custom_q_episodes", type=int, default=120)
    ap.add_argument("--custom_a2c_timesteps", type=int, default=120000)
    return ap.parse_args()


def main():
    args = parse_args()
    dataset = ensure_dataset(Path(args.log_filepath))
    os.makedirs(args.output_dir, exist_ok=True)

    if args.preset:
        key = args.preset
        if key not in PRESETS:
            raise SystemExit(f"Unknown preset {key}")
        preset = dict(PRESETS[key])  # copy
        if args.compliance:
            preset['compliance_enable'] = True
        cmd = build_experiment_cmd(dataset, preset, Path(args.output_dir), args.extra_patterns)
        rc = run_cmd(cmd)
        raise SystemExit(rc)

    interactive_loop(args)

if __name__ == "__main__":
    main()
