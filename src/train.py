# src/train.py
import argparse
from pathlib import Path
from stable_baselines3 import A2C
from .rl_environment import LogRoutingEnv
from .log_provider import LogProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _normalize_save_base(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.suffix == ".zip":
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def train_agent(
    total_timesteps: int = 50_000,
    model_save_path: str = "trained_models/a2c_log_router",
    log_source_type: str = "synthetic",
    log_filepath: str | None = None,
    sample_mode: str = "head",             # <— NEW
):
    save_base = _normalize_save_base(model_save_path)

    # pass sample_mode to LogProvider  ↓↓↓
    log_provider = LogProvider(log_source_type=log_source_type, filepath=log_filepath, sample_mode=sample_mode)
    env = LogRoutingEnv(log_provider)

    model = A2C("MlpPolicy", env, verbose=1)
    print("Starting model training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training complete.")

    model.save(str(save_base))
    print(f"Model saved to {save_base}.zip")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--log_source", choices=["synthetic", "real_world"], default="synthetic")
    ap.add_argument("--log_filepath", type=str, default=None)
    ap.add_argument("--model_path", type=str, default="trained_models/a2c_log_router")
    ap.add_argument("--sample_mode", choices=["head", "random", "balanced"], default="head")   # <— NEW
    args = ap.parse_args()

    train_agent(
        total_timesteps=args.timesteps,
        model_save_path=args.model_path,
        log_source_type=args.log_source,
        log_filepath=args.log_filepath,
        sample_mode=args.sample_mode,   
    )
