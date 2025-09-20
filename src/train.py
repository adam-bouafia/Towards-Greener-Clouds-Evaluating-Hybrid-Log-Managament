# src/train.py
import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .rl_environment import LogRoutingEnv
from .log_provider import LogProvider

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------- Utilities -----------------------------
def _set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _maybe_git_commit_hash() -> str | None:
    head = PROJECT_ROOT / ".git" / "HEAD"
    if not head.is_file():
        return None
    try:
        content = head.read_text().strip()
        if content.startswith("ref:"):
            ref_path = PROJECT_ROOT / ".git" / content.split(" ", 1)[1]
            if ref_path.is_file():
                return ref_path.read_text().strip()[:12]
        return content[:12]
    except Exception:
        return None


@dataclass
class RunMetadata:
    version: int
    timestamp: float
    obs_dim: int
    embedding_dim: int
    policy_hidden: List[int]
    total_timesteps: int
    learning_rate: float
    gamma: float
    ent_coef: float
    vf_coef: float
    n_steps: int
    n_envs: int
    scaler_present: bool
    scaler_warmup: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    eval_best_mean_reward: float | None
    seed: int | None
    git_commit: str | None
    lr_linear_decay: bool
    final_learning_rate: float
    ent_anneal: bool
    ent_target: float | None
    ent_anneal_steps: int | None
    final_ent_coef: float
    checkpoint_interval: int | None
    num_checkpoints: int


class EpisodeRewardCallback(BaseCallback):
    """Collect per-episode rewards and optionally write to CSV."""

    def __init__(self, csv_path: str | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.episode_rewards: list[float] = []
        self._fp = None
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self._fp = open(self.csv_path, "w", encoding="utf-8")
            self._fp.write("episode,reward\n")

    def _on_step(self) -> bool:
        # SB3 stores episode info in self.locals["infos"] list
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:  # triggered when an env is done
                r = float(info["episode"]["r"])  # total reward
                self.episode_rewards.append(r)
                if self._fp:
                    self._fp.write(f"{len(self.episode_rewards)-1},{r}\n")
        return True

    def _on_training_end(self) -> None:
        if self._fp:
            self._fp.flush()
            self._fp.close()


def _build_env(log_provider: LogProvider, n_envs: int):
    if n_envs <= 1:
        return LogRoutingEnv(log_provider)
    # For now use DummyVecEnv (process overhead of SubprocVecEnv + heavy model inference per step is not ideal)
    def make_env():
        return LogRoutingEnv(log_provider)
    return DummyVecEnv([make_env for _ in range(n_envs)])


class LRScheduleCallback(BaseCallback):
    """Linearly decay learning rate to zero across total timesteps."""
    def __init__(self, total_timesteps: int, initial_lr: float):
        super().__init__()
        self.total = max(1, total_timesteps)
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    def _on_step(self) -> bool:
        frac = min(max(self.num_timesteps / self.total, 0.0), 1.0)
        self.current_lr = self.initial_lr * (1.0 - frac)
        try:
            for pg in self.model.policy.optimizer.param_groups:
                pg['lr'] = self.current_lr
        except Exception:
            pass
        return True


class EntropyAnnealCallback(BaseCallback):
    """Anneal entropy coefficient from start to target over given steps."""
    def __init__(self, start_ent: float, target_ent: float, anneal_steps: int):
        super().__init__()
        self.start = start_ent
        self.target = target_ent
        self.steps = max(1, anneal_steps)
        self.current = start_ent

    def _on_step(self) -> bool:
        t = min(self.num_timesteps / self.steps, 1.0)
        self.current = self.start + (self.target - self.start) * t
        try:
            self.model.ent_coef = self.current
        except Exception:
            pass
        return True

def _normalize_save_base(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.suffix == ".zip":
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def train_agent(
    *,
    total_timesteps: int,
    model_save_path: str,
    log_source_type: str,
    log_filepath: str | None,
    sample_mode: str,
    learning_rate: float,
    gamma: float,
    ent_coef: float,
    vf_coef: float,
    n_steps: int,
    n_envs: int,
    policy_hidden: list[int],
    eval_interval: int | None,
    eval_episodes: int,
    save_best: bool,
    reward_csv_path: str | None,
    scaler_warmup: int,
    seed: int | None,
    tensorboard_log: str | None,
    lr_linear_decay: bool,
    ent_anneal: bool,
    ent_target: float | None,
    ent_anneal_steps: int | None,
    checkpoint_interval: int | None,
):
    _set_seed(seed)
    save_base = _normalize_save_base(model_save_path)

    log_provider = LogProvider(
        log_source_type=log_source_type,
        filepath=log_filepath,
        sample_mode=sample_mode,
    )
    env = _build_env(log_provider, n_envs)

    policy_kwargs = {"net_arch": dict(pi=policy_hidden, vf=policy_hidden)}

    # Optional scaler warmup: gather observations (only when n_envs == 1 for simplicity)
    scaler = None
    if scaler_warmup > 0:
        if n_envs > 1:
            print("[A2C] Scaler warmup skipped (n_envs > 1 currently not supported for warmup collection).")
        else:
            from .feature_extractor import LogFeatureExtractor
            from .utils import get_system_state
            fe = LogFeatureExtractor()
            obs_acc = []
            stream = log_provider.get_log_stream(num_logs=scaler_warmup)
            for log in stream:
                system_state = get_system_state()
                emb = fe.get_embedding(log.get("Content", ""))
                obs = np.concatenate([system_state, emb]).astype(np.float32)
                obs_acc.append(obs)
            if obs_acc:
                arr = np.vstack(obs_acc)
                mean = arr.mean(axis=0).astype(np.float32)
                std = arr.std(axis=0).astype(np.float32)
                std[std < 1e-6] = 1e-6
                scaler = {"mean": mean, "std": std}
                print(f"[A2C] Computed scaler from {len(obs_acc)} samples.")
            else:
                print("[A2C] Scaler warmup produced no samples.")

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        n_steps=n_steps,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )

    callbacks: list[BaseCallback] = []
    ep_cb = EpisodeRewardCallback(csv_path=reward_csv_path)
    callbacks.append(ep_cb)

    lr_cb = None
    if lr_linear_decay:
        lr_cb = LRScheduleCallback(total_timesteps=total_timesteps, initial_lr=learning_rate)
        callbacks.append(lr_cb)

    ent_cb = None
    if ent_anneal and ent_target is not None and ent_anneal_steps is not None:
        ent_cb = EntropyAnnealCallback(start_ent=ent_coef, target_ent=ent_target, anneal_steps=ent_anneal_steps)
        callbacks.append(ent_cb)

    # Periodic checkpointing
    checkpoints = []
    if checkpoint_interval and checkpoint_interval > 0:
        class _CheckpointCallback(BaseCallback):
            def __init__(self, interval: int, base: Path):
                super().__init__()
                self.interval = interval
                self.base = base
                self.saved = []
            def _on_step(self) -> bool:
                if self.num_timesteps > 0 and self.num_timesteps % self.interval == 0:
                    path = Path(str(self.base) + f"_ckpt_{self.num_timesteps}.zip")
                    try:
                        self.model.save(str(path))
                        self.saved.append(str(path))
                        print(f"[A2C] Checkpoint saved: {path}")
                    except Exception as e:
                        print(f"[A2C] Failed to save checkpoint ({e})")
                return True
        ckpt_cb = _CheckpointCallback(checkpoint_interval, save_base)
        callbacks.append(ckpt_cb)
    else:
        ckpt_cb = None

    eval_cb = None
    if eval_interval and eval_interval > 0:
        if n_envs > 1:
            # Build a separate eval env (single instance) sharing provider
            eval_env = _build_env(log_provider, 1)
        else:
            eval_env = env
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(save_base.parent),
            log_path=str(save_base.parent),
            eval_freq=eval_interval,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_cb)

    print("[A2C] Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    print("[A2C] Training complete.")

    model.save(str(save_base))
    print(f"[A2C] Model saved to {save_base}.zip")

    # Save best model if requested (EvalCallback already saved); create a friendly symlink/copy name
    best_mean = None
    if eval_cb and save_best:
        try:
            # EvalCallback writes best_model.zip in best_model_save_path
            best_ckpt = Path(save_base.parent) / "best_model.zip"
            if best_ckpt.is_file():
                target = Path(str(save_base) + "_best.zip")
                if target.exists():
                    target.unlink()
                best_ckpt.replace(target)
                print(f"[A2C] Best model copied to {target}")
            # Parse last eval results if log file exists
            results_path = Path(save_base.parent) / "evaluations.npz"
            if results_path.is_file():
                data = np.load(results_path)
                if "results" in data and len(data["results"]) > 0:
                    # results shape: (n_evals, n_eval_episodes, 1)
                    means = data["results"].mean(axis=1).flatten()
                    if len(means):
                        best_mean = float(np.max(means))
        except Exception as e:
            print(f"[A2C] Could not process best model info: {e}")

    # Metadata + scaler persistence
    try:
        from .feature_extractor import EMBED_MODEL
        embedding_dim = 768  # default fallback
        if hasattr(model.policy, 'mlp_extractor'):
            # We know observation dim from first layer
            obs_dim = model.policy.mlp_extractor.policy_net[0].in_features if hasattr(model.policy.mlp_extractor.policy_net[0], 'in_features') else 0
        else:
            obs_dim = 0
        rewards = np.array(ep_cb.episode_rewards, dtype=np.float32) if ep_cb.episode_rewards else np.array([])
        md = RunMetadata(
            version=1,
            timestamp=time.time(),
            obs_dim=int(obs_dim),
            embedding_dim=int(embedding_dim),
            policy_hidden=policy_hidden,
            total_timesteps=total_timesteps,
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            ent_coef=float(ent_coef),
            vf_coef=float(vf_coef),
            n_steps=int(n_steps),
            n_envs=int(n_envs),
            scaler_present=bool(scaler is not None),
            scaler_warmup=int(scaler_warmup),
            reward_mean=float(rewards.mean()) if rewards.size else 0.0,
            reward_std=float(rewards.std()) if rewards.size else 0.0,
            reward_min=float(rewards.min()) if rewards.size else 0.0,
            reward_max=float(rewards.max()) if rewards.size else 0.0,
            eval_best_mean_reward=best_mean,
            seed=seed,
            git_commit=_maybe_git_commit_hash(),
            lr_linear_decay=bool(lr_linear_decay),
            final_learning_rate=float(lr_cb.current_lr) if lr_cb else float(learning_rate),
            ent_anneal=bool(ent_anneal),
            ent_target=float(ent_target) if ent_target is not None else None,
            ent_anneal_steps=int(ent_anneal_steps) if ent_anneal_steps is not None else None,
            final_ent_coef=float(ent_cb.current) if ent_cb else float(ent_coef),
            checkpoint_interval=int(checkpoint_interval) if checkpoint_interval else None,
            num_checkpoints=len(ckpt_cb.saved) if ckpt_cb else 0,
        )
        meta_path = Path(str(save_base) + "_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(md), f, indent=2)
        print(f"[A2C] Wrote metadata to {meta_path}")
    except Exception as e:
        print(f"[A2C] Failed to write metadata: {e}")

    if scaler is not None:
        try:
            import pickle
            scaler_path = Path(str(save_base) + "_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump({"mean": scaler["mean"], "std": scaler["std"]}, f)
            print(f"[A2C] Saved scaler to {scaler_path}")
        except Exception as e:
            print(f"[A2C] Failed to save scaler: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train A2C log routing agent")
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--log_source", choices=["synthetic", "real_world"], default="synthetic")
    ap.add_argument("--log_filepath", type=str, default=None)
    ap.add_argument("--model_path", type=str, default="trained_models/a2c_log_router")
    ap.add_argument("--sample_mode", choices=["head", "random", "balanced"], default="head")
    # Hyperparams
    ap.add_argument("--learning_rate", type=float, default=7e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--n_steps", type=int, default=5)
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--policy_hidden", type=str, default="128,128", help="Comma-separated layer sizes for policy/value networks")
    # Evaluation & logging
    ap.add_argument("--eval_interval", type=int, default=0, help="Eval every N steps (0 disables)")
    ap.add_argument("--eval_episodes", type=int, default=5)
    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--reward_csv_path", type=str, default=None)
    ap.add_argument("--tensorboard_log", type=str, default=None)
    # Schedules
    ap.add_argument("--lr_linear_decay", action="store_true", help="Linearly decay learning rate to 0 over training")
    ap.add_argument("--ent_anneal", action="store_true", help="Anneal entropy coefficient toward target")
    ap.add_argument("--ent_target", type=float, default=None, help="Target entropy coefficient (with --ent_anneal)")
    ap.add_argument("--ent_anneal_steps", type=int, default=None, help="Steps over which to anneal entropy (with --ent_anneal)")
    ap.add_argument("--checkpoint_interval", type=int, default=None, help="Save intermediate checkpoints every N steps")
    # Scaler
    ap.add_argument("--scaler_warmup", type=int, default=0, help="Collect this many observations to compute mean/std before training")
    # Reproducibility
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    policy_hidden = [int(x) for x in args.policy_hidden.split(",") if x.strip()]

    train_agent(
        total_timesteps=args.timesteps,
        model_save_path=args.model_path,
        log_source_type=args.log_source,
        log_filepath=args.log_filepath,
        sample_mode=args.sample_mode,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_steps=args.n_steps,
        n_envs=args.n_envs,
        policy_hidden=policy_hidden,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_best=args.save_best,
        reward_csv_path=args.reward_csv_path,
        scaler_warmup=args.scaler_warmup,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
        lr_linear_decay=args.lr_linear_decay,
        ent_anneal=args.ent_anneal,
        ent_target=args.ent_target,
        ent_anneal_steps=args.ent_anneal_steps,
        checkpoint_interval=args.checkpoint_interval,
    )
