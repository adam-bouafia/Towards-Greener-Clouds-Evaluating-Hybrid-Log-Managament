import argparse
import os
import pickle
import time
import json
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer

from .rl_environment import LogRoutingEnv
from .log_provider import LogProvider
from .routers import StaticRouter  # teacher for guided exploration


def _unwrap_reset(ret):
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0]
    return ret


def _unwrap_step(ret):
    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, reward, bool(terminated or truncated), info
    elif len(ret) == 4:
        obs, reward, done, info = ret
        return obs, reward, bool(done), info
    raise RuntimeError("Unexpected env.step() return format")


def _collect_observations(env, warmup_steps=5000, seed=42):
    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    obs = _unwrap_reset(env.reset())
    buf = []
    for _ in range(warmup_steps):
        a = env.action_space.sample()
        obs, r, done, _ = _unwrap_step(env.step(a))
        buf.append(np.asarray(obs, dtype=np.float32))
        if done:
            obs = _unwrap_reset(env.reset())
    return np.stack(buf, axis=0)


def _fit_state_transformers(observations: np.ndarray, pca_components: int, n_bins: int):
    """Fit scaling (mean/std), PCA and discretizer.

    We standardize observations first (z-score) to stabilize PCA and binning.
    Returns (scaler_dict, pca, binner).
    """
    obs_dim = observations.shape[1]
    n_comp = min(int(pca_components), int(obs_dim))
    if n_comp <= 0:
        raise ValueError(f"Invalid PCA components={pca_components} for obs_dim={obs_dim}")

    mean = observations.mean(axis=0)
    std = observations.std(axis=0) + 1e-8
    normed = (observations - mean) / std

    pca = PCA(n_components=n_comp, random_state=0)
    reduced = pca.fit_transform(normed)
    binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    binner.fit(reduced)
    scaler = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return scaler, pca, binner


def _apply_scaler(obs: np.ndarray, scaler: dict) -> np.ndarray:
    return (obs - scaler["mean"]) / scaler["std"]


def _disc_state(obs: np.ndarray, scaler: dict, pca: PCA, binner: KBinsDiscretizer) -> tuple:
    normed = _apply_scaler(obs.reshape(1, -1), scaler)
    reduced = pca.transform(normed)
    bins = binner.transform(reduced)
    return tuple(int(x) for x in bins[0])


def train_q_learning(
    dataset_name: str,
    log_filepath: str,
    episodes: int,
    max_steps_per_episode: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    adaptive_eps_window: int,
    adaptive_eps_patience: int,
    adaptive_eps_drop: float,
    disable_static_eps_decay: bool,
    adaptive_eps_min_episodes: int,
    warmup_steps: int,
    pca_components: int,
    n_bins: int,
    model_path_prefix: str,
    guided_prob: float,
    prior_bonus: float,
    reward_history_inline_threshold: int,
    reward_history_csv_path: str | None = None,
    adaptive_events_path: str | None = None,
    sample_mode: str = "head",
):
    print(f"[QLEARN] Dataset: {dataset_name}")
    provider = LogProvider(log_source_type="real_world", filepath=log_filepath, sample_mode=sample_mode)
    env = LogRoutingEnv(provider)
    teacher = StaticRouter()
    n_actions = int(env.action_space.n)

    print(f"[QLEARN] Collecting {warmup_steps} warm-up observations...")
    warm_obs = _collect_observations(env, warmup_steps=warmup_steps)
    print(f"[QLEARN] Obs shape: {warm_obs.shape}")

    scaler, pca, binner = _fit_state_transformers(warm_obs, pca_components=pca_components, n_bins=n_bins)
    print(f"[QLEARN] PCA comps={pca.n_components_}, KBins bins={n_bins} (scaling applied)")

    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    epsilon = eps_start
    rewards_hist = []  # per-episode rewards
    recent_avgs = []   # moving window averages
    best_recent_avg = -1e9
    plateau_count = 0
    adaptive_events = []  # list of {episode, new_epsilon}
    steps_done = 0
    t0 = time.time()

    # Prepare adaptive events streaming file if provided (NDJSON)
    adaptive_events_file = None
    if adaptive_events_path:
        try:
            os.makedirs(os.path.dirname(adaptive_events_path) or '.', exist_ok=True)
            adaptive_events_file = open(adaptive_events_path, 'w')
        except Exception as e:
            print(f"[QLEARN] Failed to open adaptive events path {adaptive_events_path}: {e}")

    for ep in range(1, episodes + 1):
        obs = _unwrap_reset(env.reset())
        s = _disc_state(np.asarray(obs, dtype=np.float32), scaler, pca, binner)
        ep_reward = 0.0

        for _ in range(max_steps_per_episode):
            steps_done += 1

            # Guided exploration: follow teacher with prob guided_prob when exploring
            if np.random.rand() < epsilon:
                if np.random.rand() < guided_prob:
                    desired = teacher.get_route(env.current_log)
                    a = {"mysql": 0, "elk": 1, "ipfs": 2}[desired]
                else:
                    a = env.action_space.sample()
            else:
                a = int(np.argmax(Q[s]))

            obs2, r, done, info = _unwrap_step(env.step(a))
            ep_reward += float(r)
            s2 = _disc_state(np.asarray(obs2, dtype=np.float32), scaler, pca, binner)

            # Warm-start unseen states toward teacher's action
            if s2 not in Q:
                Q[s2] = np.zeros(n_actions, dtype=np.float32)
                desired2 = teacher.get_route(env.current_log)
                Q[s2][{"mysql": 0, "elk": 1, "ipfs": 2}[desired2]] = prior_bonus

            best_next = float(np.max(Q[s2]))
            td_target = float(r) + (0.0 if done else gamma * best_next)
            Q[s][a] += alpha * (td_target - float(Q[s][a]))

            s = s2
            if not disable_static_eps_decay:
                epsilon = max(eps_end, epsilon * eps_decay)
            if done:
                break

        rewards_hist.append(ep_reward)

        # Adaptive epsilon logic (after window filled)
        adaptive_allowed = adaptive_eps_window > 0 and ep >= adaptive_eps_min_episodes
        if adaptive_allowed and len(rewards_hist) >= adaptive_eps_window:
            window = rewards_hist[-adaptive_eps_window:]
            avg_recent = float(np.mean(window))
            recent_avgs.append(avg_recent)
            improved = avg_recent > best_recent_avg + 1e-6
            if improved:
                best_recent_avg = avg_recent
                plateau_count = 0
            else:
                plateau_count += 1
                if plateau_count >= adaptive_eps_patience:
                    old_eps = epsilon
                    epsilon = max(eps_end, epsilon * adaptive_eps_drop)
                    if epsilon < old_eps:  # record only if changed
                        event = {"episode": ep, "epsilon": float(epsilon)}
                        adaptive_events.append(event)
                        if adaptive_events_file:
                            try:
                                adaptive_events_file.write(json.dumps(event) + "\n")
                            except Exception as e:
                                print(f"[QLEARN] Failed to write adaptive event: {e}")
                    plateau_count = 0
        else:
            avg_recent = float(np.mean(rewards_hist))

        if ep % max(1, episodes // 10) == 0:
            print(f"[QLEARN] {dataset_name} | Ep {ep}/{episodes}  ep_reward={ep_reward:.2f}  avg_recent={avg_recent:.2f}  eps={epsilon:.3f}")

    print(f"[QLEARN] {dataset_name} done {episodes} episodes in {time.time()-t0:.1f}s; states={len(Q)}")

    # Close adaptive events file if used
    if adaptive_events_file:
        try:
            adaptive_events_file.close()
        except Exception:
            pass

    os.makedirs("trained_models", exist_ok=True)
    q_path = f"{model_path_prefix}_q_table.pkl"
    pca_path = f"{model_path_prefix}_pca.pkl"
    bin_path = f"{model_path_prefix}_binner.pkl"
    scaler_path = f"{model_path_prefix}_scaler.pkl"
    meta_path = f"{model_path_prefix}_metadata.json"
    with open(q_path, "wb") as f:
        pickle.dump(dict(Q), f)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    with open(bin_path, "wb") as f:
        pickle.dump(binner, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Derive embedding dim (assumes first obs shape same as scaler mean length)
    obs_dim = int(warm_obs.shape[1])
    system_state_assumed = 6 if obs_dim > 6 else obs_dim  # fallback for tiny test envs
    embedding_dim = obs_dim - system_state_assumed

    # Decide whether to embed full reward history
    episode_rewards_embedded = []
    if episodes <= reward_history_inline_threshold:
        episode_rewards_embedded = [float(x) for x in rewards_hist]
    # Optionally persist large reward histories to CSV
    if episodes > reward_history_inline_threshold and reward_history_csv_path:
        try:
            os.makedirs(os.path.dirname(reward_history_csv_path) or '.', exist_ok=True)
            with open(reward_history_csv_path, 'w') as f:
                f.write('episode,reward\n')
                for i, r in enumerate(rewards_hist, start=1):
                    f.write(f"{i},{r}\n")
            print(f"[QLEARN] Wrote reward history CSV to {reward_history_csv_path}")
        except Exception as e:
            print(f"[QLEARN] Failed to write reward history CSV: {e}")

    metadata = {
        "version": 3,
        "timestamp": time.time(),
        "obs_dim": obs_dim,
        "embedding_dim": int(max(0, embedding_dim)),
        "pca_components": int(pca.n_components_),
        "n_bins": int(n_bins),
        "episodes": int(episodes),
        "alpha": alpha,
        "gamma": gamma,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay": eps_decay,
        "adaptive_eps_window": adaptive_eps_window,
        "adaptive_eps_patience": adaptive_eps_patience,
        "adaptive_eps_drop": adaptive_eps_drop,
        "adaptive_eps_min_episodes": adaptive_eps_min_episodes,
        "disable_static_eps_decay": disable_static_eps_decay,
        "final_epsilon": float(epsilon),
        "best_recent_avg": float(best_recent_avg if best_recent_avg != -1e9 else (np.mean(rewards_hist) if rewards_hist else 0.0)),
        "adaptive_events": adaptive_events,
        "guided_prob": guided_prob,
        "prior_bonus": prior_bonus,
        "dataset_name": dataset_name,
        "sample_mode": sample_mode,
        # Reward distribution stats
        "reward_mean": float(np.mean(rewards_hist) if rewards_hist else 0.0),
        "reward_median": float(np.median(rewards_hist) if rewards_hist else 0.0),
        "reward_std": float(np.std(rewards_hist) if rewards_hist else 0.0),
        "episode_rewards": episode_rewards_embedded,
        "reward_history_inline_threshold": reward_history_inline_threshold,
    }
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[QLEARN] Failed to write metadata: {e}")
    print(f"[QLEARN] Saved: {q_path}")
    print(f"[QLEARN] Saved: {pca_path}")
    print(f"[QLEARN] Saved: {bin_path}")
    print(f"[QLEARN] Saved: {scaler_path}")
    print(f"[QLEARN] Saved: {meta_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Train a tabular Q-learning router.")
    ap.add_argument("--log_filepath", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--max_steps_per_episode", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay", type=float, default=0.997)
    ap.add_argument("--adaptive_eps_window", type=int, default=20, help="Episodes window for plateau detection (0 disables adaptive)")
    ap.add_argument("--adaptive_eps_patience", type=int, default=3, help="Consecutive non-improving windows before extra decay")
    ap.add_argument("--adaptive_eps_drop", type=float, default=0.7, help="Multiplicative factor applied to epsilon on plateau event")
    ap.add_argument("--adaptive_eps_min_episodes", type=int, default=10, help="Minimum episodes before adaptive epsilon engages")
    ap.add_argument("--no_static_eps_decay", action="store_true", help="Disable the baseline geometric epsilon decay (use adaptive only)")
    ap.add_argument("--reward_history_inline_threshold", type=int, default=120, help="If total episodes <= threshold, embed reward list in metadata; else skip.")
    ap.add_argument("--reward_history_csv_path", type=str, default=None, help="Optional path to write full reward trajectory CSV when not embedded.")
    ap.add_argument("--adaptive_events_path", type=str, default=None, help="Optional NDJSON path to stream adaptive epsilon events.")
    ap.add_argument("--warmup_steps", type=int, default=3000)
    ap.add_argument("--pca_components", type=int, default=8)
    ap.add_argument("--n_bins", type=int, default=6)
    ap.add_argument("--model_path_prefix", type=str, default="trained_models/q_learning")
    ap.add_argument("--dataset_name", type=str, default=None, help="Pretty name for logs.")
    ap.add_argument("--guided_prob", type=float, default=0.5, help="Prob of teacher action during exploration")
    ap.add_argument("--prior_bonus", type=float, default=0.3, help="Warm-start Q for new states toward teacher")
    ap.add_argument("--sample_mode", choices=["head", "random", "balanced"], default="head")  # <— NEW
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset_name or os.path.splitext(os.path.basename(args.log_filepath))[0]
    train_q_learning(
        dataset_name=dataset_name,
        log_filepath=args.log_filepath,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
    adaptive_eps_window=args.adaptive_eps_window,
    adaptive_eps_patience=args.adaptive_eps_patience,
    adaptive_eps_drop=args.adaptive_eps_drop,
        disable_static_eps_decay=args.no_static_eps_decay,
        adaptive_eps_min_episodes=args.adaptive_eps_min_episodes,
        warmup_steps=args.warmup_steps,
        pca_components=args.pca_components,
        n_bins=args.n_bins,
        model_path_prefix=args.model_path_prefix,
        guided_prob=args.guided_prob,
        prior_bonus=args.prior_bonus,
        reward_history_inline_threshold=args.reward_history_inline_threshold,
        reward_history_csv_path=args.reward_history_csv_path,
        adaptive_events_path=args.adaptive_events_path,
        sample_mode=args.sample_mode, 
    )
