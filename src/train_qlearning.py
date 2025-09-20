import argparse
import os
import pickle
import time
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
    obs_dim = observations.shape[1]
    n_comp = min(int(pca_components), int(obs_dim))
    if n_comp <= 0:
        raise ValueError(f"Invalid PCA components={pca_components} for obs_dim={obs_dim}")
    pca = PCA(n_components=n_comp, random_state=0)
    reduced = pca.fit_transform(observations)
    binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    binner.fit(reduced)
    return pca, binner


def _disc_state(obs: np.ndarray, pca: PCA, binner: KBinsDiscretizer) -> tuple:
    reduced = pca.transform(obs.reshape(1, -1))
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
    warmup_steps: int,
    pca_components: int,
    n_bins: int,
    model_path_prefix: str,
    guided_prob: float,
    prior_bonus: float,
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

    pca, binner = _fit_state_transformers(warm_obs, pca_components=pca_components, n_bins=n_bins)
    print(f"[QLEARN] PCA comps={pca.n_components_}, KBins bins={n_bins}")

    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    epsilon = eps_start
    rewards_hist = []
    steps_done = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        obs = _unwrap_reset(env.reset())
        s = _disc_state(np.asarray(obs, dtype=np.float32), pca, binner)
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
            s2 = _disc_state(np.asarray(obs2, dtype=np.float32), pca, binner)

            # Warm-start unseen states toward teacher's action
            if s2 not in Q:
                Q[s2] = np.zeros(n_actions, dtype=np.float32)
                desired2 = teacher.get_route(env.current_log)
                Q[s2][{"mysql": 0, "elk": 1, "ipfs": 2}[desired2]] = prior_bonus

            best_next = float(np.max(Q[s2]))
            td_target = float(r) + (0.0 if done else gamma * best_next)
            Q[s][a] += alpha * (td_target - float(Q[s][a]))

            s = s2
            epsilon = max(eps_end, epsilon * eps_decay)
            if done:
                break

        rewards_hist.append(ep_reward)
        if ep % max(1, episodes // 10) == 0:
            avg_recent = float(np.mean(rewards_hist[-max(1, episodes // 10):]))
            print(f"[QLEARN] {dataset_name} | Ep {ep}/{episodes}  ep_reward={ep_reward:.2f}  avg_recent={avg_recent:.2f}  eps={epsilon:.3f}")

    print(f"[QLEARN] {dataset_name} done {episodes} episodes in {time.time()-t0:.1f}s; states={len(Q)}")

    os.makedirs("trained_models", exist_ok=True)
    q_path = f"{model_path_prefix}_q_table.pkl"
    pca_path = f"{model_path_prefix}_pca.pkl"
    bin_path = f"{model_path_prefix}_binner.pkl"
    with open(q_path, "wb") as f:
        pickle.dump(dict(Q), f)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    with open(bin_path, "wb") as f:
        pickle.dump(binner, f)
    print(f"[QLEARN] Saved: {q_path}")
    print(f"[QLEARN] Saved: {pca_path}")
    print(f"[QLEARN] Saved: {bin_path}")


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
        warmup_steps=args.warmup_steps,
        pca_components=args.pca_components,
        n_bins=args.n_bins,
        model_path_prefix=args.model_path_prefix,
        guided_prob=args.guided_prob,
        prior_bonus=args.prior_bonus,
        sample_mode=args.sample_mode, 
    )
