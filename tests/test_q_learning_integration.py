import os, json, pickle
import numpy as np
import types

from src.train_qlearning import train_q_learning
from src.routers import QLearningRouter

# We create a minimal fake environment by monkeypatching LogRoutingEnv via import hooks
# Simpler: monkeypatch feature extractor & system state to shrink observation dimension.

class DummyFE:
    def get_embedding(self, content):
        # 3-dim embedding
        return np.array([0.1, -0.2, 0.05], dtype=np.float32)

# Monkeypatch targets inside modules will be done inside test function

def test_q_learning_end_to_end(tmp_path, monkeypatch):
    # Create a tiny CSV log file
    csv_path = tmp_path / 'logs.csv'
    with open(csv_path, 'w') as f:
        f.write('LineId,Time,EventId,Level,EventTemplate,Content,Component,Date,Node,LogSource\n')
        for i in range(50):
            f.write(f"{i},0,E{i},INFO,Template {i},Content {i},comp,2025-01-01,node,source\n")

    # Patch feature extractor & system state
    monkeypatch.setattr('src.feature_extractor.LogFeatureExtractor', lambda : DummyFE())
    monkeypatch.setattr('src.routers.LogFeatureExtractor', lambda : DummyFE())
    monkeypatch.setattr('src.rl_environment.LogFeatureExtractor', lambda : DummyFE())
    monkeypatch.setattr('src.routers.get_system_state', lambda : np.array([0.0,0.0,0.0], dtype=np.float32))
    monkeypatch.setattr('src.rl_environment.get_system_state', lambda : np.array([0.0,0.0,0.0], dtype=np.float32))

    # Train quickly with very small settings
    prefix = str(tmp_path / 'quick')
    train_q_learning(
        dataset_name='test',
        log_filepath=str(csv_path),
        episodes=5,
        max_steps_per_episode=20,
        alpha=0.2,
        gamma=0.9,
        eps_start=0.9,
        eps_end=0.1,
        eps_decay=0.95,
        adaptive_eps_window=2,
        adaptive_eps_patience=1,
        adaptive_eps_drop=0.5,
        disable_static_eps_decay=False,
        adaptive_eps_min_episodes=1,
        warmup_steps=10,
        pca_components=2,
        n_bins=3,
        model_path_prefix=prefix,
        guided_prob=0.4,
        prior_bonus=0.1,
        reward_history_inline_threshold=50,
        sample_mode='head'
    )

    # Check artifacts
    for suf in ["_q_table.pkl","_pca.pkl","_binner.pkl","_scaler.pkl","_metadata.json"]:
        assert os.path.isfile(prefix + suf)

    with open(prefix + '_metadata.json','r') as f:
        meta = json.load(f)
    # Basic keys
    for k in ["version","obs_dim","pca_components","reward_mean","adaptive_events"]:
        assert k in meta
    assert meta['version'] >= 3
    # Reward history embedded because episodes=5 <= threshold=50
    assert 'episode_rewards' in meta and len(meta['episode_rewards']) == 5

    # Router load smoke test
    r = QLearningRouter(model_path_prefix=prefix)
    backend = r.get_route({'Content':'hello'})
    assert backend in {'mysql','elk','ipfs'}
