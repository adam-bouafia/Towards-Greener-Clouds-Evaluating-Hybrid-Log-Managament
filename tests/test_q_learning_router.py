import os
import pickle
import json
import numpy as np
import importlib.util, pytest

if importlib.util.find_spec("sklearn") is None:
    pytest.skip("sklearn not installed; skipping q-learning router tests", allow_module_level=True)

from src.routers import QLearningRouter

# Minimal fake artifacts to test scaling + fallback logic

class FakePCA:
    def __init__(self):
        self.n_components_ = 2
    def transform(self, X):
        return X[:, :2]

class FakeBinner:
    def transform(self, X):
        return (X > 0).astype(int)

def make_artifacts(tmp_prefix):
    q_table = { (0,0): np.array([0.1, 0.2, 0.3], dtype=np.float32) }
    scaler = {
        'mean': np.zeros(4, dtype=np.float32),
        'std': np.ones(4, dtype=np.float32),
    }
    meta = {
        'version': 1,
        'obs_dim': 4,
    }
    with open(f"{tmp_prefix}_q_table.pkl", 'wb') as f: pickle.dump(q_table, f)
    with open(f"{tmp_prefix}_pca.pkl", 'wb') as f: pickle.dump(FakePCA(), f)
    with open(f"{tmp_prefix}_binner.pkl", 'wb') as f: pickle.dump(FakeBinner(), f)
    with open(f"{tmp_prefix}_scaler.pkl", 'wb') as f: pickle.dump(scaler, f)
    with open(f"{tmp_prefix}_metadata.json", 'w') as f: json.dump(meta, f)


def test_qlearning_router_scaler_and_metadata(tmp_path, monkeypatch):
    prefix = os.path.join(tmp_path, 'ql_test')
    make_artifacts(prefix)

    # Monkeypatch feature extractor + system state to fixed vector length 4
    class FakeFE:
        def get_embedding(self, content):
            return np.array([0.0, 0.0, 0.0])  # will be combined with system state 1 value -> total 4
    monkeypatch.setattr('src.routers.LogFeatureExtractor', lambda : FakeFE())
    monkeypatch.setattr('src.routers.get_system_state', lambda : np.array([0.0], dtype=np.float32))

    r = QLearningRouter(model_path_prefix=prefix)
    # Provide a log with arbitrary content
    backend = r.get_route({'Content': 'anything'})
    assert backend in {'mysql','elk','ipfs'}


def test_qlearning_router_incompatible_scaler(tmp_path, monkeypatch):
    prefix = os.path.join(tmp_path, 'ql_bad')
    make_artifacts(prefix)
    # Corrupt metadata dimension
    meta_path = f"{prefix}_metadata.json"
    with open(meta_path, 'r') as f: meta = json.load(f)
    meta['obs_dim'] = 999
    with open(meta_path, 'w') as f: json.dump(meta, f)

    class FakeFE:
        def get_embedding(self, content):
            return np.array([0.0,0.0,0.0])
    monkeypatch.setattr('src.routers.LogFeatureExtractor', lambda : FakeFE())
    monkeypatch.setattr('src.routers.get_system_state', lambda : np.array([0.0], dtype=np.float32))

    r = QLearningRouter(model_path_prefix=prefix)
    # Should fallback to static (still returns a valid backend)
    backend = r.get_route({'Content': 'x'})
    assert backend in {'mysql','elk','ipfs'}
