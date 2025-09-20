import os
import json
import pickle
import numpy as np
import pytest
import importlib.util

# Skip if heavy dependency (torch) not installed
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed; skipping A2C router test", allow_module_level=True)

from src.routers import A2CRouter

# We will mock A2C.load to avoid requiring a real trained model.
class DummyExtractor:
    def __init__(self):
        from torch import nn
        self.policy_net = [nn.Linear(10, 5)]

class DummyPolicy:
    def __init__(self):
        self.mlp_extractor = DummyExtractor()

class DummyModel:
    def __init__(self):
        self._action = 0
        self.policy = DummyPolicy()
    def predict(self, obs, deterministic=True):
        # simple heuristic: switch action based on sum sign to exercise path
        if float(np.sum(obs)) > 0:
            return 1, None
        return 0, None

# Patch stable_baselines3.A2C.load within test scope
def test_a2c_router_scaler(monkeypatch, tmp_path):
    base = tmp_path / "a2c_model"
    base.write_text("placeholder")  # ensure path exists

    def fake_load(path):
        return DummyModel()

    monkeypatch.setattr("src.routers.A2C.load", fake_load)

    # create scaler + metadata artifacts
    scaler = {"mean": np.zeros(774, dtype=np.float32), "std": np.ones(774, dtype=np.float32)}
    with open(str(base) + "_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    metadata = {"version": 1}
    with open(str(base) + "_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    router = A2CRouter(str(base))
    # build fake log shorter than embedding; router relies on feature extractor which we bypass by monkeypatch
    def fake_get_embedding(text):
        return np.zeros(768, dtype=np.float32)
    monkeypatch.setattr(router.feature_extractor, "get_embedding", fake_get_embedding)

    # monkeypatch system state
    from src import routers as routers_mod
    def fake_system_state():
        return np.zeros(6, dtype=np.float32)
    monkeypatch.setattr(routers_mod, "get_system_state", fake_system_state)

    # route
    log = {"Content": "test"}
    dest = router.get_route(log)
    assert dest in {"mysql", "elk", "ipfs"}

