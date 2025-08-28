# src/routers.py
from __future__ import annotations

import math
import pickle
from abc import ABC, abstractmethod

import numpy as np
from stable_baselines3 import A2C

from .feature_extractor import LogFeatureExtractor
from .utils import get_system_state


def _safelower(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v)
    return "" if s.lower() == "nan" else s.lower()


class BaseRouter(ABC):
    @abstractmethod
    def get_route(self, log_entry: dict) -> str:
        """Return one of: 'mysql', 'elk', 'ipfs'."""
        raise NotImplementedError


class StaticRouter(BaseRouter):
    
    def get_route(self, log_entry: dict) -> str:
        log_level = _safelower(log_entry.get("Level"))
        log_source = _safelower(log_entry.get("LogSource"))
        component = _safelower(log_entry.get("Component"))
        content = _safelower(log_entry.get("Content"))

        # High-priority security events -> IPFS
        if log_source == "openssh":
            return "ipfs"
        if component == "kernel":
            return "ipfs"
        if log_level in {"crit", "alert", "emerg"}:
            return "ipfs"

        # Application errors/warnings -> ELK
        if log_level in {"err", "error", "warn"}:
            return "elk"
        if "fail" in content or "denied" in content:
            return "elk"

        # Routine logs -> MySQL
        return "mysql"


class QLearningRouter(BaseRouter):
    """
    Loads a learned Q-table and the PCA + KBins used for state discretization.
    Uses greedy policy on known states; falls back to StaticRouter for unknown.
    """
    def __init__(self, model_path_prefix: str = "trained_models/q_learning"):
        self.log_feature_extractor = LogFeatureExtractor()
        self.backend_map = {0: "mysql", 1: "elk", 2: "ipfs"}
        self.static = StaticRouter()

        q_table_path = f"{model_path_prefix}_q_table.pkl"
        pca_path = f"{model_path_prefix}_pca.pkl"
        binner_path = f"{model_path_prefix}_binner.pkl"

        try:
            with open(q_table_path, "rb") as f:
                self.q_table = pickle.load(f)
            with open(pca_path, "rb") as f:
                self.pca = pickle.load(f)
            with open(binner_path, "rb") as f:
                self.binner = pickle.load(f)
        except FileNotFoundError as e:
            print(f"[QLRouter] Warning: {e}. Falling back to default Static routing.")
            self.q_table = {}
            self.pca = None
            self.binner = None

    def _discretize_state(self, observation: np.ndarray) -> tuple | None:
        if self.pca is None or self.binner is None:
            return None
        reduced = self.pca.transform(observation.reshape(1, -1))
        bins = self.binner.transform(reduced)
        return tuple(int(x) for x in bins[0])

    def get_route(self, log_entry: dict) -> str:
        system_state = get_system_state()
        log_embedding = self.log_feature_extractor.get_embedding(log_entry.get("Content", ""))
        observation = np.concatenate([system_state, log_embedding])

        s = self._discretize_state(observation)
        if s is not None and s in self.q_table:
            action_values = self.q_table[s]
            action = int(np.argmax(action_values))
            return self.backend_map.get(action, "mysql")

        # Fallback for unseen states: use the teacher (Static policy)
        return self.static.get_route(log_entry)


class A2CRouter(BaseRouter):
    """
    Loads an A2C policy saved by Stable-Baselines3.
    Pass either '.../a2c_xxx' or '.../a2c_xxx.zip' (we normalize).
    """
    def __init__(self, model_base_path: str):
        base = model_base_path
        if base.lower().endswith(".zip"):
            base = base[:-4]
        self.model = A2C.load(base)
        self.feature_extractor = LogFeatureExtractor()

    def get_route(self, log_entry: dict) -> str:
        system_state = get_system_state()
        log_embedding = self.feature_extractor.get_embedding(log_entry.get("Content", ""))
        observation = np.concatenate([system_state, log_embedding])

        action, _ = self.model.predict(observation, deterministic=True)
        backend_map = {0: "mysql", 1: "elk", 2: "ipfs"}
        return backend_map.get(int(action), "mysql")


class DirectRouter(BaseRouter):
    """Always route to the specified backend (baseline)."""
    def __init__(self, destination: str):
        d = (destination or "").lower()
        if d not in {"mysql", "elk", "ipfs"}:
            raise ValueError(f"Invalid destination: {destination}")
        self.destination = d

    def get_route(self, log_entry: dict) -> str:
        return self.destination
