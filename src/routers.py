# src/routers.py
from __future__ import annotations

import math
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
import random
import time

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

    def observe(self, *, log_entry: dict, destination: str, success: bool,
                routing_latency_ms: float, backend_latency_ms: float,
                energy_cpu_pkg_j: float | None = None):
        return None


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
        self._warned_incompatible = False

        q_table_path = f"{model_path_prefix}_q_table.pkl"
        pca_path = f"{model_path_prefix}_pca.pkl"
        binner_path = f"{model_path_prefix}_binner.pkl"
        scaler_path = f"{model_path_prefix}_scaler.pkl"
        meta_path = f"{model_path_prefix}_metadata.json"

        self.scaler = None
        self.metadata = {}

        try:
            with open(q_table_path, "rb") as f:
                self.q_table = pickle.load(f)
            with open(pca_path, "rb") as f:
                self.pca = pickle.load(f)
            with open(binner_path, "rb") as f:
                self.binner = pickle.load(f)
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            except FileNotFoundError:
                self.scaler = None
            try:
                import json
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)
                if self.metadata:
                    ver = self.metadata.get("version")
                    print(f"[QLRouter] Loaded metadata version={ver}")
            except FileNotFoundError:
                self.metadata = {}
        except FileNotFoundError as e:
            print(f"[QLRouter] Warning: {e}. Falling back to default Static routing.")
            self.q_table = {}
            self.pca = None
            self.binner = None
            self.scaler = None

        # Basic compatibility validation
        if self.pca is not None and self.scaler is not None:
            mean = self.scaler.get("mean")
            std = self.scaler.get("std")
            if mean is None or std is None:
                self._invalidate("Scaler missing mean/std")
            else:
                self.obs_dim = int(mean.shape[0])
                meta_dim = self.metadata.get("obs_dim")
                if meta_dim is not None and meta_dim != self.obs_dim:
                    self._invalidate(f"Obs dim mismatch meta={meta_dim} scaler={self.obs_dim}")
                # Forward compatibility notice
                meta_ver = self.metadata.get("version")
                if isinstance(meta_ver, int) and meta_ver > 3:  # supported up to version 3 currently
                    print(f"[QLRouter] Warning: metadata version {meta_ver} > supported 3; attempting best-effort load.")
        elif self.pca is not None and self.scaler is None:
            print("[QLRouter] No scaler found; will attempt raw PCA transform (may degrade accuracy).")

    def _invalidate(self, reason: str):
        print(f"[QLRouter] Incompatible artifacts: {reason}. Falling back to Static policy.")
        self.pca = None
        self.binner = None
        self.scaler = None

    def _apply_scaler(self, vec: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return vec
        mean = self.scaler.get("mean")
        std = self.scaler.get("std")
        if mean is None or std is None:
            if not self._warned_incompatible:
                print("[QLRouter] Scaler corrupt; ignoring.")
                self._warned_incompatible = True
            return vec
        return (vec - mean) / std

    def _discretize_state(self, observation: np.ndarray) -> tuple | None:
        if self.pca is None or self.binner is None:
            return None
        vec = self._apply_scaler(observation.astype(np.float32))
        reduced = self.pca.transform(vec.reshape(1, -1))
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
        # Attempt to load optional scaler & metadata (version 1)
        self.scaler = None
        self.metadata = {}
        import json, pickle, os
        scaler_path = base + "_scaler.pkl"
        meta_path = base + "_metadata.json"
        if os.path.isfile(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    sc = pickle.load(f)
                mean = sc.get('mean'); std = sc.get('std') if isinstance(sc, dict) else (None, None)
                if mean is not None and std is not None:
                    self.scaler = {'mean': np.asarray(mean, dtype=np.float32), 'std': np.asarray(std, dtype=np.float32)}
                    print(f"[A2CRouter] Loaded scaler with dim={self.scaler['mean'].shape[0]}")
            except Exception as e:
                print(f"[A2CRouter] Failed to load scaler: {e}")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
                ver = self.metadata.get('version')
                if ver is not None:
                    print(f"[A2CRouter] Loaded metadata version={ver}")
            except Exception as e:
                print(f"[A2CRouter] Failed to load metadata: {e}")

    def _apply_scaler(self, observation: np.ndarray) -> np.ndarray:
        if not self.scaler:
            return observation
        mean = self.scaler.get('mean'); std = self.scaler.get('std')
        if mean is None or std is None or mean.shape != observation.shape:
            return observation
        return (observation - mean) / std

    def get_route(self, log_entry: dict) -> str:
        system_state = get_system_state()
        log_embedding = self.feature_extractor.get_embedding(log_entry.get("Content", ""))
        observation = np.concatenate([system_state, log_embedding])
        observation = self._apply_scaler(observation.astype(np.float32))

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

# -------------- Content-Based Routing (CBR) --------------
class CBRRouter(BaseRouter):
    """Adaptive content-based router (inspired by Bizarro et al. VLDB'05).

    Simplified adaptation for log routing:
      * Treat candidate log fields as attributes (e.g., Level, Component, LogSource, first token of Content).
      * For each attribute we maintain per-bucket statistics of backend latency (and drop/success proxy).
      * Periodically (every `recompute_interval` decisions) we score attributes to pick a classifier attribute.
      * Routing decision for a log chooses backend with lowest expected latency for that attribute bucket.

    Notes:
      - We approximate selectivity with average backend latency (lower is "drop earlier").
      - Bucketing: hash(attribute_value) mod `num_buckets`.
      - Fallback order: use global backend latency averages; if none, default to StaticRouter.
    """
    def __init__(self, *, num_buckets: int = 24, sample_prob: float = 0.06,
                 warm_samples: int = 150, recompute_interval: int = 200,
                 cost_metric: str = "latency", json_dump_path: str | None = None,
                 json_dump_interval: int = 0, latency_weight: float = 1.0,
                 energy_weight: float = 1000.0, json_dump_mode: str = "overwrite",
                 state_path: str | None = None):
        self.num_buckets = int(num_buckets)
        self.sample_prob = float(sample_prob)
        self.warm_samples = int(warm_samples)
        self.recompute_interval = int(recompute_interval)
        self.cost_metric = cost_metric  # latency | energy | combined
        self.json_dump_path = json_dump_path
        self.json_dump_interval = int(json_dump_interval)
        self.latency_weight = float(latency_weight)
        self.energy_weight = float(energy_weight)
        self.json_dump_mode = json_dump_mode  # overwrite | append | timestamp
        self.state_path = state_path

        self.candidate_attributes = ["Level", "Component", "LogSource"]
        self.static = StaticRouter()
        # stats[attribute][bucket][backend] -> list of latencies
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # global_stats[backend] -> list of latencies
        self.global_stats = defaultdict(list)

        self.samples_collected = 0
        self.decisions = 0
        self.classifier_attr: str | None = None
        self.attr_scores: dict[str, float] = {}
        self.backend_order = ["mysql", "elk", "ipfs"]

        # Cache of per attribute bucket -> expected latency per backend
        self.expected_latency = defaultdict(lambda: defaultdict(lambda: float('inf')))

        self._load_state_if_any()

    def _load_state_if_any(self):
        if not self.state_path:
            return
        import os, json
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, 'r') as f:
                data = json.load(f)
            # reconstruct stats
            stats_in = data.get('stats', {})
            for attr, buckets in stats_in.items():
                for b_str, backends in buckets.items():
                    b = int(b_str)
                    for backend, lst in backends.items():
                        self.stats[attr][b][backend].extend(lst)
            global_in = data.get('global_stats', {})
            for backend, lst in global_in.items():
                self.global_stats[backend].extend(lst)
            self.samples_collected = int(data.get('samples_collected', 0))
            self.decisions = int(data.get('decisions', 0))
            self.classifier_attr = data.get('classifier_attr')
            self.attr_scores = data.get('attr_scores', {})
            self._compute_expected()
            print(f"[CBR] Loaded persisted state from {self.state_path}")
        except Exception as e:
            print(f"[CBR] Failed to load state ({e})")

    def save_state(self):
        if not self.state_path:
            return
        import json, os
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            # convert stats to pure python for JSON
            stats_out = {}
            for attr, buckets in self.stats.items():
                stats_out[attr] = {}
                for b, backend_lat in buckets.items():
                    stats_out[attr][str(b)] = {backend: lst for backend, lst in backend_lat.items()}
            data = {
                'num_buckets': self.num_buckets,
                'stats': stats_out,
                'global_stats': {k: v for k, v in self.global_stats.items()},
                'samples_collected': self.samples_collected,
                'decisions': self.decisions,
                'classifier_attr': self.classifier_attr,
                'attr_scores': self.attr_scores,
            }
            with open(self.state_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[CBR] Failed to save state ({e})")

    def _bucket(self, value: str) -> int:
        return hash(value) % self.num_buckets

    def _record(self, attr: str, bucket: int, backend: str, backend_latency_ms: float, energy_j: float | None):
        if self.cost_metric == "latency":
            cost = backend_latency_ms
        elif self.cost_metric == "energy":
            cost = energy_j if energy_j is not None else backend_latency_ms
        else:  # combined
            e = energy_j if energy_j is not None else 0.0
            cost = self.latency_weight * backend_latency_ms + self.energy_weight * e
        self.stats[attr][bucket][backend].append(cost)
        self.global_stats[backend].append(cost)

    def _compute_expected(self):
        self.expected_latency.clear()
        for attr, buckets in self.stats.items():
            for b, backend_lat in buckets.items():
                for backend, lst in backend_lat.items():
                    if lst:
                        self.expected_latency[attr][(b, backend)] = sum(lst)/len(lst)

    def _score_attributes(self):
        # Simple gain-like score: variance reduction of latency across buckets vs global variance
        global_all = []
        for v in self.global_stats.values():
            global_all.extend(v)
        if len(global_all) < 5:
            return
        gmean = sum(global_all)/len(global_all)
        gvar = sum((x-gmean)**2 for x in global_all)/len(global_all)
        if gvar <= 1e-9:
            return
        scores = {}
        for attr, buckets in self.stats.items():
            # weighted within-bucket variance
            wvar = 0.0
            total = 0
            for b, backend_lat in buckets.items():
                merged = []
                for lat_list in backend_lat.values():
                    merged.extend(lat_list)
                n = len(merged)
                if n < 2:
                    continue
                mean_b = sum(merged)/n
                var_b = sum((x-mean_b)**2 for x in merged)/n
                wvar += n * var_b
                total += n
            if total < 5:
                continue
            wvar /= max(total,1)
            scores[attr] = max(0.0, gvar - wvar) / gvar
        if scores:
            self.attr_scores = scores
            # pick highest scoring attribute
            self.classifier_attr = max(scores.items(), key=lambda x: x[1])[0]
            self._compute_expected()

    def _maybe_dump_json(self):
        if not self.json_dump_path or self.json_dump_interval <= 0:
            return
        if self.decisions % self.json_dump_interval != 0:
            return
        import json, os, time as _time
        snap = {
            "decisions": self.decisions,
            "classifier_attr": self.classifier_attr,
            "attr_scores": self.attr_scores,
            "samples_collected": self.samples_collected,
            "cost_metric": self.cost_metric,
            "latency_weight": self.latency_weight,
            "energy_weight": self.energy_weight,
        }
        try:
            os.makedirs(os.path.dirname(self.json_dump_path), exist_ok=True)
            mode = self.json_dump_mode
            if mode == 'timestamp':
                base, ext = os.path.splitext(self.json_dump_path)
                path = f"{base}_{int(_time.time())}{ext or '.json'}"
                with open(path, 'w') as f:
                    json.dump(snap, f, indent=2)
            elif mode == 'append':
                # append as NDJSON
                with open(self.json_dump_path, 'a') as f:
                    f.write(json.dumps(snap) + "\n")
            else:  # overwrite
                with open(self.json_dump_path, 'w') as f:
                    json.dump(snap, f, indent=2)
        except Exception as e:
            print(f"[CBR] JSON dump failed: {e}")

    def get_route(self, log_entry: dict) -> str:
        self.decisions += 1
        # Phase 1: choose attribute if we haven't yet and enough samples collected
        if (self.classifier_attr is None and self.samples_collected >= self.warm_samples) or \
           (self.decisions % self.recompute_interval == 0):
            self._score_attributes()
        self._maybe_dump_json()

        # If we have a classifier attribute, try to pick best backend for bucket
        if self.classifier_attr:
            val = str(log_entry.get(self.classifier_attr, ""))
            bucket = self._bucket(val)
            # Evaluate expected latency per backend for this bucket
            best_backend = None
            best_lat = float('inf')
            for backend in self.backend_order:
                el = self.expected_latency[self.classifier_attr].get((bucket, backend))
                if el is not None and el < best_lat:
                    best_lat = el
                    best_backend = backend
            if best_backend is not None:
                return best_backend

        # Fallback: if we have global stats choose backend with lowest avg latency
        if self.global_stats:
            best_backend = None
            best_lat = float('inf')
            for backend, lst in self.global_stats.items():
                if not lst:
                    continue
                mean_lat = sum(lst)/len(lst)
                if mean_lat < best_lat:
                    best_lat = mean_lat
                    best_backend = backend
            if best_backend:
                return best_backend

        # Final fallback: static rules
        return self.static.get_route(log_entry)

    def observe(self, *, log_entry: dict, destination: str, success: bool,
                routing_latency_ms: float, backend_latency_ms: float,
                energy_cpu_pkg_j: float | None = None):
        # Sampling: gather stats for a fraction of tuples
        if random.random() > self.sample_prob:
            return
        attr_values = {a: str(log_entry.get(a, "")) for a in self.candidate_attributes}
        for attr, val in attr_values.items():
            bucket = self._bucket(val)
            self._record(attr, bucket, destination, backend_latency_ms, energy_cpu_pkg_j)
        self.samples_collected += 1
