# src/rl_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math

from .feature_extractor import LogFeatureExtractor
from .utils import get_system_state
from .backends import BackendManager
from .config import MYSQL_ROOT_PASSWORD, MYSQL_DATABASE
from .metrics import EnergyMeter
from .routers import StaticRouter

SENSITIVE_LEVELS = {"crit", "alert", "emerg", "error"}
SENSITIVE_KEYWORDS = ("sshd", "kernel", "denied", "fail", "unauthorized", "forbidden")


def _safelower(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v)
    return "" if s.lower() == "nan" else s.lower()


class LogRoutingEnv(gym.Env):
    """
    Observation: [system_state(6) + log_embedding(768)] = 774-dim
    Actions: 0 = MySQL, 1 = ELK, 2 = IPFS
    Reward: policy-aligned + latency shaping + energy penalty
    """
    metadata = {"render.modes": []}

    def __init__(self, log_provider):
        super().__init__()
        self.feature_extractor = LogFeatureExtractor()
        self.backend_manager = BackendManager(type("Config", (object,), {
            "MYSQL_ROOT_PASSWORD": MYSQL_ROOT_PASSWORD,
            "MYSQL_DATABASE": MYSQL_DATABASE
        })())
        self.energy_meter = EnergyMeter()
        self.static_ref = StaticRouter()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 + 768,), dtype=np.float32
        )

        self.log_provider = log_provider
        self.log_iterator = None
        self.current_log = None

    def _get_obs(self):
        system_state = get_system_state()
        content = (self.current_log.get("Content") or "")
        log_embedding = self.feature_extractor.get_embedding(content)
        return np.concatenate([system_state, log_embedding]).astype(np.float32)

    def _is_sensitive(self, log: dict) -> bool:
        lvl = _safelower(log.get("Level", ""))
        content = _safelower(log.get("Content", ""))
        component = _safelower(log.get("Component", ""))
        return (
            lvl in SENSITIVE_LEVELS
            or component == "kernel"
            or any(k in content for k in SENSITIVE_KEYWORDS)
        )

    def _desired_destination(self, log: dict) -> str:
        return self.static_ref.get_route(log)

    def _reward(self, *, latency_ms: float, energy_cpu_j: float,
                success: bool, dest: str, sensitive: bool, desired: str) -> float:
        """
        Tuned:
          - policy: +1.2 if dest == desired, else -0.8
          - sensitive: if sensitive and not IPFS -> -0.8; if not sensitive and IPFS -> -0.2
          - latency: weight 0.3  (r = 0..1 over 0..1000ms)
          - energy: penalty = 0.05 * energy_j
          - failure: -1.0 minus latency/energy terms
        """
        if not success:
            return float(-1.0 - 0.3 * min(latency_ms / 1000.0, 1.0) - 0.05 * max(0.0, energy_cpu_j))

        r = 1.2 if dest == desired else -0.8
        if sensitive and dest != "ipfs":
            r -= 0.8
        if (not sensitive) and dest == "ipfs":
            r -= 0.2

        r += 0.3 * (1.0 - min(latency_ms / 1000.0, 1.0))
        r -= 0.05 * max(0.0, energy_cpu_j)

        return float(np.clip(r, -2.0, 2.0))

    def step(self, action):
        dest = "invalid"
        success = True
        latency_ms = 1000.0
        energy_cpu_j = 0.0

        if action == 0:
            dest = "mysql"
        elif action == 1:
            dest = "elk"
        elif action == 2:
            dest = "ipfs"

        try:
            t0 = time.perf_counter()
            self.energy_meter.start()
            if dest == "mysql":
                success = bool(self.backend_manager.write_to_mysql(self.current_log))
            elif dest == "elk":
                success = bool(self.backend_manager.write_to_elk(self.current_log))
            elif dest == "ipfs":
                cid = self.backend_manager.write_to_ipfs(self.current_log)
                success = bool(cid)
            else:
                success = False
            e = self.energy_meter.stop()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if e is not None:
                energy_cpu_j = float(getattr(e, "cpu_pkg_j", 0.0))
        except Exception as ex:
            print(f"[ENV] Error during backend write to {dest}: {ex}")
            success = False
            latency_ms = 1000.0
            energy_cpu_j = 0.0

        sensitive = self._is_sensitive(self.current_log)
        desired = self._desired_destination(self.current_log)
        reward = self._reward(
            latency_ms=latency_ms,
            energy_cpu_j=energy_cpu_j,
            success=success,
            dest=dest,
            sensitive=sensitive,
            desired=desired,
        )

        try:
            self.current_log = next(self.log_iterator)
            terminated = False
        except StopIteration:
            terminated = True
            self.current_log = {
                "LineId": "0", "Time": "0", "EventId": "0", "Level": "INFO",
                "EventTemplate": "End of stream.", "Content": "End of stream.",
                "Component": "system", "Date": "2025-01-01", "Node": "localhost",
            }

        observation = self._get_obs()
        truncated = False
        info = {
            "latency_ms": latency_ms,
            "success": success,
            "destination": dest,
            "sensitive": sensitive,
            "desired": desired,
            "energy_cpu_pkg_j": energy_cpu_j,
            "reward": reward,
        }
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.log_iterator = self.log_provider.get_log_stream()
        try:
            self.current_log = next(self.log_iterator)
        except StopIteration:
            print("[ENV] Log stream is empty at reset.")
            self.current_log = {
                "LineId": "0", "Time": "0", "EventId": "0", "Level": "INFO",
                "EventTemplate": "Empty log stream.", "Content": "Empty log stream.",
                "Component": "system", "Date": "2025-01-01", "Node": "localhost",
            }
        observation = self._get_obs()
        info = {}
        return observation, info

    def render(self): pass
    def close(self): self.backend_manager.close_connections()
