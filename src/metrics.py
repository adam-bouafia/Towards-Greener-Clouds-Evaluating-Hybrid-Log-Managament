# src/metrics.py
import os, time, json, re, threading, subprocess, shlex
from dataclasses import dataclass
from typing import Optional, Dict, Deque, Tuple, List
from collections import deque

RAPL_BASE = "/sys/class/powercap"

@dataclass
class EnergySample:
    duration_s: float
    cpu_pkg_j: float     
    psys_j: float        
    gpu_est_j: float      

class _RaplDomain:
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        self.energy_file = os.path.join(path, "energy_uj")

    def read_uj(self) -> Optional[int]:
        try:
            with open(self.energy_file, "r") as f:
                return int(f.read().strip())
        except Exception:
            return None

class GpuPowerSampler:
 
    def __init__(self, sample_ms: int = 200):
        self.sample_ms = int(sample_ms)
        self.available = False
        self._buf: Deque[Tuple[float, float]] = deque(maxlen=512)  # (ts, watts)
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._power_regex = re.compile(r'"power"\s*:\s*([0-9]+(?:\.[0-9]+)?)')

        self._start_proc()

    def _start_proc(self):
        # Needs sudoers rule: %power ALL=(root) NOPASSWD: /usr/bin/intel_gpu_top
        cmd = f"sudo -n /usr/bin/intel_gpu_top -J -s {self.sample_ms}"
        try:
            self._proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception:
            self._proc = None
            self.available = False
            return

        self.available = True
        self._thr = threading.Thread(target=self._reader_loop, daemon=True)
        self._thr.start()

    def _reader_loop(self):
        # The JSON format can be multi-line. We maintain a rolling buffer of lines
        # and try to extract a "power": <float> with a regex. If JSON parsing works, fine.
        assert self._proc and self._proc.stdout
        last_power_w = None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            ts = time.perf_counter()

            # Try a fast regex first
            m = self._power_regex.search(line)
            if m:
                try:
                    last_power_w = float(m.group(1))
                    self._buf.append((ts, last_power_w))
                    continue
                except Exception:
                    pass

            # If it's a JSON object on this line, try parse and find a numeric "power"
            line_stripped = line.strip()
            if line_stripped.startswith("{") and line_stripped.endswith("}"):
                try:
                    obj = json.loads(line_stripped)
                    val = self._find_power_in_json(obj)
                    if isinstance(val, (int, float)):
                        last_power_w = float(val)
                        self._buf.append((ts, last_power_w))
                except Exception:
                    # ignore malformed
                    pass

        # cleanup
        self.available = False

    def _find_power_in_json(self, obj) -> Optional[float]:
        # Walk dict/list recursively looking for a numeric "power" field.
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == "power" and isinstance(v, (int, float)):
                    return float(v)
                rv = self._find_power_in_json(v)
                if rv is not None:
                    return rv
        elif isinstance(obj, list):
            for x in obj:
                rv = self._find_power_in_json(x)
                if rv is not None:
                    return rv
        return None

    def energy_between(self, t0: float, t1: float) -> float:
        """
        Integrate Watts over [t0, t1] using trapezoidal rule on buffered samples.
        Returns Joules. If no samples overlap, returns 0.
        """
        if not self.available or not self._buf:
            return 0.0
        if t1 <= t0:
            return 0.0

        # Build a list of samples within a small margin around the interval
        samples: List[Tuple[float, float]] = []
        margin = 1.5 * (self.sample_ms / 1000.0)
        lo, hi = t0 - margin, t1 + margin

        for ts, w in list(self._buf):
            if ts < lo or ts > hi:
                continue
            samples.append((ts, w))

        if not samples:
            # fall back to last known watt * duration
            _, last_w = self._buf[-1]
            return max(0.0, last_w) * (t1 - t0)

        # Ensure endpoints exist at t0 and t1 (flat extrapolation)
        samples.sort(key=lambda x: x[0])
        if samples[0][0] > t0:
            samples.insert(0, (t0, samples[0][1]))
        if samples[-1][0] < t1:
            samples.append((t1, samples[-1][1]))

        # Clip to [t0, t1] and integrate
        energy_j = 0.0
        prev_t, prev_w = None, None
        for ts, w in samples:
            tt = min(max(ts, t0), t1)
            if prev_t is not None:
                dt = tt - prev_t
                if dt > 0:
                    energy_j += 0.5 * (prev_w + w) * dt
            prev_t, prev_w = tt, w
        return max(0.0, energy_j)

    def stop(self):
        self._stop.set()
        try:
            if self._proc:
                self._proc.terminate()
        except Exception:
            pass
        self.available = False

class EnergyMeter:
    """
    Real energy metering using Intel RAPL sysfs for CPU package.
    If PSYS is exposed, we’ll compute gpu_est = max(0, psys - cpu_pkg).
    Otherwise, if a GpuPowerSampler is provided, we integrate iGPU Watts over the window.
    """
    def __init__(self, gpu_sampler: Optional[GpuPowerSampler] = None):
        self.domains: Dict[str, _RaplDomain] = {}
        self._discover_domains()
        self._start_vals: Dict[str, int] = {}
        self._t0: Optional[float] = None
        self._warned = False
        self._has_psys = False
        self._gpu_sampler = gpu_sampler

    @property
    def has_psys(self) -> bool:
        return self._has_psys

    def _discover_domains(self):
        if not os.path.isdir(RAPL_BASE):
            return

        def add_dom(base_path: str):
            try:
                with open(os.path.join(base_path, "name"), "r") as f:
                    dname = f.read().strip().lower()
                self.domains[dname] = _RaplDomain(base_path, dname)
                if dname == "psys":
                    self._has_psys = True
            except Exception:
                return

            # nested subdomains
            for sub in os.listdir(base_path):
                if ":" not in sub:
                    continue
                spath = os.path.join(base_path, sub)
                try:
                    with open(os.path.join(spath, "name"), "r") as f:
                        d2 = f.read().strip().lower()
                    self.domains[f"{dname}:{d2}"] = _RaplDomain(spath, d2)
                except Exception:
                    pass

        # Add intel-rapl* top-levels
        for name in os.listdir(RAPL_BASE):
            if not name.startswith("intel-rapl"):
                continue
            add_dom(os.path.join(RAPL_BASE, name))

    def _sum_pkg_uj(self, snapshot: Dict[str, int]) -> int:
        total = 0
        for k, v in snapshot.items():
            base = k.split(":")[0]
            # count 'package-*' bases
            if base.startswith("package"):
                total += v
        return total

    def _read_snapshot(self) -> Dict[str, int]:
        snap = {}
        for key, dom in self.domains.items():
            val = dom.read_uj()
            if val is not None:
                snap[key] = val
        return snap

    def start(self):
        self._t0 = time.perf_counter()
        self._start_vals = self._read_snapshot()

    def stop(self) -> EnergySample:
        t1 = time.perf_counter()
        dt = max(0.0, t1 - (self._t0 or t1))

        now = self._read_snapshot()
        if not self._start_vals or not now:
            if not self._warned:
                print("[ENERGY] RAPL not available or unreadable; energy set to 0.")
                self._warned = True
            # If GPU sampler exists, still use it
            gpu_j = 0.0
            if self._gpu_sampler and self._gpu_sampler.available and self._t0:
                gpu_j = self._gpu_sampler.energy_between(self._t0, t1)
            return EnergySample(duration_s=dt, cpu_pkg_j=0.0, psys_j=0.0, gpu_est_j=gpu_j)

        # deltas with wrap protection
        deltas = {}
        for k, v1 in now.items():
            v0 = self._start_vals.get(k)
            if v0 is None:
                continue
            d = v1 - v0
            if d < 0:
                d = 0
            deltas[k] = d

        pkg_uj = self._sum_pkg_uj(deltas)
        psys_uj = 0
        has_psys = False
        for k, d in deltas.items():
            base = k.split(":")[0]
            name = k.split(":")[-1]
            if name == "psys" or base == "psys":
                psys_uj += d
                has_psys = True
        self._has_psys = has_psys

        cpu_pkg_j = max(0.0, pkg_uj / 1e6)
        psys_j = max(0.0, psys_uj / 1e6) if has_psys else 0.0

        if has_psys:
            gpu_est_j = max(0.0, psys_j - cpu_pkg_j)
        elif self._gpu_sampler and self._gpu_sampler.available and self._t0:
            gpu_est_j = max(0.0, self._gpu_sampler.energy_between(self._t0, t1))
        else:
            gpu_est_j = 0.0

        return EnergySample(duration_s=dt, cpu_pkg_j=cpu_pkg_j, psys_j=psys_j, gpu_est_j=gpu_est_j)
