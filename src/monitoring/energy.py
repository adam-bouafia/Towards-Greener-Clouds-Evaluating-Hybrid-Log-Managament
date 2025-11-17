"""
Energy monitoring using Intel RAPL (Running Average Power Limit).

Measures CPU energy consumption for log processing operations.
"""

import time
from pathlib import Path
from typing import Optional


class EnergyMonitor:
    """
    Monitor CPU energy consumption using Intel RAPL.
    
    RAPL provides energy counters for:
    - package: Entire CPU package (cores + uncore)
    - cores: All CPU cores
    - dram: Memory energy (if available)
    
    Energy is measured in microjoules, converted to joules for reporting.
    """
    
    RAPL_PATH = Path("/sys/class/powercap/intel-rapl")
    
    def __init__(self):
        """Initialize energy monitor."""
        self.available = self._check_rapl_available()
        self.package_path = None
        
        if self.available:
            # Find package-0 (main CPU package)
            for item in self.RAPL_PATH.iterdir():
                if item.is_dir() and item.name.startswith("intel-rapl:"):
                    name_file = item / "name"
                    if name_file.exists():
                        try:
                            with open(name_file, "r") as f:
                                if "package-0" in f.read():
                                    self.package_path = item / "energy_uj"
                                    break
                        except (IOError, ValueError):
                            continue
        
        if self.available and self.package_path:
            print(f"✅ EnergyMonitor initialized (RAPL available at {self.package_path})")
        else:
            print(f"⚠️  EnergyMonitor initialized (RAPL not available, will return 0)")
    
    def _check_rapl_available(self) -> bool:
        """
        Check if RAPL is available.
        
        Returns:
            True if RAPL sysfs is readable
        """
        return self.RAPL_PATH.exists() and self.RAPL_PATH.is_dir()
    
    def _read_energy(self) -> Optional[float]:
        """
        Read current energy counter in microjoules.
        
        Returns:
            Energy in microjoules, or None if not available
        """
        if not self.available or not self.package_path:
            return None
        
        try:
            with open(self.package_path, "r") as f:
                return float(f.read().strip())
        except (IOError, ValueError):
            return None
    
    def start_measurement(self) -> float:
        """
        Start energy measurement.
        
        Returns:
            Starting timestamp (for duration calculation)
        """
        self.start_energy = self._read_energy()
        return time.time()
    
    def end_measurement(self) -> float:
        """
        End energy measurement and calculate consumption.
        
        Returns:
            Energy consumed in joules (0 if RAPL not available)
        """
        end_energy = self._read_energy()
        
        if self.start_energy is None or end_energy is None:
            return 0.0
        
        # Handle counter wraparound (counter is 32-bit or 64-bit)
        energy_uj = end_energy - self.start_energy
        if energy_uj < 0:
            # Counter wrapped around, estimate max value
            max_energy = 2**32  # Conservative estimate
            energy_uj += max_energy
        
        # Convert microjoules to joules
        return energy_uj / 1_000_000
    
    def measure(self, func, *args, **kwargs):
        """
        Measure energy consumption of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Tuple of (result, energy_joules, duration_seconds)
        """
        start_time = self.start_measurement()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        energy = self.end_measurement()
        
        return result, energy, duration
