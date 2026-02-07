#!/usr/bin/env python3
"""
nvidia-smi Data Collector

Collects GPU statistics for old GT540M (limited query support).
Maps data to Q variables for visualization.
"""

import subprocess
import time
import re
from typing import Dict, Optional


class NvidiaCollector:
    """Collects GPU stats via nvidia-smi."""

    def __init__(self, update_interval: float = 0.5):
        """
        Args:
            update_interval: Seconds between nvidia-smi polls (default 500ms)
        """
        self.update_interval = update_interval
        self.nvidia_smi_available = self._check_nvidia_smi()

        if not self.nvidia_smi_available:
            print("WARNING: nvidia-smi not found or not working")

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--version'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_stats(self) -> Dict[str, float]:
        """
        Collect current GPU statistics.

        GT540M supports:
        - Temperature (✓)
        - Memory used/total (✓)
        - GPU utilization (✗ Not Supported)
        - Memory utilization (✗ Not Supported)

        Returns:
            Dict with keys:
            - q9: GPU temperature (°C)
            - q10: GPU memory used (MB)
            - q11: GPU memory total (MB)
            - q12: GPU memory usage %
        """
        stats = {
            'q9': 0.0,   # Temperature
            'q10': 0.0,  # Memory used MB
            'q11': 0.0,  # Memory total MB
            'q12': 0.0,  # Memory %
        }

        if not self.nvidia_smi_available:
            return stats

        try:
            # Query: temperature, memory.used, memory.total
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=temperature.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                # Parse: "41, 87, 964"
                parts = [p.strip() for p in output.split(',')]

                if len(parts) >= 3:
                    mem_used = 0.0
                    mem_total = 0.0

                    # Temperature
                    if parts[0] and parts[0] != '[Not Supported]':
                        stats['q9'] = float(parts[0])

                    # Memory used (MB)
                    if parts[1] and parts[1] != '[Not Supported]':
                        mem_used = float(parts[1])
                        stats['q10'] = mem_used

                    # Memory total (MB)
                    if parts[2] and parts[2] != '[Not Supported]':
                        mem_total = float(parts[2])
                        stats['q11'] = mem_total

                        # Calculate percentage
                        if mem_total > 0 and mem_used > 0:
                            stats['q12'] = (mem_used / mem_total) * 100.0

        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            print(f"NvidiaCollector error: {e}")

        return stats

    def get_detailed_stats(self) -> Optional[Dict]:
        """Get detailed GPU info for debugging."""
        if not self.nvidia_smi_available:
            return None

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,memory.used,memory.total',
                 '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                parts = [p.strip() for p in output.split(',')]

                return {
                    'name': parts[0] if len(parts) > 0 else 'Unknown',
                    'driver': parts[1] if len(parts) > 1 else 'Unknown',
                    'temperature_c': parts[2] if len(parts) > 2 else 'N/A',
                    'memory_used_mb': parts[3] if len(parts) > 3 else 'N/A',
                    'memory_total_mb': parts[4] if len(parts) > 4 else 'N/A',
                }
        except (subprocess.TimeoutExpired, IndexError):
            pass

        return None

    def stream_stats(self, callback):
        """
        Continuously stream GPU stats to callback function.

        Args:
            callback: Function that receives stats dict
        """
        if not self.nvidia_smi_available:
            print("NvidiaCollector: nvidia-smi not available, returning zeros")
            # Still call callback with zeros for compatibility
            while True:
                callback(self.get_stats())
                time.sleep(self.update_interval)
            return

        print(f"NvidiaCollector: Streaming stats every {self.update_interval}s")

        try:
            while True:
                stats = self.get_stats()
                callback(stats)
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\nNvidiaCollector: Stopped")


def main():
    """Test the collector."""
    collector = NvidiaCollector(update_interval=1.0)

    # Print detailed info once
    detailed = collector.get_detailed_stats()
    if detailed:
        print("\nGPU Info:")
        print(f"  Name:    {detailed['name']}")
        print(f"  Driver:  {detailed['driver']}")
        print()

    def print_stats(stats):
        print("\n" + "="*50)
        print("GPU Stats:")
        print(f"  Temperature:  {stats.get('q9', 0):.1f}°C")
        print(f"  Memory Used:  {stats.get('q10', 0):.0f} MB")
        print(f"  Memory Total: {stats.get('q11', 0):.0f} MB")
        print(f"  Memory %:     {stats.get('q12', 0):.1f}%")

    collector.stream_stats(print_stats)


if __name__ == '__main__':
    main()
