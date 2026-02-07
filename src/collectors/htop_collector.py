#!/usr/bin/env python3
"""
htop Data Collector

Collects CPU, memory, and swap statistics using psutil.
Maps data to Q variables for visualization.
"""

import psutil
import time
from typing import Dict, List


class HtopCollector:
    """Collects system stats similar to htop."""

    def __init__(self, update_interval: float = 0.1):
        """
        Args:
            update_interval: Seconds between updates (default 100ms for smooth visuals)
        """
        self.update_interval = update_interval
        self.cpu_count = psutil.cpu_count()

    def get_stats(self) -> Dict[str, float]:
        """
        Collect current system statistics.

        Returns:
            Dict with keys matching Q variable assignments:
            - q1: CPU average usage %
            - q2: Memory usage %
            - q3: Swap usage %
            - q4-q7: Individual core usage (up to 4 cores shown)
            - q8: Load average (1 min)
        """
        stats = {}

        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)

        stats['q1'] = cpu_percent  # Average CPU usage

        # Individual cores (limit to 4 for Q variables q4-q7)
        for i, core_usage in enumerate(cpu_per_core[:4]):
            stats[f'q{4+i}'] = core_usage

        # Memory stats
        mem = psutil.virtual_memory()
        stats['q2'] = mem.percent  # Memory usage %

        # Swap stats
        swap = psutil.swap_memory()
        stats['q3'] = swap.percent  # Swap usage %

        # Load average (1 min)
        load_avg = psutil.getloadavg()[0]  # 1-minute load average
        # Normalize to percentage (divide by CPU count)
        cpu_count = self.cpu_count if self.cpu_count else 1
        stats['q8'] = (load_avg / cpu_count) * 100.0

        return stats

    def get_detailed_stats(self) -> Dict:
        """
        Get detailed stats for debugging/logging.
        """
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        load_avg = psutil.getloadavg()

        return {
            'cpu': {
                'average': psutil.cpu_percent(interval=None),
                'per_core': cpu_per_core,
                'count': self.cpu_count,
            },
            'memory': {
                'total_gb': mem.total / (1024**3),
                'used_gb': mem.used / (1024**3),
                'percent': mem.percent,
            },
            'swap': {
                'total_gb': swap.total / (1024**3),
                'used_gb': swap.used / (1024**3),
                'percent': swap.percent,
            },
            'load_avg': {
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2],
            }
        }

    def stream_stats(self, callback):
        """
        Continuously stream stats to callback function.

        Args:
            callback: Function that receives stats dict
        """
        print(f"HtopCollector: Streaming stats every {self.update_interval}s")
        print(f"CPU cores detected: {self.cpu_count}")

        # Initial CPU call to initialize internal tracking
        psutil.cpu_percent(interval=None, percpu=True)
        time.sleep(0.1)

        try:
            while True:
                stats = self.get_stats()
                callback(stats)
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\nHtopCollector: Stopped")


def main():
    """Test the collector."""
    collector = HtopCollector(update_interval=1.0)

    def print_stats(stats):
        print("\n" + "="*50)
        print("System Stats:")
        print(f"  CPU Average: {stats.get('q1', 0):.1f}%")
        print(f"  Memory:      {stats.get('q2', 0):.1f}%")
        print(f"  Swap:        {stats.get('q3', 0):.1f}%")
        print(f"  Load (norm): {stats.get('q8', 0):.1f}%")

        # Per-core
        cores = [stats.get(f'q{i}', 0) for i in range(4, 8) if f'q{i}' in stats]
        if cores:
            print(f"  Cores: {', '.join(f'{c:.1f}%' for c in cores)}")

    collector.stream_stats(print_stats)


if __name__ == '__main__':
    main()
