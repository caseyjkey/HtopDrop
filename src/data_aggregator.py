#!/usr/bin/env python3
"""
Data Aggregator

Coordinates all data collectors and provides unified data stream.
"""

import threading
import time
from typing import Dict, Callable, Optional
from collections import defaultdict

from .collectors import HtopCollector, NvidiaCollector, AudioCollector


class DataAggregator:
    """Aggregates data from all collectors into unified stream."""

    def __init__(self, audio_device: Optional[int] = None):
        """
        Initialize all collectors.

        Args:
            audio_device: Audio input device index (None = default)
        """
        # Collectors
        self.htop = HtopCollector(update_interval=0.1)
        self.nvidia = NvidiaCollector(update_interval=0.5)
        self.audio = AudioCollector(update_interval=0.05, device=audio_device)

        # Shared data store (thread-safe via GIL for reads/writes of dicts)
        self.current_data: Dict[str, float] = defaultdict(float)
        self.lock = threading.Lock()

        # Threads
        self.threads = []
        self.running = False

    def _htop_callback(self, stats: Dict[str, float]):
        """Callback for htop stats."""
        with self.lock:
            self.current_data.update(stats)

    def _nvidia_callback(self, stats: Dict[str, float]):
        """Callback for nvidia stats."""
        with self.lock:
            self.current_data.update(stats)

    def _audio_callback(self, stats: Dict[str, float]):
        """Callback for audio stats."""
        with self.lock:
            self.current_data.update(stats)

    def start(self):
        """Start all collectors in background threads."""
        if self.running:
            return

        self.running = True

        print("\n" + "="*60)
        print("HtopDrop Data Aggregator Starting")
        print("="*60)

        # Start htop collector
        htop_thread = threading.Thread(
            target=self.htop.stream_stats,
            args=(self._htop_callback,),
            daemon=True
        )
        htop_thread.start()
        self.threads.append(htop_thread)

        # Start nvidia collector
        nvidia_thread = threading.Thread(
            target=self.nvidia.stream_stats,
            args=(self._nvidia_callback,),
            daemon=True
        )
        nvidia_thread.start()
        self.threads.append(nvidia_thread)

        # Start audio collector
        audio_thread = threading.Thread(
            target=self.audio.stream_stats,
            args=(self._audio_callback,),
            daemon=True
        )
        audio_thread.start()
        self.threads.append(audio_thread)

        # Give threads time to initialize
        time.sleep(0.5)

        print("\n✓ All collectors started")
        print("  - htop (CPU, memory, swap)")
        print("  - nvidia-smi (GPU temp, memory)")
        print("  - microphone (audio FFT)")
        print()

    def stop(self):
        """Stop all collectors."""
        self.running = False
        self.audio.stop_stream()

    def get_data(self) -> Dict[str, float]:
        """
        Get current aggregated data.

        Returns:
            Dict with all Q variables and audio values
        """
        with self.lock:
            return dict(self.current_data)

    def get_formatted_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get data organized by source.

        Returns:
            Dict with keys: 'htop', 'nvidia', 'audio'
        """
        data = self.get_data()

        return {
            'htop': {
                'cpu_avg': data.get('q1', 0),
                'memory': data.get('q2', 0),
                'swap': data.get('q3', 0),
                'core0': data.get('q4', 0),
                'core1': data.get('q5', 0),
                'core2': data.get('q6', 0),
                'core3': data.get('q7', 0),
                'load_avg': data.get('q8', 0),
            },
            'nvidia': {
                'temperature': data.get('q9', 0),
                'memory_used_mb': data.get('q10', 0),
                'memory_total_mb': data.get('q11', 0),
                'memory_percent': data.get('q12', 0),
            },
            'audio': {
                'bass': data.get('bass', 0),
                'mid': data.get('mid', 0),
                'treb': data.get('treb', 0),
                'band0': data.get('q17', 0),
                'band1': data.get('q18', 0),
                'band2': data.get('q19', 0),
                'band3': data.get('q20', 0),
                'band4': data.get('q21', 0),
                'band5': data.get('q22', 0),
            }
        }

    def print_status(self):
        """Print current data status (for debugging)."""
        data = self.get_formatted_data()

        print("\033[H\033[J", end='')  # Clear screen
        print("="*60)
        print("HtopDrop - Live Data")
        print("="*60)

        print("\n[HTOP - CPU & Memory]")
        print(f"  CPU Average:  {data['htop']['cpu_avg']:6.1f}%  {'█' * int(data['htop']['cpu_avg'] / 5)}")
        print(f"  Memory:       {data['htop']['memory']:6.1f}%  {'█' * int(data['htop']['memory'] / 5)}")
        print(f"  Swap:         {data['htop']['swap']:6.1f}%  {'█' * int(data['htop']['swap'] / 5)}")
        print(f"  Load (1min):  {data['htop']['load_avg']:6.1f}%")

        cores = [data['htop'][f'core{i}'] for i in range(4) if data['htop'][f'core{i}'] > 0]
        if cores:
            print(f"  Cores: {', '.join(f'{c:.1f}%' for c in cores)}")

        print("\n[NVIDIA - GPU Stats]")
        print(f"  Temperature:  {data['nvidia']['temperature']:6.1f}°C")
        print(f"  Memory:       {data['nvidia']['memory_percent']:6.1f}%  ({data['nvidia']['memory_used_mb']:.0f}/{data['nvidia']['memory_total_mb']:.0f} MB)")

        print("\n[AUDIO - Microphone]")
        print(f"  Bass:  {data['audio']['bass']:6.1f}  {'█' * int(data['audio']['bass'] / 5)}")
        print(f"  Mid:   {data['audio']['mid']:6.1f}  {'█' * int(data['audio']['mid'] / 5)}")
        print(f"  Treb:  {data['audio']['treb']:6.1f}  {'█' * int(data['audio']['treb'] / 5)}")

        print("\n(Ctrl+C to stop)")


def main():
    """Test the aggregator."""
    aggregator = DataAggregator()
    aggregator.start()

    try:
        while True:
            aggregator.print_status()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        aggregator.stop()


if __name__ == '__main__':
    main()
