#!/usr/bin/env python3
"""
Audio Collector - Microphone Input

Captures microphone audio and performs FFT analysis.
Provides bass/mid/treb values like MilkDrop's audio analysis.
"""

import numpy as np
import sounddevice as sd
import time
from typing import Dict, Optional, Callable
from collections import deque


class AudioCollector:
    """Captures microphone audio and analyzes frequency spectrum."""

    # FFT parameters (matching MilkDrop's approach)
    SAMPLE_RATE = 44100
    CHUNK_SIZE = 1024  # Samples per chunk
    FFT_SIZE = 512     # FFT bins

    # Frequency band definitions (Hz)
    BASS_MAX = 250
    MID_MAX = 2000
    TREB_MIN = 2000

    def __init__(self, update_interval: float = 0.05, device: Optional[int] = None):
        """
        Args:
            update_interval: Target time between analysis updates (default 50ms = 20fps)
            device: Audio input device index (None = default)
        """
        self.update_interval = update_interval
        self.device = device

        # Ring buffer for incoming audio
        self.buffer_size = self.CHUNK_SIZE * 4
        self.audio_buffer = deque(maxlen=self.buffer_size)

        # Smoothing for visual stability
        self.bass_smooth = 0.0
        self.mid_smooth = 0.0
        self.treb_smooth = 0.0
        self.smoothing_factor = 0.3  # Higher = more responsive, lower = smoother

        # Audio stream
        self.stream = None

        # Check available devices
        self._list_devices()

    def _list_devices(self):
        """List available audio input devices."""
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                default = " (DEFAULT)" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{default}")
        print()

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream (runs in separate thread)."""
        if status:
            print(f"Audio status: {status}")

        # Add mono audio to buffer (average channels if stereo)
        mono = np.mean(indata, axis=1)
        self.audio_buffer.extend(mono)

    def start_stream(self):
        """Start capturing audio from microphone."""
        if self.stream is not None:
            return

        print(f"AudioCollector: Starting microphone capture at {self.SAMPLE_RATE}Hz")

        try:
            # Check if device supports input
            if self.device is not None:
                dev_info = sd.query_devices(self.device)
                if dev_info['max_input_channels'] == 0:
                    print(f"WARNING: Device {self.device} has no input channels (output-only)")
                    print("AudioCollector: Running in SILENT mode (no microphone)")
                    return

            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.SAMPLE_RATE,
                blocksize=self.CHUNK_SIZE,
                callback=self._audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"WARNING: Audio stream failed to start: {e}")
            print("AudioCollector: Running in SILENT mode (no microphone)")

    def stop_stream(self):
        """Stop audio capture."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def analyze_audio(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Perform FFT and extract bass/mid/treb values.

        Args:
            audio_data: Audio samples (mono)

        Returns:
            Dict with frequency band energies
        """
        # Apply window to reduce spectral leakage
        windowed = audio_data * np.hanning(len(audio_data))

        # FFT
        fft = np.fft.rfft(windowed, n=self.FFT_SIZE)
        magnitude = np.abs(fft)

        # Frequency bins (Hz per bin)
        freq_bins = np.fft.rfftfreq(self.FFT_SIZE, 1.0 / self.SAMPLE_RATE)

        # Extract frequency bands
        bass_mask = freq_bins < self.BASS_MAX
        mid_mask = (freq_bins >= self.BASS_MAX) & (freq_bins < self.MID_MAX)
        treb_mask = freq_bins >= self.TREB_MIN

        # Calculate energy in each band (RMS)
        bass_energy = np.sqrt(np.mean(magnitude[bass_mask] ** 2))
        mid_energy = np.sqrt(np.mean(magnitude[mid_mask] ** 2))
        treb_energy = np.sqrt(np.mean(magnitude[treb_mask] ** 2))

        # Normalize to 0-100 range (approximate)
        # These constants are empirical - adjust based on your mic sensitivity
        bass = min(100.0, bass_energy * 0.5)
        mid = min(100.0, mid_energy * 0.8)
        treb = min(100.0, treb_energy * 1.2)

        return {
            'bass': bass,
            'mid': mid,
            'treb': treb,
            'waveform': audio_data[:576].tolist()  # MilkDrop uses 576 samples
        }

    def get_stats(self) -> Dict[str, float]:
        """
        Get current audio analysis mapped to Q variables.

        Standard audio variables (bass, mid, treb) are handled separately
        by the visualizer. Here we provide additional mic-specific data.

        Returns:
            Dict with:
            - bass, mid, treb: Standard audio values
            - q17-q22: Additional mic FFT bands for distinction from music
        """
        if len(self.audio_buffer) < self.CHUNK_SIZE:
            # Not enough data yet
            return {
                'bass': 0.0,
                'mid': 0.0,
                'treb': 0.0,
                'q17': 0.0, 'q18': 0.0, 'q19': 0.0,
                'q20': 0.0, 'q21': 0.0, 'q22': 0.0,
            }

        # Get latest audio chunk
        audio_chunk = np.array(list(self.audio_buffer)[-self.CHUNK_SIZE:])

        # Analyze
        analysis = self.analyze_audio(audio_chunk)

        # Smooth values for visual stability
        self.bass_smooth = (self.smoothing_factor * analysis['bass'] +
                           (1 - self.smoothing_factor) * self.bass_smooth)
        self.mid_smooth = (self.smoothing_factor * analysis['mid'] +
                          (1 - self.smoothing_factor) * self.mid_smooth)
        self.treb_smooth = (self.smoothing_factor * analysis['treb'] +
                           (1 - self.smoothing_factor) * self.treb_smooth)

        # Additional frequency bands for Q variables (finer granularity)
        # Divide spectrum into 6 bands for q17-q22
        audio_chunk_windowed = audio_chunk * np.hanning(len(audio_chunk))
        fft = np.fft.rfft(audio_chunk_windowed, n=self.FFT_SIZE)
        magnitude = np.abs(fft)

        # 6 evenly-spaced frequency bands
        band_size = len(magnitude) // 6
        q_bands = []
        for i in range(6):
            start = i * band_size
            end = start + band_size
            band_energy = np.sqrt(np.mean(magnitude[start:end] ** 2))
            q_bands.append(min(100.0, band_energy * 0.5))

        return {
            'bass': self.bass_smooth,
            'mid': self.mid_smooth,
            'treb': self.treb_smooth,
            'q17': q_bands[0],  # Lowest freq band
            'q18': q_bands[1],
            'q19': q_bands[2],
            'q20': q_bands[3],
            'q21': q_bands[4],
            'q22': q_bands[5],  # Highest freq band
        }

    def stream_stats(self, callback: Callable):
        """
        Continuously stream audio analysis to callback function.

        Args:
            callback: Function that receives stats dict
        """
        self.start_stream()

        print(f"AudioCollector: Streaming analysis every {self.update_interval}s")

        try:
            while True:
                stats = self.get_stats()
                callback(stats)
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\nAudioCollector: Stopped")
            self.stop_stream()


def main():
    """Test the collector."""
    collector = AudioCollector(update_interval=0.1)

    def print_stats(stats):
        # Clear screen-like effect
        print("\033[H\033[J", end='')

        print("="*60)
        print("Microphone Audio Analysis")
        print("="*60)

        # Main bands
        print(f"\nStandard Bands:")
        print(f"  Bass:  {stats['bass']:6.1f} {'█' * int(stats['bass'] / 5)}")
        print(f"  Mid:   {stats['mid']:6.1f} {'█' * int(stats['mid'] / 5)}")
        print(f"  Treb:  {stats['treb']:6.1f} {'█' * int(stats['treb'] / 5)}")

        # Q variable bands
        print(f"\nDetailed Bands (Q17-Q22):")
        for i in range(6):
            q_val = stats[f'q{17+i}']
            print(f"  Band {i+1}: {q_val:6.1f} {'█' * int(q_val / 5)}")

        print("\n(Ctrl+C to stop)")

    collector.stream_stats(print_stats)


if __name__ == '__main__':
    main()
