#!/usr/bin/env python3
"""
HtopDrop - Main Entry Point

Launches data collectors and visualizer.
"""

import sys
import argparse
from src.data_aggregator import DataAggregator
from src.visualizers.pygame_visualizer import HtopDropVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="HtopDrop - Real-time system monitoring visualizer"
    )
    parser.add_argument(
        '--width', type=int, default=1920,
        help='Window width (default: 1920)'
    )
    parser.add_argument(
        '--height', type=int, default=1080,
        help='Window height (default: 1080)'
    )
    parser.add_argument(
        '--fullscreen', action='store_true',
        help='Run in fullscreen mode'
    )
    parser.add_argument(
        '--audio-device', type=int, default=None,
        help='Audio input device index (see --list-devices)'
    )
    parser.add_argument(
        '--list-devices', action='store_true',
        help='List available audio devices and exit'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug mode - print data to console instead of visualizing'
    )

    args = parser.parse_args()

    # List audio devices
    if args.list_devices:
        import sounddevice as sd
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                default = " (DEFAULT)" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{default}")
        return 0

    # Create data aggregator
    print("Starting HtopDrop...")
    aggregator = DataAggregator(audio_device=args.audio_device)
    aggregator.start()

    try:
        if args.debug:
            # Debug mode - just print stats
            print("\nDebug Mode - Printing stats to console")
            print("Press Ctrl+C to stop\n")
            import time
            while True:
                aggregator.print_status()
                time.sleep(0.1)
        else:
            # Visualization mode
            visualizer = HtopDropVisualizer(
                width=args.width,
                height=args.height,
                fullscreen=args.fullscreen
            )

            print("\n" + "="*60)
            print("Visualizer Controls:")
            print("  ESC or Q : Quit")
            print("  D        : Toggle debug overlay")
            print("  F        : Toggle fullscreen")
            print("="*60 + "\n")

            # Main loop
            running = True
            while running:
                data = aggregator.get_formatted_data()
                running = visualizer.render(data)

            visualizer.cleanup()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        aggregator.stop()

    return 0


if __name__ == '__main__':
    sys.exit(main())
