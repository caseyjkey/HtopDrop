# HtopDrop Architecture

## Overview

HtopDrop is a real-time system monitoring visualizer inspired by MilkDrop. It creates distinct visual representations for different data sources: CPU/memory stats (htop), GPU stats (nvidia-smi), and microphone audio.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                             │
│  (htop_drop.py)                                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ├─── DataAggregator
                        │       │
                        │       ├─── HtopCollector (Thread 1)
                        │       │      └─> psutil → CPU, Memory, Swap
                        │       │
                        │       ├─── NvidiaCollector (Thread 2)
                        │       │      └─> nvidia-smi → GPU Temp, Memory
                        │       │
                        │       └─── AudioCollector (Thread 3)
                        │              └─> sounddevice → Mic FFT
                        │
                        └─── HtopDropVisualizer (Main Thread)
                               └─> pygame → OpenGL Rendering
```

## Data Flow

### 1. Collection Phase

Each collector runs in its own thread:

- **HtopCollector** (100ms interval)
  - Uses `psutil` library
  - Collects: CPU%, Memory%, Swap%, per-core usage
  - Outputs: Q variables q1-q8

- **NvidiaCollector** (500ms interval)
  - Executes `nvidia-smi` subprocess
  - Parses CSV output
  - Handles GT540M limitations (no utilization metrics)
  - Outputs: Q variables q9-q12

- **AudioCollector** (50ms interval)
  - Captures microphone via `sounddevice`
  - Performs FFT analysis
  - Divides spectrum into 6 bands
  - Outputs: bass, mid, treb + Q variables q17-q22

### 2. Aggregation Phase

The `DataAggregator` class:
- Receives callbacks from each collector thread
- Merges data into a unified dictionary (thread-safe with locks)
- Provides formatted data grouped by source

### 3. Visualization Phase

The `HtopDropVisualizer` class:
- Renders at 60 FPS (main thread)
- Divides screen into 3 zones:
  - **Top Third**: Audio reactive (waveforms, frequency bars)
  - **Middle Third**: CPU/Memory (rotating spokes, memory bar)
  - **Bottom Third**: GPU (heat map, memory tank)

## Q Variable Mapping

Following MilkDrop's convention, custom data is mapped to Q variables:

| Variable Range | Data Source | Description |
|----------------|-------------|-------------|
| q1-q8 | htop | CPU and memory statistics |
| q9-q12 | nvidia-smi | GPU temperature and memory |
| q13-q16 | (reserved) | Future expansion |
| q17-q24 | microphone | Additional audio frequency bands |

Standard audio variables (bass, mid, treb) are kept separate for compatibility.

## Threading Model

### Thread Safety

- **Collectors**: Each runs in daemon thread, blocking on I/O
- **Shared State**: Single `current_data` dict protected by lock
- **GIL**: Python's GIL protects dict operations
- **Callbacks**: Update shared state via lock acquisition

### Performance Considerations

- **Audio**: Highest frequency (50ms = 20Hz update) for smooth reactivity
- **htop**: Medium frequency (100ms = 10Hz) for CPU tracking
- **nvidia-smi**: Lowest frequency (500ms = 2Hz) due to subprocess overhead

## Rendering Pipeline

```
Frame N:
  1. Poll pygame events (keyboard, mouse, quit)
  2. Lock data, copy current_data snapshot
  3. Clear screen (black)
  4. Render audio zone (top)
  5. Render htop zone (middle)
  6. Render nvidia zone (bottom)
  7. Render debug overlay (if enabled)
  8. Flip buffers (display)
  9. Sleep to maintain 60 FPS
```

## Extension Points

### Adding New Data Sources

1. Create new collector in `src/collectors/`
2. Implement `get_stats()` and `stream_stats(callback)` methods
3. Add thread in `DataAggregator.start()`
4. Assign Q variables (q25-q32 available)
5. Add visualization in new zone or existing zone

### Custom Visualizations

Option 1: Modify existing zones in `pygame_visualizer.py`
Option 2: Create new visualizer class implementing `render(data)` method
Option 3: Export data via socket for external visualizers (projectM, etc.)

## Future Enhancements

### Planned Features

1. **projectM Integration**: Export Q variables to real MilkDrop presets
2. **Network Stats**: Bandwidth usage (q25-q28)
3. **Disk I/O**: Read/write rates (q29-q30)
4. **Custom Presets**: User-scriptable visualizations
5. **Recording**: Save visualization to video file

### projectM Integration Plan

```
HtopDrop → Unix Socket → projectM Patch
          (Q variables)   (reads socket,
                           injects to presets)
```

This would allow using actual .milk presets with system stats data.

## Performance Targets

- **FPS**: 60 (locked)
- **Latency**: < 100ms from data change to visual update
- **CPU Overhead**: < 5% on idle system
- **Memory**: < 100MB

## Compatibility

- **Primary Target**: Arch Linux (Gilbert)
- **GPU**: NVIDIA GT540M (limited nvidia-smi support)
- **Dependencies**: See `requirements.txt`
- **Python**: 3.9+ required (for type hints)
