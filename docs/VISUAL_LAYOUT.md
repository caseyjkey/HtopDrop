# Visual Layout

## Screen Division

```
┌─────────────────────────────────────────────────────────┐
│                                                           │
│                   AUDIO ZONE (Top 1/3)                   │
│                                                           │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                   │
│  │███│ │██ │ │█  │ │   │ │   │ │   │  ← 6 Frequency    │
│  │███│ │██ │ │█  │ │   │ │   │ │   │    Bands (q17-22) │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘                   │
│                                                           │
│           ⭕ ⭕ ⭕  ← Concentric circles                  │
│                     (bass/mid/treb)                       │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│                   HTOP ZONE (Middle 1/3)                 │
│                                                           │
│                    │                                      │
│               ─────┼─────  ← CPU core 0                  │
│                    │                                      │
│         │          │          │  ← Rotating CPU spokes   │
│         └──────────┼──────────┘    (4 cores)             │
│                    │                                      │
│                                                           │
│  ┌─────────────────────────────┐  ← Memory bar          │
│  │█████████████                │    (horizontal)         │
│  └─────────────────────────────┘                         │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│                  NVIDIA ZONE (Bottom 1/3)                │
│                                                           │
│              ┌─────────────┐                             │
│              │             │                             │
│              │             │  ← GPU Memory tank         │
│              │             │    (fills from bottom)      │
│              │█████████████│                             │
│              │█████████████│                             │
│              └─────────────┘                             │
│                                                           │
│                  41°C  ← Temperature                     │
│                                                           │
│  Background color = heat map (blue → green → red)       │
│                                                           │
└─────────────────────────────────────────────────────────┘

Debug Overlay (top-left when enabled with 'D'):
  FPS: 60
  CPU: 23.4%
  MEM: 45.2%
  GPU: 41°C
  Bass: 12.3
```

## Color Scheme

### Audio Zone (Top)
- **Background**: Pulsing black → gray (reacts to bass)
- **Frequency bars**: Rainbow gradient (red → violet)
  - Low frequencies: Red/Orange
  - Mid frequencies: Yellow/Green
  - High frequencies: Blue/Violet
- **Concentric circles**:
  - Bass (outer): Red
  - Mid (middle): Green
  - Treb (inner): Blue

### Htop Zone (Middle)
- **Background**: Darker blue (intensity = CPU load)
- **CPU spokes**: Cyan/Blue (brightness = core usage)
  - Rotates faster with higher CPU usage
  - 4 spokes for 4 CPU cores (90° apart)
- **Memory bar**: Orange/Yellow gradient
  - Fills left-to-right

### Nvidia Zone (Bottom)
- **Background**: Heat map based on GPU temperature
  - 0-50°C: Blue → Green
  - 50-80°C: Green → Yellow
  - 80-100°C: Yellow → Red
- **Memory tank**: Green fill
  - Fills bottom-to-top
  - White outline
- **Text**: White labels for temp and memory %

## Visual Characteristics

### Audio Reactivity
- **Fast response**: Updates 20 times/sec (50ms interval)
- **Smooth**: Bass values are smoothed to prevent jarring changes
- **Distinct**: 6 frequency bands provide fine-grained spectrum view

### CPU Reactivity
- **Rotation**: Visual indication of system activity
- **Per-core**: Each CPU core individually tracked
- **Load-based**: Background darkens/lightens with CPU load

### GPU Monitoring
- **Temperature mapping**: Intuitive color coding (cool = blue, hot = red)
- **Memory visualization**: Tank metaphor easy to understand
- **Low update rate**: 2Hz (500ms) to reduce nvidia-smi overhead

## Customization

All visual parameters can be customized in `pygame_visualizer.py`:

```python
# Zone layout
self.zones = {
    'audio': pygame.Rect(0, 0, width, height // 3),
    'htop': pygame.Rect(0, height // 3, width, height // 3),
    'nvidia': pygame.Rect(0, 2 * height // 3, width, height // 3)
}

# Color schemes
self.colors = {
    'audio': (255, 100, 200),
    'htop': (100, 200, 255),
    'nvidia': (100, 255, 100),
}

# Animation speeds
self.cpu_rotation += cpu_avg * 0.05  # Rotation speed
self.bass_pulse = bass * 0.3 + self.bass_pulse * 0.7  # Smoothing
```

## Alternative Layouts

### Full-Screen Single Source
Modify zones to show only one data source:
```python
self.zones = {
    'audio': pygame.Rect(0, 0, width, height),
}
```

### Side-by-Side
```python
self.zones = {
    'audio': pygame.Rect(0, 0, width // 3, height),
    'htop': pygame.Rect(width // 3, 0, width // 3, height),
    'nvidia': pygame.Rect(2 * width // 3, 0, width // 3, height),
}
```

### Quad View
```python
self.zones = {
    'audio': pygame.Rect(0, 0, width // 2, height // 2),
    'htop': pygame.Rect(width // 2, 0, width // 2, height // 2),
    'nvidia': pygame.Rect(0, height // 2, width // 2, height // 2),
    'combined': pygame.Rect(width // 2, height // 2, width // 2, height // 2),
}
```
