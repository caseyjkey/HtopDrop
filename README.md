# HtopDrop

Real-time system monitoring visualizer that maps htop stats, GPU metrics, and microphone audio to distinct visual representations in projectM/MilkDrop.

## Features

- 🎵 **Audio Reactive**: Microphone input drives music-style visualizations
- 💻 **CPU Monitoring**: Each CPU core gets unique visual representation (htop)
- 🎮 **GPU Monitoring**: Temperature and memory usage visualization (nvidia-smi)
- 🎨 **Distinct Channels**: Each data source has independent visual mapping

## Architecture

```
┌─────────────────┐
│  Microphone     │ → Audio FFT → projectM (bass, mid, treb)
└─────────────────┘

┌─────────────────┐
│   htop Parser   │ → CPU/Mem/Swap → Q Variables (q1-q8)
└─────────────────┘

┌─────────────────┐
│ nvidia-smi Poll │ → GPU Temp/Mem → Q Variables (q9-q12)
└─────────────────┘

All data → projectM Custom Presets → Distinct Visuals
```

## Data Mapping

### Q Variables (Custom Data Injection)

| Variable | Data Source | Description |
|----------|-------------|-------------|
| `q1` | htop | CPU usage % (average) |
| `q2` | htop | Memory usage % |
| `q3` | htop | Swap usage % |
| `q4-q7` | htop | Individual core usage (if multi-core) |
| `q8` | htop | Load average (1min) |
| `q9` | nvidia-smi | GPU temperature (°C) |
| `q10` | nvidia-smi | GPU memory used (MB) |
| `q11` | nvidia-smi | GPU memory total (MB) |
| `q12` | nvidia-smi | GPU memory % |
| `q17-q24` | microphone | Reserved for mic FFT bands |

### Audio Variables (Standard)

| Variable | Data Source | Description |
|----------|-------------|-------------|
| `bass` | microphone | Low frequency energy |
| `mid` | microphone | Mid frequency energy |
| `treb` | microphone | High frequency energy |

## Target Platform

- **Primary**: Gilbert (Arch Linux, GT540M)
- **Visualizer**: projectM (Linux-native MilkDrop clone)

## Installation (Gilbert)

```bash
# Install dependencies
sudo pacman -S projectm pulseaudio python python-pip

# Install Python dependencies
pip install psutil numpy sounddevice scipy

# Clone and setup
git clone git@github.com:caseyjkey/HtopDrop.git
cd HtopDrop
python htop_drop.py
```

## Usage

```bash
# Run data collector + visualizer
./run.sh

# Data collector only (for debugging)
python htop_drop.py --debug

# Custom preset directory
python htop_drop.py --presets ./custom_presets
```

## Custom Presets

HtopDrop includes custom .milk presets that specifically use system stats:

- `cpu_cores.milk` - Each CPU core drives a waveform
- `gpu_heat.milk` - GPU temperature affects color gradient
- `memory_pulse.milk` - RAM usage drives zoom/pulse
- `hybrid_system.milk` - All data sources combined

## How It Works

1. **Data Collection**: Python daemon polls htop, nvidia-smi, and microphone
2. **Data Injection**: Writes data to projectM's Q variables via shared memory/socket
3. **Visualization**: Custom presets read Q variables and render distinct visuals
4. **Real-time**: Updates 30-60 FPS for smooth reactive visuals

## GT540M Limitations

The GT540M supports limited nvidia-smi queries:
- ✅ Temperature
- ✅ Memory usage
- ❌ GPU utilization % (not supported)
- ❌ Memory utilization % (not supported)

We work with what's available!

## License

MIT
