# Quick Start Guide

## Installation on Gilbert

### 1. Clone the Repository

```bash
git clone git@github.com:caseyjkey/HtopDrop.git
cd HtopDrop
```

### 2. Run Installation Script

```bash
chmod +x install_gilbert.sh
./install_gilbert.sh
```

This will:
- Install system packages (pygame, portaudio)
- Create Python virtual environment
- Install Python dependencies
- Test the installation

### 3. List Audio Devices

```bash
python3 htop_drop.py --list-devices
```

Example output:
```
Available audio input devices:
  [0] HDA Intel PCH: ALC662 rev1 Analog (hw:0,0) (DEFAULT)
  [1] USB Audio Device
  [2] Loopback: PCM
```

### 4. Run HtopDrop

```bash
# Windowed mode
python3 htop_drop.py

# Fullscreen mode
python3 htop_drop.py --fullscreen

# Specific audio device
python3 htop_drop.py --audio-device 1

# Custom resolution
python3 htop_drop.py --width 1280 --height 720
```

## Controls

| Key | Action |
|-----|--------|
| **ESC** or **Q** | Quit |
| **D** | Toggle debug overlay |
| **F** | Toggle fullscreen |

## Debug Mode

Test data collection without visualization:

```bash
python3 htop_drop.py --debug
```

This prints stats to console every 100ms:

```
==================================================
System Stats:
  CPU Average: 23.4%
  Memory:      45.2%
  Swap:        0.0%
  Load (norm): 18.7%
  Cores: 22.1%, 24.6%, 23.8%, 23.1%

GPU Stats:
  Temperature:  41.0°C
  Memory Used:  87 MB
  Memory Total: 964 MB
  Memory %:     9.0%

Microphone Audio:
  Bass:  12.3  ██
  Mid:   8.7   █
  Treb:  5.2   █

(Ctrl+C to stop)
```

## Testing Individual Collectors

### Test htop Collector

```bash
python3 -m src.collectors.htop_collector
```

### Test nvidia Collector

```bash
python3 -m src.collectors.nvidia_collector
```

### Test Audio Collector

```bash
python3 -m src.collectors.audio_collector
```

Each collector has a `main()` function that demonstrates its functionality.

## Troubleshooting

### No Audio Input

**Problem**: `No audio devices found` or `sounddevice error`

**Solution**:
```bash
# Install PortAudio
sudo pacman -S portaudio

# Check audio devices
arecord -l

# Test microphone
arecord -d 5 test.wav
aplay test.wav
```

### nvidia-smi Not Found

**Problem**: `nvidia-smi not found or not working`

**Solution**:
```bash
# Install NVIDIA drivers
sudo pacman -S nvidia nvidia-utils

# Test nvidia-smi
nvidia-smi
```

**Note**: On GT540M, utilization metrics return `[Not Supported]`. This is expected and HtopDrop handles it gracefully.

### Low Frame Rate

**Problem**: FPS < 30 in debug overlay

**Solutions**:
1. Lower resolution: `--width 1280 --height 720`
2. Close other applications
3. Reduce audio FFT size (edit `AudioCollector.CHUNK_SIZE`)

### pygame Display Error

**Problem**: `pygame.error: No available video device`

**Solution**:
```bash
# Ensure X11 is running (not Wayland)
echo $XDG_SESSION_TYPE

# If Wayland, switch to X11 session or set:
export SDL_VIDEODRIVER=wayland
```

### Permission Denied on /dev/snd

**Problem**: Cannot access microphone

**Solution**:
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Log out and back in
```

## Running on Startup (Optional)

Create systemd user service:

```bash
mkdir -p ~/.config/systemd/user/
cat > ~/.config/systemd/user/htopdrop.service <<EOF
[Unit]
Description=HtopDrop System Visualizer
After=graphical.target

[Service]
Type=simple
WorkingDirectory=$HOME/HtopDrop
ExecStart=/usr/bin/python3 htop_drop.py --fullscreen
Restart=on-failure

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user enable htopdrop
systemctl --user start htopdrop
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system design
- Modify visualization zones in `src/visualizers/pygame_visualizer.py`
- Add custom data sources in `src/collectors/`
- Create projectM integration for real .milk presets
