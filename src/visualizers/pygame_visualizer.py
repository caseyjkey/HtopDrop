#!/usr/bin/env python3
"""
PyGame Visualizer

Creates distinct visual zones for htop, nvidia, and audio data.
Each data source has its own visual representation.
"""

import pygame
import math
import colorsys
from typing import Dict


class HtopDropVisualizer:
    """MilkDrop-style visualizer with distinct zones for each data source."""

    def __init__(self, width: int = 1920, height: int = 1080, fullscreen: bool = False):
        """
        Initialize visualizer.

        Args:
            width: Window width
            height: Window height
            fullscreen: Run in fullscreen mode
        """
        # Initialize pygame modules
        pygame.init()

        # Display setup
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode((width, height), flags | pygame.DOUBLEBUF)
        pygame.display.set_caption("HtopDrop - System Visualizer")

        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()

        # Visual zones (divide screen into regions)
        self.zones = {
            'audio': pygame.Rect(0, 0, width, height // 3),           # Top third - Audio
            'htop': pygame.Rect(0, height // 3, width, height // 3),  # Middle - CPU/Memory
            'nvidia': pygame.Rect(0, 2 * height // 3, width, height // 3)  # Bottom - GPU
        }

        # Animation state
        self.time = 0.0
        self.bass_pulse = 0.0
        self.cpu_rotation = 0.0

        # Color schemes for each zone
        self.colors = {
            'audio': (255, 100, 200),    # Pink/Magenta - Audio
            'htop': (100, 200, 255),     # Cyan/Blue - CPU
            'nvidia': (100, 255, 100),   # Green - GPU
        }

        # Font for debug info
        try:
            pygame.font.init()
            self.font = pygame.font.SysFont('monospace', 16)
            self.show_debug = True
        except Exception as e:
            print(f"WARNING: Font rendering not available: {e}")
            print("Debug overlay will be disabled")
            self.font = None
            self.show_debug = False

    def draw_audio_zone(self, surface: pygame.Surface, data: Dict, zone: pygame.Rect):
        """
        Draw audio visualization (top third).

        Microphone audio creates waveforms and frequency bars.
        """
        audio_data = data.get('audio', {})

        bass = audio_data.get('bass', 0) / 100.0
        mid = audio_data.get('mid', 0) / 100.0
        treb = audio_data.get('treb', 0) / 100.0

        # Bass pulse (affects background)
        self.bass_pulse = bass * 0.3 + self.bass_pulse * 0.7  # Smooth

        # Background glow
        glow_intensity = int(self.bass_pulse * 100)
        bg_color = (glow_intensity, glow_intensity // 2, glow_intensity)
        pygame.draw.rect(surface, bg_color, zone)

        # Frequency bands as vertical bars
        bands = [audio_data.get(f'band{i}', 0) / 100.0 for i in range(6)]
        bar_width = zone.width // len(bands)

        for i, band_val in enumerate(bands):
            x = zone.x + i * bar_width
            bar_height = int(band_val * zone.height)
            y = zone.y + zone.height - bar_height

            # Color gradient based on frequency (low = red, high = blue)
            hue = i / len(bands)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = (int(r * 255), int(g * 255), int(b * 255))

            pygame.draw.rect(surface, color, (x, y, bar_width - 2, bar_height))

        # Waveform circle (bass/mid/treb as concentric rings)
        center_x = zone.centerx
        center_y = zone.centery

        # Outer ring (bass) - Red
        bass_radius = int(50 + bass * 100)
        pygame.draw.circle(surface, (255, 50, 50), (center_x, center_y), bass_radius, 3)

        # Middle ring (mid) - Green
        mid_radius = int(40 + mid * 80)
        pygame.draw.circle(surface, (50, 255, 50), (center_x, center_y), mid_radius, 3)

        # Inner ring (treb) - Blue
        treb_radius = int(30 + treb * 60)
        pygame.draw.circle(surface, (50, 50, 255), (center_x, center_y), treb_radius, 3)

    def draw_htop_zone(self, surface: pygame.Surface, data: Dict, zone: pygame.Rect):
        """
        Draw htop visualization (middle third).

        CPU cores as rotating bars, memory as flowing particles.
        """
        htop_data = data.get('htop', {})

        cpu_avg = htop_data.get('cpu_avg', 0) / 100.0
        memory = htop_data.get('memory', 0) / 100.0
        cores = [htop_data.get(f'core{i}', 0) / 100.0 for i in range(4)]

        # Background based on CPU load
        cpu_intensity = int(cpu_avg * 80)
        bg_color = (cpu_intensity // 2, cpu_intensity // 2, cpu_intensity)
        pygame.draw.rect(surface, bg_color, zone)

        # CPU cores as rotating spokes
        center_x = zone.centerx
        center_y = zone.centery

        self.cpu_rotation += cpu_avg * 0.05  # Rotate faster with higher CPU

        for i, core_val in enumerate(cores[:4]):
            angle = self.cpu_rotation + (i * math.pi / 2)  # 4 cores = 90° apart
            length = 50 + core_val * 150

            end_x = center_x + math.cos(angle) * length
            end_y = center_y + math.sin(angle) * length

            # Color intensity based on core usage
            intensity = int(core_val * 255)
            color = (intensity, intensity, 255)

            pygame.draw.line(surface, color, (center_x, center_y), (end_x, end_y), 5)

        # Memory as horizontal bar
        mem_bar_width = int(memory * (zone.width - 40))
        mem_rect = pygame.Rect(zone.x + 20, zone.bottom - 40, mem_bar_width, 20)
        pygame.draw.rect(surface, (255, 200, 100), mem_rect)
        pygame.draw.rect(surface, (255, 255, 255), (zone.x + 20, zone.bottom - 40, zone.width - 40, 20), 2)

    def draw_nvidia_zone(self, surface: pygame.Surface, data: Dict, zone: pygame.Rect):
        """
        Draw nvidia GPU visualization (bottom third).

        Temperature as heat map, memory as filling tank.
        """
        nvidia_data = data.get('nvidia', {})

        temp = nvidia_data.get('temperature', 0)
        mem_percent = nvidia_data.get('memory_percent', 0) / 100.0

        # Background heat map based on temperature
        # 0°C = blue, 50°C = green, 80°C = yellow, 100°C = red
        temp_normalized = max(0, min(1, temp / 100.0))
        if temp_normalized < 0.5:
            # Blue to green
            r = int(temp_normalized * 2 * 255)
            g = int(100 + temp_normalized * 2 * 155)
            b = int((1 - temp_normalized * 2) * 255)
        else:
            # Green to red
            r = int(255)
            g = int((1 - (temp_normalized - 0.5) * 2) * 255)
            b = 0

        bg_color = (r // 3, g // 3, b // 3)
        pygame.draw.rect(surface, bg_color, zone)

        # Memory tank (fills from bottom)
        tank_width = zone.width // 2
        tank_height = zone.height - 40
        tank_x = zone.centerx - tank_width // 2
        tank_y = zone.y + 20

        # Filled portion
        fill_height = int(mem_percent * tank_height)
        fill_y = tank_y + tank_height - fill_height

        pygame.draw.rect(surface, (50, 200, 50), (tank_x, fill_y, tank_width, fill_height))

        # Tank outline
        pygame.draw.rect(surface, (255, 255, 255), (tank_x, tank_y, tank_width, tank_height), 3)

        # Temperature text
        temp_text = self.font.render(f"{temp:.0f}°C", True, (255, 255, 255))
        surface.blit(temp_text, (zone.centerx - 30, zone.y + 5))

        # Memory text
        mem_text = self.font.render(f"MEM {mem_percent*100:.0f}%", True, (255, 255, 255))
        surface.blit(mem_text, (zone.centerx - 40, zone.bottom - 20))

    def draw_debug_info(self, surface: pygame.Surface, data: Dict, fps: float):
        """Draw debug information overlay."""
        if not self.font:
            return  # Skip if font not available

        y_offset = 10
        line_height = 20

        # FPS
        fps_text = self.font.render(f"FPS: {fps:.0f}", True, (255, 255, 0))
        surface.blit(fps_text, (10, y_offset))
        y_offset += line_height

        # Data summary
        htop_data = data.get('htop', {})
        nvidia_data = data.get('nvidia', {})
        audio_data = data.get('audio', {})

        lines = [
            f"CPU: {htop_data.get('cpu_avg', 0):.1f}%",
            f"MEM: {htop_data.get('memory', 0):.1f}%",
            f"GPU: {nvidia_data.get('temperature', 0):.0f}°C",
            f"Bass: {audio_data.get('bass', 0):.1f}",
        ]

        for line in lines:
            text = self.font.render(line, True, (255, 255, 0))
            surface.blit(text, (10, y_offset))
            y_offset += line_height

    def render(self, data: Dict) -> bool:
        """
        Render one frame.

        Args:
            data: Data dict from DataAggregator

        Returns:
            True if should continue, False if quit requested
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()

        # Clear screen
        self.screen.fill((0, 0, 0))

        # Draw each zone
        self.draw_audio_zone(self.screen, data, self.zones['audio'])
        self.draw_htop_zone(self.screen, data, self.zones['htop'])
        self.draw_nvidia_zone(self.screen, data, self.zones['nvidia'])

        # Debug overlay
        if self.show_debug:
            fps = self.clock.get_fps()
            self.draw_debug_info(self.screen, data, fps)

        # Update display
        pygame.display.flip()

        # Tick clock (60 FPS)
        self.clock.tick(60)

        self.time += 0.016  # Approximate delta time

        return True

    def cleanup(self):
        """Clean up resources."""
        pygame.quit()
