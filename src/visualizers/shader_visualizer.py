#!/usr/bin/env python3
"""
Shader-based MilkDrop-style Visualizer

Uses OpenGL fragment shaders for flowing, psychedelic visuals.
Each data source drives distinct organic visual elements:
  - CPU: Swirling plasma warp patterns
  - Memory: Color palette shifting
  - GPU temp: Lava lamp blobs with sinusoidal edges
  - GPU memory: Blob density/size
  - Audio: Radial pulse, ripple, brightness
"""

import pygame
import moderngl
import numpy as np
import struct
from typing import Dict


# ─── GLSL Shaders ───────────────────────────────────────────────────────────

VERTEX_SHADER = """
#version 330 core
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    uv = in_uv;
}
"""

# Main visualization shader with feedback loop
FRAGMENT_SHADER = """
#version 330 core

uniform sampler2D prev_frame;
uniform float time;
uniform vec2 resolution;

// System data
uniform float cpu_avg;
uniform float mem_pct;
uniform float swap_pct;
uniform vec4 cores;       // 4 CPU core values (0-1)
uniform float load_avg;

// GPU data
uniform float gpu_temp;   // 0-100 celsius
uniform float gpu_mem;    // 0-1 percentage

// Audio data
uniform float bass;
uniform float mid;
uniform float treb;

in vec2 uv;
out vec4 fragColor;

// ── Noise functions ──────────────────────────────────────────────

vec3 mod289(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                       -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                            + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
                            dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float fbm(vec2 p, int octaves) {
    float val = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        val += amp * snoise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return val;
}

// ── Color palettes ──────────────────────────────────────────────

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

vec3 cosmicPalette(float t) {
    return palette(t,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 1.0, 1.0),
        vec3(0.0, 0.33, 0.67)
    );
}

vec3 firePalette(float t) {
    return palette(t,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 0.7, 0.4),
        vec3(0.0, 0.15, 0.20)
    );
}

vec3 oceanPalette(float t) {
    return palette(t,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 1.0, 0.5),
        vec3(0.80, 0.90, 0.30)
    );
}

// ── Lava lamp / metaball for GPU ────────────────────────────────

float metaball(vec2 p, vec2 center, float radius) {
    float d = length(p - center);
    return radius * radius / (d * d + 0.001);
}

float lavaLamp(vec2 p, float t, float temp_norm, float mem_norm) {
    float field = 0.0;

    // Number of blobs scales with memory usage (more memory = more blobs)
    int num_blobs = 3 + int(mem_norm * 5.0);

    for (int i = 0; i < 8; i++) {
        if (i >= num_blobs) break;
        float fi = float(i);

        // Each blob has unique sinusoidal motion
        float phase = fi * 1.618;  // Golden ratio spacing
        float speed = 0.3 + temp_norm * 0.5;  // Hotter = faster

        // Sinusoidal position - rises and falls like real lava
        float x = 0.3 + 0.4 * sin(t * speed * 0.7 + phase * 2.1);
        float y = 0.2 + 0.6 * (0.5 + 0.5 * sin(t * speed * 0.4 + phase));

        // Blob size oscillates with sinusoidal edge
        float base_radius = 0.06 + 0.04 * sin(t * 1.3 + fi);
        // Sinusoidal wobble on the edge
        float angle = atan(p.y - y, p.x - x);
        float wobble = 1.0 + 0.3 * sin(angle * 3.0 + t * 2.0 + fi)
                            + 0.15 * sin(angle * 5.0 - t * 1.5 + fi * 0.7);
        float radius = base_radius * wobble;

        field += metaball(p, vec2(x, y), radius);
    }

    return field;
}

// ── Main shader ─────────────────────────────────────────────────

void main() {
    vec2 st = uv;
    vec2 centered = st - 0.5;
    float aspect = resolution.x / resolution.y;
    centered.x *= aspect;

    float r = length(centered);
    float angle = atan(centered.y, centered.x);

    // ── Feedback warp (MilkDrop-style trail effect) ──

    // CPU drives warp intensity and rotation
    float warp_strength = 0.003 + cpu_avg * 0.012;
    float warp_rotation = time * (0.1 + cpu_avg * 0.3);

    // Create warped UV for sampling previous frame
    vec2 warp_offset;
    warp_offset.x = sin(angle * 2.0 + warp_rotation) * warp_strength;
    warp_offset.y = cos(angle * 3.0 + warp_rotation * 0.7) * warp_strength;

    // Zoom warp - bass makes it breathe
    float zoom = 1.0 - 0.002 - bass * 0.008;
    vec2 warped_uv = 0.5 + (st - 0.5) * zoom + warp_offset;

    // Per-core warp: each core distorts a quadrant
    if (centered.x > 0.0 && centered.y > 0.0)
        warped_uv += vec2(sin(time*3.0), cos(time*2.7)) * cores.x * 0.003;
    else if (centered.x < 0.0 && centered.y > 0.0)
        warped_uv += vec2(cos(time*2.5), sin(time*3.1)) * cores.y * 0.003;
    else if (centered.x < 0.0 && centered.y < 0.0)
        warped_uv += vec2(sin(time*2.8), cos(time*2.3)) * cores.z * 0.003;
    else
        warped_uv += vec2(cos(time*3.3), sin(time*2.1)) * cores.w * 0.003;

    // Sample previous frame with warp (creates flowing trails)
    vec3 feedback = texture(prev_frame, warped_uv).rgb;

    // Fade the feedback (trails decay)
    float fade = 0.96 + cpu_avg * 0.025;  // More CPU = longer trails
    fade = min(fade, 0.995);
    feedback *= fade;

    // ── CPU Plasma Layer ──
    // Flowing noise patterns driven by CPU activity

    float cpu_noise_speed = 0.5 + cpu_avg * 2.0;
    vec2 noise_coord = centered * (2.0 + cpu_avg * 3.0);

    float n1 = fbm(noise_coord + vec2(time * cpu_noise_speed * 0.3,
                                       time * cpu_noise_speed * 0.2), 4);
    float n2 = fbm(noise_coord * 1.5 + vec2(-time * 0.4, time * 0.3) + n1 * 0.5, 3);

    // Domain warping for extra organic feel
    float domain_warp = fbm(noise_coord + vec2(n1, n2) * 0.8 + time * 0.1, 3);

    // Color from memory-shifted palette
    float palette_shift = mem_pct * 2.0 + time * 0.05;
    vec3 plasma_color = cosmicPalette(domain_warp * 0.5 + palette_shift);

    // CPU intensity controls how much new plasma is added
    float plasma_intensity = cpu_avg * 0.15 + 0.02;
    // Add more intensity near center (radial falloff)
    plasma_intensity *= smoothstep(0.8, 0.0, r);

    // ── Audio Reactive Layer ──

    // Bass: radial pulse ring
    float bass_ring = smoothstep(0.02, 0.0, abs(r - 0.2 - bass * 0.3));
    bass_ring += smoothstep(0.03, 0.0, abs(r - 0.4 - bass * 0.15));
    vec3 bass_color = vec3(0.9, 0.2, 0.5) * bass_ring * bass * 0.8;

    // Mid: spiral arms
    float spiral = sin(angle * 3.0 + r * 10.0 - time * 2.0) * 0.5 + 0.5;
    spiral *= smoothstep(0.6, 0.1, r);
    vec3 mid_color = oceanPalette(spiral + time * 0.1) * spiral * mid * 0.3;

    // Treb: sparkle/noise at high frequencies
    float sparkle = snoise(st * 50.0 + time * 5.0);
    sparkle = pow(max(sparkle, 0.0), 3.0);
    vec3 treb_color = vec3(0.7, 0.8, 1.0) * sparkle * treb * 0.4;

    // ── GPU Lava Lamp Layer ──

    float temp_norm = clamp(gpu_temp / 100.0, 0.0, 1.0);
    float mem_norm = gpu_mem;

    float lava = lavaLamp(st, time, temp_norm, mem_norm);

    // Smooth threshold for blob edges (sinusoidal boundary feel)
    float lava_edge = smoothstep(0.8, 1.2, lava);
    float lava_glow = smoothstep(0.3, 1.0, lava) * 0.3;

    // Temperature-based color: cool blue → warm orange → hot red
    vec3 lava_cool = vec3(0.1, 0.3, 0.8);
    vec3 lava_warm = vec3(0.9, 0.5, 0.1);
    vec3 lava_hot  = vec3(1.0, 0.15, 0.05);

    vec3 lava_color;
    if (temp_norm < 0.5) {
        lava_color = mix(lava_cool, lava_warm, temp_norm * 2.0);
    } else {
        lava_color = mix(lava_warm, lava_hot, (temp_norm - 0.5) * 2.0);
    }

    // Add internal glow variation
    float lava_internal = snoise(st * 4.0 + time * 0.5) * 0.3 + 0.7;
    lava_color *= lava_internal;

    vec3 lava_layer = lava_color * (lava_edge + lava_glow);

    // ── Composite all layers ──

    vec3 color = feedback;                          // Feedback trails
    color += plasma_color * plasma_intensity;       // CPU plasma
    color += bass_color;                            // Audio bass rings
    color += mid_color;                             // Audio mid spirals
    color += treb_color;                            // Audio treb sparkle
    color = max(color, lava_layer * 0.7);           // GPU lava lamp (blended)

    // ── Vignette (subtle darkening at edges) ──
    float vignette = 1.0 - smoothstep(0.4, 1.2, r / aspect * 2.0);
    color *= mix(0.6, 1.0, vignette);

    // ── Final tone mapping ──
    color = color / (color + 1.0);  // Reinhard tone mapping
    color = pow(color, vec3(0.95));  // Slight gamma for richness

    fragColor = vec4(color, 1.0);
}
"""

# Display pass (just renders texture to screen)
DISPLAY_FRAGMENT = """
#version 330 core
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main() {
    fragColor = texture(tex, uv);
}
"""


class ShaderVisualizer:
    """MilkDrop-style shader visualizer with feedback loop."""

    def __init__(self, width: int = 1920, height: int = 1080, fullscreen: bool = False):
        pygame.init()

        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if fullscreen:
            flags |= pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("HtopDrop")

        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.time = 0.0

        # Create moderngl context from existing pygame GL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)

        # Fullscreen quad vertices (position + uv)
        vertices = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())

        # Main visualization program
        self.main_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.main_vao = self.ctx.vertex_array(
            self.main_prog,
            [(self.vbo, '2f 2f', 'in_position', 'in_uv')],
        )

        # Display program (renders FBO to screen)
        self.display_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=DISPLAY_FRAGMENT,
        )
        self.display_vao = self.ctx.vertex_array(
            self.display_prog,
            [(self.vbo, '2f 2f', 'in_position', 'in_uv')],
        )

        # Ping-pong framebuffers for feedback loop
        self.tex_a = self.ctx.texture((width, height), 4, dtype='f2')
        self.tex_b = self.ctx.texture((width, height), 4, dtype='f2')
        self.tex_a.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_a.repeat_x = False
        self.tex_a.repeat_y = False
        self.tex_b.repeat_x = False
        self.tex_b.repeat_y = False

        self.fbo_a = self.ctx.framebuffer(color_attachments=[self.tex_a])
        self.fbo_b = self.ctx.framebuffer(color_attachments=[self.tex_b])

        self.ping = True  # Toggle for ping-pong

        # Smoothed values for visual stability
        self.smooth = {
            'cpu_avg': 0.0, 'mem_pct': 0.0, 'swap_pct': 0.0,
            'core0': 0.0, 'core1': 0.0, 'core2': 0.0, 'core3': 0.0,
            'load_avg': 0.0, 'gpu_temp': 0.0, 'gpu_mem': 0.0,
            'bass': 0.0, 'mid': 0.0, 'treb': 0.0,
        }

    def _smooth(self, key: str, value: float, factor: float = 0.15) -> float:
        """Exponential smoothing for visual stability."""
        self.smooth[key] = factor * value + (1.0 - factor) * self.smooth[key]
        return self.smooth[key]

    def _set_uniform(self, name: str, value):
        """Set a uniform, silently skipping if optimized out by compiler."""
        if name in self.main_prog:
            self.main_prog[name].value = value

    def _set_uniforms(self, data: Dict):
        """Set all shader uniforms from data."""
        htop = data.get('htop', {})
        nvidia = data.get('nvidia', {})
        audio = data.get('audio', {})

        self._set_uniform('time', self.time)
        self._set_uniform('resolution', (float(self.width), float(self.height)))

        # CPU (normalize to 0-1)
        self._set_uniform('cpu_avg', self._smooth('cpu_avg', htop.get('cpu_avg', 0) / 100.0))
        self._set_uniform('mem_pct', self._smooth('mem_pct', htop.get('memory', 0) / 100.0))
        self._set_uniform('swap_pct', self._smooth('swap_pct', htop.get('swap', 0) / 100.0))
        self._set_uniform('load_avg', self._smooth('load_avg', htop.get('load_avg', 0) / 100.0))

        cores = (
            self._smooth('core0', htop.get('core0', 0) / 100.0),
            self._smooth('core1', htop.get('core1', 0) / 100.0),
            self._smooth('core2', htop.get('core2', 0) / 100.0),
            self._smooth('core3', htop.get('core3', 0) / 100.0),
        )
        self._set_uniform('cores', cores)

        # GPU
        self._set_uniform('gpu_temp', self._smooth('gpu_temp', nvidia.get('temperature', 0), 0.05))
        self._set_uniform('gpu_mem', self._smooth('gpu_mem', nvidia.get('memory_percent', 0) / 100.0, 0.05))

        # Audio (more responsive smoothing)
        self._set_uniform('bass', self._smooth('bass', audio.get('bass', 0) / 100.0, 0.4))
        self._set_uniform('mid', self._smooth('mid', audio.get('mid', 0) / 100.0, 0.35))
        self._set_uniform('treb', self._smooth('treb', audio.get('treb', 0) / 100.0, 0.3))

    def render(self, data: Dict) -> bool:
        """Render one frame. Returns False if quit requested."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()

        dt = self.clock.tick(60) / 1000.0
        self.time += dt

        # Ping-pong: read from one texture, write to the other
        if self.ping:
            read_tex, write_fbo = self.tex_a, self.fbo_b
            result_tex = self.tex_b
        else:
            read_tex, write_fbo = self.tex_b, self.fbo_a
            result_tex = self.tex_a
        self.ping = not self.ping

        # ── Main pass: render to FBO ──
        write_fbo.use()
        read_tex.use(location=0)
        self.main_prog['prev_frame'].value = 0
        self._set_uniforms(data)
        self.main_vao.render(moderngl.TRIANGLE_STRIP)

        # ── Display pass: render FBO to screen ──
        self.ctx.screen.use()
        result_tex.use(location=0)
        self.display_prog['tex'].value = 0
        self.display_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        return True

    def cleanup(self):
        """Release resources."""
        self.fbo_a.release()
        self.fbo_b.release()
        self.tex_a.release()
        self.tex_b.release()
        self.main_vao.release()
        self.display_vao.release()
        self.vbo.release()
        self.main_prog.release()
        self.display_prog.release()
        self.ctx.release()
        pygame.quit()
