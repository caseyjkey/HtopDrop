#!/usr/bin/env python3
"""
Shader-based MilkDrop-style Visualizer

Uses OpenGL fragment shaders for flowing, psychedelic visuals.
Designed to be beautiful even at idle - metrics modulate an always-alive base.

Performance target: ~30fps on GT540M (low-power GPU)
"""

import pygame
import moderngl
import numpy as np
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

# Main visualization shader - designed to be always beautiful, even at idle.
# Metrics MODULATE the base animation rather than creating it.
FRAGMENT_SHADER = """
#version 330 core

uniform sampler2D prev_frame;
uniform float time;
uniform vec2 resolution;

// System data (all 0-1 normalized)
uniform float cpu_avg;
uniform float mem_pct;
uniform vec4 cores;

// GPU data
uniform float gpu_temp;   // celsius (raw, 0-100)
uniform float gpu_mem;    // 0-1

// Audio data
uniform float bass;
uniform float mid;
uniform float treb;

in vec2 uv;
out vec4 fragColor;

// ── Lightweight noise (2 octaves max for GT540M) ────────────────

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);  // smoothstep

    float a = dot(hash22(i + vec2(0,0)) - 0.5, f - vec2(0,0));
    float b = dot(hash22(i + vec2(1,0)) - 0.5, f - vec2(1,0));
    float c = dot(hash22(i + vec2(0,1)) - 0.5, f - vec2(0,1));
    float d = dot(hash22(i + vec2(1,1)) - 0.5, f - vec2(1,1));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y) + 0.5;
}

// Simple 2-octave fbm
float fbm2(vec2 p) {
    return noise(p) * 0.65 + noise(p * 2.1 + 3.7) * 0.35;
}

// ── Color palettes (Inigo Quilez style) ─────────────────────────

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

// ── Main shader ─────────────────────────────────────────────────

void main() {
    vec2 st = uv;
    vec2 centered = st - 0.5;
    float aspect = resolution.x / resolution.y;
    centered.x *= aspect;
    float r = length(centered);
    float angle = atan(centered.y, centered.x);

    // ════════════════════════════════════════════════════════════
    // FEEDBACK: Sample previous frame with warp (MilkDrop trails)
    // ════════════════════════════════════════════════════════════

    // Always-on gentle rotation + zoom (the "alive" base motion)
    float base_speed = 0.15;
    float cpu_boost = cpu_avg * 0.6;  // CPU makes it swirl faster
    float rot_speed = base_speed + cpu_boost;

    vec2 warp;
    warp.x = sin(angle * 2.0 + time * rot_speed) * (0.003 + cpu_avg * 0.008);
    warp.y = cos(angle * 3.0 + time * rot_speed * 0.7) * (0.003 + cpu_avg * 0.008);

    // Per-core quadrant distortion (subtle but distinct per core)
    float qx = step(0.0, centered.x) * 2.0 - 1.0;
    float qy = step(0.0, centered.y) * 2.0 - 1.0;
    float core_val = (qx > 0.0 && qy > 0.0) ? cores.x :
                     (qx < 0.0 && qy > 0.0) ? cores.y :
                     (qx < 0.0 && qy < 0.0) ? cores.z : cores.w;
    warp += vec2(sin(time * 2.0 + core_val * 5.0),
                 cos(time * 1.7 + core_val * 4.0)) * core_val * 0.004;

    // Breathing zoom (always subtle, bass amplifies)
    float breathe = sin(time * 0.4) * 0.001;
    float zoom = 1.0 - 0.003 - breathe - bass * 0.012;

    vec2 warped_uv = 0.5 + (st - 0.5) * zoom + warp;
    vec3 feedback = texture(prev_frame, warped_uv).rgb;

    // Trail decay: always visible trails, CPU extends them
    float fade = 0.94 + cpu_avg * 0.04;
    fade = min(fade, 0.99);
    feedback *= fade;

    // ════════════════════════════════════════════════════════════
    // LAYER 1: Flowing plasma (always active, CPU intensifies)
    // ════════════════════════════════════════════════════════════

    // Time flows always - CPU makes it flow faster
    float flow_t = time * (0.3 + cpu_avg * 0.8);

    // Domain warping: noise fed into noise = organic shapes
    float n1 = fbm2(centered * 3.0 + vec2(flow_t * 0.4, flow_t * 0.3));
    float n2 = fbm2(centered * 2.0 + vec2(n1 * 1.5, flow_t * -0.2));

    // Memory shifts the color palette over time
    float hue_shift = mem_pct * 3.0 + time * 0.08;
    vec3 plasma_col = palette(n2 * 0.6 + hue_shift,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 1.0, 1.0),
        vec3(0.0 + mem_pct * 0.3, 0.33, 0.67 - mem_pct * 0.2)
    );

    // Always add some plasma; CPU controls how much
    float plasma_strength = 0.06 + cpu_avg * 0.15;
    // Radial falloff - brighter near center
    plasma_strength *= smoothstep(0.9, 0.1, r);

    // ════════════════════════════════════════════════════════════
    // LAYER 2: GPU lava lamp (always drifting, temp colors it)
    // ════════════════════════════════════════════════════════════

    float temp_norm = clamp(gpu_temp / 90.0, 0.0, 1.0);
    float lava_speed = 0.2 + temp_norm * 0.4;

    // 4 metaball blobs with sinusoidal paths
    float field = 0.0;
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        float phase = fi * 1.618;
        float bx = 0.3 + 0.4 * sin(time * lava_speed * 0.6 + phase * 2.1);
        float by = 0.2 + 0.6 * (0.5 + 0.5 * sin(time * lava_speed * 0.35 + phase));

        // Sinusoidal wobble on blob boundary
        float a = atan(st.y - by, st.x - bx);
        float wobble = 1.0 + 0.25 * sin(a * 3.0 + time * 1.5 + fi)
                           + 0.12 * sin(a * 5.0 - time + fi * 0.7);
        float radius = (0.05 + gpu_mem * 0.04) * wobble;
        float d = length(st - vec2(bx, by));
        field += radius * radius / (d * d + 0.001);
    }

    float lava_edge = smoothstep(0.6, 1.3, field);
    float lava_glow = smoothstep(0.2, 0.8, field) * 0.25;

    // Temperature → color: cool teal → warm amber → hot magenta
    vec3 lava_col;
    if (temp_norm < 0.4) {
        lava_col = mix(vec3(0.05, 0.4, 0.5), vec3(0.2, 0.6, 0.3), temp_norm * 2.5);
    } else if (temp_norm < 0.7) {
        lava_col = mix(vec3(0.2, 0.6, 0.3), vec3(0.9, 0.5, 0.1), (temp_norm - 0.4) * 3.3);
    } else {
        lava_col = mix(vec3(0.9, 0.5, 0.1), vec3(1.0, 0.15, 0.3), (temp_norm - 0.7) * 3.3);
    }

    vec3 lava_layer = lava_col * (lava_edge + lava_glow);

    // ════════════════════════════════════════════════════════════
    // LAYER 3: Audio reactive (rings, spirals, sparkle)
    // ════════════════════════════════════════════════════════════

    // Bass: expanding rings from center
    float ring1 = smoothstep(0.025, 0.0, abs(r - 0.15 - bass * 0.35));
    float ring2 = smoothstep(0.02, 0.0, abs(r - 0.35 - bass * 0.2));
    vec3 bass_col = vec3(0.9, 0.2, 0.6) * (ring1 + ring2 * 0.6) * (bass + 0.05);

    // Mid: spiral arms
    float spiral = sin(angle * 4.0 + r * 12.0 - time * 2.5) * 0.5 + 0.5;
    spiral *= smoothstep(0.6, 0.05, r);
    vec3 mid_col = palette(spiral + time * 0.1,
        vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 0.5), vec3(0.8, 0.9, 0.3)
    ) * spiral * (mid * 0.6 + 0.02);

    // Treb: shimmer
    float sparkle = noise(st * 30.0 + time * 4.0);
    sparkle = pow(max(sparkle - 0.3, 0.0), 2.0) * 2.0;
    vec3 treb_col = vec3(0.6, 0.8, 1.0) * sparkle * (treb * 0.5 + 0.01);

    // ════════════════════════════════════════════════════════════
    // COMPOSITE
    // ════════════════════════════════════════════════════════════

    vec3 color = feedback;                        // Trails from previous frames
    color += plasma_col * plasma_strength;        // CPU plasma (always some)
    color += bass_col;                            // Audio bass rings
    color += mid_col;                             // Audio mid spirals
    color += treb_col;                            // Audio treble sparkle
    color = max(color, lava_layer * 0.65);        // GPU lava lamp overlay

    // Gentle vignette
    float vignette = 1.0 - smoothstep(0.5, 1.4, r / aspect * 2.0);
    color *= mix(0.7, 1.0, vignette);

    // Tone mapping (prevents blowout)
    color = color / (color + 0.8);
    color = pow(color, vec3(0.92));

    fragColor = vec4(color, 1.0);
}
"""

# Pass-through display shader
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

    TARGET_FPS = 30  # 30fps to keep GT540M cool

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

        # Fullscreen quad vertices (position + uv)
        vertices = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())

        # Compile shaders
        self.main_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.main_vao = self.ctx.vertex_array(
            self.main_prog,
            [(self.vbo, '2f 2f', 'in_position', 'in_uv')],
        )

        self.display_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=DISPLAY_FRAGMENT,
        )
        self.display_vao = self.ctx.vertex_array(
            self.display_prog,
            [(self.vbo, '2f 2f', 'in_position', 'in_uv')],
        )

        # Ping-pong framebuffers for feedback loop
        # Use half resolution for performance on GT540M
        fb_w, fb_h = width // 2, height // 2
        self.fb_w, self.fb_h = fb_w, fb_h
        self.tex_a = self.ctx.texture((fb_w, fb_h), 4, dtype='f2')
        self.tex_b = self.ctx.texture((fb_w, fb_h), 4, dtype='f2')
        for tex in (self.tex_a, self.tex_b):
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False

        self.fbo_a = self.ctx.framebuffer(color_attachments=[self.tex_a])
        self.fbo_b = self.ctx.framebuffer(color_attachments=[self.tex_b])
        self.ping = True

        # Smoothed values
        self.smooth = {
            'cpu_avg': 0.0, 'mem_pct': 0.0,
            'core0': 0.0, 'core1': 0.0, 'core2': 0.0, 'core3': 0.0,
            'gpu_temp': 0.0, 'gpu_mem': 0.0,
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
        self._set_uniform('resolution', (float(self.fb_w), float(self.fb_h)))

        # CPU (0-1)
        self._set_uniform('cpu_avg', self._smooth('cpu_avg', htop.get('cpu_avg', 0) / 100.0))
        self._set_uniform('mem_pct', self._smooth('mem_pct', htop.get('memory', 0) / 100.0))

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

        # Audio (faster smoothing for responsiveness)
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

        dt = self.clock.tick(self.TARGET_FPS) / 1000.0
        self.time += dt

        # Ping-pong
        if self.ping:
            read_tex, write_fbo = self.tex_a, self.fbo_b
            result_tex = self.tex_b
        else:
            read_tex, write_fbo = self.tex_b, self.fbo_a
            result_tex = self.tex_a
        self.ping = not self.ping

        # Render to FBO (at half resolution)
        write_fbo.use()
        self.ctx.viewport = (0, 0, self.fb_w, self.fb_h)
        read_tex.use(location=0)
        self.main_prog['prev_frame'].value = 0
        self._set_uniforms(data)
        self.main_vao.render(moderngl.TRIANGLE_STRIP)

        # Display FBO to screen (upscaled by linear filtering)
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        result_tex.use(location=0)
        self.display_prog['tex'].value = 0
        self.display_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        return True

    def cleanup(self):
        """Release resources."""
        for obj in (self.fbo_a, self.fbo_b, self.tex_a, self.tex_b,
                    self.main_vao, self.display_vao, self.vbo,
                    self.main_prog, self.display_prog):
            obj.release()
        self.ctx.release()
        pygame.quit()
