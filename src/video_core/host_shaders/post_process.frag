// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D texSampler;

layout (push_constant) uniform settings {
    float gamma;
    bool hdr;
    bool srgb_input;
} pp;

const float cutoff = 0.0031308, a = 1.055, b = 0.055, d = 12.92;
vec3 gamma(vec3 rgb) {
    // The linear toe has to meet the power curve at the cutoff. The old toe (d / gamma) only
    // did so at gamma 1: with the game's brightness raised it jumped above the curve, so dark
    // values just below the cutoff came out brighter than values just above it. That inverted
    // step showed up as bands and green/purple blotches in dark gradients.
    float k = 1.0 / (2.4 + 1.0 - pp.gamma);
    float toe = pp.gamma == 1.0 ? d : (a * pow(cutoff, k) - b) / cutoff;
    return mix(
        a * pow(rgb, vec3(k)) - b,
        toe * rgb,
        lessThan(rgb, vec3(cutoff))
    );
}

// Small luma-only dither in the 8-bit input's own units. Brightness adjustment stretches the
// darkest input steps across several output levels; spreading them by half an input step hides
// the resulting bands without adding colour noise. Only used when the brightness is changed.
float dither() {
    vec2 p = gl_FragCoord.xy;
    float n1 = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    float n2 = fract(sin(dot(p, vec2(39.3468, 11.135))) * 24634.6345);
    return (n1 + n2 - 1.0) * (0.5 / 255.0);
}

// Exact inverse of gamma() at unit gamma, for buffers that are sRGB encoded but must be sampled
// through a UNORM view because Vulkan has no sRGB variant of the 10-bit format.
vec3 degamma(vec3 rgb) {
    return mix(
        pow(max(rgb + b, 0.0) / a, vec3(2.4)),
        rgb / d,
        lessThan(rgb, vec3(d * cutoff))
    );
}

// Standard sRGB encode (gamma() at unit gamma), used to dither in the input's encoding.
vec3 gamma_unit(vec3 rgb) {
    return mix(a * pow(max(rgb, 0.0), vec3(1.0 / 2.4)) - b, d * rgb, lessThan(rgb, vec3(cutoff)));
}

void main() {
    vec4 color_linear = texture(texSampler, uv);
    if (pp.hdr) {
        color = color_linear;
    } else {
        if (pp.srgb_input) color_linear.rgb = degamma(color_linear.rgb);
        if (pp.gamma != 1.0) {
            // Dither in encoded space, then return to linear for the adjustment.
            vec3 enc = clamp(gamma_unit(color_linear.rgb) + dither(), 0.0, 1.0);
            color_linear.rgb = degamma(enc);
        }
        color = vec4(gamma(color_linear.rgb), color_linear.a);
    }
}
