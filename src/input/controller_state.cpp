// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

#include "common/types.h"
#include "core/libraries/pad/pad.h"
#include "input/controller.h"

namespace Input {

using Libraries::Pad::OrbisPadButtonDataOffset;

void State::OnButton(OrbisPadButtonDataOffset button, bool is_pressed) {
    if (is_pressed) {
        buttonsState |= button;
    } else {
        buttonsState &= ~button;
    }
}

void State::OnAxis(Axis axis, int value, u64 timestamp, bool smooth) {
    const auto index = std::to_underlying(axis);
    axes[index] = axis_smoothing_end_values[index];

    axis_smoothing_start_times[index] = timestamp;
    axis_smoothing_start_values[index] = axes[index];
    axis_smoothing_end_values[index] = value;
    axis_smoothing_flags[index] = smooth;
    const auto toggle = [&](const auto button) {
        if (value > 0) {
            buttonsState |= button;
        } else {
            buttonsState &= ~button;
        }
    };
    switch (axis) {
    case Axis::TriggerLeft:
        toggle(OrbisPadButtonDataOffset::L2);
        break;
    case Axis::TriggerRight:
        toggle(OrbisPadButtonDataOffset::R2);
        break;
    default:
        break;
    }
}

void State::OnTouchpad(int touch_index, bool is_down, float x, float y) {
    touchpad[touch_index].state = is_down;
    touchpad[touch_index].x = static_cast<u16>(x * 1920);
    touchpad[touch_index].y = static_cast<u16>(y * 941);
}

void State::OnGyro(const float gyro[3]) {
    angularVelocity.x = gyro[0];
    angularVelocity.y = gyro[1];
    angularVelocity.z = gyro[2];
}

void State::OnAccel(const float accel[3]) {
    acceleration.x = accel[0];
    acceleration.y = accel[1];
    acceleration.z = accel[2];
}

void State::UpdateAxisSmoothing(u64 timestamp) {
    for (int i = 0; i < std::to_underlying(Axis::AxisMax); ++i) {
        if (!axis_smoothing_flags[i] || std::abs(axes[i] - axis_smoothing_end_values[i]) < 16) {
            axes[i] = axis_smoothing_end_values[i];
            continue;
        }
        const f32 t = std::clamp(
            (timestamp - axis_smoothing_start_times[i]) / f32{axis_smoothing_time}, 0.f, 1.f);
        axes[i] = s32(axis_smoothing_start_values[i] * (1 - t) + axis_smoothing_end_values[i] * t);
    }
    ApplyAntiSnapback(timestamp);
}

// A spring-loaded stick that is let go flies past the centre and briefly reports the opposite
// direction. Hide that rebound: for a short window after an axis was pushed far one way, a
// value under half deflection on the other side is reported as centre. A deliberate flick past
// half deflection still passes.
// The window comes from SHADPS4_ANTI_SNAPBACK_MS (0 or unset = off).
void State::ApplyAntiSnapback(u64 timestamp) {
    static const u64 window_us = [] {
        const char* env = std::getenv("SHADPS4_ANTI_SNAPBACK_MS");
        const int ms = env ? std::atoi(env) : 0;
        return ms > 0 ? static_cast<u64>(ms) * 1000 : 0;
    }();
    if (window_us == 0) {
        return;
    }
    constexpr int strong = 64; // half deflection
    for (int i = 0; i <= std::to_underlying(Axis::RightY); ++i) {
        const int d = axes[i] - 128;
        const int sign = d > 0 ? 1 : (d < 0 ? -1 : 0);
        if (std::abs(d) >= strong) {
            snapback_last_sign[i] = sign;
            snapback_last_time[i] = timestamp;
            continue;
        }
        if (sign != 0 && sign == -snapback_last_sign[i] &&
            timestamp - snapback_last_time[i] < window_us) {
            axes[i] = 128;
        }
    }
}

} // namespace Input
