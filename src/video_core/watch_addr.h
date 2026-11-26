// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Diagnostic: SHADPS4_WATCH_ADDR=addr[:size][,addr[:size]...] logs every texture/buffer cache
// event that touches the given guest ranges (default size 0x1800000), tagged with the presented
// frame number. Off when the variable is unset. Logging is rate-limited per (event, address):
// the first 40 occurrences, then every 500th, so long sessions stay small.

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/types.h"

namespace VideoCore::Watch {

struct Range {
    VAddr addr;
    u64 size;
};

inline const std::vector<Range>& Ranges() {
    static const std::vector<Range> ranges = [] {
        std::vector<Range> out;
        const char* env = std::getenv("SHADPS4_WATCH_ADDR");
        if (!env) {
            return out;
        }
        std::string_view s{env};
        while (!s.empty()) {
            const auto comma = s.find(',');
            std::string item{s.substr(0, comma)};
            s = comma == std::string_view::npos ? std::string_view{} : s.substr(comma + 1);
            if (item.empty()) {
                continue;
            }
            u64 size = 0x1800000;
            if (const auto colon = item.find(':'); colon != std::string::npos) {
                size = std::strtoull(item.c_str() + colon + 1, nullptr, 0);
                item.resize(colon);
            }
            const u64 addr = std::strtoull(item.c_str(), nullptr, 0);
            if (addr != 0 && size != 0) {
                out.push_back({addr, size});
                LOG_WARNING(Render_Vulkan, "[WATCH] watching {:#x} size {:#x}", addr, size);
            }
        }
        return out;
    }();
    return ranges;
}

inline bool Enabled() {
    return !Ranges().empty();
}

/// Returns the watched base address overlapping [addr, addr + size), or 0.
inline VAddr Hit(VAddr addr, u64 size) {
    for (const auto& r : Ranges()) {
        if (addr < r.addr + r.size && r.addr < addr + std::max<u64>(size, 1)) {
            return r.addr;
        }
    }
    return 0;
}

inline std::atomic<u64>& FrameCounter() {
    static std::atomic<u64> frame{0};
    return frame;
}

inline void NextFrame() {
    if (Enabled()) {
        ++FrameCounter();
    }
}

/// Rate limit per key: first 40, then every 500th. Returns the occurrence count or 0 to skip.
inline u64 ShouldLog(std::string_view event, VAddr base) {
    static std::mutex mutex;
    static std::unordered_map<std::string, u64> counts;
    std::scoped_lock lk{mutex};
    const u64 n = ++counts[fmt::format("{}@{:x}", event, base)];
    return (n <= 40 || n % 500 == 0) ? n : 0;
}

template <typename... Args>
inline void Log(std::string_view event, VAddr base, fmt::format_string<Args...> f,
                Args&&... args) {
    const u64 n = ShouldLog(event, base);
    if (n == 0) {
        return;
    }
    LOG_WARNING(Render_Vulkan, "[WATCH f{} {} #{} @{:#x}] {}", FrameCounter().load(), event, n,
                base, fmt::format(f, std::forward<Args>(args)...));
}

} // namespace VideoCore::Watch
