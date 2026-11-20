// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <map>
#include <span>
#include <string_view>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/ir/attribute.h"

namespace Shader {

/// One dword written by the copy shader to a stream-out buffer.
struct CopyShaderXfbOutput {
    u32 buffer;             ///< Stream-out buffer index (0-3)
    u32 offset;             ///< Byte offset inside the streamed vertex
    bool pack_half;         ///< Dword is v_cvt_pkrtz_f16_f32(src[0], src[1])
    std::array<u32, 2> src; ///< GSVS ring offsets (as read by the copy shader) of the sources
};

/// Where the copy shader takes the V# of a stream-out buffer from.
struct CopyShaderXfbBuffer {
    s32 table_sgpr{-1};       ///< User SGPR holding the pointer to the V# table (pair)
    u32 table_dword_offset{}; ///< Dword offset of the V# inside the table
    s32 direct_sgpr{-1};      ///< User SGPR where the V# is passed directly
};

struct CopyShaderData {
    std::map<u32, std::pair<Shader::IR::Attribute, u32>> attr_map;
    u32 num_attrs{0};
    u32 output_vertices{0};
    u32 num_comps{0};
    std::vector<CopyShaderXfbOutput> xfb_outputs;
    std::array<CopyShaderXfbBuffer, 4> xfb_buffers{};
    u32 xfb_buffer_mask{0};
    bool xfb_parse_failed{false};
    bool xfb_active{false}; ///< Set by the GS ring pass when outputs are captured
};

CopyShaderData ParseCopyShader(std::span<const u32> code);

/// True when SHADPS4_STRMOUT_TRACE is set (stream-out debug tracing and dumps).
bool XfbTraceEnabled();

/// Appends a line to log/strmout_trace.txt and flushes it (survives crashes).
void XfbTrace(std::string_view line);
/// Writes a binary blob to log/<name> (overwrites).
void XfbDumpBinary(std::string_view name, std::span<const u32> code);

} // namespace Shader
