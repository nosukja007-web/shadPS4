// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include "common/path_util.h"
#include "shader_recompiler/frontend/copy_shader.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/ir/attribute.h"

namespace Shader {

bool XfbTraceEnabled() {
    static const bool enabled = std::getenv("SHADPS4_STRMOUT_TRACE") != nullptr;
    return enabled;
}

void XfbTrace(std::string_view line) {
    if (!XfbTraceEnabled()) {
        return;
    }
    static std::mutex mutex;
    static std::FILE* file = [] {
        const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "strmout_trace.txt";
        return _wfopen(path.c_str(), L"w");
    }();
    std::scoped_lock lk{mutex};
    if (file) {
        std::fwrite(line.data(), 1, line.size(), file);
        std::fputc('\n', file);
        std::fflush(file);
    }
}

void XfbDumpBinary(std::string_view name, std::span<const u32> code) {
    if (!XfbTraceEnabled()) {
        return;
    }
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / std::string{name};
    if (std::FILE* f = _wfopen(path.c_str(), L"wb")) {
        std::fwrite(code.data(), sizeof(u32), code.size(), f);
        std::fclose(f);
    }
}

CopyShaderData ParseCopyShader(std::span<const u32> code) {
    Gcn::GcnCodeSlice code_slice{code.data(), code.data() + code.size()};
    Gcn::GcnDecodeContext decoder;

    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    ASSERT_MSG(code[0] == token_mov_vcchi, "First instruction is not s_mov_b32 vcc_hi, #imm");

    std::array<s32, 64> offsets{};
    offsets.fill(-1);

    std::array<s32, 256> sources{};
    sources.fill(-1);

    CopyShaderData data{};
    auto last_attr{IR::Attribute::Position0};

    // Stream-out tracking: provenance of VGPR values and sources of SGPR V#s.
    struct VgprSource {
        enum class Kind { None, Ring, PackHalf } kind{Kind::None};
        std::array<u32, 2> ring{};
    };
    std::array<VgprSource, 256> vgpr_src{};
    struct SgprSharp {
        s32 table_sgpr{-1};
        u32 dword_offset{};
    };
    std::array<SgprSharp, 128> sgpr_sharp{};
    const auto is_vgpr = [](const Gcn::InstOperand& op) {
        return op.field == Gcn::OperandField::VectorGPR && op.code < 256;
    };
    const auto add_xfb_store = [&](const Gcn::GcnInst& inst, u32 num_dwords) {
        const u32 sharp_sgpr = inst.src[2].code * 4;
        const auto& sharp = sharp_sgpr < sgpr_sharp.size() ? sgpr_sharp[sharp_sgpr] : SgprSharp{};
        u32 buffer = 0;
        CopyShaderXfbBuffer buf_src{};
        if (sharp.table_sgpr >= 0) {
            buffer = sharp.dword_offset / 4;
            buf_src.table_sgpr = sharp.table_sgpr;
            buf_src.table_dword_offset = sharp.dword_offset;
        } else {
            buf_src.direct_sgpr = static_cast<s32>(sharp_sgpr);
        }
        if (buffer >= 4) {
            data.xfb_parse_failed = true;
            return;
        }
        data.xfb_buffers[buffer] = buf_src;
        data.xfb_buffer_mask |= 1u << buffer;
        for (u32 i = 0; i < num_dwords; i++) {
            const u32 vgpr = inst.src[1].code + i;
            const auto& src = vgpr < vgpr_src.size() ? vgpr_src[vgpr] : VgprSource{};
            if (inst.src[1].field != Gcn::OperandField::VectorGPR ||
                src.kind == VgprSource::Kind::None) {
                data.xfb_parse_failed = true;
                continue;
            }
            data.xfb_outputs.push_back(CopyShaderXfbOutput{
                .buffer = buffer,
                .offset = static_cast<u32>(inst.control.mubuf.offset) + i * 4,
                .pack_half = src.kind == VgprSource::Kind::PackHalf,
                .src = src.ring,
            });
        }
    };
    while (!code_slice.atEnd()) {
        auto inst = decoder.decodeInstruction(code_slice);
        switch (inst.opcode) {
        case Gcn::Opcode::S_LSHL_B32: {
            // Stream-out copy shaders also shift SGPRs (e.g. so_offset * 4); only constant
            // shifts feed the ring offsets.
            if (inst.src[0].field != Gcn::OperandField::SignedConstIntPos ||
                inst.src[1].field != Gcn::OperandField::SignedConstIntPos) {
                if (inst.dst[0].code < sources.size()) {
                    sources[inst.dst[0].code] = -1;
                }
                break;
            }
            sources[inst.dst[0].code] =
                (inst.src[0].code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1)
                << (inst.src[1].code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1);
            break;
        }
        case Gcn::Opcode::S_MOVK_I32: {
            sources[inst.dst[0].code] = inst.control.sopk.simm;
            break;
        }
        case Gcn::Opcode::S_MOV_B32: {
            sources[inst.dst[0].code] = inst.src[0].code;
            break;
        }
        case Gcn::Opcode::S_ADDK_I32: {
            sources[inst.dst[0].code] += inst.control.sopk.simm;
            break;
        }
        case Gcn::Opcode::S_BFM_B32: {
            ASSERT(inst.src[0].field == Gcn::OperandField::SignedConstIntPos &&
                   inst.src[1].field == Gcn::OperandField::SignedConstIntPos);
            const auto src0 = inst.src[0].code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1;
            const auto src1 = inst.src[1].code - Gcn::OperandFieldRange::SignedConstIntPosMin + 1;
            sources[inst.dst[0].code] = ((1 << src0) - 1) << src1;
            break;
        }
        case Gcn::Opcode::EXP: {
            const auto& exp = inst.control.exp;
            const IR::Attribute semantic = static_cast<IR::Attribute>(exp.target);
            for (int i = 0; i < inst.src_count; ++i) {
                if ((exp.en & (1 << i)) == 0) {
                    continue;
                }
                const auto ofs = offsets[inst.src[i].code];
                if (ofs != -1) {
                    data.attr_map[ofs] = {semantic, i};
                    if (semantic > last_attr) {
                        last_attr = semantic;
                    }
                }
            }
            break;
        }
        case Gcn::Opcode::BUFFER_LOAD_DWORD: {
            ASSERT_MSG(inst.src[1].code < offsets.size(),
                       "offsets array for geometry shaders is too short");
            offsets[inst.src[1].code] = inst.control.mubuf.offset;
            if (inst.src[3].field != Gcn::OperandField::ConstZero) {
                const u32 index = inst.src[3].code;
                ASSERT(sources[index] != -1);
                offsets[inst.src[1].code] += sources[index];
            }
            vgpr_src[inst.src[1].code] = {VgprSource::Kind::Ring,
                                          {static_cast<u32>(offsets[inst.src[1].code]), 0}};
            data.num_comps++;
            break;
        }
        case Gcn::Opcode::V_MOV_B32: {
            if (is_vgpr(inst.dst[0])) {
                vgpr_src[inst.dst[0].code] =
                    is_vgpr(inst.src[0]) ? vgpr_src[inst.src[0].code] : VgprSource{};
            }
            break;
        }
        case Gcn::Opcode::V_CVT_PKRTZ_F16_F32: {
            if (is_vgpr(inst.dst[0])) {
                VgprSource src{};
                if (is_vgpr(inst.src[0]) && is_vgpr(inst.src[1]) &&
                    vgpr_src[inst.src[0].code].kind == VgprSource::Kind::Ring &&
                    vgpr_src[inst.src[1].code].kind == VgprSource::Kind::Ring) {
                    src.kind = VgprSource::Kind::PackHalf;
                    src.ring = {vgpr_src[inst.src[0].code].ring[0],
                                vgpr_src[inst.src[1].code].ring[0]};
                }
                vgpr_src[inst.dst[0].code] = src;
            }
            break;
        }
        case Gcn::Opcode::S_LOAD_DWORDX4: {
            const u32 dst = inst.dst[0].code;
            if (dst < sgpr_sharp.size() && inst.control.smrd.imm) {
                sgpr_sharp[dst] = {static_cast<s32>(inst.src[0].code * 2),
                                   static_cast<u32>(inst.control.smrd.offset)};
            }
            break;
        }
        case Gcn::Opcode::BUFFER_STORE_DWORD:
        case Gcn::Opcode::TBUFFER_STORE_FORMAT_X:
            add_xfb_store(inst, 1);
            break;
        case Gcn::Opcode::BUFFER_STORE_DWORDX2:
        case Gcn::Opcode::TBUFFER_STORE_FORMAT_XY:
            add_xfb_store(inst, 2);
            break;
        case Gcn::Opcode::BUFFER_STORE_DWORDX3:
        case Gcn::Opcode::TBUFFER_STORE_FORMAT_XYZ:
            add_xfb_store(inst, 3);
            break;
        case Gcn::Opcode::BUFFER_STORE_DWORDX4:
        case Gcn::Opcode::TBUFFER_STORE_FORMAT_XYZW:
            add_xfb_store(inst, 4);
            break;
        default:
            break;
        }
    }

    if (!IsPosition(last_attr)) {
        data.num_attrs = static_cast<u32>(last_attr) - static_cast<u32>(IR::Attribute::Param0) + 1;
    }
    if (data.attr_map.size() > 1) {
        const auto it = data.attr_map.begin();
        const u32 comp_stride = std::next(it)->first - it->first;
        data.output_vertices = comp_stride / 64;
    }

    return data;
}

} // namespace Shader
