// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <map>
#include <vector>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/frontend/copy_shader.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/operand_helper.h"
#include "shader_recompiler/ir/position.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Optimization {

void RingAccessElimination(const IR::Program& program, const RuntimeInfo& runtime_info) {
    auto& info = program.info;

    const auto& ForEachInstruction = [&](auto func) {
        for (IR::Block* block : program.blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                func(ir, inst);
            }
        }
    };

    switch (program.info.hw_stage) {
    case HwStage::Local: {
        ForEachInstruction([=](IR::IREmitter& ir, IR::Inst& inst) {
            const auto opcode = inst.GetOpcode();
            switch (opcode) {
            case IR::Opcode::WriteSharedU64:
            case IR::Opcode::WriteSharedU32: {
                bool is_composite = opcode == IR::Opcode::WriteSharedU64;
                u32 num_components = opcode == IR::Opcode::WriteSharedU32 ? 1 : 2;

                ASSERT(inst.Arg(0).IsImmediate());

                u32 offset = inst.Arg(0).U32();
                IR::Value data =
                    is_composite ? ir.UnpackUint2x32(IR::U64{inst.Arg(1)}) : inst.Arg(1);
                for (s32 i = 0; i < num_components; i++) {
                    const auto attrib = IR::Attribute::Param0 + (offset / 16);
                    const auto comp = (offset / 4) % 4;
                    const IR::U32 value =
                        IR::U32{is_composite ? ir.CompositeExtract(data, i) : data};
                    ir.SetAttribute(attrib, ir.BitCast<IR::F32, IR::U32>(value), comp);
                    offset += 4;
                }
                inst.Invalidate();
                break;
            }
            default:
                break;
            }
        });
        break;
    }
    case HwStage::Export: {
        ForEachInstruction([=](IR::IREmitter& ir, IR::Inst& inst) {
            const auto opcode = inst.GetOpcode();
            switch (opcode) {
            case IR::Opcode::StoreBufferU32: {
                const auto info = inst.Flags<IR::BufferInstInfo>();
                if (!info.system_coherent || !info.globally_coherent) {
                    break;
                }

                const auto offset = inst.Flags<IR::BufferInstInfo>().inst_offset.Value();
                ASSERT(offset < runtime_info.hw.es.vertex_data_size * 4);
                const auto data = ir.BitCast<IR::F32>(IR::U32{inst.Arg(2)});
                const auto attrib =
                    IR::Value{offset < 16 ? IR::Attribute::Position0
                                          : IR::Attribute::Param0 + (offset / 16 - 1)};
                const u32 comp = (offset / 4) % 4;

                inst.ReplaceOpcode(IR::Opcode::SetAttribute);
                inst.ClearArgs();
                inst.SetArg(0, attrib);
                inst.SetArg(1, data);
                inst.SetArg(2, ir.Imm32(comp));
                break;
            }
            default:
                break;
            }
        });
        break;
    }
    case HwStage::Geometry: {
        const auto& gs_info = runtime_info.hw.gs;
        info.gs_copy_data = Shader::ParseCopyShader(gs_info.vs_copy);

        u32 dwords_per_vertex = gs_info.out_vertex_data_size;
        if (info.gs_copy_data.num_comps && info.gs_copy_data.num_comps > dwords_per_vertex) {
            LOG_WARNING(Render_Vulkan,
                        "VERT_ITEMSIZE {} is different than actual number of dwords per vertex {}",
                        dwords_per_vertex, info.gs_copy_data.num_comps);
            dwords_per_vertex = info.gs_copy_data.num_comps;
        }

        // Stream-out: each captured dword gets its own output (Param num_attrs + k) that the
        // SPIR-V backend decorates for transform feedback.
        auto& copy_data = info.gs_copy_data;
        const bool use_xfb = gs_info.xfb_buffer_mask != 0 && !copy_data.xfb_parse_failed &&
                             !copy_data.xfb_outputs.empty() &&
                             copy_data.num_attrs + copy_data.xfb_outputs.size() <= IR::NumParams;
        if (gs_info.xfb_buffer_mask != 0 && !use_xfb) {
            LOG_ERROR(Render_Recompiler,
                      "GS stream-out requested but copy shader {:#x} could not be mapped "
                      "(failed={}, outputs={}, attrs={})",
                      gs_info.vs_copy_hash, copy_data.xfb_parse_failed,
                      copy_data.xfb_outputs.size(), copy_data.num_attrs);
        }
        copy_data.xfb_active = use_xfb;
        if (gs_info.xfb_buffer_mask != 0) {
            XfbTrace(fmt::format("GS ring pass: copy={:#x} use_xfb={} outputs={} attrs={} "
                                 "attr_map={} out_vertices={} failed={}",
                                 gs_info.vs_copy_hash, use_xfb, copy_data.xfb_outputs.size(),
                                 copy_data.num_attrs, copy_data.attr_map.size(),
                                 copy_data.output_vertices, copy_data.xfb_parse_failed));
            for (const auto& out : copy_data.xfb_outputs) {
                XfbTrace(fmt::format("  xfb out buf{} +{} {} src {:#x}/{:#x}", out.buffer,
                                     out.offset, out.pack_half ? "pkrtz" : "dword", out.src[0],
                                     out.src[1]));
            }
            for (const auto& [ofs, a] : copy_data.attr_map) {
                XfbTrace(fmt::format("  attr ring {:#x} -> attr {} comp {}", ofs, u32(a.first),
                                     a.second));
            }
        }
        std::map<u32, std::vector<u32>> xfb_users; // ring offset -> captured dwords using it
        std::vector<u32> xfb_pending(copy_data.xfb_outputs.size(), 0);
        std::map<u32, IR::F32> xfb_latest;
        if (use_xfb) {
            for (u32 k = 0; k < copy_data.xfb_outputs.size(); k++) {
                const auto& out = copy_data.xfb_outputs[k];
                xfb_users[out.src[0]].push_back(k);
                if (out.pack_half && out.src[1] != out.src[0]) {
                    xfb_users[out.src[1]].push_back(k);
                }
            }
        }
        const auto capture_xfb = [&](IR::IREmitter& ir, u32 ring_ofs, const IR::F32& data) {
            const auto it = xfb_users.find(ring_ofs);
            if (it == xfb_users.end()) {
                return;
            }
            xfb_latest.insert_or_assign(ring_ofs, data);
            for (const u32 k : it->second) {
                const auto& out = copy_data.xfb_outputs[k];
                if (out.src[0] == ring_ofs) {
                    xfb_pending[k] |= 1u;
                }
                if (out.pack_half && out.src[1] == ring_ofs) {
                    xfb_pending[k] |= 2u;
                }
                const u32 needed = out.pack_half ? 3u : 1u;
                if ((xfb_pending[k] & needed) != needed) {
                    continue;
                }
                xfb_pending[k] = 0;
                IR::F32 value = xfb_latest.at(out.src[0]);
                if (out.pack_half) {
                    const IR::Value vec =
                        ir.CompositeConstruct(xfb_latest.at(out.src[0]), xfb_latest.at(out.src[1]));
                    value = ir.BitCast<IR::F32>(ir.Pack2x16(AmdGpu::NumberFormat::Float, vec));
                }
                ir.SetAttribute(IR::Attribute::Param0 + static_cast<int>(copy_data.num_attrs + k),
                                value, 0);
            }
        };

        // A stream-out copy shader may export a constant position (e.g. 0 so nothing is
        // rasterized). Write zero then, instead of leaving the GS position undefined.
        const bool zero_position =
            use_xfb && std::ranges::none_of(copy_data.attr_map, [](const auto& entry) {
                return IR::IsPosition(entry.second.first);
            });

        ForEachInstruction([&](IR::IREmitter& ir, IR::Inst& inst) {
            const auto opcode = inst.GetOpcode();
            switch (opcode) {
            case IR::Opcode::EmitVertex: {
                if (zero_position) {
                    for (u32 comp = 0; comp < 4; comp++) {
                        ExportPosition(ir, gs_info, false, IR::Attribute::Position0, comp,
                                       ir.Imm32(0.f));
                    }
                }
                break;
            }
            case IR::Opcode::LoadBufferU32: {
                const auto info = inst.Flags<IR::BufferInstInfo>();
                if (!info.system_coherent || !info.globally_coherent) {
                    break;
                }

                const auto vertex_id = (info.index_enable ? IR::GetBufferIndexArg(&inst)
                                                          : IR::GetBufferVOffsetArg(&inst))
                                           .U32() >>
                                       2;
                const auto soffset = IR::GetBufferSOffsetArg(&inst);
                const auto bucket = soffset.U32() / 256u;
                const auto attrib = bucket < 4 ? IR::Attribute::Position0
                                               : IR::Attribute::Param0 + (bucket / 4 - 1);
                const auto comp = bucket % 4;

                auto attr_value = ir.GetAttribute(attrib, comp, vertex_id);
                inst.ReplaceOpcode(IR::Opcode::BitCastU32F32);
                inst.ClearArgs();
                inst.SetArg(0, attr_value);
                break;
            }
            case IR::Opcode::StoreBufferU32: {
                const auto buffer_info = inst.Flags<IR::BufferInstInfo>();
                if (!buffer_info.system_coherent || !buffer_info.globally_coherent) {
                    break;
                }

                const auto offset = inst.Flags<IR::BufferInstInfo>().inst_offset.Value();
                const auto data = ir.BitCast<IR::F32>(IR::U32{inst.Arg(2)});
                const auto comp_ofs = info.gs_copy_data.output_vertices * sizeof(u32);
                const auto output_size = gs_info.output_vertices * dwords_per_vertex * sizeof(u32);

                const auto vc_read_ofs = (((offset / comp_ofs) * comp_ofs) % output_size) * 16u;
                if (use_xfb) {
                    capture_xfb(ir, vc_read_ofs, data);
                }
                const auto& it = info.gs_copy_data.attr_map.find(vc_read_ofs);
                if (use_xfb && it == info.gs_copy_data.attr_map.cend()) {
                    // Component only goes to the stream-out buffer.
                    inst.Invalidate();
                    break;
                }
                ASSERT(it != info.gs_copy_data.attr_map.cend());
                const auto& [attr, comp] = it->second;

                inst.Invalidate();
                if (IsPosition(attr)) {
                    ExportPosition(ir, gs_info, false, attr, comp, data);
                } else {
                    ir.SetAttribute(attr, data, comp);
                }
                break;
            }
            default:
                break;
            }
        });
        break;
    }
    default:
        break;
    }
}

} // namespace Shader::Optimization
