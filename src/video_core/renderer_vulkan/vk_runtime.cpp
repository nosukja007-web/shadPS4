// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"

#include <vulkan/vulkan_format_traits.hpp>

namespace Vulkan {

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Cube:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE_MSG("Unexpected image type {}", u32(type));
    }
}

static std::pair<u32, u32> SanitizeCopyLayers(const VideoCore::ImageInfo& src_info,
                                              const VideoCore::ImageInfo& dst_info,
                                              const u32 depth) {
    const auto vk_src_type = ConvertImageType(src_info.type);
    const auto vk_dst_type = ConvertImageType(dst_info.type);

    u32 src_layers = src_info.resources.layers;
    u32 dst_layers = dst_info.resources.layers;

    // 3D images can only use 1 layer.
    if (vk_src_type == vk::ImageType::e3D && src_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D source layers {} to 1.", src_layers);
        src_layers = 1;
    }
    if (vk_dst_type == vk::ImageType::e3D && dst_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D destination layers {} to 1.", dst_layers);
        dst_layers = 1;
    }

    // If the image type is equal, layer count must match. Take the minimum of both.
    if (vk_src_type == vk_dst_type) {
        if (src_layers != dst_layers) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy source layers {} and destination layers {} to minimum.",
                        src_layers, dst_layers);
            src_layers = dst_layers = std::min(src_layers, dst_layers);
        }
    } else {
        // For 2D <-> 3D copies, 2D layer count must equal 3D depth.
        if (vk_src_type == vk::ImageType::e2D && vk_dst_type == vk::ImageType::e3D &&
            src_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D source layers {} to 3D destination depth {}", src_layers,
                        depth);
            src_layers = depth;
        }
        if (vk_src_type == vk::ImageType::e3D && vk_dst_type == vk::ImageType::e2D &&
            dst_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D destination layers {} to 3D source depth {}", dst_layers,
                        depth);
            dst_layers = depth;
        }
    }

    return std::make_pair(src_layers, dst_layers);
}

static u32 BufferImageCopySize(const vk::BufferImageCopy& copy, const vk::Format pixel_format) {
    const u32 row_length = copy.bufferRowLength ? copy.bufferRowLength : copy.imageExtent.width;
    const u32 height = copy.bufferImageHeight ? copy.bufferImageHeight : copy.imageExtent.height;

    const auto block = vk::blockExtent(pixel_format);
    const u32 block_size = vk::blockSize(pixel_format);
    const u32 row_pitch = (row_length / block[0]) * block_size;
    const u32 slice_pitch = (height / block[1]) * row_pitch;

    const u32 width_in_blocks = (copy.imageExtent.width + block[0] - 1) / block[0];
    const u32 height_in_blocks = (copy.imageExtent.height + block[1] - 1) / block[1];
    const u32 num_slices = copy.imageExtent.depth * copy.imageSubresource.layerCount;

    return (num_slices - 1) * slice_pitch + (height_in_blocks - 1) * row_pitch +
           width_in_blocks * block_size;
}

/// Returns the staging buffer size CopyImageWithBuffer needs for both the source download and
/// the destination upload.
static u64 ImageWithBufferCopySize(const VideoCore::Image* src, const VideoCore::Image* dst) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);
    const u32 num_layers = std::min(src->info.resources.layers, dst->info.resources.layers);
    u64 size = 0;
    for (u32 mip = 0; mip < num_mips; ++mip) {
        const vk::BufferImageCopy copy = {
            .imageSubresource{.layerCount = num_layers},
            .imageExtent = {src->HostWidth(mip), src->HostHeight(mip),
                            std::max(src->info.size.depth >> mip, 1u)},
        };
        const u64 offset = mip == 0 ? 0 : Common::AlignUp(size, 16);
        size = offset + std::max<u64>(BufferImageCopySize(copy, src->info.pixel_format),
                                      BufferImageCopySize(copy, dst->info.pixel_format));
    }
    return size;
}

/// Filter used when resampling between native and scaled copies of an image.
static vk::Filter ScaleFilter(const VideoCore::Image* image) {
    if (image->info.props.is_depth) {
        return vk::Filter::eNearest;
    }
    const std::string_view numeric = vk::componentNumericFormat(image->info.pixel_format, 0);
    if (numeric == "UINT" || numeric == "SINT") {
        return vk::Filter::eNearest;
    }
    return vk::Filter::eLinear;
}

/// Builds per-mip blit regions mapping src_width x src_height (at mip 0) to dst_width x
/// dst_height across all layers.
static SmallVector<vk::ImageBlit, 8> MakeScaleBlits(vk::ImageAspectFlags aspect, u32 num_mips,
                                                    u32 src_layer, u32 dst_layer, u32 num_layers,
                                                    u32 src_width, u32 src_height, u32 dst_width,
                                                    u32 dst_height) {
    SmallVector<vk::ImageBlit, 8> blits;
    for (u32 mip = 0; mip < num_mips; ++mip) {
        const s32 sw = static_cast<s32>(std::max(src_width >> mip, 1u));
        const s32 sh = static_cast<s32>(std::max(src_height >> mip, 1u));
        const s32 dw = static_cast<s32>(std::max(dst_width >> mip, 1u));
        const s32 dh = static_cast<s32>(std::max(dst_height >> mip, 1u));
        blits.push_back(vk::ImageBlit{
            .srcSubresource{
                .aspectMask = aspect,
                .mipLevel = mip,
                .baseArrayLayer = src_layer,
                .layerCount = num_layers,
            },
            .srcOffsets = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{sw, sh, 1}}},
            .dstSubresource{
                .aspectMask = aspect,
                .mipLevel = mip,
                .baseArrayLayer = dst_layer,
                .layerCount = num_layers,
            },
            .dstOffsets = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{dw, dh, 1}}},
        });
    }
    return blits;
}

/// Returns true and logs once if a copy between src and dst cannot be done because their host
/// scales differ.
static bool ScaleMismatch(const VideoCore::Image* src, const VideoCore::Image* dst,
                          std::string_view op) {
    if (src->scale == dst->scale) {
        return false;
    }
    LOG_WARNING(Render_Vulkan, "{}: skipping copy between scale {} {:#x} and scale {} {:#x}", op,
                src->scale, src->info.guest_address, dst->scale, dst->info.guest_address);
    return true;
}

Runtime::Runtime(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, staging_pool{instance_, scheduler_} {
    blit_helper = std::make_unique<VideoCore::BlitHelper>(instance, scheduler);

    memory_barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
    memory_barrier.dstAccessMask =
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
}

void Runtime::TickFrame() {
    staging_pool.TickFrame();
}

void Runtime::CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                         std::span<const vk::BufferCopy> copies) {
    scheduler.EndRendering();

    bool needs_flush{};
    for (const auto& copy : copies) {
        needs_flush |= IsBufferAccessed(src, copy.srcOffset, copy.size);
        needs_flush |= IsBufferAccessed(dst, copy.dstOffset, copy.size, true);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBuffer(src->Handle(), dst->Handle(), copies);

    for (const auto& copy : copies) {
        AccessBuffer(src, copy.srcOffset, copy.size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferRead);
        AccessBuffer(dst, copy.dstOffset, copy.size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferWrite);
    }
}

void Runtime::FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value) {
    scheduler.EndRendering();

    if (IsBufferAccessed(dst, offset, size, true)) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.fillBuffer(dst->Handle(), offset, size, value);

    AccessBuffer(dst, offset, size, vk::PipelineStageFlagBits2::eClear,
                 vk::AccessFlagBits2::eTransferWrite);
}

void Runtime::InlineData(VideoCore::Buffer* dst, u64 offset, u32 value) {
    scheduler.EndRendering();

    if (IsBufferAccessed(dst, offset, sizeof(value), true)) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.updateBuffer(dst->Handle(), offset, sizeof(value), &value);

    AccessBuffer(dst, offset, sizeof(value), vk::PipelineStageFlagBits2::eCopy,
                 vk::AccessFlagBits2::eTransferWrite);
}

bool Runtime::Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                      vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                      std::optional<VideoCore::SubresourceRange> subres_range) {
    const size_t prev_num_barriers = static_cast<size_t>(image_barriers.size());
    image->GetBarriers(image_barriers, dst_layout, dst_access, dst_stage, subres_range);
    return image_barriers.size() != prev_num_barriers;
}

VideoCore::UniqueImage Runtime::CreateNativeStaging(const VideoCore::Image* image) {
    auto image_ci = image->backing->image.image_ci;
    image_ci.flags = {};
    image_ci.extent.width = image->info.size.width;
    image_ci.extent.height = image->info.size.height;
    image_ci.samples = vk::SampleCountFlagBits::e1;
    image_ci.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    image_ci.initialLayout = vk::ImageLayout::eUndefined;
    VideoCore::UniqueImage staging{instance.GetDevice(), instance.GetAllocator()};
    staging.Create(image_ci);
    return staging;
}

void Runtime::TransitStaging(vk::Image image, vk::ImageAspectFlags aspect,
                             vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                             vk::AccessFlags2 src_access, vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .srcAccessMask = src_access,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
}

void Runtime::UploadScaledImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                                std::span<const vk::BufferImageCopy> upload_copies) {
    // Guest data is native sized, so upload into a native staging image and upscale it.
    auto staging = CreateNativeStaging(dst);
    const auto aspect = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllTransfer,
            vk::AccessFlagBits2::eTransferWrite);
    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        IsBufferAccessed(src, copy.bufferOffset, copy_size);
    }
    FlushBarriers();

    const auto cmdbuf = scheduler.CommandBuffer();
    TransitStaging(staging.image, aspect, vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eNone,
                   vk::AccessFlagBits2::eTransferWrite);
    cmdbuf.copyBufferToImage(src->Handle(), staging.image, vk::ImageLayout::eTransferDstOptimal,
                             upload_copies);
    TransitStaging(staging.image, aspect, vk::ImageLayout::eTransferDstOptimal,
                   vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferWrite,
                   vk::AccessFlagBits2::eTransferRead);

    const auto blits = MakeScaleBlits(aspect, dst->info.resources.levels, 0, 0,
                                      dst->info.resources.layers, dst->info.size.width,
                                      dst->info.size.height, dst->HostWidth(), dst->HostHeight());
    cmdbuf.blitImage(staging.image, vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, blits, ScaleFilter(dst));

    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        AccessBuffer(src, copy.bufferOffset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferRead);
    }
    scheduler.DeferOperation([staging = std::move(staging)] {});

    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::DownloadScaledImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                                  std::span<const vk::BufferImageCopy> download_copies) {
    // Guest memory is native sized, so downscale into a native staging image first.
    auto staging = CreateNativeStaging(src);
    const auto aspect = src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eAllTransfer,
            vk::AccessFlagBits2::eTransferRead);
    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        IsBufferAccessed(dst, copy.bufferOffset, copy_size, true);
    }
    FlushBarriers();

    const auto cmdbuf = scheduler.CommandBuffer();
    TransitStaging(staging.image, aspect, vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eNone,
                   vk::AccessFlagBits2::eTransferWrite);
    const auto blits = MakeScaleBlits(
        aspect, src->info.resources.levels, 0, 0, src->info.resources.layers, src->HostWidth(),
        src->HostHeight(), src->info.size.width, src->info.size.height);
    cmdbuf.blitImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, staging.image,
                     vk::ImageLayout::eTransferDstOptimal, blits, ScaleFilter(src));
    TransitStaging(staging.image, aspect, vk::ImageLayout::eTransferDstOptimal,
                   vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferWrite,
                   vk::AccessFlagBits2::eTransferRead);
    cmdbuf.copyImageToBuffer(staging.image, vk::ImageLayout::eTransferSrcOptimal, dst->Handle(),
                             download_copies);

    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        AccessBuffer(dst, copy.bufferOffset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferWrite);
    }
    scheduler.DeferOperation([staging = std::move(staging)] {});
}

void Runtime::BlitScaledImage(VideoCore::Image* src, VideoCore::Image* dst) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);
    const u32 num_layers = std::min(src->info.resources.layers, dst->info.resources.layers);
    const auto aspect = src->aspect_mask & dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    if (!aspect || src->info.props.is_block || dst->info.props.is_block ||
        src->info.type == AmdGpu::ImageType::Color3D ||
        dst->info.type == AmdGpu::ImageType::Color3D) {
        ScaleMismatch(src, dst, "BlitScaledImage");
        return;
    }

    scheduler.EndRendering();
    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eAllTransfer,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |=
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eAllTransfer,
                vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    // The source covers its own native size in the destination's host space.
    const u32 dst_width = std::min(src->info.size.width * dst->scale, dst->HostWidth());
    const u32 dst_height = std::min(src->info.size.height * dst->scale, dst->HostHeight());
    const auto blits = MakeScaleBlits(aspect, num_mips, 0, 0, num_layers, src->HostWidth(),
                                      src->HostHeight(), dst_width, dst_height);
    scheduler.CommandBuffer().blitImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        dst->GetImage(), vk::ImageLayout::eTransferDstOptimal,
                                        blits, ScaleFilter(dst));

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                          std::span<const vk::BufferImageCopy> upload_copies) {
    SetBackingSamples(dst, dst->info.num_samples, false);
    scheduler.EndRendering();

    if (dst->IsScaled()) {
        UploadScaledImage(dst, src, upload_copies);
        return;
    }

    bool needs_flush =
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferWrite);
    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        needs_flush |= IsBufferAccessed(src, copy.bufferOffset, copy_size);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBufferToImage(src->Handle(), dst->GetImage(), vk::ImageLayout::eTransferDstOptimal,
                             upload_copies);

    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        AccessBuffer(src, copy.bufferOffset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferRead);
    }

    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                            std::span<const vk::BufferImageCopy> download_copies) {
    SetBackingSamples(src, src->info.num_samples);
    scheduler.EndRendering();

    if (src->IsScaled()) {
        DownloadScaledImage(src, dst, download_copies);
        return;
    }

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        needs_flush |= IsBufferAccessed(dst, copy.bufferOffset, copy_size, true);
    }
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImageToBuffer(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->Handle(),
                             download_copies);

    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        AccessBuffer(dst, copy.bufferOffset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                     vk::AccessFlagBits2::eTransferWrite);
    }
}

void Runtime::CopyImage(VideoCore::Image* src, VideoCore::Image* dst) {
    if (src->scale != dst->scale) {
        BlitScaledImage(src, dst);
        return;
    }
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);

    // Format mismatch warning (safe but useful)
    if (src->info.pixel_format != dst->info.pixel_format) {
        LOG_DEBUG(Render_Vulkan,
                  "Copy between different formats: src={}, dst={}. "
                  "Result may be undefined.",
                  vk::to_string(src->info.pixel_format), vk::to_string(dst->info.pixel_format));
    }

    const u32 base_width = src->HostWidth();
    const u32 base_height = src->HostHeight();
    const u32 base_depth =
        dst->info.type == AmdGpu::ImageType::Color3D ? dst->info.size.depth : src->info.size.depth;

    // Match sample count before copying
    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    SmallVector<vk::ImageCopy, 8> regions;

    const vk::ImageAspectFlags src_aspect = src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    const vk::ImageAspectFlags dst_aspect = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    const bool src_is_2d = ConvertImageType(src->info.type) == vk::ImageType::e2D;
    const bool src_is_3d = ConvertImageType(src->info.type) == vk::ImageType::e3D;

    const bool dst_is_2d = ConvertImageType(dst->info.type) == vk::ImageType::e2D;
    const bool dst_is_3d = ConvertImageType(dst->info.type) == vk::ImageType::e3D;

    const bool is_2d_to_3d = src_is_2d && dst_is_3d;
    const bool is_3d_to_2d = src_is_3d && dst_is_2d;
    const bool is_same_type = !is_2d_to_3d && !is_3d_to_2d;

    for (u32 mip = 0; mip < num_mips; ++mip) {
        const u32 mip_w = std::max(base_width >> mip, 1u);
        const u32 mip_h = std::max(base_height >> mip, 1u);
        const u32 mip_d = std::max(base_depth >> mip, 1u);

        const auto [src_layers, dst_layers] = SanitizeCopyLayers(src->info, dst->info, mip_d);

        vk::ImageCopy region{};
        region.srcSubresource.aspectMask = src_aspect;
        region.srcSubresource.mipLevel = mip;
        region.srcSubresource.baseArrayLayer = 0;
        region.dstSubresource.aspectMask = dst_aspect;
        region.dstSubresource.mipLevel = mip;
        region.dstSubresource.baseArrayLayer = 0;

        if (is_same_type) {
            // 2D->2D OR 3D->3D
            if (src_is_3d) {
                // 3D images must use layerCount=1
                region.srcSubresource.layerCount = 1;
                region.dstSubresource.layerCount = 1;
                region.extent = vk::Extent3D(mip_w, mip_h, mip_d);
            } else {
                // Array images
                const u32 copy_layers = std::min(src_layers, dst_layers);
                region.srcSubresource.layerCount = copy_layers;
                region.dstSubresource.layerCount = copy_layers;
                region.extent = vk::Extent3D(mip_w, mip_h, 1);
            }
        } else if (is_2d_to_3d) {
            // 2D array -> 3D volume
            region.srcSubresource.layerCount = src_layers;
            region.dstSubresource.layerCount = 1;
            region.extent = vk::Extent3D(mip_w, mip_h, src_layers);
        } else if (is_3d_to_2d) {
            // 3D volume -> 2D array
            region.srcSubresource.layerCount = 1;
            region.dstSubresource.layerCount = dst_layers;
            region.extent = vk::Extent3D(mip_w, mip_h, dst_layers);
        }

        regions.push_back(region);
    }

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, regions);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                                  const VideoCore::Buffer* buffer, u64 offset) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);
    const u32 num_layers = std::min(src->info.resources.layers, dst->info.resources.layers);

    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    vk::BufferImageCopy buffer_copy = {
        .bufferOffset = offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
            .mipLevel = 0u,
            .baseArrayLayer = 0,
            .layerCount = num_layers,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {src->HostWidth(), src->HostHeight(), src->info.size.depth},
    };
    u64 copy_size = BufferImageCopySize(buffer_copy, src->info.pixel_format);
    SmallVector<vk::BufferImageCopy, 8> buffer_copies;
    buffer_copies.push_back(buffer_copy);
    for (u32 mip = 1; mip < num_mips; ++mip) {
        auto mip_copy = buffer_copy;
        mip_copy.bufferOffset = Common::AlignUp(offset + copy_size, 16);
        mip_copy.imageSubresource.mipLevel = mip;
        mip_copy.imageExtent = vk::Extent3D{src->HostWidth(mip), src->HostHeight(mip),
                                            std::max(src->info.size.depth >> mip, 1u)};
        copy_size =
            mip_copy.bufferOffset - offset + BufferImageCopySize(mip_copy, src->info.pixel_format);
        buffer_copies.push_back(mip_copy);
    }

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    needs_flush |= IsBufferAccessed(buffer, offset, copy_size);
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImageToBuffer(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             buffer->Handle(), buffer_copies);

    const vk::MemoryBarrier2 post_copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .memoryBarrierCount = 1u,
        .pMemoryBarriers = &post_copy_barrier,
    });

    for (auto& copy : buffer_copies) {
        copy.imageSubresource.aspectMask = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    }
    cmdbuf.copyBufferToImage(buffer->Handle(), dst->GetImage(),
                             vk::ImageLayout::eTransferDstOptimal, buffer_copies);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice) {
    if (ScaleMismatch(src, dst, "CopyMip")) {
        return;
    }
    const auto dst_dim = dst->info.props.is_block ? 2 : 0;
    const auto mip_block_w = std::max(dst->info.size.width >> (mip + dst_dim), 1u);
    const auto mip_block_h = std::max(dst->info.size.height >> (mip + dst_dim), 1u);
    const auto mip_block_p = std::max<u32>(dst->info.mips_layout[mip].pitch >> dst_dim, 1u);

    const auto src_dim = src->info.props.is_block ? 2 : 0;
    ASSERT(mip_block_w == (src->info.size.width >> src_dim));
    ASSERT(mip_block_h == (src->info.size.height >> src_dim));
    ASSERT(mip_block_p == (src->info.pitch >> src_dim));

    const auto [src_layers, dst_layers] =
        SanitizeCopyLayers(src->info, dst->info, src->info.size.depth);

    const vk::ImageCopy image_copy{
        .srcSubresource{
            .aspectMask = src->aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = src_layers,
        },
        .dstSubresource{
            .aspectMask = src->aspect_mask,
            .mipLevel = mip,
            .baseArrayLayer = slice,
            .layerCount = dst_layers,
        },
        .extent = {src->HostWidth(), src->HostHeight(), src->info.size.depth},
    };

    SetBackingSamples(dst, dst->info.num_samples);
    SetBackingSamples(src, src->info.num_samples);

    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, image_copy);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst) {
    if (src->info.num_samples == 1 && dst->info.num_samples == 1) {
        if ((instance.IsMaintenance8Supported() ||
             src->info.props.is_depth == dst->info.props.is_depth) &&
            src->aspect_mask == dst->aspect_mask && src->scale == dst->scale) {
            CopyImage(src, dst);
        } else if (ScaleMismatch(src, dst, "CopyColorAndDepth")) {
            return;
        } else {
            // Perform depth from/to color copy using the intermediate copy buffer.
            static constexpr size_t COPY_BUFFER_SIZE = 128_MB;
            const u64 copy_size = ImageWithBufferCopySize(src, dst);
            if (copy_size > COPY_BUFFER_SIZE) {
                LOG_WARNING(Render_Vulkan,
                            "Depth/color copy needs {:#x} bytes: src {} {}x{}x{} layers {} mips "
                            "{}, dst {} {}x{}x{} layers {} mips {}",
                            copy_size, vk::to_string(src->info.pixel_format), src->info.size.width,
                            src->info.size.height, src->info.size.depth, src->info.resources.layers,
                            src->info.resources.levels, vk::to_string(dst->info.pixel_format),
                            dst->info.size.width, dst->info.size.height, dst->info.size.depth,
                            dst->info.resources.layers, dst->info.resources.levels);
            }
            const auto copy_ref = staging_pool.Request(std::max<u64>(copy_size, COPY_BUFFER_SIZE),
                                                       VideoCore::MemoryType::DeviceLocal);
            CopyImageWithBuffer(src, dst, copy_ref.buffer, copy_ref.offset);
        }
    } else if (src->info.num_samples == 1 && dst->info.num_samples > 1 &&
               dst->info.props.is_depth) {
        // Perform a rendering pass to transfer the channels of source as samples in dest.
        bool needs_flush =
            Transit(src, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);
        needs_flush |= Transit(dst, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                               vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                   vk::PipelineStageFlagBits2::eLateFragmentTests,
                               vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
        if (needs_flush) {
            FlushBarriers();
        }
        if (ScaleMismatch(src, dst, "ReinterpretColorAsMsDepth")) {
            return;
        }

        blit_helper->ReinterpretColorAsMsDepth(
            dst->HostWidth(), dst->HostHeight(), dst->info.num_samples, src->info.pixel_format,
            dst->info.pixel_format, src->GetImage(), dst->GetImage());
    } else {
        LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
    }
}

void Runtime::CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                               const VideoCore::SubresourceRange& sub_range,
                               std::optional<vk::Rect2D> copy_rect) {
    if (ScaleMismatch(src, dst, "CopyDepthStencil")) {
        return;
    }
    scheduler.EndRendering();

    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto aspect_mask = src->aspect_mask & dst->aspect_mask;

    vk::ImageCopy region = {
        .srcSubresource{
            .aspectMask = aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = sub_range.base.layer,
            .layerCount = sub_range.extent.layers,
        },
        .srcOffset = {0, 0, 0},
        .dstSubresource{
            .aspectMask = aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = sub_range.base.layer,
            .layerCount = sub_range.extent.layers,
        },
        .dstOffset = {0, 0, 0},
        .extent = {dst->HostWidth(), dst->HostHeight(), 1},
    };
    if (copy_rect) {
        // The rectangle is in guest pixels and both images share the same scale.
        const s32 scale = static_cast<s32>(dst->scale);
        region.srcOffset = vk::Offset3D{copy_rect->offset.x * scale, copy_rect->offset.y * scale, 0};
        region.dstOffset = region.srcOffset;
        region.extent = vk::Extent3D{copy_rect->extent.width * scale,
                                     copy_rect->extent.height * scale, 1};
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, region);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyStencilToColor(VideoCore::Image* src, VideoCore::Image* dst) {
    ASSERT(src->aspect_mask & vk::ImageAspectFlagBits::eStencil);
    ASSERT(src->scale == dst->scale);
    SetBackingSamples(src, src->info.num_samples);

    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);
    const u32 num_layers = std::min(src->info.resources.layers, dst->info.resources.layers);

    // Stencil aspect data in buffers is tightly packed, one byte per texel. Offsets of
    // depth/stencil buffer copies must be multiples of 4.
    SmallVector<vk::BufferImageCopy, 8> buffer_copies;
    u64 copy_size = 0;
    for (u32 mip = 0; mip < num_mips; ++mip) {
        const u32 width = src->HostWidth(mip);
        const u32 height = src->HostHeight(mip);
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = copy_size,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        });
        copy_size = Common::AlignUp(copy_size + u64(width) * height * num_layers, 16);
    }

    const auto copy_ref = staging_pool.Request(copy_size, VideoCore::MemoryType::DeviceLocal, 16);
    for (auto& copy : buffer_copies) {
        copy.bufferOffset += copy_ref.offset;
    }

    scheduler.EndRendering();

    // Apply any pending transitions first so this image's transitions below are not batched
    // together with an earlier transition of the same image.
    FlushBarriers();
    bool needs_flush =
        Transit(src, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eCopy,
                vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite);
    needs_flush |= IsBufferAccessed(copy_ref.buffer, copy_ref.offset, copy_size, true);
    if (needs_flush) {
        FlushBarriers();
    }

    auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyImageToBuffer(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             copy_ref.buffer->Handle(), buffer_copies);

    const vk::MemoryBarrier2 post_copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1u,
        .pMemoryBarriers = &post_copy_barrier,
    });

    for (auto& copy : buffer_copies) {
        copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    }
    cmdbuf.copyBufferToImage(copy_ref.buffer->Handle(), dst->GetImage(),
                             vk::ImageLayout::eTransferDstOptimal, buffer_copies);

    AccessBuffer(copy_ref.buffer, copy_ref.offset, copy_size, vk::PipelineStageFlagBits2::eCopy,
                 vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                           const VideoCore::SubresourceRange& src_range,
                           const VideoCore::SubresourceRange& dst_range) {
    SetBackingSamples(dst, 1, false);
    scheduler.EndRendering();

    const bool needs_resolve = src->backing->num_samples != 1;
    const auto dst_stage =
        needs_resolve ? vk::PipelineStageFlagBits2::eResolve : vk::PipelineStageFlagBits2::eCopy;

    bool needs_flush = Transit(src, vk::ImageLayout::eTransferSrcOptimal, dst_stage,
                               vk::AccessFlagBits2::eTransferRead);
    needs_flush |= Transit(dst, vk::ImageLayout::eTransferDstOptimal, dst_stage,
                           vk::AccessFlagBits2::eTransferWrite);
    if (needs_flush) {
        FlushBarriers();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto [src_layers, dst_layers] = SanitizeCopyLayers(src->info, dst->info, 1);
    if (src->scale != dst->scale) {
        ResolveScaledImage(src, dst, src_range.base.layer, dst_range.base.layer,
                           std::min(src_layers, dst_layers), needs_resolve);
    } else if (!needs_resolve) {
        const vk::ImageCopy region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = src_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = dst_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = {src->HostWidth(), src->HostHeight(), 1},
        };
        cmdbuf.copyImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                         vk::ImageLayout::eTransferDstOptimal, region);
    } else {
        const vk::ImageResolve region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = src_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = dst_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = {src->HostWidth(), src->HostHeight(), 1},
        };
        cmdbuf.resolveImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                            vk::ImageLayout::eTransferDstOptimal, region);
    }

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::ResolveScaledImage(VideoCore::Image* src, VideoCore::Image* dst, u32 src_layer,
                                 u32 dst_layer, u32 num_layers, bool needs_resolve) {
    const auto cmdbuf = scheduler.CommandBuffer();
    constexpr auto aspect = vk::ImageAspectFlagBits::eColor;
    const u32 dst_width = std::min(src->info.size.width * dst->scale, dst->HostWidth());
    const u32 dst_height = std::min(src->info.size.height * dst->scale, dst->HostHeight());

    if (!needs_resolve) {
        const auto blits =
            MakeScaleBlits(aspect, 1, src_layer, dst_layer, num_layers, src->HostWidth(),
                           src->HostHeight(), dst_width, dst_height);
        cmdbuf.blitImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                         vk::ImageLayout::eTransferDstOptimal, blits, ScaleFilter(dst));
        return;
    }

    // Multisampled sources are never scaled: resolve natively, then upscale into dst.
    auto staging = CreateNativeStaging(dst);
    TransitStaging(staging.image, aspect, vk::ImageLayout::eUndefined,
                   vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eNone,
                   vk::AccessFlagBits2::eTransferWrite);
    const u32 width = std::min(src->HostWidth(), dst->info.size.width);
    const u32 height = std::min(src->HostHeight(), dst->info.size.height);
    const vk::ImageResolve region = {
        .srcSubresource{
            .aspectMask = aspect,
            .mipLevel = 0,
            .baseArrayLayer = src_layer,
            .layerCount = num_layers,
        },
        .srcOffset = {0, 0, 0},
        .dstSubresource{
            .aspectMask = aspect,
            .mipLevel = 0,
            .baseArrayLayer = dst_layer,
            .layerCount = num_layers,
        },
        .dstOffset = {0, 0, 0},
        .extent = {width, height, 1},
    };
    cmdbuf.resolveImage(src->GetImage(), vk::ImageLayout::eTransferSrcOptimal, staging.image,
                        vk::ImageLayout::eTransferDstOptimal, region);
    TransitStaging(staging.image, aspect, vk::ImageLayout::eTransferDstOptimal,
                   vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferWrite,
                   vk::AccessFlagBits2::eTransferRead);
    const auto blits = MakeScaleBlits(aspect, 1, dst_layer, dst_layer, num_layers, width, height,
                                      width * dst->scale, height * dst->scale);
    cmdbuf.blitImage(staging.image, vk::ImageLayout::eTransferSrcOptimal, dst->GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, blits, ScaleFilter(dst));
    scheduler.DeferOperation([staging = std::move(staging)] {});
}

void Runtime::ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                         const vk::ClearValue& clear_value) {
    scheduler.EndRendering();

    const bool needs_flush =
        Transit(dst, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eClear,
                vk::AccessFlagBits2::eTransferWrite, range);
    if (needs_flush) {
        FlushBarriers();
    }

    const vk::ImageSubresourceRange vk_range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = range.base.level,
        .levelCount = range.extent.levels,
        .baseArrayLayer = range.base.layer,
        .layerCount = range.extent.layers,
    };
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.clearColorImage(dst->GetImage(), vk::ImageLayout::eTransferDstOptimal, clear_value.color,
                           vk_range);

    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing) {
    auto& backing = image->backing;
    auto& backing_images = image->backing_images;
    const auto& info = image->info;
    if (!backing || backing->num_samples == num_samples) {
        return;
    }
    ASSERT_MSG(!image->info.props.is_depth, "Swapping samples is only valid for color images");
    VideoCore::Image::BackingImage* new_backing;
    auto it = std::ranges::find(backing_images, num_samples,
                                &VideoCore::Image::BackingImage::num_samples);
    if (it == backing_images.end()) {
        const auto previous_size = Common::AlignUp(image->SizeBytes(), 1024);
        auto new_image_ci = backing->image.image_ci;
        new_image_ci.samples = LiverpoolToVK::NumSamples(num_samples, image->supported_samples);

        new_backing = &backing_images.emplace_back();
        new_backing->num_samples = num_samples;
        new_backing->image = VideoCore::UniqueImage{instance.GetDevice(), instance.GetAllocator()};
        new_backing->image.Create(new_image_ci);
        if (image->memory_usage_counter) {
            *image->memory_usage_counter +=
                Common::AlignUp(image->SizeBytes(), 1024) - previous_size;
        }

        Vulkan::SetObjectName(instance.GetDevice(), new_backing->image.image,
                              "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{} (backing)",
                              info.size.width, info.size.height, info.size.depth,
                              AmdGpu::NameOf(info.tile_mode), vk::to_string(info.pixel_format),
                              info.guest_address, info.guest_size, info.resources.layers,
                              info.resources.levels, num_samples);
    } else {
        new_backing = std::addressof(*it);
    }

    if (copy_backing) {
        scheduler.EndRendering();
        ASSERT(image->info.resources.levels == 1 && image->info.resources.layers == 1);

        // Transition current backing to shader read layout
        Transit(image, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead);

        // Transition dest backing to color attachment layout, not caring of previous contents
        constexpr auto dst_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        constexpr auto dst_access =
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
        constexpr auto dst_layout = vk::ImageLayout::eColorAttachmentOptimal;
        image_barriers.push_back(vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = new_backing->image,
            .subresourceRange{
                .aspectMask = image->aspect_mask,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = image->info.resources.layers,
            },
        });
        FlushBarriers();

        // Copy between ms and non ms backing images
        blit_helper->CopyBetweenMsImages(
            image->HostWidth(), image->HostHeight(), new_backing->num_samples, info.pixel_format,
            backing->num_samples > 1, backing->image, new_backing->image);

        // Update current layout in tracker to new backings layout
        new_backing->state.layout = dst_layout;
        new_backing->state.access_mask = dst_access;
        new_backing->state.pl_stage = dst_stage;
    }

    backing = new_backing;
}

bool Runtime::IsBufferAccessed(const VideoCore::Buffer* handle, u64 offset, u64 size,
                               bool check_read_access) {
    MakeCurrent(handle);
    bool has_access = resource->write_ranges.Overlaps(offset, offset + size);
    if (check_read_access && !has_access) {
        has_access |= resource->read_ranges.Overlaps(offset, offset + size);
    }
    return has_access;
}

void Runtime::AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size,
                           vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access) {
    MakeCurrent(handle);

    const Interval range = {
        .start = offset,
        .end = offset + size,
    };

    constexpr static vk::AccessFlags2 READ_MASK =
        vk::AccessFlagBits2::eIndexRead | vk::AccessFlagBits2::eVertexAttributeRead |
        vk::AccessFlagBits2::eUniformRead | vk::AccessFlagBits2::eShaderRead |
        vk::AccessFlagBits2::eColorAttachmentRead |
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eTransferRead |
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eTransformFeedbackCounterReadEXT;

    constexpr static vk::AccessFlags2 WRITE_MASK =
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eColorAttachmentWrite |
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eTransferWrite |
        vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eTransformFeedbackWriteEXT |
        vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT;

    if (src_access & WRITE_MASK) {
        resource->write_ranges.Add(range);
    }
    if (src_access & READ_MASK) {
        resource->read_ranges.Add(range);
    }

    memory_barrier.srcStageMask |= src_stage;
    memory_barrier.srcAccessMask |= src_access & WRITE_MASK;
}

void Runtime::FlushBarriers() {
    vk::DependencyInfo dep_info{};

    if (memory_barrier.srcStageMask) {
        dep_info.pMemoryBarriers = &memory_barrier;
        dep_info.memoryBarrierCount = 1U;
    }
    if (!image_barriers.empty()) {
        dep_info.pImageMemoryBarriers = image_barriers.data();
        dep_info.imageMemoryBarrierCount = static_cast<u32>(image_barriers.size());
    }

    if (!dep_info.memoryBarrierCount && !dep_info.imageMemoryBarrierCount) {
        return;
    }

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(dep_info);

    memory_barrier.srcStageMask = vk::PipelineStageFlagBits2::eNone;
    memory_barrier.srcAccessMask = vk::AccessFlagBits2::eNone;

    image_barriers.clear();
    for (auto& resource : resources) {
        resource.read_ranges.Clear();
        resource.write_ranges.Clear();
    }
    resources.clear();
    resource = nullptr;
}

void Runtime::MakeCurrent(const VideoCore::Buffer* handle) {
    if (resource && resource->handle == handle) {
        return;
    }
    auto it = std::ranges::find(resources, handle, &BufferBarriers::handle);
    if (it != resources.end()) {
        resource = std::addressof(*it);
        return;
    }
    resource = &resources.emplace_back(handle);
}

} // namespace Vulkan
