// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_address_tracker.h"

namespace Vulkan::AddressTracker {

namespace {

struct Binding {
    vk::ObjectType type;
    u64 handle;
    u64 base;
    u64 size;
    bool bound;
    u64 seq;
};

constexpr size_t MaxBindings = 1 << 16;

std::mutex mutex;
std::deque<Binding> bindings;
std::unordered_map<u64, std::string> names;
u64 next_seq{};

bool IsTrackedType(vk::ObjectType type) {
    return type == vk::ObjectType::eBuffer || type == vk::ObjectType::eImage ||
           type == vk::ObjectType::eDeviceMemory;
}

std::string_view NameOf(u64 handle) {
    const auto it = names.find(handle);
    return it != names.end() ? std::string_view{it->second} : std::string_view{"<unnamed>"};
}

VKAPI_ATTR VkBool32 VKAPI_CALL Callback(vk::DebugUtilsMessageSeverityFlagBitsEXT,
                                        vk::DebugUtilsMessageTypeFlagsEXT type,
                                        const vk::DebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (!(type & vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding) || !data) {
        return VK_FALSE;
    }
    const auto* next = static_cast<const vk::BaseInStructure*>(data->pNext);
    while (next && next->sType != vk::StructureType::eDeviceAddressBindingCallbackDataEXT) {
        next = next->pNext;
    }
    if (!next || data->objectCount == 0) {
        return VK_FALSE;
    }
    const auto* binding = reinterpret_cast<const vk::DeviceAddressBindingCallbackDataEXT*>(next);
    const auto& object = data->pObjects[0];

    std::scoped_lock lk{mutex};
    if (object.pObjectName) {
        names[object.objectHandle] = object.pObjectName;
    }
    bindings.push_back(Binding{
        .type = object.objectType,
        .handle = object.objectHandle,
        .base = binding->baseAddress,
        .size = binding->size,
        .bound = binding->bindingType == vk::DeviceAddressBindingTypeEXT::eBind,
        .seq = next_seq++,
    });
    if (bindings.size() > MaxBindings) {
        bindings.pop_front();
    }
    return VK_FALSE;
}

} // Anonymous namespace

vk::UniqueDebugUtilsMessengerEXT CreateMessenger(vk::Instance instance) {
    const vk::DebugUtilsMessengerCreateInfoEXT msg_ci = {
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding,
        .pfnUserCallback = Callback,
    };
    auto [result, messenger] = instance.createDebugUtilsMessengerEXTUnique(msg_ci);
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Failed to create address binding messenger: {}",
                    vk::to_string(result));
        return {};
    }
    return std::move(messenger);
}

void RecordName(vk::ObjectType type, u64 handle, std::string_view name) {
    if (!IsTrackedType(type)) {
        return;
    }
    std::scoped_lock lk{mutex};
    names[handle] = std::string{name};
}

void LogAddress(u64 address) {
    std::scoped_lock lk{mutex};
    LOG_CRITICAL(Render_Vulkan, "Address bindings near {:#x} ({} recorded):", address,
                 bindings.size());
    const Binding* below{};
    const Binding* above{};
    for (const auto& b : bindings) {
        if (address >= b.base && address < b.base + b.size) {
            LOG_CRITICAL(Render_Vulkan, "  #{} {} {} {:#x} '{}' [{:#x}, {:#x}) size {:#x}", b.seq,
                         b.bound ? "BIND  " : "UNBIND", vk::to_string(b.type), b.handle,
                         NameOf(b.handle), b.base, b.base + b.size, b.size);
            continue;
        }
        if (!b.bound) {
            continue;
        }
        if (b.base + b.size <= address && (!below || b.base > below->base)) {
            below = &b;
        }
        if (b.base > address && (!above || b.base < above->base)) {
            above = &b;
        }
    }
    for (const auto* b : {below, above}) {
        if (b) {
            LOG_CRITICAL(Render_Vulkan, "  nearest #{} {} {:#x} '{}' [{:#x}, {:#x}) size {:#x}",
                         b->seq, vk::to_string(b->type), b->handle, NameOf(b->handle), b->base,
                         b->base + b->size, b->size);
        }
    }
}

} // namespace Vulkan::AddressTracker
