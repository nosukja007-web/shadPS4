// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan::AddressTracker {

/// Creates a debug messenger that records VK_EXT_device_address_binding_report events.
vk::UniqueDebugUtilsMessengerEXT CreateMessenger(vk::Instance instance);

/// Remembers the debug name of a buffer, image or memory object.
void RecordName(vk::ObjectType type, u64 handle, std::string_view name);

/// Logs every known GPU address binding that covers or neighbours the given address.
void LogAddress(u64 address);

} // namespace Vulkan::AddressTracker
