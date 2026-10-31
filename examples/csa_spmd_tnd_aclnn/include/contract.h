// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <initializer_list>
#include "policy_generated.h"
namespace csa {
// Descriptor checks only: never read device metadata to choose a variant.
template <typename Size>
bool ValidDynamicShape(int64_t variant, Size size) {
  if (variant < 0 || variant >= kVariantCount) return false;
  const auto t = size(0, 0), b = size(36, 0);
  if (t <= 0 || b <= 0) return false;
  for (int i : {10, 11, 14, 21, 24, 37, 54})
    if (size(i, 0) != t) return false;
  for (int i : {19, 23, 33, 44, 45, 46})
    if (size(i, 0) != b) return false;
  if (size(22, 0) != b + 1 || size(35, 0) != b + 1) return false;
  if (size(26, 0) != size(28, 0) || size(27, 0) != size(28, 0)) return false;
  if (size(39, 0) != size(41, 0) || size(40, 0) != size(41, 0)) return false;
  if (size(18, 1) < 8192 || size(18, 1) % 16 != 0) return false;
  if (size(32, 1) < 1040 || size(32, 1) % 16 != 0) return false;
  if (size(43, 1) < 4160 || size(43, 1) % 64 != 0) return false;
  // Wider tables use the upstream device-side logical-address fallback.
  return true;
}
}  // namespace csa
