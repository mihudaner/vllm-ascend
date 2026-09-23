/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INT32_REMAINDER_SCALAR_TILING_H_
#define INT32_REMAINDER_SCALAR_TILING_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Int32RemainderScalarTilingData)
TILING_DATA_FIELD_DEF(int64_t, elementNum);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int32_t, divisor);
TILING_DATA_FIELD_DEF(uint32_t, useAlignedCopy);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Int32RemainderScalar, Int32RemainderScalarTilingData)

struct Int32RemainderScalarCompileInfo {};
} // namespace optiling

#endif // INT32_REMAINDER_SCALAR_TILING_H_
