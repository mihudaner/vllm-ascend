/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <limits>

#include "int32_remainder_scalar_tiling.h"
#include "error/ops_error.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int64_t kInputX = 0;
constexpr int64_t kAttrDivisor = 0;
constexpr int64_t kMinDimension = 1;
constexpr int64_t kCoreMinInputElements = 2048;
constexpr int64_t kBlockAlignElements = 512;
constexpr int64_t kTileAlignElements = 128;
constexpr int64_t kDmaAlignElements = 8;
constexpr uint64_t kInputBytes = 4;
constexpr uint64_t kOutputBytes = 4;
constexpr uint64_t kCalcBufferCount = 0;
constexpr uint64_t kDoubleBufferBytesPerElement =
    2 * (kInputBytes + kOutputBytes) + kCalcBufferCount * sizeof(int32_t);

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

bool AlignUp(int64_t value, int64_t alignment, int64_t& result)
{
    if (value > std::numeric_limits<int64_t>::max() - (alignment - 1)) {
        return false;
    }
    result = CeilDiv(value, alignment) * alignment;
    return true;
}

bool CheckedProduct(const gert::Shape& shape, int64_t& elementNum)
{
    elementNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < kMinDimension ||
            elementNum > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        elementNum *= dim;
    }
    return true;
}
} // namespace

static ge::graphStatus TilingForInt32RemainderScalar(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("Int32RemainderScalar", "Tiling context is null"),
        return ge::GRAPH_FAILED);

    const gert::Tensor* xTensor = context->GetInputTensor(kInputX);
    OPS_LOG_E_IF_NULL(context, xTensor, return ge::GRAPH_FAILED);
    OPS_ERR_IF(xTensor->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(context, "Int32RemainderScalar requires INT32 input."),
        return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* divisorAttr = attrs->GetAttrPointer<int64_t>(kAttrDivisor);
    OPS_LOG_E_IF_NULL(context, divisorAttr, return ge::GRAPH_FAILED);
    OPS_ERR_IF(*divisorAttr <= 0 || *divisorAttr > std::numeric_limits<int32_t>::max(),
        OPS_LOG_E(context, "Int32RemainderScalar requires a positive int32 divisor."),
        return ge::GRAPH_FAILED);

    const gert::Shape& xShape = xTensor->GetStorageShape();
    int64_t elementNum = 0;
    OPS_ERR_IF(!CheckedProduct(xShape, elementNum),
        OPS_LOG_E(context, "Input dimensions must be positive and have an int64 element count."),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfo, return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    OPS_ERR_IF(availableCores == 0,
        OPS_LOG_E(context, "No AIV cores are available."),
        return ge::GRAPH_FAILED);

    const int64_t requestedCores = CeilDiv(elementNum, kCoreMinInputElements);
    const int64_t coreCount = std::max<int64_t>(
        1, std::min<int64_t>(availableCores, requestedCores));
    int64_t blockFormer = 0;
    OPS_ERR_IF(!AlignUp(CeilDiv(elementNum, coreCount), kBlockAlignElements, blockFormer),
        OPS_LOG_E(context, "Block size overflows int64."),
        return ge::GRAPH_FAILED);
    const int64_t blockNum = CeilDiv(elementNum, blockFormer);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OPS_ERR_IF(ubSize == 0,
        OPS_LOG_E(context, "UB size is zero."),
        return ge::GRAPH_FAILED);

    const uint64_t rawTileElements = ubSize / kDoubleBufferBytesPerElement;
    const uint64_t tileElements = rawTileElements / kTileAlignElements * kTileAlignElements;
    OPS_ERR_IF(tileElements == 0 || tileElements > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
        OPS_LOG_E(context, "UB is insufficient for a valid INT32 remainder tile."),
        return ge::GRAPH_FAILED);

    Int32RemainderScalarTilingData tiling;
    tiling.set_elementNum(elementNum);
    tiling.set_blockFormer(blockFormer);
    tiling.set_ubFormer(static_cast<int64_t>(tileElements));
    tiling.set_divisor(static_cast<int32_t>(*divisorAttr));
    tiling.set_useAlignedCopy(elementNum % kDmaAlignElements == 0);

    context->SetTilingKey(0);
    context->SetBlockDim(blockNum);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForInt32RemainderScalar(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Int32RemainderScalar)
    .Tiling(TilingForInt32RemainderScalar)
    .TilingParse<Int32RemainderScalarCompileInfo>(TilingPrepareForInt32RemainderScalar);
} // namespace optiling
