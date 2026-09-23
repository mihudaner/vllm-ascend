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

#include "int32_gather_tiling.h"
#include "error/ops_error.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int64_t kInputMapping = 0;
constexpr int64_t kInputIndices = 1;
constexpr int64_t kElementsPerCore = 24;
constexpr int64_t kElementsPerCopy = 8;

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

bool CheckedProduct(const gert::Shape& shape, int64_t& elementNum)
{
    elementNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0 || elementNum > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        elementNum *= dim;
    }
    return true;
}
} // namespace

static ge::graphStatus TilingForInt32Gather(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("Int32Gather", "Tiling context is null"),
        return ge::GRAPH_FAILED);

    const gert::Tensor* mappingTensor = context->GetInputTensor(kInputMapping);
    const gert::Tensor* indicesTensor = context->GetInputTensor(kInputIndices);
    OPS_LOG_E_IF_NULL(context, mappingTensor, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, indicesTensor, return ge::GRAPH_FAILED);
    OPS_ERR_IF(mappingTensor->GetDataType() != ge::DT_INT32 ||
                   indicesTensor->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(context, "Int32Gather requires INT32 mapping and indices."),
        return ge::GRAPH_FAILED);

    const gert::Shape& mappingShape = mappingTensor->GetStorageShape();
    OPS_ERR_IF(mappingShape.GetDimNum() != 1 || mappingShape.GetDim(0) <= 0,
        OPS_LOG_E(context, "Int32Gather requires a non-empty 1-D mapping."),
        return ge::GRAPH_FAILED);

    int64_t outputElementNum = 0;
    OPS_ERR_IF(!CheckedProduct(indicesTensor->GetStorageShape(), outputElementNum),
        OPS_LOG_E(context, "Indices dimensions must be positive and have an int64 element count."),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfo, return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    OPS_ERR_IF(availableCores == 0,
        OPS_LOG_E(context, "No AIV cores are available."),
        return ge::GRAPH_FAILED);

    const int64_t blockNum = std::min<int64_t>(
        availableCores, CeilDiv(outputElementNum, kElementsPerCore));
    const int64_t blockElementNum =
        CeilDiv(CeilDiv(outputElementNum, blockNum), kElementsPerCopy) * kElementsPerCopy;

    Int32GatherTilingData tiling;
    tiling.set_mappingElementNum(mappingShape.GetDim(0));
    tiling.set_outputElementNum(outputElementNum);
    tiling.set_blockElementNum(blockElementNum);

    context->SetTilingKey(0);
    context->SetBlockDim(blockNum);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForInt32Gather(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Int32Gather)
    .Tiling(TilingForInt32Gather)
    .TilingParse<Int32GatherCompileInfo>(TilingPrepareForInt32Gather);
} // namespace optiling
