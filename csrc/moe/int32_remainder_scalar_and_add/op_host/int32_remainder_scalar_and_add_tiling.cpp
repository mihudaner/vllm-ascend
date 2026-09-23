/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>

#include "int32_remainder_scalar_and_add_tiling.h"
#include "error/ops_error.h"

namespace optiling {
namespace {
constexpr int64_t kInputCounter = 0;
constexpr int64_t kAttrDivisor = 0;
constexpr int64_t kAttrIncrement = 1;
constexpr int64_t kMinDimension = 1;
constexpr int64_t kCounterElements = 1;

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

static ge::graphStatus TilingForInt32RemainderScalarAndAdd(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("Int32RemainderScalarAndAdd", "Tiling context is null"),
        return ge::GRAPH_FAILED);

    const gert::Tensor* counterTensor = context->GetInputTensor(kInputCounter);
    OPS_LOG_E_IF_NULL(context, counterTensor, return ge::GRAPH_FAILED);
    OPS_ERR_IF(counterTensor->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(context, "Int32RemainderScalarAndAdd requires INT32 counter."),
        return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* divisorAttr = attrs->GetAttrPointer<int64_t>(kAttrDivisor);
    OPS_LOG_E_IF_NULL(context, divisorAttr, return ge::GRAPH_FAILED);
    OPS_ERR_IF(*divisorAttr <= 0 || *divisorAttr > std::numeric_limits<int32_t>::max(),
        OPS_LOG_E(context, "Int32RemainderScalarAndAdd requires a positive int32 divisor."),
        return ge::GRAPH_FAILED);
    const int64_t* incrementAttr = attrs->GetAttrPointer<int64_t>(kAttrIncrement);
    const int64_t increment = incrementAttr == nullptr ? 1 : *incrementAttr;
    OPS_ERR_IF(increment < std::numeric_limits<int32_t>::min() ||
                   increment > std::numeric_limits<int32_t>::max(),
        OPS_LOG_E(context, "Int32RemainderScalarAndAdd requires an int32 increment."),
        return ge::GRAPH_FAILED);

    const gert::Shape& counterShape = counterTensor->GetStorageShape();
    int64_t elementNum = 0;
    OPS_ERR_IF(!CheckedProduct(counterShape, elementNum),
        OPS_LOG_E(context, "Counter dimensions must be positive and have an int64 element count."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(elementNum != kCounterElements,
        OPS_LOG_E(context, "Int32RemainderScalarAndAdd supports only a single-element counter."),
        return ge::GRAPH_FAILED);

    Int32RemainderScalarAndAddTilingData tiling;
    tiling.set_divisor(static_cast<int32_t>(*divisorAttr));
    tiling.set_increment(static_cast<int32_t>(increment));

    context->SetTilingKey(0);
    context->SetBlockDim(1);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForInt32RemainderScalarAndAdd(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Int32RemainderScalarAndAdd)
    .Tiling(TilingForInt32RemainderScalarAndAdd)
    .TilingParse<Int32RemainderScalarAndAddCompileInfo>(TilingPrepareForInt32RemainderScalarAndAdd);
} // namespace optiling
