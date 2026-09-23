/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "error/ops_error.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr int32_t kInputIndices = 1;
constexpr int32_t kOutput = 0;
}

static ge::graphStatus InferShape4Int32Gather(gert::InferShapeContext* context)
{
    const gert::Shape* indicesShape = context->GetInputShape(kInputIndices);
    OPS_LOG_E_IF_NULL(context, indicesShape, return ge::GRAPH_FAILED);
    gert::Shape* outputShape = context->GetOutputShape(kOutput);
    OPS_LOG_E_IF_NULL(context, outputShape, return ge::GRAPH_FAILED);

    const size_t dimNum = indicesShape->GetDimNum();
    outputShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        outputShape->SetDim(i, indicesShape->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4Int32Gather(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kOutput, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Int32Gather)
    .InferShape(InferShape4Int32Gather)
    .InferDataType(InferDtype4Int32Gather);
} // namespace ops
