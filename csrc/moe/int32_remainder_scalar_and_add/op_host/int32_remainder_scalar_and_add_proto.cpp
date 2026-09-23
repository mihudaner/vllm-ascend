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
constexpr int32_t kInputCounter = 0;
constexpr int32_t kOutputCurIter = 0;
constexpr int32_t kOutputCounter = 1;
}

static ge::graphStatus InferShape4Int32RemainderScalarAndAdd(gert::InferShapeContext* context)
{
    const gert::Shape* counterShape = context->GetInputShape(kInputCounter);
    OPS_LOG_E_IF_NULL(context, counterShape, return ge::GRAPH_FAILED);
    gert::Shape* curIterShape = context->GetOutputShape(kOutputCurIter);
    OPS_LOG_E_IF_NULL(context, curIterShape, return ge::GRAPH_FAILED);
    gert::Shape* counterOutShape = context->GetOutputShape(kOutputCounter);
    OPS_LOG_E_IF_NULL(context, counterOutShape, return ge::GRAPH_FAILED);

    const size_t dimNum = counterShape->GetDimNum();
    curIterShape->SetDimNum(dimNum);
    counterOutShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; ++i) {
        const int64_t dim = counterShape->GetDim(i);
        curIterShape->SetDim(i, dim);
        counterOutShape->SetDim(i, dim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4Int32RemainderScalarAndAdd(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(kOutputCurIter, ge::DT_INT32);
    context->SetOutputDataType(kOutputCounter, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Int32RemainderScalarAndAdd)
    .InferShape(InferShape4Int32RemainderScalarAndAdd)
    .InferDataType(InferDtype4Int32RemainderScalarAndAdd);
} // namespace ops
