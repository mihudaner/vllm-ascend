/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

class Int32RemainderScalarAndAddKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR counter, GM_ADDR curIter, GM_ADDR counterOut,
        const Int32RemainderScalarAndAddTilingData* tilingData)
    {
        divisor_ = tilingData->divisor;
        increment_ = tilingData->increment;
        counterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counter), 1);
        curIterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(curIter), 1);
        counterOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counterOut), 1);
    }

    __aicore__ inline void Process()
    {
        const int32_t value = counterGm_.GetValue(0);
        int32_t remainder = value % divisor_;
        if (remainder < 0) {
            remainder += divisor_;
        }
        curIterGm_.SetValue(0, remainder);
        counterOutGm_.SetValue(0, value + increment_);
    }

private:
    GlobalTensor<int32_t> counterGm_;
    GlobalTensor<int32_t> curIterGm_;
    GlobalTensor<int32_t> counterOutGm_;
    int32_t divisor_ = 1;
    int32_t increment_ = 1;
};

extern "C" __global__ __aicore__ void int32_remainder_scalar_and_add(
    GM_ADDR counter, GM_ADDR curIter, GM_ADDR counterOut, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    Int32RemainderScalarAndAddKernel op;
    op.Init(counter, curIter, counterOut, &tilingData);
    op.Process();
}
