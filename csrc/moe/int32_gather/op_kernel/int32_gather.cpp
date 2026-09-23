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

class Int32GatherKernel {
public:
    static constexpr int64_t kElementsPerCopy = 8;

    __aicore__ inline void Init(
        GM_ADDR mapping,
        GM_ADDR indices,
        GM_ADDR output,
        const Int32GatherTilingData* tilingData)
    {
        const int64_t start =
            static_cast<int64_t>(GetBlockIdx()) * tilingData->blockElementNum;
        const int64_t remaining = tilingData->outputElementNum - start;
        blockElementNum_ =
            remaining > tilingData->blockElementNum ? tilingData->blockElementNum : remaining;

        mappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(mapping));
        indicesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(indices) + start, blockElementNum_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(output) + start, blockElementNum_);
        mappingElementNum_ = tilingData->mappingElementNum;
        pipe_.InitBuffer(indicesBuf_, kElementsPerCopy * sizeof(int32_t));
        pipe_.InitBuffer(outputBuf_, kElementsPerCopy * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
        LocalTensor<int32_t> outputLocal = outputBuf_.Get<int32_t>();
        for (int64_t offset = 0; offset < blockElementNum_; offset += kElementsPerCopy) {
            const int64_t remaining = blockElementNum_ - offset;
            const int64_t validElements =
                remaining > kElementsPerCopy ? kElementsPerCopy : remaining;
            DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(validElements * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
            DataCopyPad(indicesLocal, indicesGm_[offset], copyParams, padParams);
            event_t mte2ToScalar =
                static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
            for (int64_t i = 0; i < validElements; ++i) {
                int32_t index = indicesLocal.GetValue(i);
                if (index < 0) {
                    index += mappingElementNum_;
                }
                outputLocal.SetValue(i, mappingGm_.GetValue(index));
            }
            event_t scalarToMte2 =
                static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(scalarToMte2);
            WaitFlag<HardEvent::S_MTE2>(scalarToMte2);
            event_t scalarToMte3 =
                static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(scalarToMte3);
            WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
            DataCopyPad(outputGm_[offset], outputLocal, copyParams);
            event_t mte3ToScalar =
                static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(mte3ToScalar);
            WaitFlag<HardEvent::MTE3_S>(mte3ToScalar);
        }
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECIN> indicesBuf_;
    TBuf<QuePosition::VECOUT> outputBuf_;
    GlobalTensor<int32_t> mappingGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<int32_t> outputGm_;
    int64_t mappingElementNum_ = 0;
    int64_t blockElementNum_ = 0;
};

extern "C" __global__ __aicore__ void int32_gather(
    GM_ADDR mapping, GM_ADDR indices, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    Int32GatherKernel op;
    op.Init(mapping, indices, output, &tilingData);
    op.Process();
}
