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

class Int64ToInt32StaticCastKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, const Int64ToInt32StaticCastTilingData* tilingData)
    {
        const int64_t start = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFormer;
        const int64_t remaining = tilingData->elementNum - start;
        blockElementNum_ = remaining > tilingData->blockFormer ? tilingData->blockFormer : remaining;
        tileElementNum_ = tilingData->ubFormer;
        useAlignedCopy_ = tilingData->useAlignedCopy != 0;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(x) + start, blockElementNum_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y) + start, blockElementNum_);
        pipe_.InitBuffer(xQueue_, 2, static_cast<uint32_t>(tileElementNum_ * sizeof(int64_t)));
        pipe_.InitBuffer(yQueue_, 2, static_cast<uint32_t>(tileElementNum_ * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        for (int64_t offset = 0; offset < blockElementNum_; offset += tileElementNum_) {
            const int64_t remaining = blockElementNum_ - offset;
            const uint32_t count = static_cast<uint32_t>(
                remaining > tileElementNum_ ? tileElementNum_ : remaining);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count)
    {
        LocalTensor<int64_t> xLocal = xQueue_.AllocTensor<int64_t>();
        if (useAlignedCopy_) {
            DataCopy(xLocal, xGm_[offset], count);
        } else {
            DataCopyExtParams copyParams{
                1, count * static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
            DataCopyPadExtParams<int64_t> padParams{
                false, 0, 0, static_cast<int64_t>(0)};
            DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        }
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<int64_t> xLocal = xQueue_.DeQue<int64_t>();
        LocalTensor<int32_t> yLocal = yQueue_.AllocTensor<int32_t>();
        Cast(yLocal, xLocal, RoundMode::CAST_NONE, count);
        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count)
    {
        LocalTensor<int32_t> yLocal = yQueue_.DeQue<int32_t>();
        if (useAlignedCopy_) {
            DataCopy(yGm_[offset], yLocal, count);
        } else {
            DataCopyExtParams copyParams{
                1, count * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        yQueue_.FreeTensor(yLocal);
    }

    TPipe pipe_;
    GlobalTensor<int64_t> xGm_;
    GlobalTensor<int32_t> yGm_;
    TQue<TPosition::VECIN, 1> xQueue_;
    TQue<TPosition::VECOUT, 1> yQueue_;
    int64_t blockElementNum_ = 0;
    int64_t tileElementNum_ = 0;
    bool useAlignedCopy_ = false;
};

extern "C" __global__ __aicore__ void int64_to_int32_static_cast(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    Int64ToInt32StaticCastKernel op;
    op.Init(x, y, &tilingData);
    op.Process();
}
