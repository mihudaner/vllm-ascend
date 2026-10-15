// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#define HCPRE_DEVICE __aicore__
#include "../include/contract.h"
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif
#include "pto/pto-inst.hpp"
namespace {
namespace hc_small_impl {
#include "../generated/hc_small.cpp"
}
namespace hc_medium48_impl {
#include "../generated/hc_medium48.cpp"
}
namespace hc_medium96_impl {
#include "../generated/hc_medium96.cpp"
}
namespace hc_exact96_impl {
#include "../generated/hc_exact96.cpp"
}
namespace hc_large144_impl {
#include "../generated/hc_large144.cpp"
}
namespace hc_large_impl {
#include "../generated/hc_large.cpp"
}
}
extern "C" __global__ __aicore__ void hc_pre_pto(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR base,
    GM_ADDR mixed, GM_ADDR post, GM_ADDR comb, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(data, tiling);
#if defined(__DAV_VEC__)
    const int clusterIdx = AscendC::GetBlockIdx() / 2;
    const int subblockIdx = AscendC::GetBlockIdx() % 2;
#else
    const int clusterIdx = AscendC::GetBlockIdx();
#endif
    const auto c=hcpre::MakeConfig(data.tokens);
    auto* ws=AscendC::GetUserWorkspace(workspace);
    auto* parts=reinterpret_cast<__gm__ float*>(ws);
    auto* rms=reinterpret_cast<__gm__ float*>(ws+c.partsBytes);
    auto* fifo=reinterpret_cast<__gm__ float*>(ws+c.partsBytes+c.rmsBytes)
        + clusterIdx*c.fifoElements;
#define PTR_ARGS parts, rms, reinterpret_cast<__gm__ float*>(weight), \
    reinterpret_cast<__gm__ bfloat16_t*>(x), reinterpret_cast<__gm__ float*>(scale), \
    parts, rms, reinterpret_cast<__gm__ float*>(base), reinterpret_cast<__gm__ float*>(comb), \
    reinterpret_cast<__gm__ float*>(scale), reinterpret_cast<__gm__ float*>(post), \
    reinterpret_cast<__gm__ bfloat16_t*>(mixed), fifo
#define MED48_PTR_ARGS parts, rms, reinterpret_cast<__gm__ bfloat16_t*>(x), \
    reinterpret_cast<__gm__ float*>(weight), reinterpret_cast<__gm__ float*>(scale), \
    parts, rms, reinterpret_cast<__gm__ float*>(base), reinterpret_cast<__gm__ float*>(comb), \
    reinterpret_cast<__gm__ float*>(scale), reinterpret_cast<__gm__ float*>(post), \
    reinterpret_cast<__gm__ bfloat16_t*>(mixed), fifo
#define TAIL_ARGS c.sink, c.rows, c.extraRows, c.pairsPerWorker, c.extraWorkers, \
    c.tokens, c.pairs, c.workers, c.prefix, c.workersPerPair, c.extraPairs, \
    c.padded, c.tokens, clusterIdx, 24
#define COMMON_ARGS PTR_ARGS, c.units, 24, c.padded, c.tokens, TAIL_ARGS
#define MED48_ARGS MED48_PTR_ARGS, c.units, 24, c.padded, c.tokens, TAIL_ARGS
#define EXACT_ARGS PTR_ARGS, c.units, 24, c.padded, TAIL_ARGS
    if (TILING_KEY_IS(0)) {
#if defined(__DAV_CUBE__)
        hc_small_impl::hc_small_aic(COMMON_ARGS);
#elif defined(__DAV_VEC__)
        hc_small_impl::hc_small_aiv(COMMON_ARGS, subblockIdx);
#endif
    }
    if (TILING_KEY_IS(1)) {
#if defined(__DAV_CUBE__)
        hc_medium48_impl::hc_medium48_aic(MED48_ARGS);
#elif defined(__DAV_VEC__)
        hc_medium48_impl::hc_medium48_aiv(MED48_ARGS, subblockIdx);
#endif
    }
    if (TILING_KEY_IS(2)) {
#if defined(__DAV_CUBE__)
        hc_medium96_impl::hc_medium96_aic(COMMON_ARGS);
#elif defined(__DAV_VEC__)
        hc_medium96_impl::hc_medium96_aiv(COMMON_ARGS, subblockIdx);
#endif
    }
    if (TILING_KEY_IS(3)) {
#if defined(__DAV_CUBE__)
        hc_exact96_impl::hc_exact96_aic(EXACT_ARGS);
#elif defined(__DAV_VEC__)
        hc_exact96_impl::hc_exact96_aiv(EXACT_ARGS, subblockIdx);
#endif
    }
    if (TILING_KEY_IS(4)) {
#if defined(__DAV_CUBE__)
        hc_large144_impl::hc_large144_aic(COMMON_ARGS);
#elif defined(__DAV_VEC__)
        hc_large144_impl::hc_large144_aiv(COMMON_ARGS, subblockIdx);
#endif
    }
    if (TILING_KEY_IS(5)) {
#if defined(__DAV_CUBE__)
        hc_large_impl::hc_large_aic(COMMON_ARGS);
#elif defined(__DAV_VEC__)
        hc_large_impl::hc_large_aiv(COMMON_ARGS, subblockIdx);
#endif
    }
}
