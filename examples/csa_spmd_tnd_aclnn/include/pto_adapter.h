// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
__aicore__ inline uint32_t CsaGroupIdx(){return AscendC::GetBlockIdx()/AscendC::GetTaskRatio();}
__aicore__ inline uint32_t CsaGroupCount(){return AscendC::GetBlockNum();}
#define get_block_idx CsaGroupIdx
#define get_block_num CsaGroupCount
#include "pto/pto-inst.hpp"
#undef get_block_num
#undef get_block_idx
