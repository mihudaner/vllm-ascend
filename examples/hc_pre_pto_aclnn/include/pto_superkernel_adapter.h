// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"

// PTO 生成核要求一个 AIC 和两个 AIV 共用核组编号。
// AscendC 返回逻辑核编号：AIV 除以任务比例得到核组，避免 SK 重映射后地址错位。
__aicore__ inline uint32_t HcPreGroupIdx() { return AscendC::GetBlockIdx() / AscendC::GetTaskRatio(); }
__aicore__ inline uint32_t HcPreGroupCount() { return AscendC::GetBlockNum(); }

// ISA 总头文件间接包含的辅助函数也使用原始核编号接口，SK 编译会检查它们。
// 仅在包含 ISA 时映射接口，随后撤销宏；不修改共享 ISA，也不影响后续源码。
// 适配依据固定 ISA 版本验证，不代表任意版本或所有通信接口都已验证。
#define get_block_idx HcPreGroupIdx
#define get_block_num HcPreGroupCount
#include "pto/pto-inst.hpp"
#undef get_block_num
#undef get_block_idx
