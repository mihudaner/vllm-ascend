// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "op_api_common.h"
#include "../include/contract.h"
namespace {
void Check(at::TensorList a,int64_t variant){
 TORCH_CHECK(a.size()==57,"Expected 57 diagnostic entry buffers");
 TORCH_CHECK(variant==0,"Invalid variant");
 for(const auto& t:a){TORCH_CHECK(t.numel()>0,"Empty CSA tensor");TORCH_CHECK(t.device()==a[0].device(),"Mixed devices");
 TORCH_CHECK(t.is_contiguous(),"Contiguous buffers required");}
TORCH_CHECK(a[0].scalar_type()==at::kBFloat16 && a[0].dim()==2, "Invalid dtype/rank: x");
TORCH_CHECK(a[0].size(1)==16384,"Invalid shape: x");
TORCH_CHECK(a[1].scalar_type()==at::kFloat && a[1].dim()==2, "Invalid dtype/rank: hc_fn");
TORCH_CHECK(a[1].size(0)==24,"Invalid shape: hc_fn");
TORCH_CHECK(a[1].size(1)==16384,"Invalid shape: hc_fn");
TORCH_CHECK(a[2].scalar_type()==at::kFloat && a[2].dim()==1, "Invalid dtype/rank: hc_scale");
TORCH_CHECK(a[2].size(0)==3,"Invalid shape: hc_scale");
TORCH_CHECK(a[3].scalar_type()==at::kFloat && a[3].dim()==1, "Invalid dtype/rank: hc_base");
TORCH_CHECK(a[3].size(0)==24,"Invalid shape: hc_base");
TORCH_CHECK(a[4].scalar_type()==at::kFloat && a[4].dim()==2, "Invalid dtype/rank: hc_base_2d");
TORCH_CHECK(a[4].size(0)==1,"Invalid shape: hc_base_2d");
TORCH_CHECK(a[4].size(1)==24,"Invalid shape: hc_base_2d");
TORCH_CHECK(a[5].scalar_type()==at::kBFloat16 && a[5].dim()==1, "Invalid dtype/rank: norm_w");
TORCH_CHECK(a[5].size(0)==4096,"Invalid shape: norm_w");
TORCH_CHECK(a[6].scalar_type()==at::kBFloat16 && a[6].dim()==2, "Invalid dtype/rank: wq_a");
TORCH_CHECK(a[6].size(0)==1024,"Invalid shape: wq_a");
TORCH_CHECK(a[6].size(1)==4096,"Invalid shape: wq_a");
TORCH_CHECK(a[7].scalar_type()==at::kBFloat16 && a[7].dim()==1, "Invalid dtype/rank: gamma_cq");
TORCH_CHECK(a[7].size(0)==1024,"Invalid shape: gamma_cq");
TORCH_CHECK(a[8].scalar_type()==at::kChar && a[8].dim()==2, "Invalid dtype/rank: wq_b");
TORCH_CHECK(a[8].size(0)==1024,"Invalid shape: wq_b");
TORCH_CHECK(a[8].size(1)==32768,"Invalid shape: wq_b");
TORCH_CHECK(a[9].scalar_type()==at::kFloat && a[9].dim()==1, "Invalid dtype/rank: wq_b_scale");
TORCH_CHECK(a[9].size(0)==32768,"Invalid shape: wq_b_scale");
TORCH_CHECK(a[10].scalar_type()==at::kFloat && a[10].dim()==2, "Invalid dtype/rank: rope_cos");
TORCH_CHECK(a[10].size(1)==64,"Invalid shape: rope_cos");
TORCH_CHECK(a[11].scalar_type()==at::kFloat && a[11].dim()==2, "Invalid dtype/rank: rope_sin");
TORCH_CHECK(a[11].size(1)==64,"Invalid shape: rope_sin");
TORCH_CHECK(a[12].scalar_type()==at::kBFloat16 && a[12].dim()==2, "Invalid dtype/rank: wkv");
TORCH_CHECK(a[12].size(0)==512,"Invalid shape: wkv");
TORCH_CHECK(a[12].size(1)==4096,"Invalid shape: wkv");
TORCH_CHECK(a[13].scalar_type()==at::kBFloat16 && a[13].dim()==1, "Invalid dtype/rank: gamma_ckv");
TORCH_CHECK(a[13].size(0)==512,"Invalid shape: gamma_ckv");
TORCH_CHECK(a[14].scalar_type()==at::kInt && a[14].dim()==2, "Invalid dtype/rank: native_slots");
TORCH_CHECK(a[14].size(1)==2,"Invalid shape: native_slots");
TORCH_CHECK(a[15].scalar_type()==at::kBFloat16 && a[15].dim()==2, "Invalid dtype/rank: cache_rows");
TORCH_CHECK(a[15].size(1)==512,"Invalid shape: cache_rows");
TORCH_CHECK(a[16].scalar_type()==at::kBFloat16 && a[16].dim()==2, "Invalid dtype/rank: main_wkv");
TORCH_CHECK(a[16].size(0)==1024,"Invalid shape: main_wkv");
TORCH_CHECK(a[16].size(1)==4096,"Invalid shape: main_wkv");
TORCH_CHECK(a[17].scalar_type()==at::kBFloat16 && a[17].dim()==2, "Invalid dtype/rank: main_wgate");
TORCH_CHECK(a[17].size(0)==1024,"Invalid shape: main_wgate");
TORCH_CHECK(a[17].size(1)==4096,"Invalid shape: main_wgate");
TORCH_CHECK(a[18].scalar_type()==at::kFloat && a[18].dim()==2, "Invalid dtype/rank: main_state");
TORCH_CHECK(a[19].scalar_type()==at::kInt && a[19].dim()==2, "Invalid dtype/rank: main_state_table");
TORCH_CHECK(a[20].scalar_type()==at::kFloat && a[20].dim()==2, "Invalid dtype/rank: main_ape");
TORCH_CHECK(a[20].size(0)==4,"Invalid shape: main_ape");
TORCH_CHECK(a[20].size(1)==1024,"Invalid shape: main_ape");
TORCH_CHECK(a[21].scalar_type()==at::kLong && a[21].dim()==1, "Invalid dtype/rank: positions");
TORCH_CHECK(a[22].scalar_type()==at::kInt && a[22].dim()==1, "Invalid dtype/rank: main_bounds");
TORCH_CHECK(a[23].scalar_type()==at::kInt && a[23].dim()==1, "Invalid dtype/rank: main_lengths");
TORCH_CHECK(a[24].scalar_type()==at::kInt && a[24].dim()==2, "Invalid dtype/rank: main_state_slots");
TORCH_CHECK(a[24].size(1)==2,"Invalid shape: main_state_slots");
TORCH_CHECK(a[25].scalar_type()==at::kBFloat16 && a[25].dim()==1, "Invalid dtype/rank: main_gamma");
TORCH_CHECK(a[25].size(0)==512,"Invalid shape: main_gamma");
TORCH_CHECK(a[26].scalar_type()==at::kFloat && a[26].dim()==2, "Invalid dtype/rank: main_cos");
TORCH_CHECK(a[26].size(1)==64,"Invalid shape: main_cos");
TORCH_CHECK(a[27].scalar_type()==at::kFloat && a[27].dim()==2, "Invalid dtype/rank: main_sin");
TORCH_CHECK(a[27].size(1)==64,"Invalid shape: main_sin");
TORCH_CHECK(a[28].scalar_type()==at::kInt && a[28].dim()==2, "Invalid dtype/rank: main_slots");
TORCH_CHECK(a[28].size(1)==2,"Invalid shape: main_slots");
TORCH_CHECK(a[29].scalar_type()==at::kBFloat16 && a[29].dim()==2, "Invalid dtype/rank: main_cache");
TORCH_CHECK(a[29].size(1)==512,"Invalid shape: main_cache");
TORCH_CHECK(a[30].scalar_type()==at::kBFloat16 && a[30].dim()==2, "Invalid dtype/rank: idx_wkv");
TORCH_CHECK(a[30].size(0)==256,"Invalid shape: idx_wkv");
TORCH_CHECK(a[30].size(1)==4096,"Invalid shape: idx_wkv");
TORCH_CHECK(a[31].scalar_type()==at::kBFloat16 && a[31].dim()==2, "Invalid dtype/rank: idx_wgate");
TORCH_CHECK(a[31].size(0)==256,"Invalid shape: idx_wgate");
TORCH_CHECK(a[31].size(1)==4096,"Invalid shape: idx_wgate");
TORCH_CHECK(a[32].scalar_type()==at::kFloat && a[32].dim()==2, "Invalid dtype/rank: idx_state");
TORCH_CHECK(a[33].scalar_type()==at::kInt && a[33].dim()==2, "Invalid dtype/rank: idx_state_table");
TORCH_CHECK(a[34].scalar_type()==at::kFloat && a[34].dim()==2, "Invalid dtype/rank: idx_ape");
TORCH_CHECK(a[34].size(0)==4,"Invalid shape: idx_ape");
TORCH_CHECK(a[34].size(1)==256,"Invalid shape: idx_ape");
TORCH_CHECK(a[35].scalar_type()==at::kInt && a[35].dim()==1, "Invalid dtype/rank: idx_bounds");
TORCH_CHECK(a[36].scalar_type()==at::kInt && a[36].dim()==1, "Invalid dtype/rank: idx_lengths");
TORCH_CHECK(a[37].scalar_type()==at::kInt && a[37].dim()==2, "Invalid dtype/rank: idx_state_slots");
TORCH_CHECK(a[37].size(1)==2,"Invalid shape: idx_state_slots");
TORCH_CHECK(a[38].scalar_type()==at::kBFloat16 && a[38].dim()==1, "Invalid dtype/rank: idx_gamma");
TORCH_CHECK(a[38].size(0)==128,"Invalid shape: idx_gamma");
TORCH_CHECK(a[39].scalar_type()==at::kFloat && a[39].dim()==2, "Invalid dtype/rank: idx_cos");
TORCH_CHECK(a[39].size(1)==64,"Invalid shape: idx_cos");
TORCH_CHECK(a[40].scalar_type()==at::kFloat && a[40].dim()==2, "Invalid dtype/rank: idx_sin");
TORCH_CHECK(a[40].size(1)==64,"Invalid shape: idx_sin");
TORCH_CHECK(a[41].scalar_type()==at::kInt && a[41].dim()==2, "Invalid dtype/rank: idx_slots");
TORCH_CHECK(a[41].size(1)==2,"Invalid shape: idx_slots");
TORCH_CHECK(a[42].scalar_type()==at::kBFloat16 && a[42].dim()==2, "Invalid dtype/rank: hadamard");
TORCH_CHECK(a[42].size(0)==128,"Invalid shape: hadamard");
TORCH_CHECK(a[42].size(1)==128,"Invalid shape: hadamard");
TORCH_CHECK(a[43].scalar_type()==at::kChar && a[43].dim()==2, "Invalid dtype/rank: idx_cache");
TORCH_CHECK(a[44].scalar_type()==at::kInt && a[44].dim()==2, "Invalid dtype/rank: original_table");
TORCH_CHECK(a[45].scalar_type()==at::kInt && a[45].dim()==2, "Invalid dtype/rank: compressed_table");
TORCH_CHECK(a[46].scalar_type()==at::kInt && a[46].dim()==2, "Invalid dtype/rank: indexer_table");
TORCH_CHECK(a[47].scalar_type()==at::kChar && a[47].dim()==2, "Invalid dtype/rank: idx_wq_b");
TORCH_CHECK(a[47].size(0)==1024,"Invalid shape: idx_wq_b");
TORCH_CHECK(a[47].size(1)==8192,"Invalid shape: idx_wq_b");
TORCH_CHECK(a[48].scalar_type()==at::kFloat && a[48].dim()==1, "Invalid dtype/rank: idx_wq_b_scale");
TORCH_CHECK(a[48].size(0)==8192,"Invalid shape: idx_wq_b_scale");
TORCH_CHECK(a[49].scalar_type()==at::kBFloat16 && a[49].dim()==2, "Invalid dtype/rank: weights_proj");
TORCH_CHECK(a[49].size(0)==64,"Invalid shape: weights_proj");
TORCH_CHECK(a[49].size(1)==4096,"Invalid shape: weights_proj");
TORCH_CHECK(a[50].scalar_type()==at::kFloat && a[50].dim()==2, "Invalid dtype/rank: topk_scores");
TORCH_CHECK(a[50].size(1)==512,"Invalid shape: topk_scores");
TORCH_CHECK(a[51].scalar_type()==at::kInt && a[51].dim()==2, "Invalid dtype/rank: topk_indices");
TORCH_CHECK(a[51].size(1)==512,"Invalid shape: topk_indices");
TORCH_CHECK(a[52].scalar_type()==at::kFloat && a[52].dim()==2, "Invalid dtype/rank: attn_sink");
TORCH_CHECK(a[52].size(0)==64,"Invalid shape: attn_sink");
TORCH_CHECK(a[52].size(1)==1,"Invalid shape: attn_sink");
TORCH_CHECK(a[53].scalar_type()==at::kBFloat16 && a[53].dim()==3, "Invalid dtype/rank: wo_a");
TORCH_CHECK(a[53].size(0)==8,"Invalid shape: wo_a");
TORCH_CHECK(a[53].size(1)==4096,"Invalid shape: wo_a");
TORCH_CHECK(a[53].size(2)==1024,"Invalid shape: wo_a");
TORCH_CHECK(a[54].scalar_type()==at::kChar && a[54].dim()==2, "Invalid dtype/rank: wo_b");
TORCH_CHECK(a[54].size(0)==8192,"Invalid shape: wo_b");
TORCH_CHECK(a[54].size(1)==4096,"Invalid shape: wo_b");
TORCH_CHECK(a[55].scalar_type()==at::kFloat && a[55].dim()==1, "Invalid dtype/rank: wo_b_scale");
TORCH_CHECK(a[55].size(0)==4096,"Invalid shape: wo_b_scale");
TORCH_CHECK(a[56].scalar_type()==at::kBFloat16 && a[56].dim()==2, "Invalid dtype/rank: output");
TORCH_CHECK(a[56].size(1)==16384,"Invalid shape: output");
 TORCH_CHECK(csa::ValidDynamicShape(variant, [&](int i,int axis){return a[i].size(axis);}),
 "CSA dynamic descriptor mismatch");
}
void Meta(at::TensorList a,int64_t variant){Check(a,variant);}
void Run(at::TensorList a,int64_t variant){
 Check(a,variant); const c10_npu::OptionalNPUGuard guard(a[0].device());
 EXEC_NPU_CMD(aclnnDsfCsaPtoBnn, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[52], a[53], a[54], a[55], variant, a[15], a[18], a[29], a[32], a[43], a[50], a[51], a[56]);
}
}
TORCH_LIBRARY(bytenn_pto_ops,m){m.def("dsf_csa_pto_bnn(Tensor(a!)[] buffers, int variant) -> ()");}
TORCH_LIBRARY_IMPL(bytenn_pto_ops,PrivateUse1,m){m.impl("dsf_csa_pto_bnn",&Run);}
TORCH_LIBRARY_IMPL(bytenn_pto_ops,Meta,m){m.impl("dsf_csa_pto_bnn",&Meta);}
