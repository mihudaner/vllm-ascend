// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "op_api_common.h"

namespace {
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
void Check(const at::Tensor& t, at::IntArrayRef shape, at::ScalarType dtype,
           const at::Device& device, const char* name) {
    TORCH_CHECK(t.device() == device, name, ": inputs must share a device");
    TORCH_CHECK(t.sizes() == shape, name, ": unsupported shape");
    TORCH_CHECK(t.scalar_type() == dtype, name, ": unsupported dtype");
    TORCH_CHECK(t.is_contiguous(), name, ": contiguous tensor required");
}
Outputs Allocate(const at::Tensor& x, const at::Tensor& w,
                 const at::Tensor& scale, const at::Tensor& base) {
    TORCH_CHECK(x.dim()==3 && x.size(0)>0 && x.size(0)<=1048576, "unsupported tokens");
    const auto t=x.size(0);
    Check(x, {t, 4, 4096}, at::kBFloat16, x.device(), "x");
    Check(w, {24, 16384}, at::kFloat, x.device(), "weight");
    Check(scale, {3}, at::kFloat, x.device(), "scale");
    Check(base, {24}, at::kFloat, x.device(), "base");
    return {at::empty({t,4096}, x.options()),
            at::empty({t,4}, w.options()), at::empty({t,4,4}, w.options())};
}
Outputs Run(const at::Tensor& x, const at::Tensor& w,
            const at::Tensor& scale, const at::Tensor& base) {
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto [mixed, post, comb] = Allocate(x, w, scale, base);
    EXEC_NPU_CMD(aclnnHcPrePto, x, w, scale, base, mixed, post, comb);
    return {mixed, post, comb};
}
}
TORCH_LIBRARY(hcpre_pto, m) {
    m.def("hc_pre(Tensor x, Tensor weight, Tensor scale, Tensor base) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(hcpre_pto, PrivateUse1, m) { m.impl("hc_pre", &Run); }
TORCH_LIBRARY_IMPL(hcpre_pto, Meta, m) { m.impl("hc_pre", &Allocate); }
