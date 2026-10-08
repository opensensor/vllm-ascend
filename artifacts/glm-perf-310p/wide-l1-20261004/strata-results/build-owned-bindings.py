from pathlib import Path
import torch
import torch_npu
from torch.utils.cpp_extension import load
root = Path('/srv/ai/src/glm-l1-wide-build-20261004')
build = root / 'build-strata-bindings-owned-20261004'
build.mkdir(exist_ok=True)
source = build / 'bindings.cpp'
source.write_text('''#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "aclnn_torch_adapter/op_api_common.h"
#define npu_w2_grouped_blocked_dequant_matmul_310 npu_w2_grouped_blocked_dequant_matmul_owned_310
#include "gmm/w2_grouped_blocked_dequant_matmul_v310/w2_grouped_blocked_dequant_matmul_310_torch_adpt.h"
#undef npu_w2_grouped_blocked_dequant_matmul_310
#include "gmm/w2_swiglu_v310/w2_swiglu_310_torch_adpt.h"
#include "gmm/w2_route_combine_v310/w2_route_combine_310_torch_adpt.h"
at::Tensor swiglu_meta(const at::Tensor& x) {
    return at::empty_symint(c10::SymDimVector{x.sym_size(0), x.sym_size(1) / 2}, x.options());
}
at::Tensor combine_meta(const at::Tensor& rows, const at::Tensor& inverse,
                        const at::Tensor& weights, const at::Tensor& ends) {
    return at::empty_symint(c10::SymDimVector{weights.sym_size(0), rows.sym_size(1)},
                           rows.options().dtype(at::kFloat));
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("npu_w2_grouped_blocked_dequant_matmul_310", &vllm_ascend::npu_w2_grouped_blocked_dequant_matmul_owned_310);
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("npu_w2_swiglu_310(Tensor gate_up) -> Tensor");
    m.def("npu_w2_route_combine_310(Tensor routed, Tensor inverse_order, Tensor route_weights, Tensor group_ends) -> Tensor");
    m.impl("npu_w2_swiglu_310", torch::kMeta, &swiglu_meta);
    m.impl("npu_w2_route_combine_310", torch::kMeta, &combine_meta);
    m.impl("npu_w2_swiglu_310", torch::kPrivateUse1, &vllm_ascend::npu_w2_swiglu_310);
    m.impl("npu_w2_route_combine_310", torch::kPrivateUse1, &vllm_ascend::npu_w2_route_combine_310);
}
''')
npu = Path(torch_npu.__file__).parent
load(name='glm_moe_candidates', sources=[str(source), str(root / 'aclnn_torch_adapter/NPUBridge.cpp'), str(root / 'aclnn_torch_adapter/NPUStorageImpl.cpp')], build_directory=str(build),
     extra_include_paths=[str(root), str(npu / 'include'), '/usr/local/Ascend/ascend-toolkit/latest/include'],
     extra_cflags=['-O2', '-DGLM_EXPERIMENTAL_OP_API_TENSOR_OWNERS'], extra_ldflags=[f'-L{npu / "lib"}', '-ltorch_npu',
     '-L/usr/local/Ascend/ascend-toolkit/latest/lib64', '-lascendcl'], is_python_module=False, verbose=True)
print(build / 'glm_moe_candidates.so')
