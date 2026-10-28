# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}

## CSA execution backend

The opt-in PyPTO CSA model keeps one `torch.ops.vllm.dsv4_csa_forward` interface
and one `PreparedDynamicCSA` input binding. `VLLM_ASCEND_PTO_CSA_BACKEND` selects
only the final executor:

- `python` (default): retain the Python PyPTO JIT path.
- `aclnn`: execute the precompiled `bytenn_pto_ops.dsf_csa_pto_bnn` operator.

Set the variables before the existing PyPTO-model launch command:

```bash
export VLLM_ASCEND_ENABLE_NZ=2
export PTO_CSA_VARIANT=spmd_tnd
export VLLM_ASCEND_PTO_CSA_BACKEND=aclnn
```

Use `VLLM_ASCEND_PTO_CSA_BACKEND=python` in a new process to restore PyPTO.
The previous local draft variable `VLLM_ASCEND_CSA_BACKEND` is replaced by this
upstream-compatible name; remove it from launch scripts. This variable does not
select the PyPTO model architecture or modify the native model entry.

ACLNN requires a matching CSA `bytenn_pto_ops` wheel with all 94 policies from
the d544a09a kernel baseline. HC-pre-only, old 62-policy and partial test packages
are rejected. Model preparation checks the pinned interface/policy manifests,
57 Tensor ABI, Python/Torch/TorchNPU versions, architecture and binary hashes,
then loads the operator. The service uses the current shared binding directly,
not the wheel's standalone `PreparedAclnnCSA` wrapper or its binding-source hash
check. The standalone wrapper remains available separately and must be rebuilt
with the updated source binding.

Each service fixes its backend at initialization, including bindings created for
new shapes. Both backends share Native cache/state preparation, metadata, policy,
eligibility and graph contracts. Prefill and unsupported decode inputs retain the
existing Native fallback. Dependency and policy errors do not silently switch to
Python. ACLNN rejects non-null PyPTO `config` arguments explicitly.

Normal ACLNN calls prepare two stable Top-K outputs for the 57-Tensor kernel ABI;
diagnostic calls reuse caller-provided outputs. Rebinding validates all new
arguments before replacing existing bindings. Workspace remains backend-owned.
ACLNN still imports shared PyPTO types for ABI checks, but skips PyPTO execution
and Simpler initialization. Representative NPU tests of the shared binding and
installed extension cover eager execution, repeated calls, rebinding and graph
replay, including padded token capacities. End-to-end model/service validation
is separate and remains pending; CPU/Meta tests alone do not establish device
correctness.

### CSA ACLNN execution tracing

Set `PTO_CSA_TRACE=1` before starting the service to use the existing throttled
CSA route logger. At the final ACLNN executor, the following events bracket the
actual `torch.ops.bytenn_pto_ops.dsf_csa_pto_bnn` call:

- `aclnn_op_enter`: backend/operator name, ACLNN API name, tiling key, device,
  input `x_shape`, Tensor count, and executor identity.
- `aclnn_op_returned`: the operator call returned to the host without raising.
  `status=host_call_returned_async` explicitly does not mean NPU completion.

These events are separate from `PTO_CSA_ACLNN_READY`, which only confirms package
validation/loading. Correlate them with `tnd_kernel_submit` (layer and backend)
and `tnd_kernel_returned` in the same worker. If the operator raises, no
`aclnn_op_returned` event is emitted for that call; inspect its traceback.

Mode 1 prints the first 16 occurrences and then one per 256 occurrences for each
event/device in each process, shared across bindings/layers. Mode 2 prints every
call; mode 0 (default) disables tracing. No device values are read and no NPU
synchronization is added by these executor events. Use mode 0 for performance
measurements.

ACL Graph replay bypasses this Python entry, so these events occur during eager
execution or capture, not on every replay. Use the existing graph route logs and
profiler `DsfCsaPtoBnn` kernel records to verify replay execution. Enable tracing
in a new validation process; editing source does not update an already running
worker.
