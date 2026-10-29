# Draft 无 replay：配置原因与 CSA 图键遗漏

日期：2026-10-29。仅分析与 CPU 分支复现；没有恢复已停止的 NPU 验证，没有修改或重启 C3 服务。

## 当前服务的直接原因

C3 `zgj_026` 当前启动命令（日志标识 `decode_dp16_rank0_20261029_164616.log`）包含：

```json
{"num_speculative_tokens": 5, "method": "dspark", "enforce_eager": true}
```

`/data00/zgj/scripts/npu_3_decode/runtime/decode/run_dp_template.sh:52` 写死该参数，81 行传给 `--speculative-config`。

实际容器源码 `/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:200`：

```python
self.use_cuda_graph = self.runner._use_aclgraph() and not self.speculative_config.enforce_eager
```

726 行的 draft `ACLGraphWrapper` 创建依赖 `self.use_cuda_graph`。因此当前 draft 不会因 target 启用了 FULL/SK 而自动启用图；其直接原因是配置为 eager，不是已录好图后发生策略签名不匹配。当前 C3 checkout 的 HEAD 为 `7aca9b1b3`，不是将用户提供的历史提交号当作当前运行版本。

## 用户指定提交中的独立兼容问题

分析提交：`869b1aed3dfd2f83a4b64e1b7878239715d40867`。当前 C3 的 `graph_variants.py` 仍存在相同逻辑。

1. `vllm.forward_context` 构造上下文时，用 `compilation_config.static_forward_context` 作为 `no_compile_layers`。target 与 draft 共享该层注册表；vLLM draft loader 也通过加载前后集合差识别 draft 层。
2. `graph_variants.py::csa_graph_key` 遍历全部 `context.no_compile_layers`，未排除 draft 上下文里的 target CSA 层。
3. `service.py::graph_policy_key` 对 `is_draft_model=True` 明确返回 `None`，但调用方仍把 `(target_layer, None)` 放入非空 policies，构成 `CSAGraphKey`。
4. `ACLGraphWrapper.__call__` 对没有缓存且没有 `pto_csa_policy_graph_capture` 标记的 `CSAGraphKey`，直接调用 `_run_without_graph`，发生在创建图条目之前。
5. 标记仅在 target `model_runner_v1.py::_dummy_run` 内设置。DSpark `dummy_run` 建立独立 `set_ascend_forward_context(is_draft_model=True)`，没有设置或继承该标记。

因此，在启用 draft FULL 图且共享注册表含 target CSA 时，draft 连首次 capture 都可能被拦住，持续 fallback。该问题独立于当前 `enforce_eager=true`，不能把它当成当前配置下实际触发的原因。

## CPU 复现

`analyze_draft_graph.py` 使用 `git show` 从精确提交提取真实 `graph_policy_key`、`csa_graph_key` 和 `ACLGraphWrapper.__call__`，仅替换外部上下文和回退执行体，不导入 Torch、不使用 NPU。结果见 `cpu_reproduction.json`：

- FULL draft 上下文得到 `CSAGraphKey(..., policies=(("target.csa", None),))`。
- warmup/capture/inference 三次调用均命中 eager redispatch；创建的图条目为 0。
- 仅在内存中模拟 draft 直接返回原 batch descriptor 后，可以到达正常图条目创建分支。此模拟不等于已验证真实 draft 图捕获或 replay。

## 建议的修复边界

当前 CSA runtime 明确不支持 draft 路径，因此最小修复应让 `csa_graph_key` 在 draft 上下文直接使用原 batch descriptor，避免 target CSA 影响独立 draft 图；保留 target 的 policy 图键与安全回退。不要全局删除 None policy、不要简单对所有 draft 设置 CSA 捕获标记来掩盖错误归属。

先增加共享注册表下 draft/target 隔离回归，再验证 draft 的真实 capture/replay，最后在用户决定启用 draft 图时调整 speculative 配置并重新启动服务。当前未应用任何产品补丁，也未改 `enforce_eager`。
