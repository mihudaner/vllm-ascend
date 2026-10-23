# MLOps P/D 部署

| 文件 | 关键作用 |
| --- | --- |
| `4p_ep32_3worksets.yaml` | 三个 WorkerSet：Decode ×2（合计 EP32）、Prefill ×4（各 TP4/PP4）、ACC ×1；仅 Decode 约束同超节点 |
| `acc.py` | 单文件 Ark 适配、请求日志和服务发现，复用仓库原生 proxy 转发请求 |
| `generate_hccn_conf.sh` | 在 P/D Pod 内读取本机 16 个设备地址，生成 `/etc/hccn.conf` |
| `fix_mlops_npu_routes.sh` | 在跳板机通过同节点 device-plugin 修复跨网段 NPU 路由 |
| `tests/` | HCCN、发现逻辑和模拟 P/D 的 HTTP 测试，不运行模型 |

## 使用 YAML

1. 准备 P/D 和 ACC 两类镜像，仓库路径保持 `/vllm-workspace/vllm-ascend`。P/D 包含共用启动脚本和 HCCN 脚本；ACC 包含本目录的 ACC 及原生 proxy。
2. YAML 中两处 `REPLACE_WITH_PD_IMAGE` 必须填写同一个完整镜像地址；`REPLACE_WITH_ACC_VLLM_PROXY_IMAGE` 填写独立 ACC 镜像。ACC 镜像可固定版本，不必随 P/D 镜像更新。核对 P/D 的 `MAAS_MODEL_DIR`。
3. 将 YAML 导入 MLOps 创建服务。P/D 参数直接保留在 EntryPoint，前台运行，日志进入容器标准输出。

ACC 镜像需提供原生 proxy 及其依赖、FastAPI、HTTPX、AnyIO、PyYAML 和支持 `capture_signals` 的 Uvicorn。
保留 YAML 的服务发现挂载和 P/D 内部发布配置；Decode 使用平台的 `MLP_ROLE_INDEX`、`MLP_ROLE_HEAD` 分配 rank/master。

## Pod 内脚本用法

以下命令在镜像内 `tests/dsv4/pd_deploy/` 目录执行，完整 P/D 启动命令见 YAML。

```bash
# P/D Pod：启动模型前生成本机 HCCN 配置
bash mlops/vllm_ascend_proxy/generate_hccn_conf.sh

# ACC Pod：启动适配器、proxy 和自动发现
exec python3 -u mlops/vllm_ascend_proxy/acc.py --prefill-workerset prefill --decode-workerset decode
```

ACC 每 5 秒读取 `/etc/config/discovery/application-endpoints.yaml`，按健康状态更新 P/D 列表。
默认发现 `prefill/prefill:7100` 和 `decode/decode:7100–7115`；删除或更新 Pod 后自动调整后端。
对外端口 8005，内部 proxy 8050，请求日志位于 `/app/pure_vllm/logs/requests`。

## 跳板机路由修复

将 `fix_mlops_npu_routes.sh` 放到能访问目标集群的跳板机上。机器成功调度后先执行 `verify`；验证通过无需修复，失败时再执行 `fix`，然后重新 `verify`：
`verify` 先收集 P/D 设备 IP，再逐节点检查；首次读取失败、路由不匹配或 ping 失败即退出，不再检查后续卡和节点。

```bash
bash fix_mlops_npu_routes.sh <service-id> verify
```

仅在验证失败时修复并复查：

```bash
bash fix_mlops_npu_routes.sh <service-id> fix
bash fix_mlops_npu_routes.sh <service-id> verify
```

如需撤销，在同一目录执行 `bash fix_mlops_npu_routes.sh <service-id> rollback`。

脚本启动即显示操作和服务 ID；从当前 P/D 节点的 NPU IP 自动收集对端 `/16` 网段，使用各卡网关和网口修复设备 0–15、策略表 100–115。
重复 `fix` 会重新发现当前节点；历史节点保留在状态文件中供 `rollback` 使用。
需要 kubectl 权限和插件内 Perl。保留跳板机状态文件及插件临时目录以便回滚；此修复不保证节点重建后保留。
`verify` 检查同卡号双向 ping，不代替 Mooncake KV 验收；生成 `hccn.conf` 也不代替路由修复。
