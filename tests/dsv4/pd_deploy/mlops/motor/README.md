# Motor + MLOps P/D 部署

| 文件 | 关键作用 |
| --- | --- |
| `4p_ep32_3worksets.yaml` | Decode ×2（合计 EP32）、Prefill ×4（各 TP4/PP4）、ACC ×1；仅 Decode 约束同超节点 |
| `acc.py` | 启动 Motor，提供 Ark 适配和请求日志，根据服务发现自动注册/卸载 P/D |
| `motor_config.json` | Motor 端口、KV hash affinity、tokenizer 路径等配置；不限制最小输入长度 |
| `generate_hccn_conf.sh` | 在 P/D Pod 内生成本机 `/etc/hccn.conf` |
| `fix_mlops_npu_routes.sh` | 在跳板机通过同节点 device-plugin 修复 NPU 跨网段路由 |
| `tests/` | 注册协议、HTTP 和进程生命周期测试；使用模拟 Motor，不运行模型 |

## 部署

1. 准备两类镜像，仓库路径保持 `/vllm-workspace/vllm-ascend`：P/D 镜像包含共用启动脚本和 HCCN 脚本；独立 ACC 镜像包含本目录的 ACC、配置及推理 Motor（接口按 3.1.0 包对接）。
2. YAML 中两处 `REPLACE_WITH_PD_IMAGE` 必须填写同一个完整镜像地址；`REPLACE_WITH_ACC_MOTOR_IMAGE` 填写独立 ACC 镜像。ACC 镜像可固定版本，不必随 P/D 镜像更新。
3. 核对 P/D 权重目录和 `motor_config.json` 的模型路径。ACC 已开启 SFCS 挂载，供 Motor 加载 tokenizer。
4. 导入 YAML 创建服务。P/D 参数保留在 EntryPoint；ACC 前台运行并管理 Motor 子进程，不需要手动 exec 或注册。

请求链路：`Ark → ACC:8005 → Motor:1025 → P/D`，Motor 管理接口为 `1026`。
ACC 镜像还需提供 FastAPI、HTTPX、AnyIO、PyYAML、支持 `capture_signals` 的 Uvicorn，以及 Motor 加载 DSV4 tokenizer 所需的 vLLM 依赖。

## ACC 入口

在镜像内 `tests/dsv4/pd_deploy/` 执行，YAML 已包含以下命令：

```bash
export USER_CONFIG_PATH="$PWD/mlops/motor/motor_config.json"
export KV_AFFINITY_LOG_ALL_CANDIDATES=0 MOTOR_ENABLE_DISPATCH_STOP=0
export OPENAI_STANDARD=STANDARD POD_IP=0.0.0.0
export KV_CONDUCTOR_SERVICE=127.0.0.1
exec python3 -u mlops/motor/acc.py --prefill-workerset prefill --decode-workerset decode
```

ACC 每 5 秒读取 `/etc/config/discovery/application-endpoints.yaml` 并检查后端健康状态。
每个 Pod 注册为一个 Motor 实例：P 使用 `7100`，D 使用 `7100–7115`；实例 ID 不随列表排序改变。
通过 `/instances/refresh` 的 `add`、`del` 更新实例，IP 或健康端口列表变化时先删旧记录再添加。
`/readyz` 和 `/healthcheck` 同时检查 Motor 的 `ready` 字段与已注册的 P/D；不要求固定实例数量。
Motor 异常退出会让 ACC 退出；停止容器时 ACC 关闭监听并终止 Motor 进程组。
运行日志进入标准输出，请求日志位于 `/app/pure_vllm/logs/requests`。

## HCCN 与路由

P/D 入口在模型启动前调用 `bash mlops/motor/generate_hccn_conf.sh`。
机器成功调度后，在能访问目标集群的跳板机先执行 `verify`；验证通过无需修复，失败时再执行 `fix`，然后重新 `verify`：

```bash
bash fix_mlops_npu_routes.sh <service-id> verify
```

仅在验证失败时修复并复查：

```bash
bash fix_mlops_npu_routes.sh <service-id> fix
bash fix_mlops_npu_routes.sh <service-id> verify
```

如需撤销，在同一目录执行 `bash fix_mlops_npu_routes.sh <service-id> rollback`。

适用当前资源池：D 网段 `28.103.0.0/16`，P 网段 `28.19.0.0/16`、`28.102.0.0/16`，设备 0–15、策略表 100–115。
需要 kubectl 权限和插件内 Perl。保留状态文件及插件临时目录以便回滚；此修复不保证节点重建后保留。
`verify` 检查同卡号双向 ping，不代替真实 Motor/KV/推理验收；生成 `hccn.conf` 也不代替路由修复。
