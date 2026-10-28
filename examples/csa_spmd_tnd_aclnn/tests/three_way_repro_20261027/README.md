# CSA 精度与性能复现

## 恢复记录（2026-10-28）

本目录从 c2 回收站恢复到 ACLNN 仓库，保留原始脚本、参考快照和历史结果。
原始提交为 bytenn-ascend 的 `3c8263e`、`2c6344e`；恢复不代表已完成新版 94 策略设备验收。

在宿主机进入容器后，切换到恢复后的目录：

```bash
docker exec -it wangkai_vllm026_dev72 bash
cd /data00/wangkai/aclnnWorkspace/vllm-ascend/examples/csa_spmd_tnd_aclnn/tests/three_way_repro_20261027
```

下方命令保留原有 `csa_nz1685` vendor/Torch 扩展依赖与参考快照检查。
当前源码与快照不一致时会主动拒绝执行，不能将历史结果视为当前版本复测结果。


当时完整的精度数据、指标说明与性能表见同目录的
[CSA_THREE_REFERENCES.md](CSA_THREE_REFERENCES.md)（从工程根目录移入，内容未改动）。

当前入口只执行 **ACLNN、Native 算子链、PyTorch 数学 golden**，不执行或计时 Python PTO。
`historical/` 原样保留 2026-10-27 旧测试（包含 Python PTO），不是本次复测成绩。

## 一条命令

在容器 `wangkai_vllm026_dev72` 内进入本目录，然后运行：

```bash
# 与同事命令参数风格一致：128K/B1/S2，使用空闲的逻辑 NPU 0
bash run_three_way.sh my_case 0 --history 131072 --batch 1 --query-lengths 2

# 复现原表的全部 8 档（包含 B1/S6，而非上面的 S2）
bash run_three_way.sh original_8_cases 0 --all

# 混长 T56
bash run_three_way.sh mixed_case 0 --history 131072 --batch 16 \
  --query-lengths 1,2,3,4,5,6,6,5,4,3,2,1,1,3,4,6
```

设备必须空闲，启动器检查进程并持有 device 锁；不停止其他服务。
默认 seed=1024、layer=4，非同事 bin 脚本的默认 seed=2048。
重复运行必须换 run_name，不覆盖已有结果。
`--check-only` 仅检查依赖及设备占用，不执行 NPU 算子。

输出在仓库外 `/data00/wangkai/pto_env/csa_three_way_runs/<run_name>/`：

- `RESULTS.md`：本次性能表及两条参考的精度表，逐档更新。
- `<case>/comparison.json`：完整耗时样本、误差指标与错误信息。
- `<case>/*.pt`、`generated/`、`<case>.log`：中间产物，不放入本仓库。
- `run_manifest.json`：本次脚本校验值、设备及输入清单。

ACLNN/Native：NPU Event 包围图重放，预热5次、采样10次中位数。
golden：eager 预热2次、采样3次，不能直接与图重放计算加速比。
恢复输入、编译、比较均在计时区间外。精度FAIL和程序执行失败分开报告。

## 当时的性能（历史记录，µs）

下表来自原始 `historical/<case>/comparison.json`，不是本次新入口重新测量。
按当前要求不展示 Python PTO 列。前两列为图重放，golden 为 eager。

| 输入 | ACLNN 图重放 | Native 图重放 | PyTorch golden eager |
|---|---:|---:|---:|
| 128K/B1/S6 | 384.48 | 651.50 | 15,935.08 |
| 256K/B16/S6 | 951.78 | 1,601.44 | 204,633.80 |
| 128K/B8/S6 | 661.14 | 968.82 | 96,473.88 |
| 128K/B16/S6 | 796.60 | 1,291.89 | 168,815.93 |
| 128K/B24/S6 | 1,007.99 | 1,489.49 | 242,745.54 |
| 128K/B28/S6 | 1,167.41 | 1,861.68 | 290,859.28 |
| 8K/B32/S6 | 943.78 | 1,252.82 | 321,014.83 |
| 128K/B16/混长 T56 | 738.19 | 1,307.21 | 141,854.02 |

注意：示例命令 `--query-lengths 2` 是 S2，不对应表中的 S6；
要复现首行，使用 `bash run_three_way.sh c01_repro 0 --query-lengths 6`。
历史精度未全部通过，性能数据不代表精度验收通过。

## 文件与来源

| 文件 | 用途 |
|---|---|
| `scripts/csa_compare_no_pto.py` | 当前测试主入口，跳过 Simpler 初始化和 Python PTO 执行 |
| `scripts/checks_no_pto.py` | 当前 ACLNN 对 Native/golden 的验证和计时 |
| `scripts/csa_compare_matrix.py` | 当时的主测试入口，原样归档，不由新命令执行 |
| `scripts/matrix_checks.py` | 当时的三路对照核心；新入口仅复用输入解码、计时和比较工具函数 |
| `sources/dsv4_csa.py` | 当时 Native `_native_attention_half()` 实现快照 |
| `sources/golden.py` | 当时独立数学 golden 快照 |
| `historical/CSA_THREE_REFERENCES.md` | 当时结果说明，包含指标定义与性能表 |
| `historical/<case>/*.json` | 原始8档结果与完整性能样本 |

原脚本来源：`/data00/wangkai/aclnnWorkspace/compare-20261027/`。
Native/golden 来源：`/data00/wangkai/aclnnWorkspace/vllm-ascend/` 下同名文件。
脚本快照不是自包含工具链：仍依赖此机器现有 env-dynamic、公共工具链、Native 代码、
scenarios 输入辅助、模型权重和 csa_nz1685 已安装 vendor/torch binding。
启动时校验当前 Native/golden 文件与快照一致，不静默替换数学参考。
虽然复用 PreparedDynamicCSA 的描述符准备逻辑，但不会调用 Python PTO 内核。

## 本次复测状态

2026-10-27：归档及新入口已整理；Python/shell 语法、--help、使用临时合成数据的
结果表生成冒烟检查均通过；合成数据未混入历史或实测结果。
设备预检查发现全部16张NPU均被 VLLMWorker_DP 占用，NPU 0 约56.4 GB进程显存，
因此拒绝启动。**新入口的设备执行与性能复测尚未完成**，historical 不代表新入口已验收。
