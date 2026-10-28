# CSA ACLNN 原始 ABI 与策略测试清单

这两个 JSON 文件逐字节复制自
[mihudaner/vllm-ascend@caed2c32896ddefdbcfab180b253d777bfab11e0](https://github.com/mihudaner/vllm-ascend/tree/caed2c32896ddefdbcfab180b253d777bfab11e0/examples/csa_spmd_tnd_aclnn/generated)，
未重新生成、格式化或修改内容。对应算法版本为 `1685da6914ae6196537d7f1a66e84e09a9c48ae5`。

| 文件 | SHA256 |
| --- | --- |
| interface.json | 960854881a74c05fe095bb73bc1de2ba9fe388ae972f68542a382ac0c8005848 |
| manifest.json | 8e3b3be17421adc61a1341c133c49ae045fd8313a5f29544ec78616bea3d6551 |

CPU 测试用这些独立清单对照当前 Python 根函数的 57 参数 ABI，以及原包的
62 套完整 constexpr 策略。测试不修改预期哈希来放过接口变化。
清单中的历史绝对路径仅为原始来源记录，测试不会访问那些路径。
本目录不包含设备二进制；CPU 模拟调用只验证参数顺序、布局约束、别名及状态提交，
不替代 NPU 数学精度或性能验收。
