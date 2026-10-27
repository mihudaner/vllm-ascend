# 全 policy 单卡 Native 功能与精度对照

SPMD 94 项、MPMD 55 项均完成正常入口执行，有效输出与更新状态无 NaN/Inf，
缓存非法写、只读 metadata 与输出首尾保护区检查通过。没有新增编译策略或修改算子/SDK。

**与 Native 非 bit 一致，既有严格逐元素容差未全部通过。** 下面报告实际误差，
不据此宣称模型 token/DSpark 一致，也不为通过而放宽原逐元素阈值。

同一场景两侧使用第4层真实权重、相同合成历史/有效输入；确定性 level=1、
HCCL_DETERMINISTIC=true、PTO atomic_add=0。Native eager 计算有效 token，
PTO正常入口保留容量；不测性能、不直接导出隐藏Top-K、不做16卡整模型。

| 实现 | 场景数 | 功能/保护检查通过 | 输出最大相对 RMSE | 输出最大绝对差 | 最低 cosine |
|---|---:|---:|---:|---:|---:|
| SPMD | 94 | 94 | 0.5395% | 0.03222656 | 0.999985446 |
| MPMD | 55 | 55 | 0.5397% | 0.03125000 | 0.999985437 |

相对 RMSE = RMS(PTO−Native) / RMS(Native)，按场景逐项统计后取最坏值，
不将所有场景混成一个均值。输出误差约半个百分点，未出现数量级偏离；
严格元素门槛的失败与量化缓存差异仍保留，不能替代更严格数值验收。

## 请求容量分类

Native能够模拟相同padding并提供有效输出；差别是请求描述符。
例如1个有效请求5行、T容量33：PTO扩展场景为B=1、Q=[0,5]；
Native正常padding会追加dummy，成为B=2、Q=[0,5,33]、dummy长度0。
B改变会使requests==1的专用策略改变，因此29项标为额外容量健壮性测试。

直接执行当前Native `_pad_query_start_loc_for_fia` 的CPU函数本体核对：
120项描述符相同，29项不同（SPMD20、MPMD9）。证据见
[native_padding_contract.json](native_padding_contract.json)。这不是149项真实模型入口覆盖。

初稿16项多留了失活请求槽，修正后只补测这16项；另133项完全相同输入复用。
全部最终输入固定在[function_cases.json](function_cases.json)，旧原始记录保留在runs中。

## 更新缓存/状态的最坏误差

只统计本轮允许更新的逻辑行，不用大量未变化的历史稀释误差。

| 实现 | 数据 | 最大相对 RMSE | 最大绝对差 |
|---|---|---:|---:|
| SPMD | swa.0 | 0.35138% | 0.015625 |
| SPMD | compressed.0 | 0.05459% | 0.015625 |
| SPMD | state.0 | 0.03122% | 0.00258518755 |
| SPMD | indexer.0 | 0.82012% | 2 |
| SPMD | indexer.1 | 0.32841% | 0.000228881836 |
| SPMD | indexer_state.0 | 0.02973% | 0.00259670615 |
| MPMD | swa.0 | 0.35141% | 0.015625 |
| MPMD | compressed.0 | 0.04892% | 0.0078125 |
| MPMD | state.0 | 0.03351% | 0.000716090202 |
| MPMD | indexer.0 | 0.89494% | 1 |
| MPMD | indexer.1 | 0.33529% | 6.10351562e-05 |
| MPMD | indexer_state.0 | 0.03019% | 0.000718712807 |

Indexer INT8 cache 的整数差属于量化数据，不是metadata索引；误差以整数级数原样报告。
metadata与保护区要求精确不变。首批2项检查原Native metadata，后续147项另检查PTO扩容后的metadata。
逐项覆盖范围见JSON。
正常入口未导出Top-K，未声称其整数索引与Native精确相同。

## 逐项输出误差

| 实现 | 编号 | 基础功能 | 相对 RMSE | 最大绝对差 | 严格逐元素 |
|---|---:|---|---:|---:|---|
| SPMD | 000 | PASS | 0.3491% | 0.03125000 | FAIL |
| SPMD | 001 | PASS | 0.4317% | 0.01953125 | FAIL |
| SPMD | 002 | PASS | 0.3691% | 0.03125000 | FAIL |
| SPMD | 003 | PASS | 0.4096% | 0.01757812 | FAIL |
| SPMD | 004 | PASS | 0.3496% | 0.03125000 | FAIL |
| SPMD | 005 | PASS | 0.3626% | 0.03125000 | FAIL |
| SPMD | 006 | PASS | 0.3071% | 0.01562500 | PASS |
| SPMD | 007 | PASS | 0.3245% | 0.01562500 | FAIL |
| SPMD | 008 | PASS | 0.3640% | 0.03125000 | FAIL |
| SPMD | 009 | PASS | 0.3649% | 0.03125000 | FAIL |
| SPMD | 010 | PASS | 0.2802% | 0.01562500 | PASS |
| SPMD | 011 | PASS | 0.2627% | 0.01562500 | PASS |
| SPMD | 012 | PASS | 0.3561% | 0.03125000 | FAIL |
| SPMD | 013 | PASS | 0.3535% | 0.03125000 | FAIL |
| SPMD | 014 | PASS | 0.3713% | 0.03125000 | FAIL |
| SPMD | 015 | PASS | 0.3671% | 0.03125000 | FAIL |
| SPMD | 016 | PASS | 0.3253% | 0.01562500 | FAIL |
| SPMD | 017 | PASS | 0.2728% | 0.01562500 | PASS |
| SPMD | 018 | PASS | 0.3291% | 0.01562500 | FAIL |
| SPMD | 019 | PASS | 0.4198% | 0.01953125 | FAIL |
| SPMD | 020 | PASS | 0.3297% | 0.03125000 | FAIL |
| SPMD | 021 | PASS | 0.2663% | 0.01562500 | PASS |
| SPMD | 022 | PASS | 0.3469% | 0.03125000 | FAIL |
| SPMD | 023 | PASS | 0.4040% | 0.01562500 | FAIL |
| SPMD | 024 | PASS | 0.3284% | 0.03125000 | FAIL |
| SPMD | 025 | PASS | 0.3333% | 0.03125000 | FAIL |
| SPMD | 026 | PASS | 0.3680% | 0.01562500 | FAIL |
| SPMD | 027 | PASS | 0.2946% | 0.01562500 | PASS |
| SPMD | 028 | PASS | 0.3330% | 0.03125000 | FAIL |
| SPMD | 029 | PASS | 0.3331% | 0.03125000 | FAIL |
| SPMD | 030 | PASS | 0.2798% | 0.01562500 | PASS |
| SPMD | 031 | PASS | 0.3734% | 0.01757812 | FAIL |
| SPMD | 032 | PASS | 0.3296% | 0.03125000 | FAIL |
| SPMD | 033 | PASS | 0.3320% | 0.03125000 | FAIL |
| SPMD | 034 | PASS | 0.3378% | 0.03125000 | FAIL |
| SPMD | 035 | PASS | 0.3345% | 0.03125000 | FAIL |
| SPMD | 036 | PASS | 0.2528% | 0.01562500 | PASS |
| SPMD | 037 | PASS | 0.3194% | 0.01562500 | FAIL |
| SPMD | 038 | PASS | 0.2861% | 0.01562500 | PASS |
| SPMD | 039 | PASS | 0.3537% | 0.01562500 | FAIL |
| SPMD | 040 | PASS | 0.2842% | 0.01562500 | PASS |
| SPMD | 041 | PASS | 0.3476% | 0.01562500 | FAIL |
| SPMD | 042 | PASS | 0.2901% | 0.01562500 | PASS |
| SPMD | 043 | PASS | 0.3704% | 0.03125000 | FAIL |
| SPMD | 044 | PASS | 0.3914% | 0.01562500 | FAIL |
| SPMD | 045 | PASS | 0.4631% | 0.02343750 | FAIL |
| SPMD | 046 | PASS | 0.2863% | 0.01562500 | PASS |
| SPMD | 047 | PASS | 0.4020% | 0.02343750 | FAIL |
| SPMD | 048 | PASS | 0.3129% | 0.03125000 | FAIL |
| SPMD | 049 | PASS | 0.5395% | 0.03125000 | FAIL |
| SPMD | 050 | PASS | 0.3613% | 0.01562500 | FAIL |
| SPMD | 051 | PASS | 0.3259% | 0.01953125 | FAIL |
| SPMD | 052 | PASS | 0.4761% | 0.03125000 | FAIL |
| SPMD | 053 | PASS | 0.3779% | 0.02343750 | FAIL |
| SPMD | 054 | PASS | 0.4103% | 0.03222656 | FAIL |
| SPMD | 055 | PASS | 0.4286% | 0.02441406 | FAIL |
| SPMD | 056 | PASS | 0.3740% | 0.02343750 | FAIL |
| SPMD | 057 | PASS | 0.3384% | 0.02343750 | FAIL |
| SPMD | 058 | PASS | 0.3536% | 0.03125000 | FAIL |
| SPMD | 059 | PASS | 0.4401% | 0.03125000 | FAIL |
| SPMD | 060 | PASS | 0.3228% | 0.03125000 | FAIL |
| SPMD | 061 | PASS | 0.3393% | 0.03125000 | FAIL |
| SPMD | 062 | PASS | 0.3299% | 0.03125000 | FAIL |
| SPMD | 063 | PASS | 0.3312% | 0.03125000 | FAIL |
| SPMD | 064 | PASS | 0.3376% | 0.03125000 | FAIL |
| SPMD | 065 | PASS | 0.3343% | 0.03125000 | FAIL |
| SPMD | 066 | PASS | 0.4512% | 0.03125000 | FAIL |
| SPMD | 067 | PASS | 0.3922% | 0.03125000 | FAIL |
| SPMD | 068 | PASS | 0.4051% | 0.03125000 | FAIL |
| SPMD | 069 | PASS | 0.3680% | 0.01953125 | FAIL |
| SPMD | 070 | PASS | 0.3806% | 0.02392578 | FAIL |
| SPMD | 071 | PASS | 0.3437% | 0.03125000 | FAIL |
| SPMD | 072 | PASS | 0.3894% | 0.03125000 | FAIL |
| SPMD | 073 | PASS | 0.3624% | 0.01953125 | FAIL |
| SPMD | 074 | PASS | 0.3675% | 0.03125000 | FAIL |
| SPMD | 075 | PASS | 0.4658% | 0.02734375 | FAIL |
| SPMD | 076 | PASS | 0.3704% | 0.02343750 | FAIL |
| SPMD | 077 | PASS | 0.3513% | 0.02343750 | FAIL |
| SPMD | 078 | PASS | 0.3453% | 0.03125000 | FAIL |
| SPMD | 079 | PASS | 0.3846% | 0.02343750 | FAIL |
| SPMD | 080 | PASS | 0.3512% | 0.01855469 | FAIL |
| SPMD | 081 | PASS | 0.3764% | 0.02514648 | FAIL |
| SPMD | 082 | PASS | 0.3963% | 0.03125000 | FAIL |
| SPMD | 083 | PASS | 0.4254% | 0.02343750 | FAIL |
| SPMD | 084 | PASS | 0.3751% | 0.02343750 | FAIL |
| SPMD | 085 | PASS | 0.3468% | 0.02343750 | FAIL |
| SPMD | 086 | PASS | 0.3845% | 0.03125000 | FAIL |
| SPMD | 087 | PASS | 0.3702% | 0.02343750 | FAIL |
| SPMD | 088 | PASS | 0.4010% | 0.02343750 | FAIL |
| SPMD | 089 | PASS | 0.3619% | 0.01953125 | FAIL |
| SPMD | 090 | PASS | 0.3795% | 0.03125000 | FAIL |
| SPMD | 091 | PASS | 0.3693% | 0.01953125 | FAIL |
| SPMD | 092 | PASS | 0.4131% | 0.02343750 | FAIL |
| SPMD | 093 | PASS | 0.3819% | 0.03125000 | FAIL |
| MPMD | 000 | PASS | 0.4653% | 0.02343750 | FAIL |
| MPMD | 001 | PASS | 0.4317% | 0.01953125 | FAIL |
| MPMD | 002 | PASS | 0.4137% | 0.01614380 | FAIL |
| MPMD | 003 | PASS | 0.4096% | 0.01757812 | FAIL |
| MPMD | 004 | PASS | 0.3702% | 0.03125000 | FAIL |
| MPMD | 005 | PASS | 0.3644% | 0.01562500 | FAIL |
| MPMD | 006 | PASS | 0.3071% | 0.01562500 | PASS |
| MPMD | 007 | PASS | 0.3245% | 0.01562500 | FAIL |
| MPMD | 008 | PASS | 0.4660% | 0.01953125 | FAIL |
| MPMD | 009 | PASS | 0.2774% | 0.01562500 | FAIL |
| MPMD | 010 | PASS | 0.3053% | 0.01562500 | FAIL |
| MPMD | 011 | PASS | 0.2166% | 0.01562500 | PASS |
| MPMD | 012 | PASS | 0.2744% | 0.01562500 | PASS |
| MPMD | 013 | PASS | 0.2151% | 0.01562500 | PASS |
| MPMD | 014 | PASS | 0.1755% | 0.01562500 | PASS |
| MPMD | 015 | PASS | 0.2120% | 0.01562500 | PASS |
| MPMD | 016 | PASS | 0.3079% | 0.01562500 | FAIL |
| MPMD | 017 | PASS | 0.3077% | 0.01562500 | FAIL |
| MPMD | 018 | PASS | 0.2794% | 0.01562500 | FAIL |
| MPMD | 019 | PASS | 0.3994% | 0.01562500 | FAIL |
| MPMD | 020 | PASS | 0.3416% | 0.02343750 | FAIL |
| MPMD | 021 | PASS | 0.3048% | 0.01562500 | FAIL |
| MPMD | 022 | PASS | 0.3320% | 0.01562500 | FAIL |
| MPMD | 023 | PASS | 0.4331% | 0.01953125 | FAIL |
| MPMD | 024 | PASS | 0.4462% | 0.02343750 | FAIL |
| MPMD | 025 | PASS | 0.3578% | 0.01953125 | FAIL |
| MPMD | 026 | PASS | 0.3500% | 0.01562500 | FAIL |
| MPMD | 027 | PASS | 0.4127% | 0.01757812 | FAIL |
| MPMD | 028 | PASS | 0.5397% | 0.03125000 | FAIL |
| MPMD | 029 | PASS | 0.3576% | 0.02343750 | FAIL |
| MPMD | 030 | PASS | 0.3879% | 0.02343750 | FAIL |
| MPMD | 031 | PASS | 0.3527% | 0.02343750 | FAIL |
| MPMD | 032 | PASS | 0.3394% | 0.01953125 | FAIL |
| MPMD | 033 | PASS | 0.2960% | 0.01562500 | FAIL |
| MPMD | 034 | PASS | 0.4176% | 0.02343750 | FAIL |
| MPMD | 035 | PASS | 0.3523% | 0.03125000 | FAIL |
| MPMD | 036 | PASS | 0.3277% | 0.01806641 | FAIL |
| MPMD | 037 | PASS | 0.3903% | 0.03125000 | FAIL |
| MPMD | 038 | PASS | 0.3973% | 0.02343750 | FAIL |
| MPMD | 039 | PASS | 0.3789% | 0.02343750 | FAIL |
| MPMD | 040 | PASS | 0.3263% | 0.03125000 | FAIL |
| MPMD | 041 | PASS | 0.3397% | 0.03125000 | FAIL |
| MPMD | 042 | PASS | 0.3326% | 0.03125000 | FAIL |
| MPMD | 043 | PASS | 0.3400% | 0.03125000 | FAIL |
| MPMD | 044 | PASS | 0.3487% | 0.02343750 | FAIL |
| MPMD | 045 | PASS | 0.3892% | 0.03125000 | FAIL |
| MPMD | 046 | PASS | 0.3839% | 0.02343750 | FAIL |
| MPMD | 047 | PASS | 0.3424% | 0.01807404 | FAIL |
| MPMD | 048 | PASS | 0.3671% | 0.03125000 | FAIL |
| MPMD | 049 | PASS | 0.4238% | 0.02343750 | FAIL |
| MPMD | 050 | PASS | 0.3848% | 0.02343750 | FAIL |
| MPMD | 051 | PASS | 0.3711% | 0.02343750 | FAIL |
| MPMD | 052 | PASS | 0.4709% | 0.03125000 | FAIL |
| MPMD | 053 | PASS | 0.4211% | 0.03125000 | FAIL |
| MPMD | 054 | PASS | 0.4021% | 0.03125000 | FAIL |

完整逐元素统计、差异位置、状态与任务来源见[function_results.json](function_results.json)。
首批代表场景误用了重新编译的artifact加载入口；已改为SDK只读prebuilt加载，后续没有重编译。
原始编译计时独立保留，不以这里的执行耗时计算编译或性能收益。
