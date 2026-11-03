# Qwen3.5 Ascend TP=2 推理正确性实验记录

**项目：** TokenSpeed — Qwen3.5 Ascend NPU 适配  
**工作分支：** `feat/qwen35-9b-27b-tp2`  
**实验基线 Commit：** `731a8f4ca9f77c43c994867b38ddeefe6139396e`  
**启动脚本修复 Commit：** `1bf93e32`  
**记录日期：** 2026-11-03  
**负责人：** 成员 B  
**状态：** 四模型 TP=2 推理与正确性对照已完成，性能评测由成员 C 负责

## 1. 实验目标与结论

本阶段主要验证 Qwen3.5-0.8B、4B、9B、27B 在 Ascend NPU 上使用 TokenSpeed 实现 TP=2 推理的可运行性、输出稳定性与数值一致性。

**主要结论：**

1. 四个模型均成功运行 TP=2 推理，15/15 条测试请求全部完成。
2. 0.8B、4B、9B 完成 TP=1 与 TP=2 的输出对照；27B 因当前单卡推理配置无法提供足够缓存容量，使用独立 Hugging Face 实现作为参考。
3. 各模型均存在部分生成 Token 不一致的情况，但不能直接认定为 TP 实现错误。
4. 4B 三方对照证明：即使不启用 TP，TokenSpeed 与 Hugging Face 也可能产生输出差异。
5. 27B 的一个首 Token 分歧发生在两个高分候选之间，HF Logit Margin 仅为 0.125，与数值扰动导致 Argmax 翻转的解释相容，但根因未被完全证明。
6. 静态代码检查未发现 TP=1 和 TP=2 异常切换主要 Attention/GDN Backend 的证据。

本记录证明当前实现能够完成指定测试负载，并提供了独立参考和数值差异证据；不代表所有算子均已证明逐元素等价，也不代表所有输入均能保证正确。

## 2. 实验环境与配置

| 项目 | 配置 |
|---|---|
| 硬件 | Ascend 910C，双 NPU |
| 使用设备 | `ASCEND_RT_VISIBLE_DEVICES=6,7` |
| NPU 显存 | 每卡约 64 GB（设备标称值） |
| CANN | 9.0.0 |
| PyTorch | 2.9.0 |
| torch-npu | 2.9.0 |
| Transformers | 5.12.0 |
| 模型精度 | BF16 |
| Full Attention Backend | `mha` |
| Sampling Backend | `greedy` |
| 张量并行 | TP=1 / TP=2 |
| TP=2 通信 | HCCL |
| All-Reduce Fusion | 4B TP=1 和 TP=2 均为 `False` |

原始启动配置以对应模型的 `server.log` 为准，不应假设所有模型具有完全相同的缓存或资源参数。

## 3. 测试方法

使用统一的 Batch 测试脚本：

`~/tp2_evidence_v2/environment/run_batch.py`

测试集合共 15 条请求：

- 12 条不同测试 Prompt，包括简单任务和较长生成任务。
- 3 条状态隔离测试，按 A1 → B → A2 顺序交错执行。

主要参数：

- `temperature=0`，使用确定性 Greedy 解码。
- 关闭模型的 Thinking 模式。
- 记录 API 响应、实际生成 Token IDs，以及可用的 Logprob 信息。
- 实际生成 Token IDs 从 TokenSpeed 服务端日志按 Request ID 提取，而非仅依赖客户端文本重新编码。

**一致性指标定义：** 整条生成序列的 Token IDs 完全相同才记为一次 Exact Match。

该指标较严格。由于某个生成 Token 一旦发生分歧，后续生成可能沿不同轨迹继续，因此整条序列不一致不等于整个回答错误。

15 条请求包含重复的隔离测试输入，不能视为 15 个统计独立样本。本文的一致率属于描述性结果，不用于推断哪种计算路径在统计上更准确。

## 4. 四模型推理结果

| 模型 | TP=2 请求成功 | TP=1 请求成功 | 正确性参考 |
|---|---:|---:|---|
| Qwen3.5-0.8B | 15/15 | 15/15 | TokenSpeed TP=1 |
| Qwen3.5-4B | 15/15 | 15/15 | TokenSpeed TP=1 + HF |
| Qwen3.5-9B | 15/15 | 15/15 | TokenSpeed TP=1 |
| Qwen3.5-27B | 15/15 | 无法完成 | HF BF16 |

### 4.1 Token 一致性

| 模型 | 对照对象 | 完全一致 |
|---|---|---:|
| 0.8B | TokenSpeed TP=1 vs TP=2 | 11/15 |
| 4B | TokenSpeed TP=1 vs TP=2 | 8/15 |
| 4B | HF vs TokenSpeed TP=1 | 6/15 |
| 4B | HF vs TokenSpeed TP=2 | 8/15 |
| 9B | TokenSpeed TP=1 vs TP=2 | 10/15 |
| 27B | HF vs TokenSpeed TP=2 | 8/15 |

所有已完成的请求均返回成功，A1 → B → A2 测试中，相同的 A 请求输出保持一致。

该隔离检查覆盖当前测试场景，不等于证明所有并发、批处理和 Cache 调度情况均不存在状态污染。

### 4.2 27B TP=1 的限制

尝试通过调整最大模型长度、缓存与回收配置运行 27B TP=1，均因当前单卡资源和缓存容量不足而未能成功完成推理。

保留了三次失败日志：

- `~/tp2_acceptance/27b/tp1/server_first_failure.log`
- `~/tp2_acceptance/27b/tp1/server_second_failure.log`
- `~/tp2_acceptance/27b/tp1/server.log`

该结果说明当前测试环境及配置下 TP=1 无法完成 27B 推理，不应扩展为 27B 在任何单卡设备上均无法运行。

## 5. 4B 三方正确性实验

为区分 TP 路径差异和框架差异，使用相同模型权重进行三方对照：

1. Hugging Face BF16 参考实现。
2. TokenSpeed TP=1。
3. TokenSpeed TP=2。

**输入一致性验证：** 三方实际输入 Token IDs 全部匹配，15/15。

| 对照组合 | 完整生成序列一致 |
|---|---:|
| HF vs TP=1 | 6/15 |
| HF vs TP=2 | 8/15 |
| TP=1 vs TP=2 | 8/15 |
| 三方首 Token 一致 | 15/15 |

观察到：

- `case_03`、`case_10`：TP=1 和 TP=2 的完整输出一致，但 HF 不同。
- `case_04`、`case_09`：HF 和 TP=2 的完整输出一致，但 TP=1 不同。
- 三种实现的首 Token 在全部 15 条测试中一致。

**解释：** 框架实现之间的数值差异确实存在，且 HF 不总是更接近 TokenSpeed TP=1。因此，不能简单把 HF 当作逐 Token 完全一致的标准，也不能根据 6/15 与 8/15 判断 TP=2 比 TP=1 更准确。

## 6. 数值误差专项分析

### 6.1 0.8B：TP=1 与 TP=2 的数值差异

针对 `case_09` 进行深入排查：

- TP=1 和 TP=2 的实际输入 IDs 完全一致，共 142 个。
- 首 Token 发生不同选择。
- 对比首步 Logits，得到 MAE 约 0.03114、RMSE 约 0.03974、最大绝对误差约 0.203125。
- 最终 Hidden States 的 MAE 约 0.05214。
- 分层比较发现误差从早期层开始出现，并随网络深度逐渐增长。
- 独立 BF16 HCCL All-Reduce 数值测试也观察到非零误差；FP32 路径具有不同的误差表现。
- 检查过 GDN `out_proj` 的权重分片重组，未发现该检查范围内的权重不一致。

这些结果支持“分片计算、归约顺序及有限精度运算可引入数值差异”的解释，但没有证明所有生成分歧都仅由 BF16 导致。

应区分两种误差传播：

- **Layer-wise：** 同一次前向传播经过多个网络层，数值差异可能在网络深度方向累积，影响首 Token。
- **Autoregressive：** 生成过程中某一步 Token 不同，后续输入随之变化，使生成轨迹继续分叉。

### 6.2 27B：首 Token 分歧

27B 的 `case_06` 中：

- 实际输入长度：21 Tokens。
- HF 首 Token：`332`。
- TokenSpeed TP=2 首 Token：`49338`。

使用 Hugging Face 实现对相同输入执行单次 Forward，获得：

| 候选 Token ID | HF Logit |
|---|---:|
| `332` | 30.625 |
| `49338` | 30.500 |
| `13962` | 27.625 |
| `37405` | 27.500 |
| `8160` | 26.875 |

两个首选候选的 Logit Margin：

\[
\Delta = 30.625 - 30.500 = 0.125
\]

HF 的前两个候选非常接近，而第三名明显更低。不同计算路径产生的数值扰动可能改变两个候选的相对排序。

**限制：** 目前没有取得 TP=2 对应的完整首步 Logits，因此无法量化两种实现之间的实际偏移，也无法确定其根因。该 Case 记录为 near-tie 数值分歧，而不是已确认的 TP Bug 或已证明的 BF16 舍入问题。

## 7. TP=1 与 TP=2 计算路径检查

对模型层、Linear 层、通信管理器、GDN 配置和 Backend 注册代码进行了静态检查。

主要观察：

1. TP=1 与 TP=2 配置的 Full Attention Backend 均为 `mha`。
2. Qwen3.5 GDN 使用 `hybrid_linear_attn` 路径；TP Size 用于调整 Head 数量、张量分片和状态形状。
3. Linear/MLP 在 TP=2 下使用分片矩阵乘法，并在必要位置通过通信完成归约。
4. 4B TP=1 与 TP=2 的 `enable_allreduce_fusion` 均为 `False`。
5. 未在检查范围内发现因 TP Size 不同而异常切换主要 GDN/Attention Backend 的证据。

但相同 Backend 不代表底层编译 Kernel 完全相同。TP Size 改变张量形状和 GEMM 规模，也可能影响底层 Kernel 选择或计算顺序。

因此，本次静态检查不构成底层 Kernel 完全等价的证明。

## 8. HF 参考实现与权重核验

27B 使用 Hugging Face `Qwen3_5ForCausalLM` 作为独立参考。

- 将多模态 Checkpoint 中的语言模型权重映射至 Text-only 模型结构。
- HF 27B 模型成功加载 851/851 项预期权重。
- 无 Missing、Unexpected 或 Mismatched 权重。
- 抽样权重与原始 Checkpoint 核对一致。
- 使用两张 NPU 进行层级 Device Mapping，不使用 Tensor Parallelism。
- 使用从 TokenSpeed TP=2 服务端日志提取的实际 Prompt Token IDs 作为生成输入。

HF 参考与 TokenSpeed 使用不同实现及计算路径，因此不应预期逐 Token 必然完全一致。

## 9. 原始证据索引

主要实验目录：

```text
~/tp2_evidence_v2/
~/tp2_acceptance/
```

关键结果：

| 证据 | 路径 |
|---|---|
| Batch 测试脚本 | `~/tp2_evidence_v2/environment/run_batch.py` |
| TP 对照脚本 | `~/tp2_evidence_v2/environment/compare_tp.py` |
| 4B 三方对照 | `~/tp2_evidence_v2/4b/hf_reference/three_way_summary.json` |
| 4B HF 完整对照 | `~/tp2_evidence_v2/4b/hf_reference/comparison_15.jsonl` |
| 27B HF 完整对照 | `~/tp2_evidence_v2/27b/hf_reference/comparison_15.jsonl` |
| 27B HF 权重核验 | `~/tp2_evidence_v2/27b/hf_reference/hf_load_check.json` |
| 27B 首 Token Logits | `~/tp2_evidence_v2/27b/hf_reference/case06_logits.json` |
| 27B TP=1 失败日志 | `~/tp2_acceptance/27b/tp1/` |
| 各模型启动日志与运行记录 | `~/tp2_acceptance/{0.8b,4b,9b,27b}/` |

上述路径为实验服务器上的本地路径，**并非 GitHub 仓库内的附件**。成员 C 需要具有相应服务器文件访问权限，或单独取得精简证据包。

## 10. 交接说明与剩余工作

**成员 B 已完成：**

- 四个模型的 TP=2 推理可运行性验证。
- 0.8B、4B、9B 的 TP=1 对照。
- 4B 三方独立参考对照。
- 27B HF 权重加载与推理验证。
- 数值差异专项排查。
- 请求状态隔离测试及 TP 计算路径静态检查。

**成员 C 后续需要：**

1. 基于相同代码版本完成吞吐量、延迟、显存及性能优化实验。
2. 记录 TP=1 和 TP=2 的运行条件与硬件资源占用，避免将不同负载条件下的结果直接比较。
3. 在最终报告中分别呈现推理成功率、生成一致率、数值误差分析和性能数据。
4. 如实保留 27B TP=1 的资源限制，以及未完全解释的数值差异。

**最终判断：** 本阶段四模型 TP=2 推理与既定正确性对照已完成，可以交接性能评测与报告整合。尚未证明所有生成分歧的精确根因，不应表述为完全数值一致。
