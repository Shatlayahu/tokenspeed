# Qwen3.5 Ascend TP=2 推理正确性实验记录

**项目：** TokenSpeed — Qwen3.5 Ascend NPU 适配

**工作分支：** `feat/qwen35-9b-27b-tp2`

**记录日期：** 2026-11-03

**实验记录及正确性对照：** 成员 B

**状态：** 四模型 TP=2 连续推理及已规划的正确性对照已完成；成员 B 的 9B/27B 启动、分片及设备快照证据已整理。

## 1. 实验目标与结论

本阶段验证 Qwen3.5-0.8B、4B、9B、27B 在 Ascend NPU 上使用 TokenSpeed 实现 TP=2 推理的可运行性、连续请求稳定性和与参考计算路径的输出一致性，并为课程验收保留真实双 Rank 参与计算的证据。

**主要结论：**

1. 四个模型均成功执行 TP=2 的 15 条连续测试请求；这不是仅启动进程或加载权重的测试。

2. 0.8B、4B、9B 完成 TokenSpeed TP=1/TP=2 对照；27B 在当前单卡缓存/资源配置下不能完成 TP=1 推理，故以独立 Hugging Face（HF）实现作参考，**没有编造 27B 单卡基线**。

3. 9B、27B 的服务日志分别确认 `world_size=2`、Rank 0/1 初始化以及双方加载同一模型的局部权重；设备快照记录了两张目标物理 NPU 的运行期占用。

4. 各模型有部分生成 Token 序列不完全一致；这一事实本身不能证明 TP 实现错误，也不能证明差异全部由 BF16 引起。

5. 4B 的 HF/TP=1/TP=2 三方对照（15/15 输入 Token IDs 一致）显示：即使不启用 TP，HF 与 TokenSpeed 也可能产生序列分歧。

6. 27B `case_06` 首 Token 的两个主要候选在 HF 中相差 0.125 Logit，属于较接近的候选；但尚未取得 TokenSpeed 同一步的完整 Logits，根因仍未确定。

7. 静态检查未发现 TP=1 与 TP=2 异常切换主要 Attention/GDN Backend 的证据；两者必然存在分片 GEMM 和跨卡归约的计算差异。

**结论范围：** 已证明上述测试负载在真实双 Rank TP=2 下可完成，并保留正确性对照；不意味着每个算子逐元素等价、所有输入均正确或所有数值差异已解释。

## 2. 实验环境与配置

| 项目 | 配置 |
|---|---|
| 硬件 | Ascend 910 系列 NPU（课程使用的 910C 环境），每个目标 Chip 的 `npu-smi` 容量为 65,536 MB |
| 使用设备 | `ASCEND_RT_VISIBLE_DEVICES=6,7`（启动时设置） |
| CANN | 9.0.0 |
| PyTorch / torch-npu | 2.9.0 / 2.9.0 |
| Transformers | 5.12.0 |
| 模型权重及计算精度 | 相同模型 Checkpoint，BF16 |
| Full Attention Backend | `mha` |
| Sampling Backend | `greedy`，`temperature=0` |
| 张量并行 | TP=1 / TP=2；TP=2 `world_size=2` |
| 分布式 NPU 通信 | HCCL 后端（框架路径）；Gloo 日志只证明相应进程组连接，不能代替 HCCL Profiling |
| All-Reduce + Norm Fusion | 已核对 4B TP=1 与 TP=2 均为 `False` |
| 模型输入模式 | `language_model_only=True`，关闭 Thinking |

启动参数以每次实验对应的 `server.log` 为准，不同模型的显存利用率、端口和缓存设置不应混写为相同配置。9B TP=2 日志记录 `max_model_len=4096`、`gpu_memory_utilization=0.55`；27B TP=2 记录 `max_model_len=4096`、`gpu_memory_utilization=0.65`。

## 3. 测试方法与对照条件

统一测试脚本：`~/tp2_evidence_v2/environment/run_batch.py`。

测试集为 15 次请求：12 条不同 Prompt（含简短任务及较长生成任务）和 A1 → B → A2 三次交错状态隔离请求。测试以相同 Prompt、输出长度设置、Greedy 解码及 Thinking 关闭配置运行；可运行 TP=1 的模型采用同一模型目录的权重与 BF16 精度。HF 参考使用对应的 BF16 权重和实际 Prompt Token IDs，并关闭采样。

TokenSpeed 的真实生成 Token IDs 按 Request ID 从服务端日志提取，而非仅将 API 文本重新编码。**Exact Match** 指整个输出 Token ID 序列完全一致，不仅是答案语义相近。

需要注意：15 次请求包含重复的状态隔离样本，并非 15 个统计独立样本；完整序列的某处发生分歧后，自回归输入会改变，因此 Exact Match 不宜直接解释为“模型正确率”或用于断定哪种计算路径更准确。A1 → B → A2 是交错请求隔离测试，不能等同于覆盖所有并发调度情形。

## 4. 四模型连续推理和正确性结果

| 模型 | TP=2 连续请求 | TP=1 连续请求 | 参考方案 |
|---|---:|---:|---|
| Qwen3.5-0.8B | 15/15 成功 | 15/15 成功 | TokenSpeed TP=1 |
| Qwen3.5-4B | 15/15 成功 | 15/15 成功 | TokenSpeed TP=1 + HF |
| Qwen3.5-9B | 15/15 成功 | 15/15 成功 | TokenSpeed TP=1 |
| Qwen3.5-27B | 15/15 成功 | 未能完成 | HF BF16（双 NPU 层级 Device Mapping，非 TP） |

### 4.1 输出 Token IDs 一致性

| 模型 | 对照路径 | 完整序列一致 |
|---|---|---:|
| 0.8B | TP=1 vs TP=2 | 11/15 |
| 4B | TP=1 vs TP=2 | 8/15 |
| 4B | HF vs TP=1 | 6/15 |
| 4B | HF vs TP=2 | 8/15 |
| 9B | TP=1 vs TP=2 | 10/15 |
| 27B | HF vs TP=2 | 8/15 |

已完成的测试请求均有成功响应；A1 与 A2 的输出在相应测试中一致。此项仅说明当前交错测试没有观察到状态串扰，不代表所有并发请求与 Cache 调度路径均已验证。

### 4.2 27B TP=1 不可用的原因

对 27B TP=1 先后尝试调整最大模型长度、缓存/快照回收相关参数，但运行时报告快照池或可用单请求 Token 容量不足，未成功完成单卡推理。原始失败日志：

- `~/tp2_acceptance/27b/tp1/server_first_failure.log`

- `~/tp2_acceptance/27b/tp1/server_second_failure.log`

- `~/tp2_acceptance/27b/tp1/server.log`

**记录范围：** 当前共享服务器的单 Chip 资源及所试配置不足；不能推断该模型在所有单卡条件下一定无法运行，更不能填写一个不存在的 TP=1 性能或 Token 对照数据。

## 5. 9B/27B 双 Rank 运行、设备映射和内存证据（成员 B）

本节专门对应课程要求中“两个 Rank 的启动与运行记录、设备映射和内存占用”。0.8B/4B 的部署和双 Rank 运行证据由成员 A 维护；它们的正确性对照汇总保留在第 4 节。

### 5.1 两个 Rank 确实参与同一模型的 TP=2

| 证据 | 9B | 27B |
|---|---|---|
| 服务参数 | `world_size=2`, `attn_tp_size=2`, `nprocs_per_node=2`, `device='npu'` | 相同 |
| Rank 初始化 | `global rank: 0 / attn_tp_rank: 0`，`global rank: 1 / attn_tp_rank: 1` | 相同 |
| 双方权重加载 | Rank 0/1 均出现 `Load weight begin` 和 `[TP_SHARD_CHECK]` | 相同 |
| GDN `out_proj` 完整 Checkpoint 权重 | `[4096,4096]` | `[5120,6144]` |
| 每 Rank 实际本地权重 | `[4096,2048]`，沿输入维切分 | `[5120,3072]`，沿输入维切分 |
| 连续请求 | 15/15 完成 | 15/15 完成 |

例如 `server.log` 内的 `[TP_SHARD_CHECK]` 同时给出 `rank=0/2` 与 `rank=1/2` 对同一层的 `checkpoint_shape`、`local_shape`、`input_dim=1`。这是**同一模型权重按 TP 规则在两 Rank 上加载局部形状**的直接证据，与运行期双设备占用和 15 条推理结果共同支持真实 TP=2，而非两份完整模型副本。

**交付给成员 C 的精简摘要：**[`9b_tp2_runtime.txt`](evidence/9b_tp2_runtime.txt)、[`27b_tp2_runtime.txt`](evidence/27b_tp2_runtime.txt)。15 条 Case 名称见 [`9b_tp2_cases.json`](evidence/9b_tp2_cases.json) 和 [`27b_tp2_cases.json`](evidence/27b_tp2_cases.json)；这些 Case 清单本身不包含成功状态，成功率证据仍在原始 API 响应中。

**原始证据文件：**

- 9B：`~/tp2_acceptance/9b/server.log`、`~/tp2_evidence_v2/9b/tp2/responses.jsonl`

- 27B：`~/tp2_acceptance/27b/server.log`、`~/tp2_evidence_v2/27b/tp2/responses.jsonl`

上述日志清楚证明双 Rank 初始化和加载不同分片，响应文件证明服务完成连续请求；**尚未通过逐 Rank 的逐 Token Trace 或 HCCL Profiler 直接测量每步执行和通信次数**，不可把上述证据表述为已取得这类 Profiling 数据。

### 5.2 设备映射

启动设置为 `ASCEND_RT_VISIBLE_DEVICES=6,7`，服务器参数为 `base_gpu_id=0`、`gpu_id_step=1`、`world_size=2`。按可见设备顺序及框架的常规 Rank 设备选择规则，推导映射如下：

| Rank | 进程内可见设备编号 | 物理 NPU（推导） |
|---|---:|---:|
| Rank 0 | 0 | 6 |
| Rank 1 | 1 | 7 |

**证据边界：** 服务日志确认两个逻辑 Rank 和 `base_gpu_id/gpu_id_step` 参数，`npu-smi` 确认物理 NPU 6/7 上有 Scheduler 进程；当前摘录没有单行日志直接打印 `rank → physical NPU` 的对应关系，因此表内映射标为**依据启动配置推导**，不是直接实测绑定结果。

### 5.3 运行期 NPU HBM 占用快照

| 模型 | 物理 NPU 6 总 HBM | 物理 NPU 7 总 HBM | 每 Chip 容量 |
|---|---:|---:|---:|
| 9B TP=2 | 41,932 MB | 41,333 MB | 65,536 MB |
| 27B TP=2 | 48,973 MB | 48,377 MB | 65,536 MB |

数据来自模型运行期间保存的 `npu-smi` 快照：

- `~/tp2_acceptance/9b/npu_running.txt`

- `~/tp2_acceptance/27b/npu_running.txt`

这两个数值是**设备级总占用的单次观测**，包括同设备其他进程，**不是模型独占 HBM，也不是单模型峰值 HBM**。快照的进程表显示物理 NPU 6/7 上除本次服务进程外仍有其他进程，故不能用总占用减法在没有一致基线和进程归属核对的情况下推导模型专属显存。

## 6. 4B 三方正确性实验

为区分 TP 切分带来的差异与框架实现差异，比较 HF BF16、TokenSpeed TP=1、TokenSpeed TP=2 三种路径。**三方实际输入 Token IDs 全部匹配：15/15。三方逐 Case 汇总见 [`4b_three_way_summary.json`](evidence/4b_three_way_summary.json)（精简证据）。**

| 对照组合 | 完整序列一致 |
|---|---:|
| HF vs TP=1 | 6/15 |
| HF vs TP=2 | 8/15 |
| TP=1 vs TP=2 | 8/15 |
| 三方首 Token | 15/15 一致 |

- `case_03`、`case_10`：TP=1 与 TP=2 整条输出一致，HF 不同。

- `case_04`、`case_09`：HF 与 TP=2 整条输出一致，TP=1 不同。

- 15 条三方对照的首 Token 全部一致。

**解释：** HF 不总是更接近 TokenSpeed TP=1，且完整序列一致率差异较小、样本有重复；不能据此声称 TP=2 比 TP=1 更准确，也不能把所有 HF/TP 分歧归因于张量并行错误。

## 7. 数值差异专项分析

### 7.1 0.8B：TP=1 与 TP=2 的差异

对 `case_09` 的已有排查显示：

- TP=1/TP=2 实际输入 Token IDs 完全一致，共 142 个，但首 Token 选择不同。

- 首步 Logits 差异：MAE ≈ 0.03114、RMSE ≈ 0.03974、最大绝对误差 ≈ 0.203125。

- 最终 Hidden States MAE ≈ 0.05214；不同层的误差随深度逐渐增长。

- BF16 HCCL All-Reduce 的合成数值实验出现非零误差，FP32 对照亦有差异；检查范围内的 GDN `out_proj` 分片权重重组一致。

这些观察与分片计算、运算顺序及有限精度舍入导致偏移的解释相容，但不证明 BF16 是全部分歧的唯一原因。**逐层误差积累**（同一次 Forward 内）与**自回归分叉**（后续生成步）是两种不同机制；首 Token 不同不能用后者解释，但可以与前者相容。

### 7.2 27B：`case_06` 首 Token near-tie

该请求输入长度为 21 Tokens；HF 首 Token ID 为 `332`，TokenSpeed TP=2 为 `49338`。HF 单步 Forward 的 Logits：

| 候选 Token ID | HF Logit |
|---|---:|
| `332` | 30.625 |
| `49338` | 30.500 |
| `13962` | 27.625 |
| `37405` | 27.500 |
| `8160` | 26.875 |

两名最高候选在 HF 下的差值为 **0.125**。这说明 HF 对两个候选的偏好较接近，计算路径的数值偏移**可能**改变 Argmax。尚无 TP=2 同一步完整 Logits，不能确认差异幅度与根因；本例应记为**尚未解释的 near-tie 输出分歧**，而不是已经定位的 BF16 错误或 TP Bug。

证据：仓库精简副本 [`27b_case06_logits.json`](evidence/27b_case06_logits.json)；服务器原始文件 `~/tp2_evidence_v2/27b/hf_reference/case06_logits.json`。

## 8. TP=1 / TP=2 计算路径及切分通信说明

### 8.1 Backend 静态检查

已检查模型层、Linear、`CommManager`、GDN 配置和 Backend 注册逻辑：

1. TP=1/TP=2 的 Full Attention Backend 均配置为 `mha`；Qwen3.5 GDN 经 `hybrid_linear_attn` 路径。

2. TP Size 影响局部 Head、通道、缓存 State 形状及分片 GEMM；TP=2 增加集合通信。

3. 4B TP=1 和 TP=2 均设置 `enable_allreduce_fusion=False`。

4. 检查范围内未发现因 TP Size 而异常切换主要 Attention/GDN Backend 的证据。

**限制：** 相同 Backend 名称不保证相同底层编译 Kernel；不同张量形状、Tile、计算顺序与归约精度仍可能导致输出差异。这是静态分析，不是 Kernel Trace。

### 8.2 主要通信位置、内容和集合通信类型

详细的 4B 分片张量表、两 Rank 权重/输入输出形状，以及 Conv State、Recurrent State、KV Cache 的切分方式单独整理在：

**`docs/qwen35_4b_tp2_partition_communication.md`**

当前标准非 Token-Sharded TP=2 的主要**层内输出归约路径**为：

| 位置 | 待合并数据（每 Rank） | 集合通信 | 代码入口 |
|---|---|---|---|
| GDN `out_proj` 后 | `[T,2560]` 的 BF16 局部部分和 | All-Reduce | `CommManager.post_attn_comm()` |
| Full Attention `o_proj` 后 | `[T,2560]` 的 BF16 局部部分和 | All-Reduce | `CommManager.post_attn_comm()` |
| Dense MLP Down Projection 后 | `[T,2560]` 的 BF16 局部部分和 | All-Reduce | `CommManager.post_mlp_comm()` → `post_dense_comm()` |

GDN/Attention 的输出投影设置 `reduce_results=False`，由 `CommManager` 在后续通信边界执行归约；Dense MLP 在满足 `mapping.dense.has_tp`、`\_shard(ctx) is None` 和相应 TP 分组条件时执行 All-Reduce。框架还支持 Reduce-Scatter、All-Gather、All-to-All 等其他分支，但不能在没有运行时证据的情况下把它们说成当前实验实际调用的通信。

GDN Conv State、Recurrent State 及 Full Attention KV Cache 按各 Rank 的本地通道或 Head 保存和更新，标准 Prefill/Decode 中不需要每步跨 Rank 合并完整 State。**这不意味着整个模型只有三处通信**：Embedding、输出及其他框架级操作也可能有通信；此表只汇总三类典型层的主要归约点。

对 4B 而言，每次上述 All-Reduce 的每 Rank 输入逻辑大小为 `T × 2560 × 2 = 5120T` Bytes（BF16），只是参与归约的张量规模，**不是实际链路流量、耗时或整个模型的总通信量**。

## 9. HF 参考实现与权重核验

27B 使用独立 HF `Qwen3_5ForCausalLM` 参考：

- 将原多模态 Checkpoint 的语言模型权重映射到 Text-only 模型结构，加载 851/851 项预期权重。

- 无 Missing、Unexpected 或 Mismatched 权重；抽样参数与原始 Checkpoint 对照一致。

- BF16，采用两张 NPU 的**层级 Device Mapping**，不是 TP；不能把它作为 27B 单 Chip TP=1 基线。

- 从 TokenSpeed TP=2 服务端日志获取实际 Prompt Token IDs，按相同任务设置完成 15 条 HF 参考生成。

不同框架的算子实现与数值路径可能不同，因此独立参考主要用于发现显著不一致，不意味着必须得到逐 Token 完全相同的输出。

## 10. 交付证据与原始文件索引

本次交接采用**仓库内精简证据 + 服务器原始证据**两级结构。成员 C 可以直接通过当前分支阅读精简证据；如需核验完整 API 响应、原始服务日志或失败调用，再使用服务器原始文件。下列仓库路径均相对于仓库根目录；Markdown 链接相对于本文所在的 `docs/` 目录。

### 10.1 已整理到 Git 工作区的精简证据

| 证据及用途 | 仓库内文件 | 能证明什么／限制 |
|---|---|---|
| 9B 双 Rank 启动及切分摘要 | [`docs/evidence/9b_tp2_runtime.txt`](evidence/9b_tp2_runtime.txt) | `world_size=2`、Rank 0/1 初始化、同层权重在两 Rank 的局部形状、物理 NPU 6/7 的设备级 HBM 快照；不是逐步执行 Trace |
| 27B 双 Rank 启动及切分摘要 | [`docs/evidence/27b_tp2_runtime.txt`](evidence/27b_tp2_runtime.txt) | 同上；不包含全部服务日志 |
| 9B 请求 Case 清单 | [`docs/evidence/9b_tp2_cases.json`](evidence/9b_tp2_cases.json) | 列出 15 条请求的 Case 名称和记录数；**不能单独证明 15/15 成功** |
| 27B 请求 Case 清单 | [`docs/evidence/27b_tp2_cases.json`](evidence/27b_tp2_cases.json) | 同上；**不能单独证明 15/15 成功** |
| 4B HF / TP=1 / TP=2 三方汇总 | [`docs/evidence/4b_three_way_summary.json`](evidence/4b_three_way_summary.json) | 每条 Case 的三方输出一致性及首 Token 比较；不包含完整生成文本 |
| 27B `case_06` HF Logits | [`docs/evidence/27b_case06_logits.json`](evidence/27b_case06_logits.json) | HF 首步前五候选和 0.125 Margin；不包含 TP=2 完整首步 Logits |
| 典型层切分与通信分析 | [`docs/qwen35_4b_tp2_partition_communication.md`](qwen35_4b_tp2_partition_communication.md) | 4B GDN、Full Attention、Dense MLP 的分片形状、状态分布及静态通信路径 |

这些文件是在服务器工作区生成或准备提交的交付材料；**只有完成 Git Commit 和 Push 后，成员 C 才能从远端分支获取**。本记录不将这些精简摘要称为完整原始实验记录。

### 10.2 仅保留在实验服务器上的原始证据

原始目录：

```text
~/tp2_evidence_v2/
~/tp2_acceptance/
```

| 原始证据 | 服务器路径 |
|---|---|
| Batch 请求脚本 | `~/tp2_evidence_v2/environment/run_batch.py` |
| TP=1 / TP=2 对照脚本 | `~/tp2_evidence_v2/environment/compare_tp.py` |
| 9B TP=2 原始启动、Rank 和分片日志 | `~/tp2_acceptance/9b/server.log` |
| 9B TP=2 原始 NPU 快照 | `~/tp2_acceptance/9b/npu_running.txt` |
| 9B TP=2 完整 API 响应（验证请求成功） | `~/tp2_evidence_v2/9b/tp2/responses.jsonl` |
| 9B TP=1 服务日志 | `~/tp2_acceptance/9b/tp1/server.log` |
| 27B TP=2 原始启动、Rank 和分片日志 | `~/tp2_acceptance/27b/server.log` |
| 27B TP=2 原始 NPU 快照 | `~/tp2_acceptance/27b/npu_running.txt` |
| 27B TP=2 完整 API 响应（验证请求成功） | `~/tp2_evidence_v2/27b/tp2/responses.jsonl` |
| 27B TP=1 三次失败日志 | `~/tp2_acceptance/27b/tp1/` |
| 4B 三方完整对照 | `~/tp2_evidence_v2/4b/hf_reference/comparison_15.jsonl` |
| 27B HF 完整输出对照 | `~/tp2_evidence_v2/27b/hf_reference/comparison_15.jsonl` |
| 27B HF 权重加载核验 | `~/tp2_evidence_v2/27b/hf_reference/hf_load_check.json` |
| 27B 首步 HF Logits 原始结果 | `~/tp2_evidence_v2/27b/hf_reference/case06_logits.json` |

成员 C 可直接查看仓库中的精简记录；如需独立复核 `15/15` 成功率、TP=1 与 TP=2 的全部输出序列或 27B 单卡启动失败原因，应取得相关原始文件的读取权限或由成员 B 单独导出。**远端仓库的六份精简证据并不包含所有成功状态及全部原始对照日志**。

### 10.3 证据解读和共享限制

1. 27B TP=1 在当前单卡测试条件下无法完成推理，因此不存在该模型的 TP=1 正确性或性能基线。
2. 9B/27B HBM 数字为**运行时单次设备总占用**，包括其他进程，不是模型独占显存或峰值显存。
3. Rank 0 → 物理 NPU 6、Rank 1 → 物理 NPU 7 为根据可见设备设置和启动参数**推导**的映射，当前证据不包含逐 Rank 物理绑定的直接打印记录。
4. 目前还没有逐 Token 的逐 Rank Trace，也没有完成 HCCL 通信耗时 Profiling；后者属于成员 C 的性能分析工作。
5. Exact Match 不等于模型语义正确率；现有数值分歧的根因并未全部确定。
6. 原始日志包含共享服务器进程、运行参数等信息；共享或公开上传前需要审查。不要直接将完整原始日志或模型输出批量提交到公共仓库。
