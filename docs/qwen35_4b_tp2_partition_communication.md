# 切分与通信的对应关系（Qwen3.5-4B，典型层分析）

## 阅读提示

本节分析 **Qwen3.5-4B 在 TokenSpeed 标准 TP=2（两个 Rank、非 Query-Sharded、无 Token Sharding）配置下的典型层**。核心结论是：每个 Rank 持有输入 Hidden States 的逻辑副本，按输出通道或 Head 执行各自的 QKV、GDN、Attention 和 MLP 局部计算；需要在同一请求的各生成步骤间持续维护的 Conv State、Recurrent State 和 KV Cache，则按本地通道或 Head 分片保存和更新，正常 Prefill/Decode 不需要跨 Rank 交换这些状态本身。

**在本节讨论的层内主路径上，主要集合通信发生在 GDN/Full Attention 输出投影及 Dense MLP Down Projection 的局部贡献需要合并时。** 对当前标准 TP=2 的非 Token-Sharded 路径，通信管理器使用 All-Reduce，使两个 Rank 获得相同的完整 Hidden States。这里**不声称整个模型只有这些通信**：Embedding、输出处理及其他运行模式的通信不属于本节的典型层分析范围。

下文按 GDN / Full Attention / Dense MLP 三类结构，逐一列出两个 Rank 的权重、输入输出与中间张量形状，并集中汇总推理状态的分片与更新方式。

**证据标记：**

- **【形状推导】**：从本地 `config.json` 及 `qwen3_5.py` 并行层构造代码推导出的**逻辑形状**；未通过运行时 Hook 逐张量核实。
- **【代码核查】**：从 `CommManager`、并行 Linear 和相关调用点静态核查得到的条件分支；未用 HCCL Profiling 核对运行时每一次通信的次数、次序或耗时。
- **【已有实验】**：之前已保存的双 Rank 推理、TP 对照等测试证据；不等于本节逐算子 Profile。

**符号约定：** `T` 为某次 Forward 中的 Token 行数；`S` 为单请求的已缓存序列长度。权重按 `[out_features, in_features]` 表示；Conv1D 权重保留其额外维度。表中“完整尺寸”指未切分时的**逻辑张量**，不表示每个 Rank 都保留完整张量。两 Rank 的尺寸可以相同，但其分片内容不同。

模型基础配置：Hidden Size **2560**；**32 层**（**24 层 GDN + 8 层 Full Attention**，每四层 3 层 GDN + 1 层 Full Attention）；MLP Intermediate Size **9216**。

---

## 1. GDN 层

### 1.1 输入投影（Column Parallel）【形状推导】

\[
D_K = 16\times128 = 2048,\qquad D_V = 32\times128 = 4096
\]

| 张量 | 完整尺寸 | 分布方式 | Rank 0 | Rank 1 |
|---|---|---|---|---|
| 输入 X | `[T,2560]` | 逻辑复制 | `[T,2560]` | `[T,2560]` |
| Q | `[T,2048]` | 按 Key Head / 输出通道切分 | `[T,1024]` | `[T,1024]` |
| K | `[T,2048]` | 按 Key Head / 输出通道切分 | `[T,1024]` | `[T,1024]` |
| V | `[T,4096]` | 按 Value Head / 输出通道切分 | `[T,2048]` | `[T,2048]` |
| Z（Gate） | `[T,4096]` | 按 Value Head / 输出通道切分 | `[T,2048]` | `[T,2048]` |
| B、A | 各 `[T,32]` | 按 Value Head / 输出通道切分 | 各 `[T,16]` | 各 `[T,16]` |
| 合并投影权重 | `[12352,2560]` | 按输出通道分片 | `[6176,2560]` | `[6176,2560]` |

合并权重的输出维度为 `2048+2048+4096+4096+32+32=12352`。`qwen3_5.py` 中存在合并的 `MergedColumnParallelLinear` 路径，也存在取决于量化配置的拆分投影路径；**表中合并权重代表等价的逻辑总尺寸，不据此断言所有配置都执行一次物理融合 GEMM**。

两个 Rank 持有相同的输入 X，使用不同的本地权重算出对应输出通道；**此投影阶段不需要跨 Rank 集合通信**。

### 1.2 Causal Conv1D【形状推导】

GDN 卷积覆盖 Q/K/V 对应的通道：`2048+2048+4096=8192`，卷积核宽度为 4。

| 张量 | 完整逻辑尺寸 | 分布方式 | Rank 0 | Rank 1 |
|---|---|---|---|---|
| Conv1D 权重 | `[8192,1,4]` | 按 Q/K/V 通道切分 | `[4096,1,4]` | `[4096,1,4]` |
| Conv1D 输入通道 | `[T,8192]` | 按通道切分 | `[T,4096]` | `[T,4096]` |

各 Rank 对本地通道执行因果卷积，无须交换对方的卷积输入通道。**Conv State 的逻辑切分和 Decode 更新见第 4 节。**

### 1.3 Output Projection 与通信边界【形状推导 + 代码核查】

GDN 共有 `32×128=4096` 个 Value 输出通道，因此 `out_proj` 的**完整输入是 `[T,4096]`，每 Rank 的局部输入才是 `[T,2048]`**。

| 张量 | 完整尺寸 | 分布方式 | Rank 0 | Rank 1 |
|---|---|---|---|---|
| `out_proj` 输入 | `[T,4096]` | 按 Value Head / 输入通道切分 | `[T,2048]` | `[T,2048]` |
| `out_proj` 权重 | `[2560,4096]` | 按输入通道切分（Row Parallel） | `[2560,2048]` | `[2560,2048]` |
| 投影局部贡献 | `[T,2560]` | **局部部分和，待归约；不是输出特征切分** | `[T,2560]` | `[T,2560]` |
| All-Reduce 后结果 | `[T,2560]` | 逻辑复制 | `[T,2560]` | `[T,2560]` |

令本地输入为 `X_0、X_1`，对应分片权重为 `W_0、W_1`，则

\[
Y_0=X_0W_0^\top,\qquad Y_1=X_1W_1^\top,\qquad Y=Y_0+Y_1.
\]

**【代码核查】** `out_proj` 配置 `reduce_results=False`：局部贡献并非在 Linear 内立即归约，而是由后续 `CommManager.post_attn_comm()` 处理。在当前非 Query-Sharded、Attention/Dense TP 宽度一致且选择 All-Reduce 的路径中，执行 `all_reduce(hidden_states, attn.tp_group)`，使两个 Rank 都拿到完整输出。此处所说“`out_proj` 后通信”是**逻辑通信边界**，不表示 Linear 自身调用 HCCL。

---

## 2. Full Attention 层

模型具有 **16 个 Query Heads、4 个 KV Heads、Head Dimension 256**，配置开启 Attention Output Gate。

### 2.1 QKV 与 Attention Gate 切分【形状推导】

| 张量 | 完整尺寸 | 分布方式 | Rank 0 | Rank 1 |
|---|---|---|---|---|
| 输入 X | `[T,2560]` | 逻辑复制 | `[T,2560]` | `[T,2560]` |
| Q | `[T,16,256]` | 按 Query Head 切分 | `[T,8,256]` | `[T,8,256]` |
| K | `[T,4,256]` | 按 KV Head 切分 | `[T,2,256]` | `[T,2,256]` |
| V | `[T,4,256]` | 按 KV Head 切分 | `[T,2,256]` | `[T,2,256]` |
| Attention Output Gate | `[T,16,256]` | 按 Query Head 切分 | `[T,8,256]` | `[T,8,256]` |
| Attention 输出 | `[T,4096]` | 按 Query Head 切分 | `[T,2048]` | `[T,2048]` |
| QKV+Gate 合并投影权重（逻辑） | `[10240,2560]` | 按输出通道 / Head 切分 | `[5120,2560]` | `[5120,2560]` |

`10240=4096(Q)+1024(K)+1024(V)+4096(Gate)`。两个 Rank 分别计算本地 Head 的 QK、Softmax、Value 加权和 Gate 处理，**此过程中不需交换对方的 KV Heads**。此处未展开 Paged Attention 的具体物理缓存布局。

### 2.2 Output Projection 与通信边界【形状推导 + 代码核查】

Full Attention 输出为 `16×256=4096` 通道，`o_proj` 同样采用 Row Parallel：

| 张量 | 完整尺寸 | 分布方式 | Rank 0 | Rank 1 |
|---|---|---|---|---|
| Attention 输出 / `o_proj` 输入 | `[T,4096]` | 按 Query Head / 输入通道切分 | `[T,2048]` | `[T,2048]` |
| `o_proj` 权重 | `[2560,4096]` | 按输入通道切分 | `[2560,2048]` | `[2560,2048]` |
| 投影局部贡献 | `[T,2560]` | **局部部分和，待归约；不是输出特征切分** | `[T,2560]` | `[T,2560]` |
| All-Reduce 后结果 | `[T,2560]` | 逻辑复制 | `[T,2560]` | `[T,2560]` |

**【代码核查】** `o_proj` 配置 `reduce_results=False`，与 GDN 一样通过 `CommManager.post_attn_comm()` 的相应分支执行 All-Reduce。两种结构**复用同一个 Attention 类通信入口**，区别在于局部输出的计算方式。KV Cache 的逻辑切分与更新见**第 4 节**。

---

## 3. Dense MLP 层

MLP Intermediate Size 为 **9216**。

### 3.1 Gate / Up / Down 切分【形状推导】

| 张量 | 完整尺寸 | 分布方式 | Rank 0 | Rank 1 |
|---|---|---|---|---|
| 输入 X | `[T,2560]` | 逻辑复制 | `[T,2560]` | `[T,2560]` |
| Gate Projection 权重 | `[9216,2560]` | 按输出通道切分 | `[4608,2560]` | `[4608,2560]` |
| Up Projection 权重 | `[9216,2560]` | 按输出通道切分 | `[4608,2560]` | `[4608,2560]` |
| Gate Projection 输出 | `[T,9216]` | 按输出通道切分 | `[T,4608]` | `[T,4608]` |
| Up Projection 输出 | `[T,9216]` | 按输出通道切分 | `[T,4608]` | `[T,4608]` |
| 激活/逐元素乘积 | `[T,9216]` | 按 Intermediate Channel 切分 | `[T,4608]` | `[T,4608]` |
| Down Projection 权重 | `[2560,9216]` | 按输入通道切分（Row Parallel） | `[2560,4608]` | `[2560,4608]` |
| Down Projection 局部贡献 | `[T,2560]` | **局部部分和，待归约** | `[T,2560]` | `[T,2560]` |
| All-Reduce 后结果 | `[T,2560]` | 逻辑复制 | `[T,2560]` | `[T,2560]` |

Gate/Up 属于 Column Parallel，两个 Rank 各自计算和激活本地的 Intermediate Channels；Down Projection 属于 Row Parallel，对局部激活进行投影，得到尺寸相同但数值不同的两个部分和，必须合并才能恢复完整输出。

### 3.2 通信边界【代码核查】

MLP 局部贡献由 `CommManager.post_mlp_comm()` 转至 `post_dense_comm()`。在 **Dense TP 启用、`_shard(ctx) is None` 且 `use_all_reduce(False)` 为真** 的分支中，执行 `all_reduce(hidden_states, dense.tp_group)`。

这与 GDN / Full Attention 的 `post_attn_comm()` 属于不同入口，但在本节标准 TP=2、非 Token-Sharded 的 Dense 模型配置中均归约局部部分和。其他分支是否触发需结合具体运行时配置判断。

---

## 4. TP=2 推理状态分布与更新

这里的“状态”指**同一请求从 Prefill 延续到后续 Decode 步骤**必须维护的缓存或递归信息。**不同请求需要独立管理其状态**，而不是让一个请求的状态跨请求复用。

| 状态 | 所属层 | 分布方式 | 单请求完整逻辑形状 | 每 Rank 逻辑形状 | Decode 时如何更新 | 常规 TP 路径是否同步该状态本身 |
|---|---|---|---|---|---|---|
| Conv State | GDN | 按 Q/K/V 通道切分 | `[8192,3]` | `[4096,3]` | 滚动/更新固定长度历史窗口，写入最新通道值 | 否 |
| Recurrent State | GDN | 按 Value Head 切分 | `[32,128,128]` | `[16,128,128]` | 按递归更新规则修改已有 State，**不是追加新的一行** | 否 |
| KV Cache（分别对 K、V） | Full Attention | 按 KV Head 切分 | 各 `[S,4,256]` | 各 `[S,2,256]` | 为新 Token 写入本地 K/V，序列逻辑长度增加 | 否 |

- **Prefill**：处理 Prompt Token 并建立/更新对应的本地状态；具体内核可能按 Chunk 处理，不能将所有 State 简化为“直接追加”。
- **Decode**：各 Rank 读取和更新自己负责的 State 分片；常规 Head/Channel TP 不需要逐 Token 对 State 本身进行跨 Rank All-Reduce、All-Gather 或交换完整缓存。
- **请求隔离**：每个请求的状态必须通过各自的 Cache/State 槽位区分；“两个 Rank 各管一个分片”与“不同请求之间彼此隔离”是两个独立要求。
- **逻辑形状与物理布局**：上述是单请求的逻辑视图；实际 Cache Pool、Paged KV Cache 的存储形式还包括页表、槽位和其他布局维度。Conv State 的 `[C,3]` 表示 kernel width=4 时保留前三步上下文的抽象历史窗口，不承诺物理缓存一定恰好以 `[C,3]` 布局实现。

因此，本节所谓“状态不通信”专指**这三类 Head/Channel 局部状态没有逐步跨 Rank 同步的必要**；并不否定同一次 Forward 中投影局部贡献需要经集合通信合并。

---

## 5. 通信路径汇总

| 位置 | 数据分布 | 主要集合通信 | 代码位置 / 适用说明 |
|---|---|---|---|
| GDN 输入投影 | X 逻辑复制，输出 Q/K/V/Z/B/A 分片 | 无 | `MergedColumnParallelLinear`，输出按 Head/Channel 切分 |
| GDN Conv / Recurrent 更新 | State 按本地通道/Head 分片 | 不同步 State 本身 | 本地 GDN 计算 |
| GDN `out_proj` 的局部输出边界 | 两份 `[T,2560]` 局部部分和 | **All-Reduce** | `post_attn_comm()`，满足标准 TP 的 AR 分支条件 |
| Full Attention QKV / Gate | 输入复制，Head 分片 | 无 | `QKVParallelLinear` |
| Full Attention 计算 | 本地 KV Heads 和 Cache | 不同步 KV Cache 本身 | `PagedAttention` 本地 Head 计算 |
| Full Attention `o_proj` 的局部输出边界 | 两份 `[T,2560]` 局部部分和 | **All-Reduce** | `post_attn_comm()`，同上述 AR 分支 |
| MLP Gate/Up | 输入复制，Intermediate Channels 分片 | 无 | Column Parallel |
| MLP Down 的局部输出边界 | 两份 `[T,2560]` 局部部分和 | **All-Reduce** | `post_mlp_comm()` → `post_dense_comm()`，须满足 `_shard(ctx) is None` 等条件 |
| Residual / Norm | 在上述归约后处理完整 Hidden States | 此处无需为本地 Norm 单独同步 | 不表示整个模型不存在其他通信 |

**【代码核查】通信分支条件：**

- `post_attn_comm()` 在 `query_sharded` 或没有 Attention TP 时直接返回；否则，若 `use_all_reduce(self.is_moe)` 为真，使用 `all_reduce(..., attn.tp_group)`。不满足时可走 Token Reduce-Scatter 等其他路径。
- `post_dense_comm()` 在未启用 Dense TP 时不通信；在 `_shard(ctx) is None` 且 `use_all_reduce(False)` 为真时执行 `all_reduce(..., dense.tp_group)`，其他配置可能执行 All-to-All 或 Token Reduce-Scatter。
- 4B 已检查的 TP=1 与 TP=2 启动配置都包含 `enable_allreduce_fusion=False`；本分析不将融合 All-Reduce + Norm 视为本次已执行路径。

**范围限定：** 此表描述所选 GDN、Full Attention、Dense MLP 层的主路径。TokenSpeed 源码中还存在 Embedding 归约、Reduce-Scatter、All-Gather、All-to-All 等能力；不能从本表推断整个模型运行期间“仅有”三类 All-Reduce，也不能把支持的其他分支写成当前实验已实际触发。

## 6. 通信数据量（量级估计）

对于本节三类典型层的单次 `[T,2560]` BF16 All-Reduce，**每 Rank 参与归约的输入张量**大小为

\[
T\times2560\times2 = 5120T\ \text{bytes}.
\]

- Decode：若本轮只有 1 个 Token 行，`T=1`，输入张量约 **5 KiB**。
- Prefill：若本轮有 128 个 Token 行，`T=128`，输入张量约 **640 KiB**。

这是**每个归约调用的逻辑输入 Payload**，不是所有层通信的总量，也不是 HCCL 链路上传输的实际字节数或测得的延迟。真实时延、通信与计算重叠情况需要 Profiling 或分段计时实验支持。

## 7. 验证状态与限制

**已有证据：**

- **【已有实验】** 四模型 TP=2 双 Rank 推理均成功完成既定 15 条请求；4B 的 HF / TokenSpeed TP=1 / TP=2 三方实际输入 Token IDs 相同（15/15）。
- **【代码核查】** 4B TP=1 与 TP=2 的主要 Attention/GDN Backend 配置相同；启动日志显示两者 `enable_allreduce_fusion=False`。
- **【代码核查】** GDN `out_proj`、Full Attention `o_proj` 均由 `CommManager.post_attn_comm()` 选择通信路径；Dense MLP 的局部结果由 `post_mlp_comm()` / `post_dense_comm()` 处理。标准非 Token-Sharded TP 分支可执行 All-Reduce。
- **【已有实验】** 先前完成过 GDN `out_proj` 的权重分片和 BF16/FP32 数值比较；其结果属于局部算子证据，**不能据此声称对整层及所有状态更新路径完成验证**。

**未验证或不在本节范围内：**

- 张量形状为静态逻辑推导，未逐项 Hook Dump 实际运行时张量；不同量化/融合配置可能改变物理张量与权重布局。
- 未使用 HCCL Profiling 逐次统计通信算子、调用次数及执行耗时；因此“实际执行了多少次 All-Reduce”未在本节证明。
- Cache Pool 内三类 State 的物理布局、请求调度及全部并发场景未在此逐项展开。
- 生成 Token 序列在不同 TP / HF 路径间并非全部一致，差异与有限精度路径变化相容，但不能归因于单一原因或直接证明实现绝对正确。
