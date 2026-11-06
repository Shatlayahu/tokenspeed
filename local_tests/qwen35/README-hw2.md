# HW2 0.8B TP 性能对照

从仓库根目录运行。服务使用 `test/ci_system/serve_qwen35_tp1.sh` 或
`test/ci_system/serve_qwen35_08b_tp2.sh`；先后启动，不能同时占用同一设备。
服务脚本自行创建 Scheduler 进程，不使用 `torchrun`。两个服务均设置相同权重、
BF16、`mha`、greedy、关闭 Thinking/前缀缓存、`TS_MAX_MODEL_LEN=4096`、
`TS_MAX_NUM_SEQS=8`、`TS_GPU_MEMORY_UTILIZATION=0.30`。实验服务器的
`TS_PORT=8010` 对应 HTTP 控制端口 `8011`，应以实际启动日志为准。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source ~/venvs/tokenspeed/bin/activate
python local_tests/qwen35/benchmark_tp.py \
  --base-url http://127.0.0.1:8011 \
  --model qwen3.5-0.8b \
  --model-dir ~/models/Qwen3.5-0.8B \
  --tp 2 --physical-devices 6,7 \
  --input-lengths 32,1024 --concurrencies 1,4 \
  --output-tokens 32 --warmups 1 --repeats 3 \
  --output-dir ~/hw2-c-results/08b-tp2/matrix
```

TP=1 使用同一命令，只改 `--tp 1 --physical-devices 6` 和输出目录。
每个配置独立预热，然后按并发数同时发起一波请求；重复三波。
`benchmark_tp.json` 保存逐请求数据、每波合并后的统计及两颗物理 NPU 的
`npu-smi` 样本。TTFT 从请求发出到第一个非空内容块；TPOT 用首末内容块时间差
除以服务端报告的生成 token 数减一；系统吞吐为本配置全部成功生成 token 数除以
三波的总墙钟时间。失败请求不计入吞吐分子，但保留在原始记录并计入墙钟时间。
P50/P95 取排序后最邻近的实际观测值；每格样本数较少，应同时查看逐请求记录。

通信相关实验固定 1024 输入 Token，只改变并发为 1/4/8：

```bash
python local_tests/qwen35/benchmark_tp.py \
  --base-url http://127.0.0.1:8011 \
  --model qwen3.5-0.8b --model-dir ~/models/Qwen3.5-0.8B \
  --tp 2 --physical-devices 6,7 \
  --input-lengths 1024 --concurrencies 1,4,8 \
  --output-tokens 32 --warmups 1 --repeats 3 \
  --output-dir ~/hw2-c-results/08b-tp2/communication
```

HBM 是物理设备总占用，包含其他进程；`hbm_pre_run_total_mb` 在服务加载完成后
采集。它不能直接当作模型独占显存或可用 KV cache。长输入并发下应同时查看
TTFT P95，避免只用吞吐概括排队体验。Profiling 会改变运行开销，时间线数据
只用于计算、通信与等待的归因，不作为未加 profiler 的延迟基线。

需要采集 CANN/HCCL 时间线时，在启动 TP=2 服务前设置以下环境变量，
使用 `profile_tp2_msprof.sh` 替代普通服务启动脚本。另一个终端用上述
benchmark 发请求，结束后以 `msprof --export=on --output=<PROF_...目录>`
导出 `op_summary`、`task_time`、`communication_statistic`，逐 Rank 检查
是否生成设备数据。`--delay` 从服务启动算起，需按实际启动时间调节。

```bash
export TS_REPO_ROOT=$PWD TS_VENV_ROOT=$HOME/venvs/tokenspeed
export TS_MODEL_DIR=$HOME/models/Qwen3.5-0.8B
export TS_PROFILE_OUTPUT=$HOME/hw2-c-results/08b-tp2/profile
export TS_PROFILE_DELAY=30 TS_PROFILE_DURATION=180
export ASCEND_RT_VISIBLE_DEVICES=6,7 TS_PORT=8010
bash local_tests/qwen35/profile_tp2_msprof.sh
```

本次采集中只有 Rank 1 导出完整设备时间线，且 profiler 显著增加请求耗时。
仓库保留精简 HCCL 原表和分析窗口摘要，服务器保留完整原始 profiler 目录。
汇报时不能用单 Rank 时间线推算双 Rank 端到端通信占比，也不能把任务等待
字段当成服务排队时间。

较大模型仍需一份 TP=2 吞吐/延迟原始记录。取得 9B 或 27B 权重和对应已启动
服务后，复用 `benchmark_tp.py`：设置正确的 `--model` 服务注册名、
`--model-dir`、`--tp 2 --physical-devices 6,7`，选择两档输入与并发，并记录
两 Rank 日志及运行期 HBM。当前账号没有这些权重，仓库现有 27B 证据只覆盖
连续请求与设备总 HBM 快照，尚不足以填写性能数值。
