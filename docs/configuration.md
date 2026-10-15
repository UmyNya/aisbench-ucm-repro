# 配置与脚本阅读顺序

配置采用 JSON。未写出的字段继承 [DEFAULTS](../src/aisbench_ucm/config.py)，未知字段报错，避免拼写错误悄悄失效。`--set dotted.key=value` 可重复使用；数值、布尔值、数组和对象按 JSON 解析，其余按字符串处理。相对路径相对于仓库根目录，输出目录相对于当前工作目录。路径支持环境变量，如 `$MODEL_PATH`。

## 常用参数

| 字段 | 含义 / 冻结默认值 |
| --- | --- |
| workload.n | 请求数，40 |
| workload.source | auto / frozen / synthetic / jsonl |
| workload.input_tokens / output_tokens | 40000 / 1 |
| workload.prefix_ratio | 0.8992，按 block_size 向下对齐 |
| workload.block_size | 输入对齐和观测单位，64；不等于强行传入 vLLM block-size |
| workload.seed / shuffle | 20260911 / true |
| load.concurrency / request_rate | 8 / 0；0 为闭环 |
| load.stream | true |
| experiment.repetitions | 3 |
| experiment.early_prefetch | [false, true] |
| experiment.alternate_order | true，偶数轮反转顺序 |
| server.max_num_batched_tokens / max_num_seqs | 8192 / 16 |
| server.max_model_len | 40960，至少覆盖输入+输出长度 |
| server.kv_cache_memory_bytes | 5372160000；null 让 vLLM 自动分配 |
| server.gpu_memory_utilization | 0.92 |
| server.gpu | 本机 GPU 索引或 UUID，多个用逗号隔开 |
| server.hf_overrides | 前14层、MTP 0；更换模型时可设 {} |
| server.block_size / long_prefill_token_threshold | null，保持 vLLM 默认 |
| server.enforce_eager / async_scheduling | false / false |
| ucm.l1_gib / l2_gib | 19 / 128 |
| ucm.posix_io_engine / io_direct | aio / true |
| preparation.enabled | 是否执行一次无关预热及逐条目标预热 |
| preparation.wait_dump | 用观测 RPC 确认原生写盘任务完成 |
| observation.enabled | 注入缓存来源和调度队列观测 |
| observation.require_exact_prefix_hits | 每条正式请求必须复用指定的前缀 token 数 |
| observation.reject_compilation_during_measurement | 测量窗口有新 GPU 二进制写入则不通过 |
| runtime.compile_cache | 可持久保留的编译缓存目录 |
| execution.*_timeout_s | 启动、请求、落盘、AISBench 阶段超时 |

缓存容量是配置的字节容量，不能对其他模型直接换算成原实验的 KV token 数。详细观测会保存实际布局；冻结模型实际 block=64，HBM 5000 blocks=320000 tokens，L1普通容量1079552 tokens，另有1024加载预留分片。

## 输入与顺序

冻结池含 51 项：0…49 是候选目标，50 是无关预热。N40/N35 分别顺序预热前40/35项；正式顺序先对50个索引使用同一个 Python Random seed 洗牌，再保留索引小于 N 的项。正式文本与预热文本不同，前35968 tokens按64对齐后可复用。N40正式输入 SHA256：

`40d9acae334b4f6654b9df8cca1e480cce65b50d982c8f564e00079412a569a5`

改变输入条件后，auto 使用本地 tokenizer 和固定 GSM8K 来源生成新输入，验证文本编码回 token 的长度、前缀及请求身份。生成失败会报错，不会悄悄截短或减少 N。

已有 JSONL 的每行需要 `question` 或 `prompt` 字符串。示例：

~~~bash
python3 scripts/run.py run --config configs/external.json \
  --set endpoint.tokenizer=/models/tokenizer \
  --set endpoint.model=my-model --set endpoint.base_url=http://localhost:8000 \
  --set workload.source=jsonl --set workload.dataset=/data/requests.jsonl \
  --set workload.n=100 --set workload.enforce_input_tokens=false \
  --output results/custom
~~~

使用 JSONL 并启用目标预热时，必须提供 `workload.warmup_dataset`。允许变长输入时仍记录并核对每条实际 token 数；关闭长度约束不等于忽略响应中的错误计数。

## 更换模型

已有服务模式仅需要 tokenizer 与接口。自动启动其他模型时，应修改模型路径、移除14层覆盖、调整长度和缓存容量，禁用仅在固定接口验证过的详细观测：

~~~bash
python3 scripts/run.py run \
  --set server.model_path=/models/other --set endpoint.tokenizer=/models/other \
  --set endpoint.model=other --set server.hf_overrides='{}' \
  --set server.enable_expert_parallel=false \
  --set ucm.layerwise_sync_fix=false \
  --set observation.enabled=false --set preparation.wait_dump=false \
  --set workload.input_tokens=4096 --set workload.output_tokens=128 \
  --set workload.n=35 --set experiment.repetitions=1 \
  --output results/other
~~~

该示例仍启用 UCM，因此模型需受所选 UCM/vLLM 支持。普通 vLLM 压测可额外设置 `ucm.enabled=false` 和 `experiment.early_prefetch=[false]`。详细来源与 waiting 观测仅对固定版本、layerwise、TP=DP=PP=1 进行验证，不承诺所有模型兼容。

## 脚本结构

| 文件 | 职责 |
| --- | --- |
| scripts/run.py、cli.py | 参数解析与命令分发 |
| config.py | 默认配置、覆盖、校验、实验顺序 |
| data.py、data_worker.py | 冻结输入或 tokenizer 生成输入 |
| service.py | 服务参数、独占检查、启动、布局验收、清理 |
| benchmark.py | 对固定上游 wrapper 的最小适配、AISBench 子进程 |
| runner.py | 预热→落盘等待→正式请求→验收的顺序 |
| metrics.py | 从 AISBench 时间戳与 SQLite 数组计算指标 |
| observation.py、queue_analysis.py | 缓存来源对账、队列积分、编译窗口验收 |
| runtime_hooks/ | 服务端轻量观测，绝不改变缓存命中或调度决策 |
| scripts/setup.py | 固定源码、隔离环境、UCM 原生编译 |
| scripts/artifacts.py | 下载校验、恢复归档编译缓存 |

每个运行目录只能创建一次。启动失败、中断和发压错误都会进入清理路径；不会终止其他人的 GPU 进程。编译缓存保留供后续使用，当前服务的临时 L2、共享内存和进程会清理。已有服务模式不管理远端服务生命周期。

## 指标口径

输入吞吐包含复用前缀，不等于新计算 token/s。TTFT、E2E来自 AISBench 实际请求时间戳；单 token 输出没有 TPOT；非流式调用不报告TTFT/TPOT，只报告E2E。多轮展示算术均值、样本标准差和合并请求分位数，不因某轮速度慢而剔除。

HBM来自最终匹配，L1/L2按原始物理分片来源计；L2预取到L1后仍计L2。off路径只能精确归因整组L1/L2，不虚构逐请求来源。waiting时间积分为观测间状态保持的估计；调度事件占比是反复观测的计数，不能当作独立请求概率或精确I/O阻塞时间。


## 失败后续跑

运行失败后可执行 `python3 scripts/run.py resume results/目录`。已合格组跳过，未合格组创建新的 attempt 目录；原尝试保留，汇总仅纳入合格结果。续跑会检查运行代码、源锁文件和正式/预热输入哈希没有变化。已安装的依赖环境也必须保持不变；如果需要修改代码、UCM版本或实验条件，应使用新的输出目录，不能把不同条件混成一组对照。
