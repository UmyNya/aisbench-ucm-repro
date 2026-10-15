# aisbench-ucm-repro

独立的 AISBench / vLLM / UCM 性能实验仓库。保留原 N40 实验的配置、输入、预热规则与汇总，用同一套脚本运行 N35 或其他负载。

- **完整实验**：逐组启动 vLLM/UCM，准备缓存，执行 Early Prefetch 开关对照，验收并清理资源。
- **已有服务**：连接 OpenAI completions 接口运行 AISBench，无需本地 GPU 或 UCM。
- **可读脚本**：入口是 [scripts/run.py](scripts/run.py)，流程在 [runner.py](src/aisbench_ucm/runner.py)；配置、输入、服务、发压和分析各自独立。没有 N35 专用脚本。

[English quickstart](docs/quickstart-en.md) · [配置与流程](docs/configuration.md) · [环境与复现边界](docs/environment.md) · [原实验汇总](reports/reference-n40.md) · [历史分析回归](reports/historical-analysis-validation.json) · [本次验证](reports/validation.md)

## 安装

需要 Linux x86_64、Python 3.12、Git。服务端还需要 C/C++ 编译器、CMake、Ninja、libaio/libnuma 开发包，以及兼容 CUDA 13 的 NVIDIA 驱动。运行目录建议放在 SSD。

~~~bash
git clone https://github.com/UmyNya/aisbench-ucm-repro.git
cd aisbench-ucm-repro
python3 scripts/setup.py sources
python3 scripts/setup.py client
# 仅完整实验需要：
python3 scripts/setup.py server --jobs 4
~~~

源码固定在 [sources.lock.json](sources.lock.json)，两个独立 Python 环境使用 [env/](env/) 中的精确版本清单。首次安装会下载较多 CUDA / PyTorch 依赖；有 wheel 缓存时可加 `--wheelhouse /path/to/wheels`。安装不会修改用户已有的 Python 环境。

默认 UCM 为 **520428c4d2eaa23cb12e4d47d70a7bb820f8d4e0**。显式切换方式及历史差异见[环境文档](docs/environment.md)。模型权重需自行获取，不包含在 Git 或 Release 中。

## 运行冻结 N40 对照

原实验使用 GLM-5.2-FP8 **前 14 层**，不是完整模型的性能结果。模型来源和硬件需求见[环境文档](docs/environment.md)。

~~~bash
export MODEL_PATH=/path/to/GLM-5.2-FP8
python3 scripts/run.py plan --config configs/n40-b8192.json
python3 scripts/run.py doctor --config configs/n40-b8192.json
python3 scripts/run.py run --config configs/n40-b8192.json --output results/n40
~~~

默认三轮，顺序 off→on、on→off、off→on；每组使用独立服务和空 UCM 缓存。每次先做一次无关预热，再按索引 0…39 顺序预热，确认写盘完成后正式发压。不清空 HBM，不固定 L1 热点，不插入补热请求。

首次运行可先恢复归档编译缓存，见[环境文档](docs/environment.md)。测量窗口内出现新 GPU 二进制写入会使该组验收失败；失败记录保留，不能根据性能好坏挑选重跑结果。

## 修改 N 或其他参数

N35 只改变 N。下面各执行一轮 off/on，其他条件保持相同：

~~~bash
python3 scripts/run.py run --config configs/n40-b8192.json \
  --set workload.n=35 --set experiment.repetitions=1 --output results/n35
python3 scripts/run.py run --config configs/n40-b8192.json \
  --set experiment.repetitions=1 --output results/n40-once
~~~

可用 `--set load.concurrency=4`、`--set load.request_rate=2`、`--set workload.seed=123`、`--set server.max_num_batched_tokens=4096` 等覆盖，也可复制 JSON 配置修改。完整参数及更换模型示例见[配置文档](docs/configuration.md)。

满足冻结输入条件时，N≤50 使用同一输入池：先打乱全部 50 个索引，再保留索引小于 N 的请求。改变长度、前缀比例、tokenizer 或 N>50 时自动生成新输入，并记录实际 token 数和哈希。

## 连接已有服务

~~~bash
python3 scripts/run.py run --config configs/external.json \
  --set endpoint.base_url=http://127.0.0.1:8000 \
  --set endpoint.model=my-model \
  --set endpoint.tokenizer=/path/to/local/tokenizer \
  --set workload.n=35 --set workload.input_tokens=4096 \
  --set workload.output_tokens=128 --set workload.prefix_ratio=0.5 \
  --output results/external
~~~

URL 是服务根地址，不带 `/v1`。使用 completions 接口和原始文本，不添加聊天模板。需要认证时设置 `AISBENCH_API_KEY`；客户端实例化时读取密钥，不写入配置。已有服务模式不会切换服务端 Early 开关，也不能验收其 UCM 落盘与缓存来源。

## 输出与验证

每次生成独立目录：配置、输入及哈希、逐组启动参数、AISBench 输出、分析、清理记录及 `summary.json`。汇总只纳入通过验收的组。

~~~bash
python3 scripts/run.py analyze results/n40
# 失败后保持代码、配置、输入和依赖不变，仅续跑未合格组：
python3 scripts/run.py resume results/n40
python3 -m unittest discover -s tests -v
# 真实 AISBench 客户端测试本地模拟接口，不启动模型：
.venv-bench/bin/python tests/check_client_integration.py \
  --tokenizer "$MODEL_PATH" --output .work/client-check
~~~

发布输入、配置、环境清单和汇总；完整原始日志及逐请求结果不公开。原实验和新验证分开记录，历史吞吐提升不作为新环境的通过门槛。新脚本采用 [Apache-2.0](LICENSE)，第三方许可见 [NOTICE](NOTICE)。
