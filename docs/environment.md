# 环境保存与复现边界

采用独立实验仓库，UCM/vLLM/AISBench 保持为锁定依赖。这样升级业务代码、调 N 或连接已有服务时，不必把实验配置混入 UCM 功能分支。

## 固定来源

[sources.lock.json](../sources.lock.json) 记录完整 commit、来源和版本：

| 组件 | 默认版本 |
| --- | --- |
| UCM | StorTecDev/ucm-prefetch，520428c4d2eaa23cb12e4d47d70a7bb820f8d4e0 |
| vLLM | 0.28.0，2cf0a6915ce544dc493a0990f2ea38d81601128a |
| AISBench wrapper | rayn-zzz/aisbench_auto_tools_prefix，ef2b6d8653d57487b622b0dc9db458f8a1e8dec6 |
| ais_bench_benchmark | 3.1.20260630，wheel SHA256见锁文件 |
| 模型 | zai-org/GLM-5.2-FP8，revision f33c6dc501ee5a2c7e35155653b1b1abbc320951 |
| Python | 3.12 |
| 容器基础镜像 | Ubuntu 24.04，Dockerfile固定 OCI digest |

安装脚本获取固定源码，使用发布的 vLLM wheel，并从干净 UCM 源码构建原生库。Python依赖按原实验记录逐项锁版本，原生构建依赖按归档 SHA256 校验。两套环境分开安装；上游 AISBench wrapper 会修改包内数据链接，因此同一 benchmark 环境采用锁串行使用，退出后恢复链接。

显式切换 UCM 时先修改源码选择，再重新构建：

~~~bash
python3 scripts/setup.py sources --ucm-commit FULL_40_CHARACTER_COMMIT
python3 scripts/setup.py server --jobs 4
~~~

实际使用的 commit 和原生库哈希记录在环境及每组结果里。切换版本产生新的实验条件，不能把结果标为冻结历史版本。

原实验记录的 UCM HEAD 为 692c3d231861be577d0283f222f638360202571e，但其归档执行源码已包含 READY query-range 修复。用户选定的520428提交包含该修复，并集成当时的 layerwise 同步补丁；保持 `ENABLE_UCM_PATCH=1` 和 `GLM51_LAYERWISE_SYNC_FIX=1`，不再次安装本地重复补丁。本仓库新构建原生库，不能宣称与原实验二进制字节一致。

## 模型与硬件

模型从 [官方 Hugging Face 仓库](https://huggingface.co/zai-org/GLM-5.2-FP8/tree/f33c6dc501ee5a2c7e35155653b1b1abbc320951) 获取，使用该 revision 的 tokenizer。完整模型与仅保留所需 tensor 的14层子集均需符合模型加载器的布局要求；本次验证使用经过来源与张量核验的14层子集，约113.6GB权重。仓库不分发权重，也不从其他人的运行目录复制权重。

可用 Hugging Face CLI 获取完整固定版本，先确认磁盘空间：

~~~bash
.venv-server/bin/hf download zai-org/GLM-5.2-FP8 \
  --revision f33c6dc501ee5a2c7e35155653b1b1abbc320951 \
  --local-dir /models/GLM-5.2-FP8
~~~

模型只运行前14层、MTP关闭，不用于完整模型质量或解码性能推断。tokenizer.json SHA256固定为 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`。

原机器为单 L20X，HBM约140GiB；原实验记录 GPU UUID 与本次验证机器不同。冻结配置需要19GiB L1共享内存加元数据空间，建议 `/dev/shm` 至少24GiB；L2配置128GiB，需要 SSD 及 O_DIRECT/aio 支持。还应预留模型、两个 Python 环境和编译缓存的磁盘空间。CPU/内存限制、驱动、存储和并发干扰都会影响吞吐，配置相同不保证相同绝对性能。

## 编译缓存归档

Release保留能取得的原实验编译缓存，SHA256在 [artifacts.lock.json](../artifacts.lock.json)。它属于原实验已有的二进制证据，不是完整运行镜像。使用固定配置与兼容GPU时，可在首次运行前恢复：

~~~bash
python3 scripts/artifacts.py restore-cache
# 已有本地归档时：
python3 scripts/artifacts.py restore-cache \
  --archive /path/to/compiled-kernel-cache.tar.gz --output .work/compile-cache
~~~

目标目录必须不存在；不合并或覆盖已有缓存。安装路径、编译器或GPU条件变化仍可能触发重新编译，首次冷启动较慢时可用 --set execution.startup_timeout_s=3600 增加等待时间。原归档来自 sm90a / CUDA13 环境；不兼容硬件应使用新的编译缓存目录，让运行时重新编译。测量前预热无法保证覆盖任意新负载的所有kernel形状，因此正式窗口仍验收新GPU二进制写入。失败组保留证据并单独处理，不纳入统计。

已从其他历史归档找回与目标实验 SHA256 完全一致的3个UCM原生库，作为 original-native-partial.tar.gz 保存；其余10个仅保留原哈希。见 env/original-native-manifest.json。可通过 `python3 scripts/artifacts.py fetch --artifact original_native_partial` 下载。这个部分归档不是可安装的完整运行时；当前520428环境仍使用源码重建的13个库。

## Docker

[Dockerfile](../Dockerfile) 提供固定基础镜像与相同安装流程。宿主机需要 NVIDIA Container Toolkit；容器内不安装驱动。APT包由Ubuntu仓库解析，未声称整个镜像达到逐字节重建。

~~~bash
docker build -t aisbench-ucm-repro:0.1.0 .
docker run --rm --gpus device=0 --shm-size=24g \
  -v /models/GLM-5.2-FP8:/models/model:ro \
  -v "$PWD/results":/opt/aisbench-ucm-repro/results \
  aisbench-ucm-repro:0.1.0 run --set experiment.repetitions=1 \
  --output results/container-n40
~~~

当前交付主验证路径为本地独立环境；Docker是否实测以验证报告为准。模型不打入镜像。

## 发布范围

Git包含代码、配置、输入压缩包、哈希、来源、许可证和汇总。Release包含冻结输入、汇总、可取得的原编译缓存和校验清单。不发布对话提示词与澄清回答、完整原实验结果、逐请求日志、模型权重、密钥或虚拟环境。运行路径由配置提供，不依赖原机器目录。

原始结果保存在本地原实验目录；新运行结果默认进入被Git忽略的 results/。发布参考汇总不能代替原始逐请求证据的独立审计。


## 依赖检查记录

独立服务端环境的 pip check 通过。冻结的 benchmark 环境保留原记录版本，pip check 报告两项已有元数据冲突：ml-dtypes 0.6.0要求NumPy≥2，而AISBench要求NumPy<2；vLLM 0.28.0要求较新的OpenCV，而benchmark环境保留4.11.0.86。benchmark环境只运行AISBench客户端，模型服务使用独立server环境。真实AISBench流式、非流式completions调用已验证；本锁文件不宣称支持benchmark环境中所有未使用的多模态/本地推理包。不要在benchmark环境运行vLLM服务或自动升级NumPy以消除警告。
