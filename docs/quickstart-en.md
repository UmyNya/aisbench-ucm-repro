# Quickstart

Run configurable AISBench completion benchmarks and paired UCM Early Prefetch experiments. The reference is GLM-5.2-FP8 with **14 layers**, 40 requests, 40,000 input tokens, one output token, concurrency 8, and 35,968 reusable prefix tokens.

Use Linux x86_64 and Python 3.12. Managed runs also require a compatible NVIDIA GPU/driver, CMake, Ninja, a C++ compiler, libaio-dev and libnuma-dev. Setup creates separate virtual environments and installs the recorded CUDA 13 dependencies.

~~~bash
python3 scripts/setup.py sources
python3 scripts/setup.py client
python3 scripts/setup.py server --jobs 4
export MODEL_PATH=/path/to/GLM-5.2-FP8
python3 scripts/run.py doctor
python3 scripts/run.py run --output results/n40
~~~

The reference defaults to three alternating off/on pairs. N35 uses the same implementation and frozen input pool:

~~~bash
python3 scripts/run.py run --set workload.n=35 \
  --set experiment.repetitions=1 --output results/n35
~~~

For an existing completion service, install only sources/client and supply a local tokenizer:

~~~bash
python3 scripts/run.py run --config configs/external.json \
  --set endpoint.base_url=http://127.0.0.1:8000 \
  --set endpoint.model=my-model \
  --set endpoint.tokenizer=/path/to/tokenizer \
  --set workload.n=35 --set workload.input_tokens=4096 \
  --set workload.output_tokens=128 --output results/external
~~~

The URL must not end in `/v1`. Set `AISBENCH_API_KEY` for authentication. External mode cannot toggle Early Prefetch or verify native cache accounting. Input generation supports arbitrary N, token lengths, prefix ratios, seeds, and a custom JSONL source.

`plan` prints configuration; `prepare` materializes inputs; `doctor` checks prerequisites; `run` executes and cleans up its service; `analyze` recomputes saved summaries; `resume results/path` retries only unqualified arms of an unchanged failed campaign. Relative configuration paths resolve from the checkout.

Default UCM: `520428c4d2eaa23cb12e4d47d70a7bb820f8d4e0`. This reproduces workload/protocol/configuration; identical historical timing or original native-library bytes are not promised. See [environment](environment.md), [configuration](configuration.md), and [reference results](../reports/reference-n40.md).

Offline tests: `python3 -m unittest discover -s tests -v`. New code is Apache-2.0; third-party licenses are in [NOTICE](../NOTICE).
