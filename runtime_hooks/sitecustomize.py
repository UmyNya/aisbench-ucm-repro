"""Experiment-local startup manifest and low-volume scheduler/request observations."""

import json
import os
import time
from pathlib import Path

if os.environ.get("MIXED_OBSERVE") == "1":
    import natural_observe

    identity_block = int(os.environ["AISBENCH_IDENTITY_BLOCK"])
    prefix_tokens = int(os.environ["AISBENCH_PREFIX_TOKENS"])
    import waiting_observe
    from wrapt import when_imported

    root = Path(os.environ["MIXED_RUN"])

    def event(file, data):
        with (root / file).open("a") as f:
            f.write(
                json.dumps(
                    dict(monotonic_ns=time.monotonic_ns(), pid=os.getpid(), **data),
                    default=str,
                )
                + "\n"
            )

    @when_imported("ucm.integration.vllm.ucm_connector")
    def patch_ucm(module):
        cls = module.UCMDirectConnector
        original = cls.register_kv_caches

        def register(self, kv_caches):
            result = original(self, kv_caches)
            natural_observe.register(self)
            import torch, vllm, ucm
            import vllm._version

            layout = self.kv_cache_layout
            cfg = self._vllm_config

            def desc(t):
                if isinstance(t, (tuple, list)):
                    return [desc(x) for x in t]
                return dict(
                    shape=list(t.shape),
                    dtype=str(t.dtype),
                    bytes=t.numel() * t.element_size(),
                    ptr=t.data_ptr(),
                )

            shard, block = module._get_store_io_sizes(
                layout.shard_size, layout.block_size
            )
            data = dict(
                pid=os.getpid(),
                connector_class=type(self).__name__,
                layout_class=type(layout).__name__,
                tensors={k: desc(v) for k, v in kv_caches.items()},
                block_size=cfg.cache_config.block_size,
                user_specified_block_size=cfg.cache_config.user_specified_block_size,
                kv_cache_memory_bytes=cfg.cache_config.kv_cache_memory_bytes,
                compilation_mode=str(cfg.compilation_config.mode),
                cudagraph_mode=str(cfg.compilation_config.cudagraph_mode),
                num_blocks=layout.num_blocks,
                layer_ids=self.layer_ids,
                layer_name_to_id=layout.layer_name_to_id,
                physical_shard_bytes=shard,
                physical_block_bytes=block,
                logical_shard_bytes=layout.shard_size,
                logical_block_bytes=layout.block_size,
                unique_id=self.unique_id,
                launch_config=self.launch_config,
                use_layerwise=self.use_layerwise,
                enable_event_sync=self.enable_event_sync,
                dtype=str(cfg.model_config.dtype),
                kv_dtype=str(cfg.cache_config.cache_dtype),
                num_hidden_layers=cfg.model_config.hf_config.num_hidden_layers,
                num_nextn_predict_layers=getattr(
                    cfg.model_config.hf_config, "num_nextn_predict_layers", 0
                ),
                speculative_config=str(cfg.speculative_config),
                seed=cfg.model_config.seed,
                tensor_parallel_size=cfg.parallel_config.tensor_parallel_size,
                data_parallel_size=cfg.parallel_config.data_parallel_size,
                pipeline_parallel_size=cfg.parallel_config.pipeline_parallel_size,
                enable_expert_parallel=cfg.parallel_config.enable_expert_parallel,
                enforce_eager=cfg.model_config.enforce_eager,
                max_model_len=cfg.model_config.max_model_len,
                enable_prefix_caching=cfg.cache_config.enable_prefix_caching,
                enable_chunked_prefill=cfg.scheduler_config.enable_chunked_prefill,
                max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens,
                max_num_seqs=cfg.scheduler_config.max_num_seqs,
                long_prefill_token_threshold=cfg.scheduler_config.long_prefill_token_threshold,
                async_scheduling=cfg.scheduler_config.async_scheduling,
                world_size=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                vllm_version=vllm.__version__,
                vllm_commit=vllm._version.__commit_id__,
                vllm_path=vllm.__file__,
                ucm_path=ucm.__file__,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                sync_fix=os.environ.get("GLM51_LAYERWISE_SYNC_FIX") == "1",
                sync_hook_installed=bool(
                    getattr(module.UCMLayerWiseConnector, "_glm52_sync_fix", False)
                ),
                gpu_name=torch.cuda.get_device_name(),
                gpu_capability=torch.cuda.get_device_capability(),
            )
            (root / "actual-layout.json").write_text(
                json.dumps(data, indent=2, default=str) + "\n"
            )
            print("MIXED_ACTUAL_LAYOUT " + str(root / "actual-layout.json"), flush=True)
            return result

        cls.register_kv_caches = register
        prefix_file = root / "executed-code/inputs/request-prefix-ids.json"
        prefix_indices = {
            tuple(x["token_ids"]): x["index"]
            for x in json.loads(prefix_file.read_text())
        }
        case_file = root / "executed-code/inputs/request-case-ids.json"
        case_indices = (
            {
                tuple(x["token_ids"]): x["case_index"]
                for x in json.loads(case_file.read_text())
            }
            if case_file.exists()
            else {}
        )
        on_new = cls.on_new_request

        def on_new_request(self, request):
            event(
                "new_requests.jsonl",
                dict(
                    request_id=request.request_id,
                    prompt_tokens=request.num_tokens,
                    input_index=prefix_indices.get(
                        tuple(request.all_token_ids[:identity_block])
                    ),
                    case_index=case_indices.get(
                        tuple(request.all_token_ids[:identity_block])
                        + tuple(
                            request.all_token_ids[
                                prefix_tokens : prefix_tokens + identity_block
                            ]
                        )
                    ),
                ),
            )
            return on_new(self, request)

        cls.on_new_request = on_new_request
        matched = cls.get_num_new_matched_tokens

        def get_match(self, request, num_computed_tokens):
            result = matched(self, request, num_computed_tokens)
            waiting_observe.record_lookup(request, result)
            # One compact record per distinct lookup result; no per-block logging.
            state = (num_computed_tokens, result[0])
            seen = getattr(self, "_mixed_matches", None)
            if seen is None:
                seen = {}
                self._mixed_matches = seen
            if seen.get(request.request_id) != state:
                seen[request.request_id] = state
                event(
                    "matches.jsonl",
                    dict(
                        request_id=request.request_id,
                        prompt_tokens=request.num_tokens,
                        hbm_tokens=num_computed_tokens,
                        external_tokens=result[0],
                    ),
                )
            return result

        cls.get_num_new_matched_tokens = get_match

    @when_imported("ucm.integration.vllm.early_connector")
    def patch_early_wait(module):
        cls = module._EarlyConnectorMixin
        original = cls.get_num_new_matched_tokens

        def get_match(self, request, num_computed_tokens):
            result = original(self, request, num_computed_tokens)
            # PENDING returns before the native/base method is called.
            waiting_observe.record_lookup(request, result, early=True)
            return result

        cls.get_num_new_matched_tokens = get_match

    @when_imported("vllm.v1.core.kv_cache_manager")
    def patch_allocation_wait(module):
        original = module.KVCacheManager.allocate_slots

        def allocate(self, request, *args, **kwargs):
            result = original(self, request, *args, **kwargs)
            waiting_observe.record_allocation(request, result)
            return result

        module.KVCacheManager.allocate_slots = allocate

    @when_imported("vllm.v1.core.sched.scheduler")
    def patch_scheduler(module):
        def state(self):
            return waiting_observe.snapshot(self)

        original = module.Scheduler.schedule

        def schedule(self, *args, **kwargs):
            before = state(self)
            start = time.monotonic_ns()
            waiting_observe.begin_step()
            try:
                result = original(self, *args, **kwargs)
            finally:
                waiting_observe.finish_step()
            after = state(self)
            if (
                before["running"]
                + before["waiting"]
                + after["running"]
                + after["waiting"]
            ):
                event(
                    "queues.jsonl",
                    dict(
                        kind="schedule",
                        start_ns=start,
                        running_before=before["running"],
                        waiting_before=before["waiting"],
                        **after,
                        scheduled_tokens=result.num_scheduled_tokens,
                    ),
                )
            return result

        module.Scheduler.schedule = schedule
        update = module.Scheduler.update_from_output

        def update_from_output(self, *args, **kwargs):
            result = update(self, *args, **kwargs)
            event("queues.jsonl", dict(kind="output", **state(self)))
            return result

        module.Scheduler.update_from_output = update_from_output
        add = module.Scheduler.add_request

        def add_request(self, *args, **kwargs):
            result = add(self, *args, **kwargs)
            event("queues.jsonl", dict(kind="arrival", **state(self)))
            return result

        module.Scheduler.add_request = add_request


# The selected UCM commit installs layerwise synchronization via ucm_patch.pth.
