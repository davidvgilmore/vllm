# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TypeAlias

import torch
import torch.nn as nn

from vllm.config.pooler import SequencePoolingType
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.tasks import PoolingTask
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.pool.metadata import PoolingCursor, PoolingMetadata

SequencePoolingMethodOutput: TypeAlias = torch.Tensor | list[torch.Tensor | None]

_MEAN_POOL_ACCUMULATION_CHUNK_BYTES = 16 * 1024 * 1024  # 16MB


class SequencePoolingMethod(nn.Module, ABC):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify", "embed", "classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        raise NotImplementedError


class CLSPool(SequencePoolingMethod):
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        if pooling_cursor.is_partial_prefill():
            raise RuntimeError("partial prefill is not supported with CLS pooling")

        return hidden_states[pooling_cursor.first_token_indices_gpu]


class LastPool(SequencePoolingMethod):
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        return hidden_states[pooling_cursor.last_token_indices_gpu]


class MeanPool(SequencePoolingMethod):
    @staticmethod
    def _segment_sums(
        hidden_states: torch.Tensor,
        num_scheduled_tokens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """FP32 sum of this step's scheduled tokens, per request.

        One segmented reduction for the whole batch keeps the kernel count
        independent of batch size, so a single partially prefilled request
        cannot turn a large co-scheduled batch into per-request launches.
        """
        num_seqs = num_scheduled_tokens_cpu.numel()
        hidden_size = hidden_states.shape[-1]
        segment_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden_states.device, dtype=torch.long),
            num_scheduled_tokens_cpu.to(hidden_states.device),
            output_size=hidden_states.shape[0],
        )
        segment_sums = torch.zeros(
            (num_seqs, hidden_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )

        bytes_per_token = hidden_size * torch.finfo(torch.float32).bits // 8
        chunk_size = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // bytes_per_token)
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            segment_sums.index_add_(
                0,
                segment_ids[start:end],
                hidden_states[start:end].to(dtype=torch.float32),
            )
        return segment_sums

    @staticmethod
    def _forward_single_step(
        hidden_states: torch.Tensor,
        prompt_lens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        num_seqs = prompt_lens_cpu.numel()
        hidden_size = hidden_states.shape[-1]
        prompt_lens = async_tensor_h2d(
            prompt_lens_cpu, device=hidden_states.device, dtype=torch.int64
        )
        # eg. [2, 1, 3] -> [0, 0, 1, 2, 2, 2]
        segment_ids = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden_states.device, dtype=torch.long),
            prompt_lens,
            output_size=int(prompt_lens_cpu.sum()),
        )
        segment_sums = torch.zeros(
            (num_seqs, hidden_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )

        bytes_per_token = hidden_size * torch.finfo(torch.float32).bits // 8
        chunk_size = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // bytes_per_token)

        # iterate over the batch in chunks
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            # using index_add_ to accumulate for each segment
            segment_sums.index_add_(
                0,
                segment_ids[start:end],
                hidden_states[start:end].to(dtype=torch.float32),
            )

        return segment_sums / prompt_lens.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolingMethodOutput:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        prompt_lens_cpu = pooling_cursor.prompt_lens_cpu
        num_seqs = prompt_lens_cpu.numel()
        hidden_size = hidden_states.shape[-1]

        if num_seqs == 0:
            # early return for empty batch
            return hidden_states.new_empty((0, hidden_size), dtype=torch.float32)

        # Fast path: every prompt is fully scheduled in this step and no
        # request carries accumulated state, i.e. the exact conditions under
        # which MEAN pooling ran before chunked accumulation existed.
        if not pooling_cursor.is_partial_prefill() and all(
            state.mean_pool_sum is None and state.mean_pool_count == 0
            for state in pooling_metadata.pooling_states
        ):
            return self._forward_single_step(hidden_states, prompt_lens_cpu)

        return self._forward_accumulate(hidden_states, pooling_metadata, pooling_cursor)

    def _forward_accumulate(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
        pooling_cursor: PoolingCursor,
    ) -> SequencePoolingMethodOutput:
        prompt_lens_cpu = pooling_cursor.prompt_lens_cpu
        chunk_sums = self._segment_sums(
            hidden_states, pooling_cursor.num_scheduled_tokens_cpu
        )
        output_list: list[torch.Tensor | None] = []
        for index, (state, scheduled, prompt_len, finished) in enumerate(
            zip(
                pooling_metadata.pooling_states,
                pooling_cursor.num_scheduled_tokens_cpu,
                prompt_lens_cpu,
                pooling_cursor.is_finished(),
            )
        ):
            chunk_sum = chunk_sums[index]
            if state.mean_pool_sum is None:
                # Clone: the row is a view into this step's batch tensor, which
                # a retained accumulator would keep alive across steps.
                state.mean_pool_sum = chunk_sum.clone()
            else:
                if (
                    state.mean_pool_sum.shape != chunk_sum.shape
                    or state.mean_pool_sum.device != chunk_sum.device
                ):
                    state.clean()
                    raise RuntimeError(
                        "MEAN pooling accumulator does not match the "
                        "scheduled hidden states"
                    )
                state.mean_pool_sum.add_(chunk_sum)
            state.mean_pool_count += int(scheduled)

            if not finished:
                output_list.append(None)
                continue

            if state.mean_pool_count != int(prompt_len):
                actual_count = state.mean_pool_count
                state.clean()
                raise RuntimeError(
                    "MEAN pooling accumulated an unexpected number of tokens: "
                    f"{actual_count} != {int(prompt_len)}"
                )
            if state.mean_pool_count == 0:
                state.clean()
                raise RuntimeError("MEAN pooling requires at least one token")

            output_list.append(state.mean_pool_sum / state.mean_pool_count)
            state.clean()

        if all(output is not None for output in output_list):
            return torch.stack([output for output in output_list if output is not None])
        return output_list


def get_seq_pooling_method(
    pooling_type: SequencePoolingType | str,
) -> SequencePoolingMethod:
    if pooling_type == "CLS":
        return CLSPool()
    if pooling_type == "LAST":
        return LastPool()
    if pooling_type == "MEAN":
        return MeanPool()

    raise NotImplementedError(f"Unknown sequence pooling type: {pooling_type!r}")
