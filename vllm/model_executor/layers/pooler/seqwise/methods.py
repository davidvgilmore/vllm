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
from vllm.v1.pool.metadata import PoolingMetadata

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
    def _sum_chunk(hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        bytes_per_token = hidden_size * torch.finfo(torch.float32).bits // 8
        chunk_size = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // bytes_per_token)
        result = torch.zeros(
            hidden_size, dtype=torch.float32, device=hidden_states.device
        )
        for start in range(0, hidden_states.shape[0], chunk_size):
            end = min(start + chunk_size, hidden_states.shape[0])
            result.add_(hidden_states[start:end].sum(dim=0, dtype=torch.float32))
        return result

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

        hidden_states_list = torch.split(
            hidden_states, pooling_cursor.num_scheduled_tokens_cpu.tolist()
        )
        output_list: list[torch.Tensor | None] = []
        for state, hidden_states_chunk, scheduled, prompt_len, finished in zip(
            pooling_metadata.pooling_states,
            hidden_states_list,
            pooling_cursor.num_scheduled_tokens_cpu,
            prompt_lens_cpu,
            pooling_cursor.is_finished(),
        ):
            chunk_sum = self._sum_chunk(hidden_states_chunk)
            if state.mean_pool_sum is None:
                state.mean_pool_sum = chunk_sum
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
