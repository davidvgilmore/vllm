# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for sequence and token pooling methods and their factories."""

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.pooler.seqwise.methods import (
    CLSPool,
    LastPool,
    MeanPool,
    get_seq_pooling_method,
)
from vllm.model_executor.layers.pooler.tokwise.methods import (
    AllPool,
    StepPool,
    get_tok_pooling_method,
)
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingCursor, PoolingMetadata, PoolingStates

_CPU = torch.device("cpu")


def _make_pooling_cursor(
    prompt_lens: list[int],
    *,
    num_scheduled_tokens: list[int] | None = None,
    seq_lens: list[int] | None = None,
    device: torch.device = _CPU,
) -> PoolingCursor:
    """Build a PoolingCursor from a list of per-sequence prompt lengths."""
    prompt_lens_cpu = torch.tensor(prompt_lens, dtype=torch.long)
    if num_scheduled_tokens is None:
        num_scheduled_tokens_cpu = prompt_lens_cpu.clone()
    else:
        num_scheduled_tokens_cpu = torch.tensor(num_scheduled_tokens, dtype=torch.long)
    if seq_lens is None:
        seq_lens_cpu = prompt_lens_cpu.clone()
    else:
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.long)

    cumsum = torch.zeros(len(prompt_lens) + 1, dtype=torch.long, device=device)
    torch.cumsum(num_scheduled_tokens_cpu, dim=0, out=cumsum[1:])

    return PoolingCursor(
        first_token_indices_gpu=cumsum[: len(prompt_lens)].to(device),
        last_token_indices_gpu=(cumsum[1:] - 1).to(device),
        prompt_lens_cpu=prompt_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        num_scheduled_tokens_cpu=num_scheduled_tokens_cpu,
    )


def _make_metadata(
    prompt_lens: list[int],
    *,
    tasks: list[str] | None = None,
    token_ids: list[list[int]] | None = None,
    pooling_params: list[PoolingParams] | None = None,
    num_scheduled_tokens: list[int] | None = None,
    seq_lens: list[int] | None = None,
    pooling_states: list[PoolingStates] | None = None,
    device: torch.device = _CPU,
) -> PoolingMetadata:
    """Build a minimal PoolingMetadata for testing pooling methods."""
    n_seqs = len(prompt_lens)
    if tasks is None:
        tasks = ["embed"] * n_seqs
    if pooling_params is None:
        pooling_params = [PoolingParams(task=t) for t in tasks]

    prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.long)

    prompt_token_ids_cpu = None
    prompt_token_ids = None
    if token_ids is not None:
        max_len = max(len(t) for t in token_ids)
        padded = [t + [0] * (max_len - len(t)) for t in token_ids]
        prompt_token_ids_cpu = torch.tensor(padded, dtype=torch.long)
        prompt_token_ids = prompt_token_ids_cpu.to(device)

    cursor = _make_pooling_cursor(
        prompt_lens,
        num_scheduled_tokens=num_scheduled_tokens,
        seq_lens=seq_lens,
        device=device,
    )

    if pooling_states is None:
        pooling_states = [PoolingStates() for _ in range(n_seqs)]

    return PoolingMetadata(
        prompt_lens=prompt_lens_tensor,
        prompt_token_ids=prompt_token_ids,
        prompt_token_ids_cpu=prompt_token_ids_cpu,
        pooling_params=pooling_params,
        pooling_states=pooling_states,
        pooling_cursor=cursor,
    )


# ---------------------------------------------------------------------------
# CLSPool
# ---------------------------------------------------------------------------
class TestCLSPool:
    def test_extracts_first_token(self):
        hidden = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
        metadata = _make_metadata([2, 3])
        pooler = CLSPool()
        out = pooler(hidden, metadata)
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        assert torch.equal(out, expected)

    def test_rejects_partial_prefill(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metadata = _make_metadata([3], num_scheduled_tokens=[2])
        pooler = CLSPool()
        with pytest.raises(RuntimeError, match="partial prefill"):
            pooler(hidden, metadata)


# ---------------------------------------------------------------------------
# LastPool
# ---------------------------------------------------------------------------
class TestLastPool:
    def test_extracts_last_token(self):
        hidden = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
        metadata = _make_metadata([2, 3])
        pooler = LastPool()
        out = pooler(hidden, metadata)
        expected = torch.tensor([[3.0, 4.0], [9.0, 10.0]])
        assert torch.equal(out, expected)

    def test_partial_prefill_extracts_last_scheduled(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        metadata = _make_metadata([4], num_scheduled_tokens=[2])
        pooler = LastPool()
        out = pooler(hidden, metadata)
        expected = torch.tensor([[3.0, 4.0]])
        assert torch.equal(out, expected)
        assert metadata.pooling_states[0].mean_pool_sum is None
        assert metadata.pooling_states[0].mean_pool_count == 0


# ---------------------------------------------------------------------------
# MeanPool
# ---------------------------------------------------------------------------
class TestMeanPool:
    def test_computes_mean(self):
        hidden = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]], dtype=torch.float32
        )
        metadata = _make_metadata([2, 1])
        pooler = MeanPool()
        out = pooler(hidden, metadata)
        expected = torch.tensor([[2.0, 3.0], [10.0, 20.0]], dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_single_token_is_identity(self):
        hidden = torch.tensor([[5.0, 10.0]], dtype=torch.float32)
        metadata = _make_metadata([1])
        pooler = MeanPool()
        out = pooler(hidden, metadata)
        assert torch.allclose(out, hidden, atol=1e-5)

    def test_uniform_values_return_same(self):
        hidden = torch.full((4, 3), 7.0, dtype=torch.float32)
        metadata = _make_metadata([4])
        pooler = MeanPool()
        out = pooler(hidden, metadata)
        expected = torch.full((1, 3), 7.0, dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_multiple_sequences(self):
        hidden = torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 4.0],
                [4.0, 8.0],
                [10.0, 10.0],
            ],
            dtype=torch.float32,
        )
        metadata = _make_metadata([3, 1])
        pooler = MeanPool()
        out = pooler(hidden, metadata)
        expected = torch.tensor([[2.0, 4.0], [10.0, 10.0]], dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_empty_batch(self):
        hidden = torch.empty((0, 8), dtype=torch.float32)
        metadata = _make_metadata([])
        pooler = MeanPool()
        out = pooler(hidden, metadata)
        assert out.shape == (0, 8)

    @pytest.mark.parametrize("chunk_sizes", [[2, 2], [1, 1, 2]])
    def test_chunked_prefill_matches_single_shot(self, chunk_sizes):
        hidden = torch.arange(16, dtype=torch.float16).reshape(4, 4)
        pooler = MeanPool()
        expected = pooler(hidden, _make_metadata([4]))

        states = [PoolingStates()]
        offset = 0
        for chunk_size in chunk_sizes:
            offset += chunk_size
            out = pooler(
                hidden[offset - chunk_size : offset],
                _make_metadata(
                    [4],
                    num_scheduled_tokens=[chunk_size],
                    seq_lens=[offset],
                    pooling_states=states,
                ),
            )
            if offset < len(hidden):
                assert out == [None]
                assert states[0].mean_pool_sum is not None
                assert states[0].mean_pool_sum.dtype == torch.float32
                assert states[0].mean_pool_count == offset

        assert isinstance(out, torch.Tensor)
        assert torch.allclose(out, expected)
        assert states[0].mean_pool_sum is None
        assert states[0].mean_pool_count == 0

    def test_chunked_prefill_mixed_completion(self):
        pooler = MeanPool()
        states = [PoolingStates(), PoolingStates()]
        hidden = torch.tensor(
            [[1.0, 3.0], [3.0, 5.0], [10.0, 20.0]], dtype=torch.float32
        )
        out = pooler(
            hidden,
            _make_metadata(
                [2, 3],
                num_scheduled_tokens=[2, 1],
                seq_lens=[2, 1],
                pooling_states=states,
            ),
        )

        assert isinstance(out, list)
        assert torch.equal(out[0], torch.tensor([2.0, 4.0]))
        assert out[1] is None
        assert states[0].mean_pool_sum is None
        assert states[1].mean_pool_count == 1

    def test_mixed_batch_accumulator_is_independent_of_batch_buffer(self):
        """One partial request must not alias or corrupt sibling state, and a
        retained accumulator must not hold this step's batch tensor alive."""
        pooler = MeanPool()
        states = [PoolingStates(), PoolingStates()]
        hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)

        out = pooler(
            hidden,
            _make_metadata(
                [1, 4],
                num_scheduled_tokens=[1, 2],
                seq_lens=[1, 2],
                pooling_states=states,
            ),
        )

        # First request finished in one step; second is still accumulating.
        assert isinstance(out, list)
        assert torch.equal(out[0], torch.tensor([0.0, 1.0]))
        assert out[1] is None
        assert states[0].mean_pool_sum is None
        assert states[1].mean_pool_count == 2
        assert states[1].mean_pool_sum.data_ptr() != hidden.data_ptr()
        assert torch.equal(states[1].mean_pool_sum, torch.tensor([6.0, 8.0]))

    def test_pooling_state_clean_releases_mean_accumulator(self):
        state = PoolingStates()
        state.mean_pool_sum = torch.ones(4, dtype=torch.float32)
        state.mean_pool_count = 3

        state.clean()

        assert state.mean_pool_sum is None
        assert state.mean_pool_count == 0

    def test_single_step_fast_path_matches_accumulate(self):
        """The vectorized single-step path and the per-request accumulate path
        must agree so results do not depend on which path a batch takes."""
        hidden = torch.randn(6, 8)
        pooler = MeanPool()
        fast = pooler(hidden, _make_metadata([2, 4]))

        # Pre-seeding one zero-valued accumulator forces the accumulate path
        # without changing the mathematical result.
        states = [PoolingStates(), PoolingStates()]
        states[0].mean_pool_sum = torch.zeros(8, dtype=torch.float32)
        accumulated = pooler(
            hidden,
            _make_metadata(
                [2, 4],
                num_scheduled_tokens=[2, 4],
                seq_lens=[2, 4],
                pooling_states=states,
            ),
        )

        assert isinstance(fast, torch.Tensor)
        assert isinstance(accumulated, torch.Tensor)
        assert torch.allclose(fast, accumulated, atol=1e-6)

    def test_chunked_accumulation(self):
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        metadata = _make_metadata([3, 2])
        pooler = MeanPool()
        with patch(
            "vllm.model_executor.layers.pooler.seqwise.methods"
            "._MEAN_POOL_ACCUMULATION_CHUNK_BYTES",
            16,
        ):
            out = pooler(hidden, metadata)
        expected_seq0 = hidden[:3].float().mean(dim=0, keepdim=True)
        expected_seq1 = hidden[3:].float().mean(dim=0, keepdim=True)
        expected = torch.cat([expected_seq0, expected_seq1], dim=0)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_upcasts_to_float32(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
        metadata = _make_metadata([2])
        pooler = MeanPool()
        out = pooler(hidden, metadata)
        assert out.dtype == torch.float32
        expected = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-2)


# ---------------------------------------------------------------------------
# get_seq_pooling_method factory
# ---------------------------------------------------------------------------
class TestGetSeqPoolingMethod:
    def test_cls(self):
        assert isinstance(get_seq_pooling_method("CLS"), CLSPool)

    def test_last(self):
        assert isinstance(get_seq_pooling_method("LAST"), LastPool)

    def test_mean(self):
        assert isinstance(get_seq_pooling_method("MEAN"), MeanPool)

    def test_unknown_raises(self):
        with pytest.raises(NotImplementedError, match="UNKNOWN"):
            get_seq_pooling_method("UNKNOWN")


# ---------------------------------------------------------------------------
# AllPool
# ---------------------------------------------------------------------------


@dataclass
class _FakeSchedulerConfig:
    enable_chunked_prefill: bool = False


@dataclass
class _FakeVllmConfig:
    scheduler_config: _FakeSchedulerConfig


class TestAllPool:
    @staticmethod
    def _make_all_pool(*, chunked: bool = False) -> AllPool:
        fake_config = _FakeVllmConfig(
            scheduler_config=_FakeSchedulerConfig(
                enable_chunked_prefill=chunked,
            ),
        )
        with patch(
            "vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config",
            return_value=fake_config,
        ):
            return AllPool()

    def test_splits_by_sequence(self):
        pooler = self._make_all_pool()
        hidden = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
        metadata = _make_metadata([2, 3])
        out = pooler(hidden, metadata)
        assert len(out) == 2
        assert torch.equal(out[0], hidden[:2])
        assert torch.equal(out[1], hidden[2:])

    def test_single_sequence(self):
        pooler = self._make_all_pool()
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metadata = _make_metadata([3])
        out = pooler(hidden, metadata)
        assert len(out) == 1
        assert torch.equal(out[0], hidden)

    def test_chunked_prefill_returns_none_for_unfinished(self):
        pooler = self._make_all_pool(chunked=True)
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        metadata = _make_metadata(
            [4],
            num_scheduled_tokens=[2],
            seq_lens=[2],
        )
        out = pooler(hidden, metadata)
        assert len(out) == 1
        assert out[0] is None

    def test_chunked_prefill_returns_concat_when_finished(self):
        pooler = self._make_all_pool(chunked=True)

        chunk1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        metadata1 = _make_metadata(
            [4],
            num_scheduled_tokens=[2],
            seq_lens=[2],
        )
        out1 = pooler(chunk1, metadata1)
        assert out1[0] is None

        chunk2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        metadata2 = _make_metadata(
            [4],
            num_scheduled_tokens=[2],
            seq_lens=[4],
        )
        metadata2.pooling_states = metadata1.pooling_states
        out2 = pooler(chunk2, metadata2)
        assert out2[0] is not None
        expected = torch.cat([chunk1, chunk2], dim=0)
        assert torch.equal(out2[0], expected)

    def test_chunked_prefill_single_shot_matches_non_chunked(self):
        pooler = self._make_all_pool(chunked=True)
        hidden = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
        metadata = _make_metadata([2, 3])
        out = pooler(hidden, metadata)
        assert len(out) == 2
        assert torch.equal(out[0], hidden[:2])
        assert torch.equal(out[1], hidden[2:])

    def test_chunked_prefill_mixed_finished_unfinished(self):
        pooler = self._make_all_pool(chunked=True)
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metadata = _make_metadata(
            [2, 4],
            num_scheduled_tokens=[2, 1],
            seq_lens=[2, 1],
        )
        out = pooler(hidden, metadata)
        assert len(out) == 2
        assert torch.equal(out[0], hidden[:2])
        assert out[1] is None


# ---------------------------------------------------------------------------
# StepPool
# ---------------------------------------------------------------------------
class TestStepPool:
    @staticmethod
    def _make_step_pool(*, chunked: bool = False) -> StepPool:
        fake_config = _FakeVllmConfig(
            scheduler_config=_FakeSchedulerConfig(
                enable_chunked_prefill=chunked,
            ),
        )
        with patch(
            "vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config",
            return_value=fake_config,
        ):
            return StepPool()

    def test_filters_by_step_tag_id(self):
        pooler = self._make_step_pool()
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        token_ids = [[10, 99, 10, 20]]
        params = [PoolingParams(task="token_classify", step_tag_id=10)]
        metadata = _make_metadata([4], token_ids=token_ids, pooling_params=params)
        out = pooler(hidden, metadata)
        assert len(out) == 1
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        assert torch.equal(out[0], expected)

    def test_filters_by_returned_token_ids(self):
        pooler = self._make_step_pool()
        hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        token_ids = [[10, 20]]
        params = [PoolingParams(task="token_classify", returned_token_ids=[0, 2])]
        metadata = _make_metadata([2], token_ids=token_ids, pooling_params=params)
        out = pooler(hidden, metadata)
        assert len(out) == 1
        expected = torch.tensor([[1.0, 3.0], [4.0, 6.0]])
        assert torch.equal(out[0], expected)

    def test_no_filtering_without_params(self):
        pooler = self._make_step_pool()
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        token_ids = [[10, 20]]
        params = [PoolingParams(task="token_classify")]
        metadata = _make_metadata([2], token_ids=token_ids, pooling_params=params)
        out = pooler(hidden, metadata)
        assert len(out) == 1
        assert torch.equal(out[0], hidden)

    def test_combined_step_tag_and_returned_token_ids(self):
        pooler = self._make_step_pool()
        hidden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        token_ids = [[99, 10, 99]]
        params = [
            PoolingParams(
                task="token_classify",
                step_tag_id=10,
                returned_token_ids=[0, 2],
            )
        ]
        metadata = _make_metadata([3], token_ids=token_ids, pooling_params=params)
        out = pooler(hidden, metadata)
        assert len(out) == 1
        expected = torch.tensor([[4.0, 6.0]])
        assert torch.equal(out[0], expected)

    def test_step_tag_id_no_match_returns_empty(self):
        pooler = self._make_step_pool()
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        token_ids = [[10, 20]]
        params = [PoolingParams(task="token_classify", step_tag_id=999)]
        metadata = _make_metadata([2], token_ids=token_ids, pooling_params=params)
        out = pooler(hidden, metadata)
        assert len(out) == 1
        assert out[0].shape == (0, 2)

    def test_chunked_prefill_propagates_none_for_unfinished(self):
        pooler = self._make_step_pool(chunked=True)
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        token_ids = [[10, 20, 30, 40]]
        params = [PoolingParams(task="token_classify", step_tag_id=10)]
        metadata = _make_metadata(
            [4],
            token_ids=token_ids,
            pooling_params=params,
            num_scheduled_tokens=[2],
            seq_lens=[2],
        )
        out = pooler(hidden, metadata)
        assert len(out) == 1
        assert out[0] is None

    def test_chunked_prefill_filters_when_finished(self):
        pooler = self._make_step_pool(chunked=True)
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        token_ids = [[10, 99, 10, 20]]
        params = [PoolingParams(task="token_classify", step_tag_id=10)]
        metadata = _make_metadata([4], token_ids=token_ids, pooling_params=params)
        out = pooler(hidden, metadata)
        assert len(out) == 1
        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        assert torch.equal(out[0], expected)

    def test_requires_token_ids_update(self):
        pooler = self._make_step_pool()
        update = pooler.get_pooling_updates("token_classify")
        assert update.requires_token_ids is True


# ---------------------------------------------------------------------------
# get_tok_pooling_method factory
# ---------------------------------------------------------------------------
class TestGetTokPoolingMethod:
    def test_all(self):
        fake_config = _FakeVllmConfig(
            scheduler_config=_FakeSchedulerConfig(
                enable_chunked_prefill=False,
            ),
        )
        with patch(
            "vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config",
            return_value=fake_config,
        ):
            assert isinstance(get_tok_pooling_method("ALL"), AllPool)

    def test_step(self):
        fake_config = _FakeVllmConfig(
            scheduler_config=_FakeSchedulerConfig(
                enable_chunked_prefill=False,
            ),
        )
        with patch(
            "vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config",
            return_value=fake_config,
        ):
            assert isinstance(get_tok_pooling_method("STEP"), StepPool)

    def test_unknown_raises(self):
        with pytest.raises(NotImplementedError, match="UNKNOWN"):
            get_tok_pooling_method("UNKNOWN")
