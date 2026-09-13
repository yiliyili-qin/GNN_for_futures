"""Shared LSTM encoding and cross-futures temporal dot-product attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CrossFuturesTemporalAttentionOutput:
    """Intermediate tensors retained for composition with later graph modules."""

    hidden_states: Tensor
    queries: Tensor
    attention_scores: Tensor
    peer_indices: Tensor


class CrossFuturesTemporalAttention(nn.Module):
    """Encode each future independently, then score every other future's history.

    Input shape is ``[num_futures, sequence_length, input_size]``. A shared
    single-layer, unidirectional LSTM produces ``[N, t, F]`` hidden states.
    For each target future i, its final state is projected to a query and
    dotted with the final L states of every source future j != i. The raw
    scores have shape ``[N, N - 1, L]`` and are not scaled or normalized.
    """

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 13,
        history_length: int = 16,
        expected_num_futures: int | None = 44,
        lstm_bias: bool = True,
        query_bias: bool = False,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if expected_num_futures is not None and expected_num_futures < 2:
            raise ValueError("expected_num_futures must be at least 2 or None")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.expected_num_futures = expected_num_futures
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=lstm_bias,
            batch_first=True,
            bidirectional=False,
        )
        self.query_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=query_bias,
        )

    def forward(self, inputs: Tensor) -> CrossFuturesTemporalAttentionOutput:
        """Return LSTM states, final queries, raw pair scores, and peer indices."""
        self._validate_inputs(inputs)
        hidden_states, _ = self.lstm(inputs)
        queries = self.query_projection(hidden_states[:, -1, :])
        keys = hidden_states[:, -self.history_length :, :]

        all_scores = torch.einsum("if,jlf->ijl", queries, keys)
        num_futures = inputs.shape[0]
        peer_mask = ~torch.eye(num_futures, dtype=torch.bool, device=inputs.device)
        attention_scores = all_scores[peer_mask].reshape(
            num_futures,
            num_futures - 1,
            self.history_length,
        )
        all_indices = torch.arange(num_futures, device=inputs.device)
        peer_indices = all_indices.expand(num_futures, -1)[peer_mask].reshape(
            num_futures,
            num_futures - 1,
        )
        return CrossFuturesTemporalAttentionOutput(
            hidden_states=hidden_states,
            queries=queries,
            attention_scores=attention_scores,
            peer_indices=peer_indices,
        )

    def _validate_inputs(self, inputs: Tensor) -> None:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [num_futures, sequence_length, input_size]")
        num_futures, sequence_length, input_size = inputs.shape
        if self.expected_num_futures is not None and num_futures != self.expected_num_futures:
            raise ValueError(
                f"Expected {self.expected_num_futures} futures, got {num_futures}"
            )
        if num_futures < 2:
            raise ValueError("inputs must contain at least two futures")
        if sequence_length < self.history_length:
            raise ValueError(
                f"sequence_length must be at least history_length={self.history_length}"
            )
        if input_size != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {input_size}")
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype")