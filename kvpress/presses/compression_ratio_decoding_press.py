# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.scorer_press import ScorerPress


class CompressionRatioDecodingPress(DecodingPress):
    """
    A decoding press that keeps a fixed fraction of all tokens seen so far.

    Unlike `DecodingPress`, which compresses to a fixed absolute `target_size`,
    this subclass derives the target size from the full sequence length observed
    so far during decoding.

    Parameters
    ----------
    base_press : ScorerPress
        The scorer press used to compute importance scores for tokens.
    target_compression_ratio : float
        Fraction of all tokens seen so far to remove during decoding.
    compression_interval : int, default=512
        Number of decoding steps between compression.
    hidden_states_buffer_size : int, default=256
        Maximum number of hidden states to keep before compression.

    Notes
    -----
    This press requires logical `position_ids` to be passed through the model
    forward call. Otherwise an exception will be raised.
    """

    target_compression_ratio: float

    def __init__(
        self,
        base_press: ScorerPress | AdaKVPress,
        target_compression_ratio: float,
        compression_interval: int = 512,
        hidden_states_buffer_size: int = 256,
    ):
        self.target_compression_ratio = target_compression_ratio
        super().__init__(
            base_press=base_press,
            compression_interval=compression_interval,
            target_size=1,
            hidden_states_buffer_size=hidden_states_buffer_size,
        )
        assert 0 <= self.target_compression_ratio < 1, "target_compression_ratio must be between 0 and 1"

    def _resolve_target_size(self, kwargs: dict) -> int:
        total_tokens_seen = self._resolve_total_tokens_seen(kwargs)
        return max(1, int(total_tokens_seen * (1 - self.target_compression_ratio)))

    def _resolve_total_tokens_seen(self, kwargs: dict) -> int:
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            return int(kwargs["position_ids"].reshape(-1)[-1].item()) + 1

        raise NotImplementedError("CompressionRatioDecodingPress requires logical position_ids in kwargs")
