# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for DSA cross-layer top-k index sharing (IndexShare, GLM-5.2)."""

import pytest

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    validate_dsa_index_share_pipeline_split,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    is_dsa_skip_topk_layer,
    source_dsa_compute_layer,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig

# GLM-5.2: index_topk_freq=4, index_skip_topk_offset=3 over 78 decoder layers.
GLM52_FREQ = 4
GLM52_OFFSET = 3
GLM52_NUM_LAYERS = 78
# HF `indexer_types` marks these 0-indexed layers "full"; the rest are "shared".
GLM52_FULL_LAYERS_0IDX = [0, 1, 2] + list(range(6, GLM52_NUM_LAYERS, 4))


class TestIndexSharePattern:
    """The freq/offset pattern must reproduce HF's `indexer_types` exactly."""

    def test_full_layers_match_hf_indexer_types(self):
        full = [
            i
            for i in range(GLM52_NUM_LAYERS)
            if not is_dsa_skip_topk_layer(i + 1, GLM52_OFFSET, GLM52_FREQ)
        ]
        assert full == GLM52_FULL_LAYERS_0IDX
        assert len(full) == 21

    def test_shared_layer_count_matches_weight_delta(self):
        shared = [
            i
            for i in range(GLM52_NUM_LAYERS)
            if is_dsa_skip_topk_layer(i + 1, GLM52_OFFSET, GLM52_FREQ)
        ]
        # GLM-5 has 59870 tensors, GLM-5.2 has 59585; the 285 difference is
        # 57 shared layers x 5 indexer tensors (wq_b, wk, k_norm.weight, k_norm.bias,
        # weights_proj).
        assert len(shared) == 57
        assert len(shared) * 5 == 59870 - 59585

    def test_source_layer_is_nearest_preceding_full_layer(self):
        for layer_id in range(GLM52_NUM_LAYERS):
            layer_number = layer_id + 1
            source = source_dsa_compute_layer(layer_number, GLM52_OFFSET, GLM52_FREQ)
            # A source layer must itself own an indexer, and never lie ahead.
            assert not is_dsa_skip_topk_layer(source, GLM52_OFFSET, GLM52_FREQ)
            assert source <= layer_number
            if not is_dsa_skip_topk_layer(layer_number, GLM52_OFFSET, GLM52_FREQ):
                assert source == layer_number
            else:
                # Nothing between source and this layer may be a full layer.
                for between in range(source + 1, layer_number):
                    assert is_dsa_skip_topk_layer(between, GLM52_OFFSET, GLM52_FREQ)

    def test_groups_are_four_layers_wide_after_the_offset(self):
        for start in range(6, GLM52_NUM_LAYERS, 4):
            group = range(start + 1, min(start + 5, GLM52_NUM_LAYERS + 1))
            sources = {source_dsa_compute_layer(n, GLM52_OFFSET, GLM52_FREQ) for n in group}
            assert sources == {start + 1}

    def test_freq_one_disables_sharing(self):
        for layer_number in range(1, 20):
            assert not is_dsa_skip_topk_layer(layer_number, 0, 1)
            assert source_dsa_compute_layer(layer_number, 0, 1) == layer_number

    def test_config_rejects_invalid_arguments(self):
        # The pure helpers assume validated input; TransformerConfig.__post_init__ is the single
        # place that range-checks these two knobs.
        base = dict(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            experimental_attention_variant="dsa",
        )
        with pytest.raises(ValueError, match="dsa_indexer_skip_topk_offset"):
            MLATransformerConfig(**base, dsa_indexer_skip_topk_offset=-1)
        with pytest.raises(ValueError, match="dsa_indexer_topk_freq"):
            MLATransformerConfig(**base, dsa_indexer_topk_freq=0)


class _LayoutConfig:
    """Minimal stand-in for TransformerConfig in the layout validator."""

    def __init__(self, freq=GLM52_FREQ, offset=GLM52_OFFSET, variant="dsa"):
        self.experimental_attention_variant = variant
        self.dsa_indexer_topk_freq = freq
        self.dsa_indexer_skip_topk_offset = offset


class TestIndexSharePipelineSplit:
    """Index sharing never crosses a pipeline stage, so layouts must be group-aligned."""

    @staticmethod
    def _chunks(sizes):
        start = 0
        for size in sizes:
            yield range(start, start + size)
            start += size

    def test_aligned_vpp_layout_passes(self):
        # examples/glm5.2: every VPP chunk starts on a full layer.
        sizes = [6, 4, 4, 4, 8, 4, 4, 4, 4, 4, 8, 4, 4, 4, 8, 4]
        assert sum(sizes) == GLM52_NUM_LAYERS
        config = _LayoutConfig()
        for chunk in self._chunks(sizes):
            validate_dsa_index_share_pipeline_split(config, chunk)

    def test_misaligned_layout_is_rejected(self):
        # The GLM-5 layout (10 layers per PP rank, VPP=2) gives 5-layer chunks, which
        # is coprime with the 4-layer group width.
        sizes = [5] * 7 + [4] + [5] * 7 + [4]
        config = _LayoutConfig()
        rejected = 0
        for chunk in self._chunks(sizes):
            try:
                validate_dsa_index_share_pipeline_split(config, chunk)
            except RuntimeError:
                rejected += 1
        assert rejected == 11

    def test_stage_starting_on_shared_layer_is_rejected(self):
        config = _LayoutConfig()
        with pytest.raises(RuntimeError, match="index-share pipeline split is invalid"):
            validate_dsa_index_share_pipeline_split(config, range(7, 10))

    def test_validator_is_noop_without_index_share(self):
        # freq=1 or a non-DSA variant must not constrain the layout at all.
        validate_dsa_index_share_pipeline_split(_LayoutConfig(freq=1), range(7, 10))
        validate_dsa_index_share_pipeline_split(_LayoutConfig(variant=None), range(7, 10))
