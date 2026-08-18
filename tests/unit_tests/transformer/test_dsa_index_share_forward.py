# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Functional check: DSAttention index sharing reuses the producer layer's top-k indices.

Builds a small stack of DSAttention modules with GLM-5.2's freq/offset, runs one forward,
and asserts that (a) only the computing layers own indexer weights, (b) each shared layer
passes the exact tensor its source layer published to sparse attention, and (c) a shared layer
whose source did not run in the same forward fails loudly instead of silently mis-computing.
"""

import pytest
import torch

import megatron.core.transformer.experimental_attention_variant.dsa as dsa_module
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAIndexShareCarrier,
    DSAttention,
    DSAttentionSubmodules,
    is_dsa_skip_topk_layer,
    source_dsa_compute_layer,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

FREQ = 4
OFFSET = 3
NUM_LAYERS = 12
SEQ_LEN = 32
BATCH = 2


def _build_config():
    return MLATransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=256,
        num_attention_heads=16,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=16,
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=False,
        # IndexShare, as in GLM-5.2.
        dsa_indexer_topk_freq=FREQ,
        dsa_indexer_skip_topk_offset=OFFSET,
        # GLM-5.x indexer numerics.
        dsa_indexer_rope_interleaved=True,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_k_norm_epsilon=1e-6,
    )


def _build_layer(config, layer_number, pg_collection):
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    indexer_spec = ModuleSpec(
        module=DSAIndexer,
        submodules=DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        ),
    )
    return DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=indexer_spec),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type='self',
        pg_collection=pg_collection,
    )


class TestDSAttentionIndexShare:
    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        self.config = _build_config()
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp']
        )
        self.layers = [
            _build_layer(self.config, n, self.pg_collection) for n in range(1, NUM_LAYERS + 1)
        ]
        yield
        Utils.destroy_model_parallel()

    def test_only_computing_layers_own_an_indexer(self):
        for layer in self.layers:
            expected_skip = is_dsa_skip_topk_layer(layer.layer_number, OFFSET, FREQ)
            assert layer.skip_topk == expected_skip
            if expected_skip:
                assert layer.indexer is None
                assert layer.source_layer == source_dsa_compute_layer(
                    layer.layer_number, OFFSET, FREQ
                )
            else:
                assert isinstance(layer.indexer, DSAIndexer)
                assert layer.source_layer == layer.layer_number
        assert sum(1 for l in self.layers if l.indexer is not None) == 5  # layers 1,2,3,7,11

    def test_mtp_layer_always_owns_an_indexer(self):
        # layer_number is offset by num_layers for MTP, and MTP must keep its indexer
        # regardless of where that lands in the pattern.
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_spec = ModuleSpec(
            module=DSAIndexer,
            submodules=DSAIndexerSubmodules(
                linear_wq_b=ModuleSpec(module=TELinear),
                linear_wk=ModuleSpec(module=TELinear),
                k_norm=ModuleSpec(module=TENorm),
                linear_weights_proj=ModuleSpec(module=TELinear),
            ),
        )
        for depth in range(1, 5):
            mtp = DSAttention(
                config=self.config,
                submodules=DSAttentionSubmodules(indexer=indexer_spec),
                layer_number=depth,
                attn_mask_type=AttnMaskType.causal,
                attention_type='self',
                pg_collection=self.pg_collection,
                is_mtp_layer=True,
            )
            assert mtp.layer_number == NUM_LAYERS + depth
            assert not mtp.skip_topk, f"MTP depth {depth} must own an indexer"
            assert mtp.indexer is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shared_layers_consume_the_source_layer_indices(self, monkeypatch):
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        def rand(*shape):
            return torch.randn(*shape, dtype=torch.bfloat16).cuda()

        query = rand(SEQ_LEN, BATCH, num_heads, head_dim)
        key = rand(SEQ_LEN, BATCH, num_heads, head_dim)
        value = rand(SEQ_LEN, BATCH, num_heads, head_dim)
        x = rand(SEQ_LEN, BATCH, self.config.hidden_size)
        qr = rand(SEQ_LEN, BATCH, self.config.q_lora_rank)

        carrier = DSAIndexShareCarrier()
        received = {}
        current_layer = None
        original_unfused_dsa_fn = dsa_module.unfused_dsa_fn

        def record_topk_indices(query, key, value, topk_indices, softmax_scale):
            received[current_layer] = topk_indices
            return original_unfused_dsa_fn(query, key, value, topk_indices, softmax_scale)

        monkeypatch.setattr(dsa_module, "unfused_dsa_fn", record_topk_indices)
        for layer in self.layers:
            layer.cuda().eval()
            current_layer = layer.layer_number
            with torch.no_grad():
                out = layer(
                    query=query,
                    key=key,
                    value=value,
                    x=x,
                    qr=qr,
                    attention_mask=None,
                    attn_mask_type=AttnMaskType.causal,
                    index_share_carrier=carrier,
                )
            assert out.shape == (SEQ_LEN, BATCH, num_heads * head_dim)

        holder = carrier._dsa_index_share_topk_holder
        # Only computing layers publish.
        assert sorted(holder) == [1, 2, 3, 7, 11]
        # Every shared layer receives the exact tensor its source published.
        for layer in self.layers:
            if layer.skip_topk:
                assert layer.source_layer in holder
                assert received[layer.layer_number] is holder[layer.source_layer]
                assert holder[layer.source_layer].shape == (
                    BATCH,
                    SEQ_LEN,
                    min(self.config.dsa_indexer_topk, SEQ_LEN),
                )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shared_layer_without_its_source_fails_loudly(self):
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads
        shared = next(l for l in self.layers if l.skip_topk)
        shared.cuda().eval()

        def rand(*shape):
            return torch.randn(*shape, dtype=torch.bfloat16).cuda()

        with pytest.raises(RuntimeError, match="needs top-k indices from source"):
            with torch.no_grad():
                shared(
                    query=rand(SEQ_LEN, BATCH, num_heads, head_dim),
                    key=rand(SEQ_LEN, BATCH, num_heads, head_dim),
                    value=rand(SEQ_LEN, BATCH, num_heads, head_dim),
                    x=rand(SEQ_LEN, BATCH, self.config.hidden_size),
                    qr=rand(SEQ_LEN, BATCH, self.config.q_lora_rank),
                    attention_mask=None,
                    attn_mask_type=AttnMaskType.causal,
                    index_share_carrier=DSAIndexShareCarrier(),
                )
