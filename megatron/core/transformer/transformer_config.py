# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F

from megatron.core.enums import Fp8Recipe
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.transformer.enums import AttnBackend, LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

from ..fusions.fused_bias_geglu import quick_gelu
from ..model_parallel_config import ModelParallelConfig
from ..utils import (
    get_te_version,
    init_method_normal,
    is_te_min_version,
    is_torch_min_version,
    scaled_init_method_normal,
)
from ..parallel_state import get_pipeline_model_parallel_rank

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except ImportError:
    HAVE_PACKAGING = False


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Configuration object for megatron-core transformers.

    The initialization function has an argument for each parameter,
    including those in ModelParallelConfig.
    """

    ####################
    # model architecture
    ####################

    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    mtp_num_layers: Optional[int] = None
    """Number of Multi-Token Prediction (MTP) Layers."""

    mtp_loss_scaling_factor: Optional[float] = None
    """Weighting factor of Multi-Token Prediction (MTP) loss."""

    mtp_loss_scaling_factor_decay_ratio: Optional[float] = None
    """Weighting factor decay ratio for Multi-Token Prediction (MTP) loss."""

    mtp_shared_layers: bool = False
    """Share (tie) all MTP layers. A single MTP layer is created and recurrently
    forwarded multiple times instead of creating multiple independent layers."""

    mtp_connection_type: str = 'sequential'
    """Connection type of Multi-Token Prediction (MTP). 
    'sequential': each MTP layer takes the previous MTP layer's hidden states as input (chain).
    'parallel': every MTP layer takes the main model's hidden states as input (fan-out)."""

    num_layers_in_first_pipeline_stage: Optional[int] = None
    """Number of transformer layers on first pipeline stage.
    None implies equal layer division across PP ranks."""

    num_layers_in_last_pipeline_stage: Optional[int] = None
    """Number of transformer layers on last pipeline stage.
    None implies equal layer division across PP ranks."""

    pipeline_model_parallel_layout: Optional[Union[str, list, PipelineParallelLayerLayout]] = None
    """Custom definition of the pipeline parallel partitioning.
    Support type:
    - str: e.g., 'Et*3|(tt|)*29,m|L'. Stages are split by '|', replicated stages or layers
    can be described with multiplication. Commas can be used cosmetically.
    - list: e.g., [['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']].
    - PipelineParallelLayerLayout: a PipelineParallelLayerLayout object.
    If given either a string or a list, it will be transferred into a PipelineParallelLayerLayout
    in post init. Let i = a * pp_size + b, then layout[i] gives a list of the layers 
    in the a-th vpp stage and the b-th pp stage, i.e., vpp(0)pp(0), vpp(0)pp(1), ..., 
    vpp(i)pp(j), vpp(i)pp(j+1), ..., vpp(-1)pp(-2), vpp(-1)pp(-1).
    In the inner lists of layers, 'embedding' or 'E' denotes the embedding layer, 'loss' or 'L'
    denotes the loss function, and 'decoder' or 't' denotes the transformer decoder layer.
    Examples:
        [['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']]:
        pp = 2, vpp = None
        pp rank 0 holds: embedding, decoder
        pp rank 1 holds: decoder*3, loss
        'E|(tt|)*2,(t|)*4,mL':
        pp = 2, vpp = 4
        vpp rank 0 pp rank 0 holds: embedding
        vpp rank 0 pp rank 1~2 holds: decoder*2
        vpp rank 0 pp rank 3 holds: decoder
        vpp rank 1 pp rank 0~2 holds: decoder
        vpp rank 1 pp rank 3 holds: mtp, loss"""

    account_for_embedding_in_pipeline_split: bool = False
    """If set, the embedding layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    account_for_loss_in_pipeline_split: bool = False
    """If set, the loss layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    attention_backend: AttnBackend = AttnBackend.auto
    """Attention backend to run. By default we let transformer engine
    decide the best backend to run (except in the case of local).
    If attention backend is local we use the local pytorch implementation in mcore.
    Users can specify exact backend by changing this config. """

    softmax_scale: Optional[float] = None
    """Softmax scale for attention scaling."""

    softmax_type: Literal['vanilla', 'off-by-one', 'learnable'] = 'vanilla'
    """Applies modified softmax from https://www.evanmiller.org/attention-is-off-by-one.html. 
       Supports both TE FusedAttention and local unfused attention. Supports both a fixed offset and 
       and learnable offset."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: Optional[int] = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size
    if not provided."""

    kv_channels: Optional[int] = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residule connection ordering."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm operations."""

    layernorm_zero_centered_gamma: bool = False
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves
    numerical stability."""

    add_bias_linear: bool = True
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in
    MLP layer)."""

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    activation_func: Callable = F.gelu
    """Activation function to use for the non-linearity in the MLP."""

    activation_func_fp8_input_store: bool = False
    """Store the input of MLP activation function in FP8 for backprop to save memory.
    The stored input is casted back to the original precision before backprop compuatation."""

    glu_linear_offset: float = 0.0
    """Offset term in the GLU activation function: activation_func(x[0]) * (x[1] + offset). Only 
    used when gated_linear_unit is True"""

    activation_func_clamp_value: Optional[float] = None
    """Clamp the output of the linear_fc1 in the activation function. Only used when activation_func
    is quick_gelu or weighted SwiGLU (MoE only)."""

    num_moe_experts: Optional[int] = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    window_size: Optional[Tuple[int, int]] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    window_attn_skip_freq: Optional[Union[int, List[int]]] = None
    """Frequency of full attention layers among sliding window attention layers. Accepts either:
    - An integer N: Represents a (N-1):1 ratio, one full attention layer after (N-1) SWA layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,1,0,0,0,0], where 1 represents SWA. """

    normalization: str = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""

    qk_layernorm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    test_mode: bool = False
    """Whether to run real-time tests."""

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    multi_latent_attention: bool = False
    """Whether to use multi-latent attention."""

    no_rope_freq: Optional[Union[int, List[int]]] = None
    """Controls which layers perform Rotary Position Embedding (RoPE). Accepts either:
    An integer N: Creates a pattern where RoPE is skipped every N-1 layers. For example,
    no_rope=4 means RoPE is applied for 3 layers, then skipped for 1 layer, repeating this pattern.
    A list of integers: Defines a custom pattern where 1 means skip RoPE and 0 means apply RoPE.
    For example, [0,1,1,0] means: apply RoPE, skip RoPE, skip RoPE, apply RoPE."""
    
    ####################
    # attention variant
    ####################
    experimental_attention_variant: Optional[str] = None
    """Type of attention variant to use. Currently support gated_delta_net and dsa."""

    dsa_indexer_n_heads: Optional[int] = None
    """Number of DSA indexer heads."""

    dsa_indexer_head_dim: Optional[int] = None
    """Dimension per DSA indexer head."""

    dsa_indexer_topk: Optional[int] = None
    """Number of top-k tokens to select in DSA indexer."""

    dsa_indexer_loss_coeff: Optional[float] = None
    """Coefficient for the DSA indexer KL divergence loss. Set to 0 to disable indexer loss."""

    dsa_indexer_use_sparse_loss: bool = False
    """Whether to use sparse DSA indexer loss. If True, the indexer loss will be computed using the
    top-k indices."""

    dsa_indexer_topk_freq: int = 1
    """Cross-layer top-k index sharing (IndexShare) period. 1 disables sharing: every layer owns an
    indexer. N > 1 means only one layer in every N owns an indexer; the remaining layers reuse the
    top-k indices computed by the nearest preceding computing layer. Used by GLM-5.2
    (index_topk_freq=4)."""

    dsa_indexer_skip_topk_offset: int = 0
    """Layer offset at which the IndexShare period starts. Layers at or before the offset always own
    an indexer. GLM-5.2 uses index_skip_topk_offset=3, which together with topk_freq=4 gives
    computing layers [1, 2, 3, 7, 11, ..., 75] (1-indexed)."""

    dsa_indexer_rope_interleaved: bool = False
    """Whether the DSA indexer applies interleaved RoPE instead of the non-interleaved half-split
    layout. GLM-5.x sets this to True (HF indexer_rope_interleave)."""

    dsa_indexer_rotate_activation: bool = True
    """Whether to apply the Hadamard transform to the DSA indexer query/key. DeepSeek-V3.2 uses
    True; GLM-5.x uses False."""

    dsa_indexer_k_norm_epsilon: Optional[float] = None
    """Epsilon for the DSA indexer key LayerNorm. None falls back to layernorm_epsilon. GLM-5.x
    requires 1e-6, which differs from the model-level norm epsilon."""

    apply_dsa_kernel_fusion: bool = False
    """Whether to use fused DSA kernel"""
    ####################
    # DeepSeek-v4 hybrid attention
    ####################
    csa_window_size: int = 128
    """Sliding window size for compressed sparse attention."""

    csa_compress_ratios: Optional[List[int]] = None
    """Per-layer compress ratios for DSv4 hybrid attention, e.g. [0, 0, 4, 128, 4, 128, ...]."""

    csa_compress_rotary_base: float = 40000.0
    """RoPE base for compressed KV positions in compressed sparse attention."""

    csa_dense_mode: bool = False
    """Whether to use dense mode for compressed sparse attention. If True, the CSA indexer will be
    disabled."""

    moe_deepep_num_sms: int = 20
    """Number of SMs to use for DeepEP."""

    moe_hybridep_num_sms: int = 16
    """Number of SMs to use for HybridEP. In pure NVL scenarios, 
    16 SMs can generally achieve good bandwidth."""

    ####################
    # initialization
    ####################
    init_method: Optional[Callable] = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with
    mean=0.0 and std=init_method_std."""

    output_layer_init_method: Optional[Callable] = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    embedding_init_method: Optional[Callable] = None
    """
    Method to initialize weights of the embedding layer. If None, will be set as described 
    in init_method above.
    """

    embedding_init_method_std: Optional[float] = None
    """
    Standard deviation of the zero mean normal for the default initialization method for the 
    embedding layer. If None, will be set to init_method_std.
    """

    init_model_with_meta_device: bool = False
    """
    If True, initializes the model with the meta device. This is helpful for
    training of very large models. This feature is only works when megatron fsdp is turned on.
    """

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    disable_bf16_reduced_precision_matmul: bool = False
    """If True, sets torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False to
    prevent matmul from using reduced precision accumulation when using BF16."""

    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    persist_layer_norm: bool = False
    """If True, uses the persistent fused layer norm kernel. This kernel only supports a fixed set
    of hidden sizes."""

    memory_efficient_layer_norm: bool = False
    """If True, and using local layers (not from TransformerEngine), tells Apex to use the memory
    efficient fused LayerNorm kernel. Ignored if not using LayerNorm."""

    bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    use_fused_weighted_squared_relu: bool = False
    """If True, uses fused weighted squared relu kernel when using MoE."""

    fused_single_qkv_rope: bool = False
    """If set, avoid splitting QKV before ROPE forward and avoid concatenating ROPE dgrads."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: Optional[str] = None
    """Determines which type of activation recompute to use.  Megatron-core supports 'selective'
    activation checkpointing where the submodules set in --recompute-modules is checkpointed.
    The default is "core_attn" which is the memory intensive part of attention.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_method: Optional[str] = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: Optional[int] = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    distribute_saved_activations: Optional[bool] = None
    """If True, distribute recomputed activations across the model parallel group."""

    recompute_modules: Optional[List[str]] = None
    """The submodules to recompute.
    choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts",
             "a2a_overlap_attn","a2a_overlap_post_attn", "a2a_overlap_mlp".
    default: ["core_attn"].
    "core_attn": recompute the core attention part of the transformer layer.
    "moe_act": recompute the MoE MLP activation function.
    "layernorm": recompute the input_layernorm and pre_mlp_layernorm.
    "mla_up_proj": recompute the MLA up projection and RoPE applying parts.
    "mlp": recompute the dense MLP submodule.
    "moe": recompute the MoE layer.
    "shared_experts": recompute the shared experts in the MoE layer.
    "moe_act", "layernorm", and "mla_up_proj" use output-discarding checkpointing,
    "core_attn", "mlp", "moe", and "shared_experts" use normal checkpointing.
    "a2a_overlap_attn", "a2a_overlap_post_attn", "a2a_overlap_mlp": recompute computation segments
    that are split to enable expert-parallel All-to-All communication overlap; only valid when EP
    A2A overlap is enabled.
    """
    ####################
    # fp8 related
    ####################
    fp8: Optional[str] = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    fp8_recipe: Optional[str] = "delayed"
    """If set, enables the use of FP8 precision through Transformer Engine. There are 3 predefined
    choices (1) 'tensorwise' uses per tensor current scaling recipe, (2) 'delayed'
    uses delayed scaling recipe, 3) 'mxfp8' for Blackwell architecture only,
    4) 'blockwise' for blockwise scaling recipe."""

    fp8_param: bool = False
    """If set, keep the parameters in fp8 precision to save memory. This option must be used
    together with fp8 mode (i.e., TransformerConfig.fp8 is not None). Note that not all parameters
    will be converted to fp8; for example, biases will remain unchanged. The parameters affected are
    primarily the weights of GEMMs. The specific parameters that will be converted to fp8 are
    determined by TE."""

    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    fp8_interval: int = 1
    """DEPRECATED from TransformerEngine v1.8.0. This flag is ignored.
    Controls how often the scaling factor is recomputed.
    """

    fp8_amax_history_len: int = 1
    """The length of the amax history window used for scaling factor computation."""

    fp8_amax_compute_algo: str = "most_recent"
    """Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
    predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
    always chooses the most recently seen value.

    """

    fp8_wgrad: bool = True
    """When set to False, override FP8 config options and do the wgrad computation
    in higher precision."""

    fp8_dot_product_attention: bool = False
    """When set to True, use the FP8 implementation of Dot Product Attention."""

    fp8_multi_head_attention: bool = False
    """When set to True, use the FP8 implementation of Multi Head Attention."""

    tp_only_amax_red: bool = False
    """When set to True, reduce the FP8 AMAX only in the TP or TP-CP domain"""

    first_last_layers_bf16: bool = False
    """If True, retains first and last N TransformerBlocks in BF16 as opposed to FP8."""

    num_layers_at_start_in_bf16: int = 1
    """Number of layers at the start of the model to keep in BF16 precision when
    first_last_layers_bf16 is True."""

    num_layers_at_end_in_bf16: int = 1
    """Number of layers at the end of the model to keep in BF16 precision when
    first_last_layers_bf16 is True."""

    selective_fp8: bool = False
    """If True, enable selective FP8 training: only whitelisted modules
    (determined by selective_fp8_allowed_ub_names) run in FP8, all other
    modules (MLP, norms, etc.) stay in BF16.  Each component's config
    (e.g. LLM foundation, ViT image encoder) can independently set this
    flag via its own YAML."""

    selective_fp8_allowed_ub_names: Optional[List[str]] = None
    """Userbuffer names (TE linear layer names) that are allowed to run in FP8 during
    selective FP8 training. Defaults to empty (no modules enabled) if not specified."""

    use_kitchen: bool = False
    """Use the kitchen extension for transformer quantization."""

    ####################
    # fp4 related
    ####################
    fp4: Optional[str] = None
    """If set, enables the use of FP4 precision through Transformer Engine. Currently only 
    supports 'nvfp4' which uses NVFP4BlockScaling recipe (requires TE >= 2.7.0.dev0)."""

    fp4_recipe: Optional[str] = "nvfp4"
    """If set, enables the use of FP4 precision through Transformer Engine. Currently only
    'nvfp4' is supported which uses NVFP4BlockScaling recipe for Blackwell+ architecture."""

    fp4_param: bool = False
    """If set, keep the parameters in fp4 precision to save memory. This option must be used
    together with fp4 mode (i.e., TransformerConfig.fp4 is not None). Note that not all parameters
    will be converted to fp4; for example, biases will remain unchanged."""

    ####################
    # chunkpipe related
    ####################
    enable_chunkpipe: bool = False
    """when set to true, split sequence into multiple chunks"""

    chunksize: int = 0
    """size for each chunk"""

    chunk_num_per_seq: int = 0
    """number of chunks per sequence, calculated as seq_length // chunksize"""

    keep_activations_chunks: int = 0
    """num of chunks of which activations will be retained"""

    chunkpipe_forward_microbatch: int = 0
    """microbatch num for chunk pipe forward"""

    chunkpipe_backward_microbatch: int = 0
    """microbatch num for chunk pipe backward"""

    chunkpipe_current_group_size: int = 0
    """Runtime mutable field. The chunk_group_size of the group currently
    being processed by the scheduler. For pretrain this equals chunk_num_per_seq;
    for SFT it varies per group (1 for binpacked, >1 for long sequences)."""

    chunkpipe_chunk_idx_in_group: int = 0
    """Runtime mutable field. The index of the current chunk within its group
    (0-based). Set by the scheduler before each forward_step. Avoids relying
    on chunkpipe_forward_microbatch % group_size which breaks when group sizes
    vary across groups in SFT."""

    chunk_keys: dict[int, Any] = None
    """caches for keys"""

    chunk_values: dict[int, Any] = None
    """caches for values"""

    chunkpipe_forward: bool = False
    """chunkpipe forward"""

    sft_chunkpipe_mode: bool = False
    """Whether chunkpipe is running in SFT mode (dynamic group_size).
    Enabled when training_phase == 'sft' and enable_chunkpipe is True."""


    ####################
    # MoE related
    ####################
    moe_shared_expert_intermediate_size: Optional[int] = None
    """Shared expert total ffn hidden size.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    None means no shared expert.
    By default, the shared experts execute before the router. However, when
    moe_shared_expert_overlap or overlap_moe_expert_parallel_comm is set,
    the shared experts execute after the router, before the routed experts.
    This makes the gradients from the router and the shared experts added in
    different orders to the hidden_states, causing minor numerical differences
    in the hidden_states gradient."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and dispatcher communications.
    Without this, the shared experts execute before the router."""

    moe_layer_freq: Union[int, List[int]] = 1
    """Frequency between MoE layers and Dense layers. Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers.
    - A list that defines a custom pattern, e.g.: [1,1,1,0,1,1,1,0,1,1,1,0]"""

    moe_ffn_hidden_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size"""

    moe_router_load_balancing_type: Union[str, List[str]] = "aux_loss"
    """The load balancing strategy for the router.
    Options:
    - "aux_loss": Load balancing loss used in GShard and SwitchTransformer, calculated at
    micro-batch level.
    - "seq_aux_loss": Load balancing loss used in DeepSeekV2 and DeepSeekV3, computes loss
    for each individual sample.
    - "global_aux_loss": Load balancing loss calculated at global batch level.
    - "sinkhorn": Balancing algorithm used in S-BASE.
    - "none": No load balancing.
    A list of strings can be provided to combine multiple aux-loss load balancing types.
    The default is "aux_loss".
    """

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_topk_limited_devices: Optional[int] = None
    """Number of EP ranks to consider for each token in group-limited routing,
    DEPRECATED and replaced by moe_router_num_groups and moe_router_group_topk.
    """

    moe_router_padding_for_fp8: Optional[bool] = False
    """Whether to pad the routing_map to make sure the number of tokens each expert received
    is a multiple of 16/32 for FP8 precision. This can remove the explicit padding in the
    GroupedMLP layer."""

    moe_router_num_groups: Optional[int] = None
    """Number of groups to divide experts into for group-limited routing.
    When using group-limited routing:
    1. Experts are divided into 'moe_router_num_groups' equal-sized groups
    2. For each token, 'moe_router_group_topk' groups are selected based on sum of
    top-('moe_router_topk'/'moe_router_group_topk') routing scores within each group
    3. From these selected groups, 'moe_router_topk' individual experts are chosen
    Two common use cases:
    - Device-limited routing: Set 'moe_router_num_groups' equal to expert parallel size (EP)
    to limit each token to experts on a subset of devices
    (See DeepSeek-V2: https://arxiv.org/pdf/2405.04434)
    - Node-limited routing: Set 'moe_router_num_groups' equal to number of nodes in EP group
    to limit each token to experts on a subset of nodes
    (See DeepSeek-V3: https://arxiv.org/pdf/2412.19437)
    """

    moe_router_group_topk: Optional[int] = None
    """Number of selected groups for group-limited routing."""

    moe_router_pre_softmax: bool = False
    """Enable pre-softmax(pre-sigmoid) routing for MoE, which means softmax is before the 
    top-k selection.
    By default, softmax is done after top-k."""

    moe_router_topk_scaling_factor: Optional[float] = None
    """Scaling factor for routing score in top-k selection, only works when moe_router_pre_softmax
    enabled. Defaults to None, which means no scaling."""

    moe_router_score_function: str = "softmax"
    """Score function for MoE routing. Can be "softmax" or "sigmoid"."""

    moe_router_dtype: Optional[str] = None
    """Data type for routing and expert output weighted averaging. Using fp32 or fp64 can
    improve stability especially when the number of experts is large (e.g. finegrained-moe).
    None means no changes for dtype."""

    moe_router_enable_expert_bias: bool = False
    """TopK routing with dynamic per-expert bias in the aux-loss-free load balancing strategy.
    The routing decision is based on the sum of the routing scores and the expert bias.
    See https://arxiv.org/abs/2408.15664 for details."""

    moe_router_bias_update_rate: float = 1e-3
    """The expert bias is updated based on the number of assigned tokens to each expert
    in a global batch, where the bias is increased for the experts with less assigned tokens
    and decreased for the experts with more assigned tokens.
    The default value 1e-3 is same as that used in DeepSeekV3."""

    moe_router_force_load_balancing: bool = False
    """[Experimental] Force load balancing with random logits for MoE router, supports naive topk
    and group-limited topk. This is an experimental feature and only for benchmark."""

    moe_router_force_hotspot_ratio: float = 0.0
    """[Experimental] Force a ratio of router tokens to route to the first EP rank.
    This is an experimental feature and only for benchmark."""
    moe_n_hash_layers: int = 0
    """Number of leading transformer layers that use hash-based MoE routing.
    Layers with layer_number <= moe_n_hash_layers use a pre-computed tid2eid
    lookup table for expert selection instead of learned top-k routing."""

    actual_vocab_size: Optional[int] = None
    """Padded actual vocabulary size. Required when moe_n_hash_layers > 0 for the
    tid2eid lookup buffer in hash-based MoE routing."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).
    """

    moe_use_legacy_grouped_gemm: bool = False
    """Use legacy GroupedMLP rather than TEGroupedMLP.
    Note: The legacy one will be deprecated soon."""

    moe_aux_loss_coeff: Union[float, List[float]] = 0.0
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended.
    If a list of load balancing types is provided for `moe_router_load_balancing_type`,
    a corresponding list of coefficients should be provided here."""

    moe_z_loss_coeff: Optional[float] = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: Optional[float] = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    moe_token_dropping: bool = False
    """This feature involves selectively dropping and padding tokens for each expert to achieve a
    specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note that this is
    currently unsupported so should remain False."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'.
    Options are 'allgather','alltoall' and 'flex'."""

    moe_enable_echo: bool = False
    """[Experimental] Enable Elastic Cloning for Hot Experts."""

    moe_echo_dump_dir: Optional[str] = None
    """The directory to dump the echo routing data."""

    moe_echo_log_steps: Optional[str] = None
    """Comma-separated list of training steps to log echo expert stats, e.g. "1,3,5"."""

    moe_echo_log_layers: Optional[str] = None
    """Comma-separated list of layer numbers to log echo expert stats, e.g. "10,15,20,25"."""

    moe_echo_log_file: Optional[str] = None
    """Path to the output log file for echo expert stats."""

    moe_num_echo_experts: Optional[int] = None
    """[Experimental] Number of echo experts to use. If None, the number of echo experts is set to
    the number of experts."""

    moe_echo_expert_dispatch_overlap: bool = False
    """Enable overlap of echo expert dispatch and expert computation."""

    moe_echo_expert_dispatcher_type: str = "hybridep"
    """The type of expert dispatcher to use for echo experts. Can be either "hybridep" or "alltoall"."""

    moe_echo_algorithm: str = "sinkhorn"
    """Algorithm used for echo expert token assignment when moe_enable_echo is True.
    Options:
      - "sinkhorn": topology-aware Sinkhorn-Knopp optimal transport + iterative col-top1 matching.
      - "greedy": one_shot_greedy (K=1) or approx_bin_packing (K>1).
    It is only effective when moe_enable_echo is enabled."""

    moe_enable_deepep: bool = False
    """[Experimental] Enable DeepEP for efficient token dispatching and combine in MoE models."""

    moe_flex_dispatcher_backend: str = "deepep"
    """[Experimental] The backend to use for flex token dispatcher. The default is "deepep".
    Options are "deepep" and "hybridep". Currently only "hybridep" backend supports 
    the MNNVL case."""

    moe_received_token_capacity: Optional[float] = None
    """The capacity of total received tokens on each ep rank."""

    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: Optional[float] = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match
    the expert capacity length, effective only after the moe_expert_capacity_factor is set. The
    default setting is False."""

    moe_token_drop_policy: str = "probs"
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with
    the lowest probabilities will be dropped. If "position", tokens at the end of each batch will
    be dropped.
    """

    moe_layer_recompute: bool = False
    """Memory optimization: checkpointing moe_layer to save actiavtion memory."""

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_router_fusion: bool = False
    """Fuse ops in routing and aux loss calculation."""

    moe_apply_probs_on_input: bool = False
    """Apply probs on input of experts instead of applying after activation and glu."""

    ### moe memory monitor ###
    enable_moe_mem_monitor: bool = False
    """ Whether to enable memory monitor. """

    print_moe_mem_monitor_interval: int = 1000
    """ Interval to print memory monitor. """

    moe_mem_monitor_log: str = None
    """ Path to log memory monitor. """

    moe_mem_monitor_force_print_token_threshold: int = 100000000
    """ Threshold to force print memory monitor. """

    ##################
    # Context Parallel
    ##################
    cp_comm_type: Optional[Union[str, List[str]]] = None
    """Inter-gpu communication type for context parallelism.
    str: all layers share same communication type.
    List[str]: each layer has its separate communication type.
    cp_comm_type of each layer can be "p2p" or "all_gather" or "a2a" or "a2a+p2p".
    "p2p": Exchange KV chunks with P2P communications in ring topology. P2P is async and can be
    overlapped with attention compute.
    "all_gather": All-gather to get full sequence of KV before attention. The all-gather is not
    async, and cannot be overlapped.
    "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP group, and gather to get
    full sequence of QKV.
    "a2a+p2p": A hierarchical implementation of context parallelism to attention.
    It uses A2A communications in low-level CP groups (e.g., via NVLink),
    and P2P communications in high-level CP groups (e.g., via IBLink).
    """

    ##################
    # Cuda Graphs
    ##################
    enable_cuda_graph: bool = False
    """DEPRECATED and replaced by cuda_graph_impl.
    When set to true, either partial CUDA graph (1/many CUDA graph per layer) or full iteration
    CUDA graph (1 CUDA graph for whole iteration excluding optimizer) is enabled. --cuda-graph-scope
    determines the scope of graph capture."""

    cuda_graph_use_single_mempool: bool = False
    """When set to true, cudagraphs will be captured inside a single mempool, in which all
    cudagraphs may only be used once per step. If false, cudagraphs may be reused across
    microbatches. Enabling may reduce cudagraph memory overheads due to memory fragmentation,
    however may greatly increase the number of cudagraphs created when the number of microbatches
    is high."""

    cuda_graph_retain_backward_graph: bool = False
    """When set to true, cudagraph backward passes will be graph captured with 'retain_grad=True'
    This may enable cudagraphs for certain modules that are not completely cudagraph safe. For
    more details, see: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html."""

    cuda_graph_warmup_steps: int = 3
    """Number of warmup steps for CUDA graphs"""

    external_cuda_graph: bool = False
    """DEPRECATED and replaced by cuda_graph_impl.
    When set to true, TransformerLayer layers are swapped with user provided CUDA graphs."""

    cuda_graph_impl: str = "none"
    """Determines the CUDA graph capture implementation.
    "none": no CUDA graph.
    "local": capture the CUDA graph using MCore local implementation. Either partial CUDA graph
    (1/many CUDA graph per layer) or full iteration CUDA graph (1 CUDA graph for whole iteration
    excluding optimizer) is enabled.
    "transformer_engine": capture the CUDA graph using TE make_graphed_callables()."""

    cuda_graph_scope: str = "full"
    """Determines the CUDA graphs capturing scope.
    When cuda_graph_impl is set to "transformer_engine", valid values are "full" and "attn".
    "Full" scope captures a whole Transformer layer. "Attn" scope only captures operations in
    TransformerLayer._forward_attention().
    When cuda_graph_impl is set to "local", "full_iteration" can be specified as cuda_graph_scope
    to enable whole iteration CUDA graph. All other values enable layerwise CUDA graph."""

    ####################
    # Hyper-Connection Configuration
    ####################
    enable_hyper_connections: bool = False
    """Enable mHC residual connections."""

    num_residual_streams: int = 4
    """Number of residual streams (n in paper)."""

    mhc_sinkhorn_iterations: int = 20
    """Number of Sinkhorn-Knopp iterations for doubly stochastic projection."""

    mhc_init_gating_factor: float = 0.01
    """Initial value of Gating Factor (alpha in paper)."""

    recompute_hyper_connections: bool = False
    """Enable recomputation for HyperConnection intermediate activations.
    
    When enabled, all HyperConnection operations (compute_mappings, aggregate, apply_h_res, 
    apply_h_post) are wrapped with CheckpointWithoutOutput and managed by MHCBlockRecomputeManager.
    This significantly reduces memory usage by discarding intermediate activations and 
    recomputing them during backward pass.
    
    Requirements:
    - Only effective when enable_hyper_connections=True and training=True
    - Must use recompute_granularity='selective'
    - Cannot be used together with recompute_mlp=True (they use different checkpoint mechanisms)
    
    The last layer in each recompute block's final MLP BDA output is NOT checkpointed and 
    serves as the hook_tensor for registering the unified recompute hook."""

    mhc_recompute_layer_num: Optional[int] = None
    """Number of layers per MHC recompute block.
    
    When set, every `mhc_recompute_layer_num` layers form a recompute block. The last layer
    in each recompute block (i.e., layer_number % mhc_recompute_layer_num == 0 or the final
    layer in the transformer block) will:
    - NOT checkpoint its final MLP BDA
    - Register the unified recompute hook on its MLP BDA output
    - A new MHCBlockRecomputeManager is created for subsequent layers
    
    If None, all layers in the transformer block share a single recompute block."""

    mhc_use_perm_decomposition: bool = False
    """Enable mHC perm decomposition. 
    https://arxiv.org/abs/2601.05732"""

    mhc_use_triton_fused_kernel: bool = False
    """Enable mHC Triton fused kernel.
    https://github.com/WithNucleusAI/mHC-triton/tree/main"""

    use_fused_mhc: bool = False
    """Use the fused mHC pre/post forward kernels in megatron.core.transformer.fused_mhc_kernels.
    Read by HyperConnectionModule when enable_hyper_connections=True."""

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    disable_parameter_transpose_cache: bool = False
    """When set to true, the parameter transposes are not cached for subsequent iterations."""

    config_logger_dir: str = ""
    """When non-empty, dumps entry-point configs to config_logger_dir"""

    flash_decode: bool = False
    """ Use the optimized flash decoding kernel during inference. """

    use_te_activation_func: bool = False
    """Whether to use ffn activation functions implemented by TransformerEngine"""

    use_te_rng_tracker: bool = False
    """ Whether to use the TE or MCore version of the RNG tracker. """

    inference_rng_tracker: bool = False
    """ Whether we should instantiate a separate RNG tracker for inference. """

    inference_sampling_seed: int = 42
    """ Random seed to use for sampling during inference. """

    symmetric_ar_type: Optional[str] = None
    """Type of symmetric all reduce to use"""

    mrope_section: Optional[List[int]] = None
    """ Multimodal rope section is for channel dimension of temporal, height and width
    in rope calculation. """

    is_hybrid_model: bool = False
    """ Indicates whether this is a hybrid model. """

    mamba_state_dim: int = 128
    """The dimensionality of the state representation in Mamba layers."""

    mamba_head_dim: int = 64
    """The dimensionality of the heads in the Mamba layers."""

    mamba_num_groups: int = 8
    """The number of groups used in Mamba layers."""

    mamba_num_heads: Optional[int] = None
    """The number of heads used in Mamba layers. 
    If None, the number of heads will be hidden_size * expand // mamba_head_dim."""

    use_mamba_mem_eff_path: bool = True
    """If True, use the memory efficient path for Mamba layers."""

    mlp_chunks_for_prefill: int = 1
    """The number of chunks along the sequence dimension to use for MLP computation
    during prefill."""

    heterogeneous_block_specs: bool = False
    """Whether to use heterogeneous block specs (nemotron-nas architecture)."""

    hetereogenous_dist_checkpoint: bool = False
    """Whether to use heterogenous layers in distributed checkpoint."""

    ####################
    # Quantization
    ####################
    quant_recipe: Optional[RecipeConfig] = None
    """Configuration of any quantization to be applied to the model"""

    transformer_impl: str = "transformer_engine"
    """Transformer implementation to use.
    Options are 'transformer_engine' for Transformer Engine and 'local' for MCore."""

    # reduce variable seq shape p2p comm
    p2p_comm_fixed_seq_lengths_per_rank: int = 0
    """If enabled, all the lengths of p2p data will be padding to fixed lengths;
    It can be uesed for sft, to reduce communication shape of  variable seq.
    the fixed lengths is seq-length-per-rank + 1."""
    micro_batch_size: int = 1
    """The micro batch size to use for training."""

    #####################################
    # Fine-grained Activation Offloading
    #####################################
    fine_grained_activation_offloading: bool = False
    """If True, offload the input of the specified modules to the CPU."""

    offload_modules: Optional[list[str]] = None
    """The submodules to offload its input.
    choices: "attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act".
    "attn_norm": offload the input of the normalization in the attention part.
    "core_attn": offload the input of the core attention part.
    "mlp_norm": offload the input of the normalization in the mlp part.
    "attn_proj": offload the input of the attn linear projection part.
    "expert_fc1": offload the input of the expert fc1 part.
    "moe_act": offload the input of the moe act part.
    """
    offload_tensors: Optional[list[str]] = None
    """Tensors to offload to CPU during forward pass and reload during backward pass.
    Choices: "dispatched_input", "pre_mlp_layernorm_output".
    "dispatched_input": This tensor is the output of pre_routed_experts_compute() and 
    serves as the input to routed_experts_compute(). It contains the token data that 
    has been routed and prepared to be sent to individual experts.
    "pre_mlp_layernorm_output": This tensor is the output of pre_mlp_layernorm and 
    serves as the input to shared_experts_compute().
    """
    min_offloaded_tensor_size: int = 1024 * 1024
    """The minimum size of the tensor to be offloaded."""


    use_fp32_dtype_for_param_pattern: Optional[List[str]] = None
    """The module list for fp32 param training"""

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more
        details.
        """
        super().__post_init__()
        self.chunk_keys = {}
        self.chunk_values = {}

        if self.experimental_attention_variant == "dsv4_hybrid":
            assert self.multi_latent_attention, "DSv4 Hybrid requires multi_latent_attention."
            assert self.csa_compress_ratios is not None, "csa_compress_ratios must be set"
            mtp_layers = getattr(self, "mtp_num_layers", 0) or 0
            expected_len = self.num_layers + mtp_layers
            assert len(self.csa_compress_ratios) == expected_len, (
                f"csa_compress_ratios length ({len(self.csa_compress_ratios)}) must equal "
                f"num_layers + mtp_num_layers ({self.num_layers} + {mtp_layers} = {expected_len})"
            )
            assert all(
                ratio in [0, 4, 128] for ratio in self.csa_compress_ratios
            ), "csa_compress_ratios must be 0, 4, or 128"
            # sequence_parallel is supported for DSv4 Hybrid Attention
            assert not getattr(self, "qk_clip", False), (
                "QK clipping is not supported with DSv4 Hybrid Attention."
            )

        if self.experimental_attention_variant == "dsa":
            if self.dsa_indexer_topk_freq < 1:
                raise ValueError(
                    f"dsa_indexer_topk_freq must be positive, got {self.dsa_indexer_topk_freq}."
                )
            if self.dsa_indexer_skip_topk_offset < 0:
                raise ValueError(
                    "dsa_indexer_skip_topk_offset must be non-negative, got "
                    f"{self.dsa_indexer_skip_topk_offset}."
                )

        if self.fp16 and self.bf16:
            raise ValueError(
                f"Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True."
            )

        # Apply BF16 matmul precision setting if needed
        if self.bf16 and self.disable_bf16_reduced_precision_matmul:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if (
            self.num_query_groups % self.tensor_model_parallel_size != 0
            and self.experimental_attention_variant != "dsv4_hybrid"
        ):
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.fp8:
            # cannot support first last layer bf16 with delayed scaling
            if self.first_last_layers_bf16 and self.fp8_recipe == Fp8Recipe.delayed:
                raise ValueError("Delayed scaling does not support first / last layer in BF16.")

            # max bf16 layers per pipeline stage
            max_bf16_layers_per_pipeline_stage = (
                self.num_layers // self.pipeline_model_parallel_size
            )

            # check start/end bf16 layer counts are valid
            if self.first_last_layers_bf16:
                if (
                    self.num_layers_at_start_in_bf16 < 0
                    or self.num_layers_at_start_in_bf16 > max_bf16_layers_per_pipeline_stage
                ):
                    raise ValueError(
                        f"num_layers_at_start_in_bf16 ({self.num_layers_at_start_in_bf16}) must be "
                        f"between 0 and number of layers per pipeline stage "
                        f"({max_bf16_layers_per_pipeline_stage})."
                    )
                if (
                    self.num_layers_at_end_in_bf16 < 0
                    or self.num_layers_at_end_in_bf16 > max_bf16_layers_per_pipeline_stage
                ):
                    raise ValueError(
                        f"num_layers_at_end_in_bf16 ({self.num_layers_at_end_in_bf16}) must be "
                        f"between 0 and number of layers per pipeline stage "
                        f"({max_bf16_layers_per_pipeline_stage})."
                    )

        if self.fp8_param and not self.fp8:
            raise ValueError("fp8_param must be used together with fp8 mode.")

        # FP4 validation
        if self.fp4_param and not self.fp4:
            raise ValueError("fp4_param must be used together with fp4 mode.")

        if self.fp4 and self.fp8:
            raise ValueError("fp4 and fp8 cannot be used simultaneously. Please choose one.")

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError("num_moe_experts must be non None to use expert-parallel.")

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError("num_moe_experts must be non-negative.")

        if self.num_moe_experts is not None and self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size
            warnings.warn("moe_ffn_hidden_size is not set, using ffn_hidden_size instead.")

        if self.num_moe_experts is None:
            assert (
                self.moe_ffn_hidden_size is None
            ), "moe_ffn_hidden_size must be None when num_experts is not set."

        if self.moe_enable_deepep:
            if self.moe_token_dispatcher_type != "flex":
                raise ValueError("DeepEP backend is only supported with flex token dispatcher.")

            if self.moe_flex_dispatcher_backend == "hybridep":
                raise ValueError(
                    "deepep and hybridep backends cannot be enabled at the same time "
                    "for flex token dispatcher."
                )
            self.moe_flex_dispatcher_backend = "deepep"
            warnings.warn(
                "moe_enable_deepep is deprecated."
                "Please use --moe-flex-dispatcher-backend=deepep instead."
            )

        if self.moe_token_dispatcher_type == "flex":
            if self.moe_pad_expert_input_to_capacity and (
                self.moe_enable_deepep or self.moe_flex_dispatcher_backend == "deepep"
            ):
                raise ValueError(
                    "Flex token dispatcher with deepep backend does not support "
                    "moe_pad_expert_input_to_capacity"
                )

        if self.moe_shared_expert_intermediate_size is not None:
            if self.moe_shared_expert_intermediate_size <= 0:
                raise ValueError(
                    f"moe_shared_expert_intermediate_size must be "
                    f"num_shared_experts * ffn_size_of_each_shared_expert, "
                    f"but got {self.moe_shared_expert_intermediate_size}"
                )
            if self.moe_shared_expert_overlap and self.moe_token_dispatcher_type not in [
                "alltoall"
            ]:
                raise ValueError(
                    f"moe_shared_expert_overlap only works with alltoall token dispatcher."
                )

        if isinstance(self.moe_router_load_balancing_type, list):
            assert isinstance(self.moe_aux_loss_coeff, list) and len(
                self.moe_aux_loss_coeff
            ) == len(self.moe_router_load_balancing_type), (
                "moe_aux_loss_coeff must be a list of the same length as "
                "moe_router_load_balancing_type"
            )

        if self.moe_expert_capacity_factor is not None:
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if isinstance(self.moe_router_load_balancing_type, list):
                for load_balancing_type in self.moe_router_load_balancing_type:
                    if load_balancing_type not in [
                        "aux_loss",
                        "seq_aux_loss",
                        "global_aux_loss",
                        "none",
                    ]:
                        raise ValueError(
                            "moe_expert_capacity_factor only works with aux_loss, "
                            "seq_aux_loss, global_aux_loss or none load balancing"
                        )
            elif self.moe_router_load_balancing_type not in [
                "aux_loss",
                "seq_aux_loss",
                "global_aux_loss",
                "none",
            ]:
                raise ValueError(
                    "moe_expert_capacity_factor only works with aux_loss, "
                    "seq_aux_loss, global_aux_loss or none load balancing"
                )

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    "moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity"
                )

        if self.cpu_offloading and (
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(
                f"CPU offloading can be done only for layers less than {self.num_layers}"
            )
        
        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                "Currently there is no support for Pipeline parallelism with CPU offloading"
            )

        if self.cpu_offloading and self.recompute_granularity is not None:
            raise ValueError(
                "CPU offloading does not work when activation recomputation is enabled"
            )

        if self.recompute_granularity is not None:
            if self.recompute_granularity not in ["full", "selective"]:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full"'
                    'or "selective".'
                )

            if self.recompute_method is not None:
                if self.recompute_method not in ["block", "uniform"]:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != "selective":
                raise ValueError(
                    f"Using recompute_granularity: {self.recompute_granularity} so "
                    'recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != "selective" and self.recompute_num_layers is None \
                    and self.custom_pipeline_recompute_layers is None:
                raise ValueError(
                    f"When using recompute_granularity: {self.recompute_granularity} "
                    "recompute_num_layers must be between "
                    "1 and num_layers_per_pipeline_rank: "
                    f"{self.num_layers // self.pipeline_model_parallel_size}"
                )
            elif (
                self.recompute_granularity == "selective" and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f"When using recompute_granularity: {self.recompute_granularity} "
                    "recompute_num_layers must be None."
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f"distribute_saved_activations: {self.distribute_saved_activations} must be "
                    f"false when sequence parallel is enabled: {self.sequence_parallel}"
                )

        if self.recompute_modules is None:
            self.recompute_modules = ["core_attn"]

        if self.recompute_granularity == "selective":
            if len(self.recompute_modules) > 0:
                allowed_modules = {
                    "core_attn",
                    "moe_act",
                    "mlp_act",
                    "layernorm",
                    "mla_up_proj",
                    "pre_mlp",
                    "mlp",
                    "moe",
                    "shared_experts",
                    "routed_experts",
                    "a2a_overlap_attn",
                    "a2a_overlap_post_attn",
                    "a2a_overlap_mlp",
                }
                invalid_modules = set(self.recompute_modules) - allowed_modules
                assert not invalid_modules, (
                    f"Invalid choices for recompute_modules: {invalid_modules}. "
                    f"Allowed modules are: {allowed_modules}"
                )

            if "moe_act" in self.recompute_modules and not self.moe_grouped_gemm:
                raise ValueError(
                    "moe_act in recompute_modules is only supported with moe_grouped_gemm."
                )

            if "mla_up_proj" in self.recompute_modules and not self.multi_latent_attention:
                raise ValueError(
                    "mla_up_proj in recompute_modules is only supported with "
                    "multi_latent_attention."
                )

            if "core_attn" in self.recompute_modules:
                warnings.warn(
                    "If you are using transformer_engine as the transformer implementation, "
                    "the core_attn is from transformer_engine and may be the fused version. "
                    "For fused attention, you have no need to set 'core_attn' to recompute. "
                    "Please check that the core_attn recompute is really needed."
                )

            if "shared_experts" in self.recompute_modules:
                if (
                    self.moe_shared_expert_intermediate_size is not None
                    and self.moe_shared_expert_overlap
                ):
                    raise ValueError(
                        "shared_experts recompute cannot work with --moe-shared-expert-overlap."
                    )
            if "a2a_overlap_attn" in self.recompute_modules and not self.overlap_moe_expert_parallel_comm:
                raise ValueError(
                    "a2a_overlap_attn recompute cannot work with --overlap-moe-expert-parallel-comm."
                )
            if "a2a_overlap_post_attn" in self.recompute_modules and not self.overlap_moe_expert_parallel_comm:
                raise ValueError(
                    "a2a_overlap_post_attn recompute cannot work with --overlap-moe-expert-parallel-comm."
                )
            if "a2a_overlap_mlp" in self.recompute_modules and not self.overlap_moe_expert_parallel_comm:
                raise ValueError(
                    "a2a_overlap_mlp recompute cannot work with --overlap-moe-expert-parallel-comm."
                )

            if self.fp8:
                if "moe_act" in self.recompute_modules or "layernorm" in self.recompute_modules:
                    if self.fp8_recipe == 'delayed':
                        raise ValueError(
                            "Delayed scaling does not support moe_act and layernorm recompute "
                            "for fp8."
                        )
                    if not is_te_min_version("2.6.0dev0"):
                        raise ValueError(
                            "moe_act and layernorm recompute for fp8 needs "
                            "transformer-engine>=2.6.0dev0, "
                            f"but your version is {get_te_version()}."
                        )

        if self.moe_layer_recompute:
            warnings.warn(
                "--moe-layer-recompute is deprecated. "
                "Use --recompute-granularity selective --recompute-modules moe_layer instead."
            )
            if self.recompute_granularity == "full":
                raise ValueError(
                    "Do not set --moe-layer-recompute with full recompute granularity. "
                )
            self.recompute_granularity = "selective"
            if "moe" not in self.recompute_modules:
                self.recompute_modules.append("moe")

        # Validation for recompute_hyper_connections
        if self.recompute_hyper_connections:
            if not self.enable_hyper_connections:
                raise ValueError(
                    "recompute_hyper_connections requires enable_hyper_connections=True."
                )
            if self.recompute_granularity != "selective":
                raise ValueError(
                    "recompute_hyper_connections requires recompute_granularity='selective'. "
                    f"Got recompute_granularity={self.recompute_granularity}."
                )
            if "mlp" in self.recompute_modules:
                raise ValueError(
                    "recompute_hyper_connections cannot be used together with 'mlp' in "
                    "recompute_modules. They use different checkpoint mechanisms that may conflict."
                )

        # Validation for hyper_connections with tensor parallelism
        # When hyper connections are enabled with TP > 1, sequence_parallel must be True.
        # This is because HyperConnectionModule uses non-TP-aware layers (nn.Linear, nn.RMSNorm),
        # and their gradients need to be synchronized across TP ranks via the sequence_parallel
        # attribute mechanism.
        if self.enable_hyper_connections and self.tensor_model_parallel_size > 1:
            if not self.sequence_parallel and self.experimental_attention_variant != "dsv4_hybrid":
                raise ValueError(
                    "When enable_hyper_connections=True and tensor_model_parallel_size > 1, "
                    "sequence_parallel must be True. HyperConnectionModule parameters require "
                    "gradient synchronization across TP ranks, which is handled by the "
                    "sequence_parallel mechanism."
                )

        if self.fine_grained_activation_offloading:
            # At least one of offload_modules or offload_tensors must be specified
            has_modules = self.offload_modules is not None and len(self.offload_modules) > 0
            has_tensors = self.offload_tensors is not None and len(self.offload_tensors) > 0
            
            assert has_modules or has_tensors, (
                "fine_grained_activation_offloading requires at least one of "
                "offload_modules or offload_tensors to be specified."
            )
            
            # Validate offload_modules if specified
            if has_modules:
                allowed_modules = {
                    "core_attn",
                    "attn_proj",
                    "expert_fc1",
                    "moe_act",
                    "attn_norm",
                    "mlp_norm",
                }
                invalid_modules = set(self.offload_modules) - allowed_modules
                assert not invalid_modules, (
                    f'Invalid choices for offload_modules: {invalid_modules}. '
                    f'Allowed modules are: {allowed_modules}'
                )
                if "attn_proj" in self.offload_modules and "core_attn" not in self.offload_modules:
                    raise ValueError(
                        "attn_proj cannot be set to offload_modules alone without core_attn "
                        "because the input of attn_proj is the output of core_attn, "
                        "which is needed in core_attn.backward()."
                    )
            
            # Validate offload_tensors if specified
            if has_tensors:
                allowed_tensors = {
                    "dispatched_input",
                    "pre_mlp_layernorm_output",
                }
                invalid_tensors = set(self.offload_tensors) - allowed_tensors
                assert not invalid_tensors, (
                    f'Invalid choices for offload_tensors: {invalid_tensors}. '
                    f'Allowed tensors are: {allowed_tensors}'
                )
                if "dispatched_input" in self.offload_tensors and "a2a_overlap_mlp" not in self.recompute_modules:
                    raise ValueError(
                        "Offloading 'dispatched_input' is only supported when 'a2a_overlap_mlp' recomputation "
                        "is enabled. Please add 'a2a_overlap_mlp' to --recompute-modules."
                    )
                if (
                    "pre_mlp_layernorm_output" in self.offload_tensors
                    and "a2a_overlap_mlp" not in self.recompute_modules
                ):
                    raise ValueError(
                        "Offloading 'dispatched_input' is only supported when 'a2a_overlap_mlp' recomputation "
                        "is enabled. Please add 'a2a_overlap_mlp' to --recompute-modules."
                    )

        if (
            self.num_layers_in_first_pipeline_stage is not None
            or self.num_layers_in_last_pipeline_stage is not None
        ) and (
            self.account_for_embedding_in_pipeline_split or self.account_for_loss_in_pipeline_split
        ):
            raise ValueError(
                "num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage cannot be"
                "set at the same time with account_for_embedding_in_pipeline_split"
                "and account_for_loss_in_pipeline_split"
            )

        # PP layout
        if self.pipeline_model_parallel_layout is not None:
            # If pipeline layout is set, we will check the conflicts
            # with other pipeline layout arguments.
            any_conflict = (
                self.num_layers_in_first_pipeline_stage is not None
                or self.num_layers_in_last_pipeline_stage is not None
                or self.account_for_embedding_in_pipeline_split
                or self.account_for_loss_in_pipeline_split
            )
            if any_conflict:
                raise ValueError(
                    "pipeline_model_parallel_layout cannot be set"
                    " with other pipeline layout arguments."
                    f" {self.num_layers_in_first_pipeline_stage=},"
                    f" {self.num_layers_in_last_pipeline_stage=},"
                    f" {self.account_for_embedding_in_pipeline_split=},"
                    f" {self.account_for_loss_in_pipeline_split=}."
                )

            # Transfer pipeline_model_parallel_layout from str or list to
            # PipelineParallelLayerLayout
            if isinstance(self.pipeline_model_parallel_layout, str):
                self.pipeline_model_parallel_layout = PipelineParallelLayerLayout.from_str(
                    layout=self.pipeline_model_parallel_layout,
                    pipeline_model_parallel_size=self.pipeline_model_parallel_size,
                )
            elif isinstance(self.pipeline_model_parallel_layout, list):
                # Since list is not hashable, the initialization will not be cached.
                self.pipeline_model_parallel_layout = PipelineParallelLayerLayout(
                    layout=self.pipeline_model_parallel_layout,
                    pipeline_model_parallel_size=self.pipeline_model_parallel_size,
                )

            # Check whether the input VPP size conflicts with the PP layout
            detected_vpp_size = (
                self.pipeline_model_parallel_layout.virtual_pipeline_model_parallel_size
            )
            if self.virtual_pipeline_model_parallel_size is not None:
                assert self.virtual_pipeline_model_parallel_size == detected_vpp_size, (
                    f"virtual_pipeline_model_parallel_size conflicts with"
                    f" pipeline_model_parallel_layout,"
                    f" ({self.virtual_pipeline_model_parallel_size=}, "
                    f" {detected_vpp_size=})"
                )
            elif detected_vpp_size > 1:
                self.virtual_pipeline_model_parallel_size = detected_vpp_size

            # Check whether the layout is valid.
            self.mtp_standalone = self.pipeline_model_parallel_layout.validate_layer_layout(
                num_layers=self.num_layers, mtp_num_layers=self.mtp_num_layers
            )

        # Uneven PP
        elif (
            self.num_layers_in_first_pipeline_stage is not None
            or self.num_layers_in_last_pipeline_stage is not None
        ):
            pipeline_parallel_size = self.pipeline_model_parallel_size
            num_layers = self.num_layers

            if self.num_layers_in_first_pipeline_stage is not None:
                if self.num_layers_in_first_pipeline_stage <= 0:
                    raise ValueError("num_layers_in_first_pipeline_stage must be larger than 0")

                if self.virtual_pipeline_model_parallel_size is not None:
                    if (
                        self.num_layers_in_first_pipeline_stage
                        % self.virtual_pipeline_model_parallel_size
                        != 0
                    ):
                        raise ValueError(
                            f"number of layers at first stage: "
                            f"{self.num_layers_in_first_pipeline_stage}"
                            f"must be divisible by virtual pipeline"
                            f"parallel degree {self.virtual_pipeline_model_parallel_size}"
                        )
                num_layers -= self.num_layers_in_first_pipeline_stage
                pipeline_parallel_size -= 1

            if self.num_layers_in_last_pipeline_stage is not None:
                if self.num_layers_in_last_pipeline_stage <= 0:
                    raise ValueError("num_layers_in_last_pipeline_stage must be larger than 0")

                if self.virtual_pipeline_model_parallel_size is not None:
                    if (
                        self.num_layers_in_last_pipeline_stage
                        % self.virtual_pipeline_model_parallel_size
                        != 0
                    ):
                        raise ValueError(
                            f"number of layers at last stage: "
                            f"{self.num_layers_in_last_pipeline_stage}"
                            f"must be divisible by virtual pipeline"
                            f"parallel degree {self.virtual_pipeline_model_parallel_size}"
                        )
                num_layers -= self.num_layers_in_last_pipeline_stage
                pipeline_parallel_size -= 1

            # Here pipeline_parallel_size is the number of middle PP stages. If there are middle
            # PP stages, check number of layers at middle stage is divisible by middle PP size.
            if pipeline_parallel_size and not num_layers % pipeline_parallel_size == 0:
                raise ValueError(
                    f"number of layers at middle stage: {num_layers} must be divisible by"
                    f"the middle pipeline model parallel size {pipeline_parallel_size}"
                )

            # If there are middle PP stages, check number of layers
            # on each middle PP rank is divisible by VPP size.
            if pipeline_parallel_size and self.virtual_pipeline_model_parallel_size is not None:
                num_layers_per_middle_pipeline_rank = num_layers // pipeline_parallel_size
                if (
                    not num_layers_per_middle_pipeline_rank
                    % self.virtual_pipeline_model_parallel_size
                    == 0
                ):
                    raise ValueError(
                        f"number of layers on each middle pipeline rank:"
                        f"{num_layers_per_middle_pipeline_rank} must be divisible by virtual"
                        f"pipeline parallel degree {self.virtual_pipeline_model_parallel_size}"
                    )

        elif (
            self.account_for_embedding_in_pipeline_split or self.account_for_loss_in_pipeline_split
        ):
            if self.virtual_pipeline_model_parallel_size is None:
                num_layers = self.num_layers

                if self.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if self.account_for_loss_in_pipeline_split:
                    num_layers += 1

                if not num_layers % self.pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f"number of middle layers: {num_layers} must be divisible by "
                        f"middle pipeline_model_parallel_size {self.pipeline_model_parallel_size}"
                    )
            else:
                num_layers = self.num_layers
                if self.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if self.account_for_loss_in_pipeline_split:
                    num_layers += 1

                if not num_layers % self.pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f"num_layers: {num_layers} after enable"
                        f"account_for_embedding_in_pipeline_split or "
                        f"account_for_loss_in_pipeline_split must be divisible"
                        f"by pipeline_model_parallel_size "
                        f"{self.pipeline_model_parallel_size}"
                    )

                num_layers_per_pipeline_rank = num_layers // self.pipeline_model_parallel_size
                if (
                    not num_layers_per_pipeline_rank % self.virtual_pipeline_model_parallel_size
                    == 0
                ):
                    raise ValueError(
                        f"number of layers on each pipeline rank: {num_layers_per_pipeline_rank}"
                        f"(after enable account_for_embedding_in_pipeline_split or "
                        f"account_for_loss_in_pipeline_split) must be divisible by"
                        f"virtual_pipeline_model_parallel_size"
                        f"{self.virtual_pipeline_model_parallel_size}"
                    )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_activation_fusion:
            if self.activation_func not in [F.gelu, F.silu, quick_gelu]:
                raise ValueError(
                    "When bias_activation_fusion is True, activation function should be either "
                    "gelu, swiglu, or quick_geglu"
                )
            if (
                self.activation_func == F.gelu
                and not self.gated_linear_unit
                and not self.add_bias_linear
            ):
                raise ValueError(
                    "When bias_activation_fusion is True, gated_linear_unit is False "
                    "and activation function is gelu, add_bias_linear must also be True."
                )
            if self.activation_func == quick_gelu and not self.gated_linear_unit:
                raise ValueError(
                    "When bias_activation_fusion is True and activation function is quick_gelu, "
                    "gated_linear_unit must be True."
                )
            if self.glu_linear_offset != 0.0 and self.activation_func != quick_gelu:
                raise ValueError(
                    "When bias_activation_fusion is True and glu_linear_offset is non-zero, "
                    "activation function must be quick_gelu."
                )

            if self.use_te_activation_func:
                raise ValueError(
                    "bias_activation_fusion and use_te_activation_func cannot be both true. "
                    "If you use bias in MLP FC1, we recommend setting bias_activation_fusion "
                    "to True and use_te_activation_func to False."
                )

        if self.use_te_activation_func:
            if self.activation_func not in (F.gelu, F.silu, F.relu):
                raise ValueError(
                    "TransformerEngine only support gelu, geglu, silu, swiglu, relu, reglu. "
                    "If you don't want to use TransformerEngine activation function, set "
                    "use_te_activation_func to False"
                )

        if self.activation_func_fp8_input_store:
            if self.activation_func != F.silu or not self.gated_linear_unit:
                raise ValueError("Storing activation input in FP8 is supported only for SwiGLU.")

        if self.activation_func_clamp_value is not None:
            # swiglu
            if self.activation_func == F.silu and self.gated_linear_unit:
                if self.num_moe_experts is None:
                    raise ValueError(
                        "activation_func_clamp_value for SwiGLU is only supported with MoE."
                    )
                if self.use_te_activation_func:
                    raise ValueError(
                        "use_te_activation_func must be False "
                        "when activation_func_clamp_value is not None for SwiGLU"
                    )

        if self.apply_rope_fusion:
            if self.multi_latent_attention:
                warnings.warn(
                    "apply_rope_fusion for multi-latent attention only supports training. "
                    "It is experimental and may change in future versions."
                )
                if self.enable_chunkpipe:
                    self.apply_rope_fusion = False
            else:
                if self.rotary_interleaved:
                    if not is_te_min_version("2.3.0"):
                        raise ValueError(
                            "rotary_interleaved does not work with apply_rope_fusion for "
                            "TE < 2.3.0. Please install TE >= 2.3.0"
                        )

                from megatron.core.models.common.embeddings.rope_utils import (
                    fused_apply_rotary_pos_emb,
                    fused_apply_rotary_pos_emb_thd,
                )

                if fused_apply_rotary_pos_emb is None and fused_apply_rotary_pos_emb_thd is None:
                    raise ValueError(
                        "apply_rope_fusion is not available. Please install TE >= 1.4."
                    )

        if self.multi_latent_attention and self.rotary_interleaved:
            raise ValueError("rotary_interleaved does not work with multi_latent_attention.")

        # Set the embedding init method
        if self.embedding_init_method_std is None:
            # By default, use the same init std as you use for every other non-output layer.
            self.embedding_init_method_std = self.init_method_std

        if self.embedding_init_method is None:
            if self.init_method is None or (self.embedding_init_method_std != self.init_method_std):
                # In this case, we set both the init method and the embedding init method to
                #  whatever std value requested (or defaulted) for the embedding_init_layer
                self.embedding_init_method = init_method_normal(self.embedding_init_method_std)
            else:
                # Replicate the current behavior where if you are not changing the std of the
                #  embedding init differently and the init method is set, we fallback to the
                #  init method for this layer. Since we are here after an OR we know that
                #  init_method is not None
                self.embedding_init_method = self.init_method

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std,
                self.num_layers,
                multiplier=2.0 if not self.is_hybrid_model else 1.0,
            )

        if self.num_moe_experts is not None and self.add_bias_linear:
            assert (
                self.expert_tensor_parallel_size == 1
            ), "Bias in Moe is only supported when ETP==1"

        #if self.moe_router_enable_expert_bias and self.moe_router_score_function != "sigmoid":
        #    raise ValueError(
        #        "Expert bias for aux-loss-free routing only supports sigmoid score function."
        #        "Please set --moe-router-score-function sigmoid for sigmoid score function."
        #    )

        if self.moe_n_hash_layers > 0:
            assert (
                self.actual_vocab_size is not None
            ), "actual_vocab_size must be set when moe_n_hash_layers > 0."
            if self.pipeline_model_parallel_size > 1:
                assert self.pipeline_model_parallel_layout is not None, (
                    "pipeline_model_parallel_layout must be set when using hash MoE "
                    "layers with pipeline parallelism (PP > 1)."
                )
                # The embedding is always in layout[0][0] (PP rank 0, VPP rank 0).
                # All hash MoE layers must be in the same virtual pipeline stage.
                embedding_stage = self.pipeline_model_parallel_layout.layout[0][0]
                n_decoders_with_embedding = embedding_stage.count(LayerType.decoder)
                assert self.moe_n_hash_layers <= n_decoders_with_embedding, (
                    f"Currently, All hash MoE layers must be in the same virtual pipeline stage "
                    f"as the embedding. The embedding stage has "
                    f"{n_decoders_with_embedding} decoder layers, but "
                    f"moe_n_hash_layers={self.moe_n_hash_layers}."
                )
            assert (
                not self.overlap_moe_expert_parallel_comm
            ), "overlap_moe_expert_parallel_comm does not support moe_n_hash_layers > 0 for now."
            warnings.warn(
                "Hash MoE layer initialized with placeholder round-robin tid2eid. "
                "For real training, you MUST either (a) load tid2eid from a "
                "pre-trained DSv4 checkpoint, or (b) provide a frequency-aware "
                "initialization (e.g., Sinkhorn-balanced over token frequency). "
                "Round-robin will cause severe expert imbalance."
            )

        if self.num_moe_experts and self.fp8:
            # TE version below 1.7.0 will raise Error when handle zeros tokens for expert
            if not is_te_min_version("1.7.0.dev0"):
                raise ValueError(
                    "Only transformer-engine>=1.7.0 supports MoE FP8 training, "
                    f"but your version is {get_te_version()}."
                )

            if self.moe_grouped_gemm and not is_te_min_version("1.11.0"):
                raise ValueError(
                    "Only transformer-engine>=1.11.0 supports FP8 grouped gemm, "
                    f"but your version is {get_te_version()}."
                )

        if self.moe_router_padding_for_fp8:
            if self.fp8 is None:
                raise ValueError("fp8 must be specified when moe_router_padding_for_fp8 is True.")

            if self.moe_token_dispatcher_type in ["allgather", "alltoall_seq"]:
                raise ValueError(
                    "allgather and alltoall_seq dispatcher does not support "
                    "moe_router_padding_for_fp8."
                )

        if (
            self.moe_router_topk == 1
            and self.moe_router_score_function == "softmax"
            and not self.moe_router_pre_softmax
            and self.moe_router_load_balancing_type != "sinkhorn"
        ):
            # Requires applying softmax before selecting the top-k when k is 1,
            # since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")

        if self.moe_router_group_topk:
            if self.moe_router_topk_limited_devices:
                raise ValueError(
                    "moe_router_topk_limited_devices is deprecated and replaced by "
                    "moe_router_group_topk and moe_router_num_groups."
                )
            if not self.moe_router_num_groups:
                raise ValueError(
                    "When using group limited routing, moe_router_num_groups must be specified."
                )
            else:
                assert self.num_moe_experts % self.moe_router_num_groups == 0, (
                    f"num_moe_experts ({self.num_moe_experts}) should be divisible by "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
                assert self.moe_router_group_topk <= self.moe_router_num_groups, (
                    f"moe_router_group_topk ({self.moe_router_group_topk}) should be smaller than "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
        elif self.moe_router_topk_limited_devices:
            warnings.warn(
                "moe_router_topk_limited_devices is deprecated. Use moe_router_group_topk and "
                "moe_router_num_groups instead."
            )
            self.moe_router_group_topk = self.moe_router_topk_limited_devices
            self.moe_router_num_groups = self.expert_model_parallel_size

        if self.enable_cuda_graph or self.external_cuda_graph:
            assert (
                self.cuda_graph_impl == "none"
            ), "Do not use enable_cuda_graph or external_cuda_graph with cuda_graph_impl."
            assert (
                not self.enable_cuda_graph or not self.external_cuda_graph
            ), "enable_cuda_graph and external_cuda_graph cannot be enabled at the same time."

            if self.enable_cuda_graph:
                warnings.warn('enable_cuda_graph is deprecated, use cuda_graph_impl=local instead.')
                self.cuda_graph_impl = "local"
            if self.external_cuda_graph:
                warnings.warn(
                    'external_cuda_graph is deprecated, '
                    'use cuda_graph_impl=transformer_engine instead.'
                )
                self.cuda_graph_impl = "transformer_engine"
        if self.cuda_graph_impl != "none":
            assert self.cuda_graph_impl in [
                "transformer_engine",
                "local",
            ], f"Invalid cuda graph implementation: {self.cuda_graph_impl}"
            if self.cpu_offloading:
                raise ValueError("CUDA graphs not supported with CPU offloading.")
            if self.recompute_granularity:
                if (
                    self.recompute_granularity != "selective"
                    or self.cuda_graph_impl != "transformer_engine"
                    or self.cuda_graph_scope != "attn"
                ):
                    raise ValueError("CUDA graphs not supported with activation recomputation.")
                else:
                    for module in self.recompute_modules:
                        if module in ['core_attn', 'mla_up_proj']:
                            raise ValueError(
                                f'attn cuda graph is not supported with {module} recompute.'
                            )
                    if "layernorm" in self.recompute_modules:
                        warnings.warn(
                            "input_layernorm recompute is not supported with attention "
                            "cudagraph. Will only recompute the pre_mlp_layernorm."
                        )

        if self.moe_token_dispatcher_type in ["allgather"]:
            if self.variable_seq_lengths is True:
                raise ValueError(
                    f"Token dispatcher type: {self.moe_token_dispatcher_type} does not support "
                    f"variable sequence length, please use alltoall dispatcher instead."
                )

        if self.moe_enable_echo:
            assert self.gradient_accumulation_fusion is True, "MoE Echo only support gradient accumulation fusion."
            assert (
                self.moe_num_echo_experts is not None
            ), "moe_num_echo_experts must be specified when moe_enable_echo is True"
            assert (
                self.moe_num_echo_experts % self.expert_model_parallel_size == 0
            ), "moe_num_echo_experts must be divisible by expert_model_parallel_size when moe_enable_echo is True"
            
        if self.moe_permute_fusion:
            from megatron.core.transformer.moe.moe_utils import (
                fused_permute,
                fused_permute_with_probs,
                fused_sort_chunks_by_index,
                fused_sort_chunks_by_index_with_probs,
                fused_unpermute,
            )

            if (
                fused_permute is None
                or fused_permute_with_probs is None
                or fused_sort_chunks_by_index is None
                or fused_sort_chunks_by_index_with_probs is None
                or fused_unpermute is None
            ):
                raise ValueError("fused permutation is not available. Please install TE >= 2.1.0.")

        if self.overlap_moe_expert_parallel_comm:
            # TODO: remove this after we fix the hang issue with torch version < 2.6.0
            assert is_torch_min_version(
                "2.6.0"
            ), "A2A Overlap encounters hang issue with torch version < 2.6.0"
            if self.pipeline_model_parallel_size > 1:
                assert self.virtual_pipeline_model_parallel_size is not None, (
                    "If enabling EP A2A overlap, virtual_pipeline_model_parallel_size "
                    "must be specified when pipeline_model_parallel_size > 1"
                )
            # Expert model parallelism requirements
            assert (
                self.expert_model_parallel_size > 1
            ), 'overlap_moe_expert_parallel_comm is only supported with expert model parallelism'
            assert self.moe_token_dispatcher_type in [
                'alltoall',
                'flex',
            ], 'overlap_moe_expert_parallel_comm is supported with alltoall/flex token dispatcher'

            assert (
                self.recompute_granularity != 'full'
            ), 'disable full recomputation when enabling overlap_moe_expert_parallel_comm'
            assert (
                self.recompute_method is None
            ), 'disable recomputation method when enabling overlap_moe_expert_parallel_comm'
            assert (
                self.recompute_num_layers is None
            ), 'recompute_num_layers must be None when enabling overlap_moe_expert_parallel_comm'

            # Check if bf16 or fp16 is used
            assert (
                self.bf16 or self.fp16
            ), 'overlap_moe_expert_parallel_comm is only supported with bf16 or fp16 model'

            assert (
                not self.moe_shared_expert_overlap
            ), 'disable moe_shared_expert_overlap when enabling overlap_moe_expert_parallel_comm'
            assert (
                self.mtp_num_layers is None or self.mtp_num_layers == 1
            ), 'MTP layernum only supports 1 when enabling overlap_moe_expert_parallel_comm.'

        # Check delay_wgrad_compute compatibility
        if self.delay_wgrad_compute:
            assert (
                self.overlap_moe_expert_parallel_comm
            ), 'overlap_moe_expert_parallel_comm must be enabled when enabling delay_wgrad_compute'
            assert (
                not self.moe_use_legacy_grouped_gemm
            ), 'delay_wgrad_compute is not supported with legacy groupedgemm implementation'

        if self.context_parallel_size > 1 and self.cp_comm_type is not None:
            if isinstance(self.cp_comm_type, list):
                assert len(self.cp_comm_type) == self.num_layers, (
                    f"Length of cp_comm_type ({len(self.cp_comm_type)}) should equal to "
                    f"the total number of transformer layers ({self.num_layers})!"
                )
            else:
                assert isinstance(
                    self.cp_comm_type, str
                ), "Unsupported communication type for context parallelism!"

        assert (
            self.pipeline_model_parallel_size > 0
        ), f"Pipeline model parallel size must be larger than 0 \
            when enable --standalone-embedding-stage and --standalone-loss-stage"

        if (
            self.num_moe_experts is not None
            and self.num_moe_experts >= 32
            and not self.moe_router_dtype
        ):
            warnings.warn(
                "Using a large number of experts (e.g. >=32) without fp32 routing. "
                "Consider enabling moe_router_dtype for better numerical stability."
            )
        if self.symmetric_ar_type is not None:
            if not HAVE_PACKAGING:
                raise ImportError(
                    "packaging is not installed. Please install it with `pip install packaging`."
                )
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"

        if self.no_rope_freq:
            assert not self.flash_decode, "flash_decode cannot be used with no_rope."
            if isinstance(self.no_rope_freq, int):
                assert self.num_layers % self.no_rope_freq == 0, (
                    f"no_rope_freq={self.no_rope_freq} should be "
                    f"divisible by num_layers={self.num_layers}."
                )
                # Convert integer pattern to list pattern
                # e.g. no_rope=4 with num_layers=8 becomes [0,0,0,1,0,0,0,1]
                pattern = [0] * (self.no_rope_freq - 1) + [1]
                self.no_rope_freq = pattern * (self.num_layers // self.no_rope_freq)
            else:
                assert len(self.no_rope_freq) == self.num_layers, (
                    f"Length of no_rope list ({len(self.no_rope_freq)}) must match "
                    f"the number of layers ({self.num_layers})"
                )

        if self.use_fp32_dtype_for_param_pattern is not  None:
            allowed_modules = {
                "expert_bias",
                "output_layer",
                "final_layernorm",
                "input_layernorm",
                "pre_mlp_layernorm",
                "router",
                "self_attention_hyper_connection",
                "mlp_hyper_connection",
                "attn_hc",
                "ffn_hc",
                "hc_head",
                "sinks",
                "position_bias",
                "e_score_correction_bias",
                "q_a_norm",
                "kv_norm",
                "post_attention_layernorm",
                "norm",
            }
            invalid_modules = set(self.use_fp32_dtype_for_param_pattern) - allowed_modules
            assert not invalid_modules, (
                f"Invalid choices for recompute_modules: {invalid_modules}. "
                f"Allowed modules are: {allowed_modules}"
            )
        
@dataclass
class MLATransformerConfig(TransformerConfig):
    """Configuration object for megatron-core Multi-Latent Attention (MLA) transformers.

    The initialization function has an argument for each parameter, including those in
    ModelParallelConfig. Included YaRN RoPE parameters that is fused in MLA.
    """

    multi_latent_attention: bool = True
    """Whether to use Multi-Latent Attention."""

    q_lora_rank: int = 512
    """Rank of Query tensor's low rank representation."""

    kv_lora_rank: int = 512
    """Rank of Key and Value tensors' low rank representation."""

    qk_head_dim: int = 128
    """Dimension of the head in the QK projection. q_head_dim = qk_head_dim + qk_pos_emb_head_dim"""

    qk_pos_emb_head_dim: int = 64
    """Dimension of the position embedding in the QK projection."""

    v_head_dim: int = 128
    """Dimension of the head in the V projection."""

    normalization: str = "RMSNorm"
    """Default normalization layer for MLA models is RMSNorm."""

    rope_type: str = "yarn"
    """Type of RoPE to use. Default to yarn, options are rope and yarn."""

    rotary_base: float = 10000
    """Rotary base for the rotary embeddings, used by rope and yarn."""

    rotary_percent: float = 1.0
    """Rotary percent for the rotary embeddings, used by rope."""

    rotary_scaling_factor: float = 40
    """Rotary scaling factor for the rotary embeddings, used by yarn."""

    original_max_position_embeddings: int = 4096
    """Original maximum position embeddings for the original model, used by yarn."""

    beta_fast: float = 32
    """Beta fast for YaRN RoPE, used by yarn."""

    beta_slow: float = 1
    """Beta slow for YaRN RoPE, used by yarn."""

    mscale: float = 1.0
    """Mscale for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    mscale_all_dim: float = 0.0
    """Mscale all dimensions for YaRN RoPE in Multi-Latent Attention, used by yarn."""

    o_groups: int = 8
    """Number of groups for grouped low-rank output projection (wo_a) in DSv4 hybrid attention."""

    o_lora_rank: int = 1024
    """Low-rank dimension per group for grouped output (wo_a). Used when o_groups > 0."""

    cache_mla_latents: bool = False
    """Cache the low dimensional tensors for MLA rather than full KV cache.
       This is only for the dynamic inference backend and requires that 
       Flash MLA is installed."""
    
    padding_v_to_qk_dim: bool = False
    """Pad the value dimension to query and key dimension"""

    def __post_init__(self):
        super().__post_init__()
        if self.multi_latent_attention and self.apply_rope_fusion and self.rope_type != "yarn":
            raise ValueError("apply_rope_fusion for MLA only works with YARN RoPE.")

        # DSv4 hybrid: derive qk_head_dim and kv_lora_rank from v_head_dim and qk_pos_emb_head_dim.
        if self.experimental_attention_variant == "dsv4_hybrid":
            assert (
                not getattr(self, "mla_down_proj_fusion", False)
            ), "MLA down projection fusion must be disabled for DSv4 hybrid mode."
            derived = self.v_head_dim - self.qk_pos_emb_head_dim
            self.qk_head_dim = derived
            self.kv_lora_rank = derived
            self.hetereogenous_dist_checkpoint = True

        if self.cache_mla_latents:
            assert (
                self.apply_rope_fusion is False
            ), "Rope Fusion is not compatible with caching latents"
