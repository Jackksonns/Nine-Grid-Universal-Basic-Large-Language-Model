# typing: strict
# coding=utf-8
# Copyright 2024 QiYuan Inc.

import math
from typing import Optional
from typing import Tuple

import bmtrain as bmt
import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_func

from .configuration_dragonfly import DragonflyConfig  # from fm9g.utils import Config

# TODO:
# 1. add scale_emb to embed and layernorm
# 2. add scale_width to all layers
# 3. add scale_depth to residual


class ScaledRotaryEmbeddingESM(bmt.DistributedModule):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    Add multiple Positional Interpolation methods:
    + [Linear](http://arxiv.org/abs/2306.15595)
    + [NTK-aware](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
    + [Dynamic Scaling](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)
    + [NTK-by-parts](https://github.com/jquesnelle/yarn/pull/1)
    + [YaRN](http://arxiv.org/abs/2309.00071)
    Args:
        dim: Dimension of the input, attn_dim // n_heads.
        max_position_embeddings: Maximum number of positions to be embedded.
        base: Base of the positional encoding function.
        pose_prob: Probability of using PoSE.
        pose_scaling_factor: max_position_embeddings scaling factor for PoSE.
        scaling_type: Type of scaling to use, one of ["Linear", "NTK-aware", "Dynamic NTK", "NTK-by-parts", "YaRN", "Dynamic YaRN", ""].
        rope_scaling_factor: RoPE Scaling factor for scaling type, new max length / before extend max length.
        beta_fast: Number of rotations to use for fast angular velocity.
        beta_slow: Number of rotations to use for slow angular velocity.
        extrapolation_factor: [0, 1], 0 is fully extrapolation, 1 is fully NTK-by-parts/YaRN.
        attn_factor: Uniform attn scale factor for tuning YaRN, 1 is best for LLaMA-1/2.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        pose_prob: float = 0.0,
        pose_scaling_factor: float = 1.0,
        scaling_type: str = "",
        rope_scaling_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        extrapolation_factor: int = 1,
        attn_factor: int = 1,
        original_max_position_embeddings: int = 2048,
        persistent: bool = True,
        dynamic_scaling_seq_len: int = 512,
        device=None,
    ):
        assert scaling_type in ["Linear", "NTK-aware", "Dynamic NTK", "NTK-by-parts", "YaRN", ""]
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.persistent = persistent
        self.device = device
        # scaling config
        self.scaling_type = scaling_type
        self.pose_scaling_factor = pose_scaling_factor
        self.rope_scaling_factor = rope_scaling_factor
        # PoSE
        self.pose_prob = pose_prob
        # NTK-by-parts and YaRN args
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        if pose_prob > 0:
            self.scaled_max_position_embeddings = int(max_position_embeddings * pose_scaling_factor)
        else:
            self.scaled_max_position_embeddings = max_position_embeddings

        if self.scaling_type == "NTK-aware":
            base = self.base * (self.rope_scaling_factor ** (self.dim / (self.dim - 2)))
        else:
            base = self.base
        # TODO: Implement base NTK-aware in NTK-by-parts
        if self.scaling_type in ["NTK-by-parts", "YaRN"]:
            self._ntk_parts_update_inv_freq(self.scaled_max_position_embeddings)
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Get n-d magnitude scaling corrected for interpolation
        self.m_scale = float(self._get_m_scale(self.rope_scaling_factor) * self.attn_factor)
        self._set_cos_sin_cache(dynamic_scaling_seq_len)

    def _get_m_scale(self, scale=1.0):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def _ntk_parts_update_inv_freq(self, seq_len):
        # Inverse dim formula to find dim based on number of rotations
        def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        # Find dim range bounds based on rotations
        def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))

            return max(low, 0), min(high, dim - 1)  # Clamp values just in case

        def linear_ramp_mask(min, max, dim):
            if min == max:
                max += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.rope_scaling_factor * pos_freqs)
        low, high = find_correction_range(
            self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1 - linear_ramp_mask(low, high, self.dim // 2).float().to(self.device)
        ) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=self.persistent)

    def _set_cos_sin_cache(self, seq_len, device=None):
        self.max_seq_len_cached = seq_len
        if device is not None:
            self.device = device

        if self.scaling_type == "Dynamic NTK" and seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.rope_scaling_factor * seq_len / self.max_position_embeddings) - (self.rope_scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=self.persistent)

        t = torch.arange(self.max_seq_len_cached, device=self.device).type_as(self.inv_freq)
        if self.scaling_type == "Linear":
            freqs = torch.outer(t / self.rope_scaling_factor, self.inv_freq.to(device=t.device).to(t.dtype))
        else:
            freqs = torch.outer(t, self.inv_freq.to(device=t.device).to(t.dtype))
        emb = torch.cat((freqs, freqs), dim=-1)
        if self.scaling_type == "YaRN":
            self.register_buffer("cos_cached", (emb.cos() * self.m_scale), persistent=self.persistent)
            self.register_buffer("sin_cached", (emb.sin() * self.m_scale), persistent=self.persistent)
        else:
            self.register_buffer("cos_cached", emb.cos(), persistent=self.persistent)
            self.register_buffer("sin_cached", emb.sin(), persistent=self.persistent)

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids) -> Tuple[torch.Tensor, torch.Tensor]:
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        orig_dtype = k.dtype
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_fp32 = q.to(dtype=torch.float32, device=q.device)
        k_fp32 = k.to(dtype=torch.float32, device=k.device)
        q_embed = (q_fp32 * cos) + (self._rotate_half(q_fp32) * sin)
        k_embed = (k_fp32 * cos) + (self._rotate_half(k_fp32) * sin)
        return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_dim, offset=0, cu_seqlens=None, max_length=None, position_ids=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_dim = (seq_dim + k.dim()) % k.dim()
        # get max current seq len from all workers
        if self.pose_prob > 0.0:
            seq_len = torch.max(position_ids) + 1
        else:
            seq_len = k.size(seq_dim) + offset
        seq_len_tensor = torch.tensor(seq_len, device=self.device)
        seq_len_tensor_reduced = bmt.distributed.all_reduce(seq_len_tensor, op="max")
        seq_len_reduced = seq_len_tensor_reduced.item()
        # update cache if needed
        if seq_len_reduced > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos, sin = (
            self.cos_cached[:seq_len_reduced],
            self.sin_cached[:seq_len_reduced],
        )
        if position_ids.dtype != torch.long:  # 231108 input is int32
            position_ids = position_ids.to(dtype=torch.long)
        if cu_seqlens is None:
            q_embed, k_embed = self._apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        else:
            assert offset == 0, "past kv is not supported in flash attn"
            q_embed, k_embed = self._apply_rotary_pos_emb(q, k, cos, sin, position_ids.view(-1))

        return q_embed, k_embed


def Linear(*args, **kwargs):
    tp = kwargs.pop("tp", 0)
    if tp == 0:
        return NormalLinear(*args, **kwargs)
    if tp == 1:
        return ColumnParallelLinear(*args, **kwargs)
    if tp == 2:
        return RowParallelLinear(*args, **kwargs)


class NormalLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out

        # TODO:init
        # init_std = 1 / ((dim_in + dim_out) ** 0.5)

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501

        x = F.linear(x, self.weight, None)

        return x


class ColumnParallelLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 0.02,
        gather_output=False,
        gather_input=True,
    ):
        super().__init__()
        assert dim_out % bmt.config["tp_size"] == 0

        # TODO: init
        # init_std = 1 / ((dim_in + dim_out) ** 0.5)

        dim_out = dim_out // bmt.config["tp_size"]
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.gather_input = gather_input
        self.gather_output = gather_output

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=0,
            tp_mode=True,
        )
        self.bias = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501

        x = bmt.nn.OpParallelLinear.apply(
            x, self.weight, self.bias, self.gather_input, self.gather_output, False, None, 1
        )

        return x


class RowParallelLinear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 0.02,
        split_input=False,
        all_reduce_output=False,
    ):
        super().__init__()
        assert dim_in % bmt.config["tp_size"] == 0
        # init_std = 1 / ((dim_in + dim_out) ** 0.5)

        dim_in = dim_in // bmt.config["tp_size"]
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out

        self.split_input = split_input
        self.all_reduce_output = all_reduce_output

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            tp_split_dim=1,
            tp_mode=True,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if not self.all_reduce_output:
            x = x.view(x.shape[0] * bmt.config["tp_size"], -1, x.shape[-1])

        x = bmt.nn.OpParallelLinear.apply(
            x, self.weight, None, self.split_input, False, self.split_input, 1 if self.all_reduce_output else 2, 1
        )

        return x


@torch.jit.script
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class LayerNorm(bmt.DistributedModule):
    """RMS LayerNorm"""

    def __init__(
        self,
        dim_norm: int,
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        init_var: float = 1.0,
    ):
        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = bmt.DistributedParameter(torch.full((dim_norm,), init_var, dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output.
        """  # noqa: E501
        assert x.size(-1) == self.dim_norm
        return rms_layernorm(x, self.weight, self.eps)


class DenseGatedACT(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
        activate_fn: str = "silu",
        dtype=torch.bfloat16,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
    ):
        super().__init__()

        _std = init_std / math.sqrt(scale_width) if scale else init_std

        self.w_0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            tp=tp,
            init_std=_std,
        )

        self.w_1 = Linear(dim_in=dim_in, dim_out=dim_ff, dtype=dtype, tp=tp, init_std=_std)

        if activate_fn == "gelu":
            self.act = torch.nn.GELU()
        elif activate_fn == "silu":
            self.act = torch.nn.functional.silu
        else:
            raise NotImplementedError(f"{activate_fn} is not supported")

    def forward(self, x: torch.Tensor):
        """This model inherits from bmt.DistributedModule.
            Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_ff)``)

        """  # noqa: E501
        gate_score = self.act(self.w_0(x))
        x = self.w_1(x)

        x = gate_score * x
        return x


class FeedForward(bmt.DistributedModule):
    r"""FeedForward module

    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.bfloat16.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str = "silu",
        dtype=torch.bfloat16,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
    ):
        super().__init__()

        self.w_in = DenseGatedACT(
            dim_in=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        _std = init_std / math.sqrt(scale_width) if scale else init_std
        self.w_out = Linear(dim_in=dim_ff, dim_out=dim_model, dtype=dtype, init_std=_std)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """  # noqa: E501
        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x)

        return x


class Embedding(bmt.DistributedModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.bfloat16,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale: bool = False,
        scale_emb: float = 1.0,
        scale_width: float = 1.0,
        tp: int = 0,
    ):
        super().__init__()

        self.dim_model = embedding_size
        self.weight = bmt.DistributedParameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
        )
        self.tp = tp
        self.scale = scale
        self.scale_emb = scale_emb
        self.scale_width = scale_width

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        if self.tp:
            x = x.view(-1).chunk(bmt.config["tp_size"])[bmt.config["tp_rank"]].view(x.size(0), -1)

        embeds = F.embedding(x, self.weight)

        if self.scale:
            embeds = embeds * self.scale_emb

        return embeds

    def projection(self, x: torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501

        if self.scale:
            x = x / self.scale_width  # TODO: check if it is ok to add before all_gather

        logits = F.linear(x, self.weight)
        return logits


class Attention(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype: torch.dtype = torch.bfloat16,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_groups = num_heads // num_kv_heads
        self.dim_head = dim_head

        self.scale = scale
        _std = init_std / math.sqrt(scale_width) if scale else init_std

        self.project_q = Linear(
            self.dim_model,
            self.num_heads * self.dim_head,
            dtype=dtype,
            tp=tp,
            init_std=_std,
        )
        self.project_k = Linear(
            self.dim_model,
            self.num_kv_heads * self.dim_head,
            dtype=dtype,
            tp=tp,
            init_std=_std,
        )
        self.project_v = Linear(
            self.dim_model,
            self.num_kv_heads * self.dim_head,
            dtype=dtype,
            tp=tp,
            init_std=_std,
        )

        self.attention_out = Linear(
            self.num_heads * self.dim_head,
            self.dim_model,
            dtype=dtype,
            tp=tp * 2,  # TODO
            init_std=_std,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
            self.dropout_p = dropout_p
        else:
            self.dropout = None

        self.tp = tp

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        position_bias: torch.Tensor,  # TODO
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: int = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """This model inherits from bmt.DistributedModule.
        Args:
            hidden_q (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        batch_size = hidden_q.size(0)

        if self.tp:
            assert hidden_q.data_ptr() == hidden_kv.data_ptr()

            hidden_q = bmt.nn.OpParallelLinear.apply(
                hidden_q,
                torch.cat([self.project_q.weight, self.project_k.weight, self.project_v.weight], dim=0),
                torch.cat([self.project_q.bias, self.project_k.bias, self.project_v.bias], dim=0)
                if self.project_q.bias is not None
                else None,
                True,
                False,
                False,
                None,
                1,
            )

            hidden_q = hidden_q.view(batch_size, -1, hidden_q.shape[-1])

            block_size = hidden_q.shape[-1] // (self.head_groups + 1 + 1)
            h_q = hidden_q[..., : block_size * self.head_groups]
            h_k = hidden_q[..., block_size * self.head_groups : block_size * (self.head_groups + 1)]
            h_v = hidden_q[..., block_size * (self.head_groups + 1) :]
        else:
            h_q = self.project_q(hidden_q)
            h_k = self.project_k(hidden_kv)
            h_v = self.project_v(hidden_kv)

        len_q = h_q.size(1)
        len_k = h_k.size(1)

        h_q = h_q.view(batch_size * len_q, -1, self.dim_head)
        h_k = h_k.view(batch_size * len_k, -1, self.dim_head)
        h_v = h_v.view(batch_size * len_k, -1, self.dim_head)
        h_q, h_k = position_bias(h_q, h_k, -3, cu_seqlens=cu_seqlens, max_length=max_seqlen, position_ids=position_ids)
        score = flash_attn_varlen_func(
            h_q,
            h_k,
            h_v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            self.dropout_p,
            causal=True,
            deterministic=True,
        )

        #print("DEBUG! use flash!!!!!! ARQ")
        score = score.view(batch_size, len_q, -1)
        score = self.attention_out(score)

        return score


class SelfAttentionBlock(bmt.DistributedModule):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        qk_norm: bool = False,
        layer_id: int = 0,
        num_layers: int = 0,
    ):
        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.self_attention = Attention(
            dim_model=dim_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dim_head=dim_head,
            dtype=dtype,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            qk_norm=qk_norm,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.scale = scale
        self.scale_depth = scale_depth
        self.num_layers = num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: ScaledRotaryEmbeddingESM,
        cu_seqlens: torch.Tensor,
        max_seqlen: int = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of self-attention block. It can be the embedding of a batch of sequences.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation.
            position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of attention block.

        """  # noqa: E501
        x = self.layernorm_before_attention(hidden_states)
        x = self.self_attention(
            x,
            x,
            position_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )

        if self.dropout is not None:
            x = self.dropout(x)

        if self.scale_depth > 0:
            hidden_states = hidden_states + x * (
                self.scale_depth / math.sqrt(self.num_layers)
            )  # https://arxiv.org/pdf/2310.02244.pdf
        else:
            hidden_states = hidden_states + x

        return hidden_states


class FFNBlock(torch.nn.Module):
    """The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str,
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        layer_id: int = 0,
        num_layers: int = 0,
    ):
        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.ffn = FeedForward(
            dim_model,
            dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.scale = scale
        self.scale_depth = scale_depth
        self.num_layers = num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """  # noqa: E501
        x = self.layernorm_before_ffn(hidden_states)
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)

        if self.scale_depth > 0:
            hidden_states = hidden_states + x.view_as(hidden_states) * (
                self.scale_depth / math.sqrt(self.num_layers)
            )  # https://arxiv.org/pdf/2310.02244.pdf
        else:
            hidden_states = hidden_states + x.view_as(hidden_states)

        return hidden_states


class TransformerBlock(torch.nn.Module):
    """The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        activate_fn: str = "silu",
        dtype=torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        qk_norm: bool = False,
        layer_id: int = 0,
        num_layers: int = 0,
    ):
        super().__init__()

        self.self_att = SelfAttentionBlock(
            dim_model=dim_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dim_head=dim_head,
            dtype=dtype,
            eps=eps,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            scale_depth=scale_depth,
            qk_norm=qk_norm,
            layer_id=layer_id,
            num_layers=num_layers,
        )

        self.ffn = FFNBlock(
            dim_model=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            eps=eps,
            dropout_p=dropout_p,
            tp=tp,
            scale=scale,
            init_std=init_std,
            scale_width=scale_width,
            scale_depth=scale_depth,
            layer_id=layer_id,
            num_layers=num_layers,
        )

    def forward(
        self,
        self_hidden_states: torch.Tensor,
        self_position_bias: Optional[torch.Tensor] = None,  # TODO
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            self_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            self_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation of self-attention.
            self_position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of transformer block.

        """  # noqa: E501
        # (batch, dim_model, seq_self)
        hidden_states = self.self_att(
            self_hidden_states,
            position_bias=self_position_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )

        # (batch, dim_model, seq_self)
        hidden_states = self.ffn(hidden_states)

        return hidden_states


class Encoder(bmt.DistributedModule):
    """Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): number of layers.
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.bfloat16.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        num_kv_heads: int = -1,
        activate_fn: str = "silu",
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        tp: int = 0,
        scale: bool = False,
        init_std: float = 0.02,
        scale_width: float = 1.0,
        scale_depth: float = -1,
        qk_norm: bool = False,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        if num_kv_heads == -1:
            num_kv_heads = num_heads
        self.num_layers = num_layers

        self.layers = bmt.TransformerBlockList(
            [
                bmt.CheckpointBlock(
                    TransformerBlock(
                        dim_model=dim_model,
                        dim_ff=dim_ff,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        dim_head=dim_head,
                        activate_fn=activate_fn,
                        dtype=dtype,
                        eps=eps,
                        dropout_p=dropout_p,
                        tp=tp,
                        scale=scale,
                        init_std=init_std,
                        scale_width=scale_width,
                        scale_depth=scale_depth,
                        qk_norm=qk_norm,
                        layer_id=layer_id,
                        num_layers=num_layers,
                    ),
                    use_checkpoint=use_checkpoint
                )
                for layer_id in range(num_layers)
            ]
        )
        self.output_layernorm = LayerNorm(dim_norm=dim_model, dtype=dtype, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden-states (:obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``): Input of encoder, might be the embedding of a batch of sequences.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc, seq_enc)``): Avoid invalid areas to participate in the calculation
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_enc, seq_enc)``) Provides position information to attention mechanism.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``: The encoder output.

        """  # noqa: E501
        hidden_states = self.layers(
            hidden_states,
            position_bias,
            cu_seqlens,
            max_seqlen,
            position_ids,
        )
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states


class Dragonfly(bmt.DistributedModule):
    def __init__(self, config: DragonflyConfig):
        super().__init__()

        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dim_head=config.dim_head,
            activate_fn=config.activate_fn,
            dtype=config.dtype,
            eps=config.eps,
            dropout_p=config.dropout_p,
            tp=config.tp,
            scale=config.scale,
            init_std=config.init_std,
            scale_width=config.scale_width,
            scale_depth=config.scale_depth,
            qk_norm=config.qk_norm,
            use_checkpoint=config.use_checkpoint,
        )

        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
            init_std=config.init_std,
            tp=config.tp,
            scale=config.scale,
            scale_emb=config.scale_emb,
            scale_width=config.scale_width,
        )

        self.position_bias = ScaledRotaryEmbeddingESM(
            dim=config.dim_head,
            max_position_embeddings=config.max_length,
            base=config.base,
            pose_prob=config.pose_prob,
            pose_scaling_factor=config.pose_scaling_factor,
            scaling_type=config.rope_scaling_type,
            rope_scaling_factor=config.rope_scaling_factor,
            original_max_position_embeddings=config.orig_max_length,
            dynamic_scaling_seq_len=config.max_length,  # disable dynamic scaling
            persistent=False,
            device="cuda",
        )

        if config.tie_lm_head is False:
            self.lm_head = Embedding(
                vocab_size=config.vocab_size,
                embedding_size=config.dim_model,
                dtype=config.dtype,
                init_std=config.init_std,
                scale=config.scale,
                scale_width=config.scale_width,
                tp=config.tp,
            )

        self.config = config

    def forward(
        self,
        input: torch.Tensor,  # (batch, seqlen) int32
        cu_seqlens: torch.Tensor = None,  # (real_batch+2) int32
        max_seqlen: int = None,
        position_ids: torch.Tensor = None,  # (batch, seqlen) int32
    ):
        hidden_states = self.input_embedding(input)

        assert cu_seqlens is not None, "cu_seqlens are needed in Flash Attention cuda impl"
        hidden_states = self.encoder(
            hidden_states,
            position_bias=self.position_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )

        if self.config.tie_lm_head is True:
            logits = self.input_embedding.projection(hidden_states)
        else:
            logits = self.lm_head.projection(hidden_states)

        return logits
