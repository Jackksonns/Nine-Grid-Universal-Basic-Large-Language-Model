import math
from typing import Optional
from typing import Tuple

try:
    from .flash_triton import FlashAttnFunc
except:
    FlashAttnFunc = None
import bmtrain as bmt
import torch
from einops import rearrange

from .linear import ColumnParallelLinear
from .linear import Linear
from .position_embedding import apply_chatglm_rotary_pos_emb

try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except:
    flash_attn_varlen_func = None

try:
    from flash_attn.bert_padding import pad_input
    from flash_attn.bert_padding import unpad_input
except:
    pad_input = None
    unpad_input = None


class OpFlash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, record, q, k, v, cu_seqlens, max_seqlen, dropout_p, causal):
        ctx.self = self
        ctx.cu_seqlens = cu_seqlens
        ctx.max_length = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.softmax_scale = q.shape[-1] ** (-0.5)
        if not record and "out" in self._layer_dict:
            out = self._layer_dict.pop("out")
            softmax_lse = self._layer_dict.pop("softmax_lse")
            rng_state = self._layer_dict.pop("rng_state")
        else:
            out, _, _, _, _, softmax_lse, _, rng_state = _flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p,
                ctx.softmax_scale,
                causal=causal,
                return_softmax=False,
            )
            if record:
                self._layer_dict["out"] = out
                self._layer_dict["softmax_lse"] = softmax_lse
                self._layer_dict["rng_state"] = rng_state

        ctx.save_for_backward(q, k, v, out, softmax_lse, rng_state)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.cu_seqlens,
            ctx.cu_seqlens,
            ctx.max_length,
            ctx.max_length,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
        )
        return None, None, dq, dk, dv, None, None, None, None


class Attention(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_kv_heads: int,
        dim_head: int,
        dtype: torch.dtype = torch.half,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        add_qkv_bias: bool = False,
        use_flash_attn: bool = False,
        tp: int = 0,
    ) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_groups = num_heads // num_kv_heads
        self.dim_head = dim_head

        self.project_q = Linear(
            self.dim_model,
            self.num_heads * self.dim_head,
            bias=add_qkv_bias,
            dtype=dtype,
            scale=scale,
            tp=tp,
        )
        self.project_k = Linear(
            self.dim_model,
            self.num_kv_heads * self.dim_head,
            bias=add_qkv_bias,
            dtype=dtype,
            scale=scale,
            tp=tp,
        )
        self.project_v = Linear(
            self.dim_model,
            self.num_kv_heads * self.dim_head,
            bias=add_qkv_bias,
            dtype=dtype,
            scale=scale,
            tp=tp,
        )

        self.attention_out = Linear(
            self.num_heads * self.dim_head,
            self.dim_model,
            dtype=dtype,
            scale=scale,
            tp=tp * 2,
        )

        self.softmax = torch.nn.Softmax(dim=-1)

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
            self.dropout_p = dropout_p
        else:
            self.dropout = None

        self.use_flash_attn = use_flash_attn
        self._layer_dict = {}

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_bias_type: Optional[str] = "relative",
        length_mask: Optional[torch.Tensor] = None,
        attention_mask_bias: Optional[torch.Tensor] = None,
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

        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)

        if isinstance(self.project_q, ColumnParallelLinear):
            assert hidden_q.data_ptr() == hidden_kv.data_ptr()
            if self.project_q.scale and self.project_q.scale_before:
                hidden_q = hidden_q / math.sqrt(self.project_q.dim_in)
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
            )
            if self.project_q.scale and not self.project_q.scale_before:
                hidden_q = hidden_q / math.sqrt(self.project_q.dim_in)

            block_size = hidden_q.shape[-1] // (self.head_groups + 1 + 1)
            h_q = hidden_q[..., : block_size * self.head_groups]
            h_k = hidden_q[..., block_size * self.head_groups : block_size * (self.head_groups + 1)]
            h_v = hidden_q[..., block_size * (self.head_groups + 1) :]
        else:
            h_q = self.project_q(hidden_q)
            h_k = self.project_k(hidden_kv)
            h_v = self.project_v(hidden_kv)

        batch_size = h_q.size(0)

        if not self.use_flash_attn:
            h_q = h_q / math.sqrt(math.sqrt(self.dim_head))
            h_k = h_k / math.sqrt(math.sqrt(self.dim_head))

            h_q = h_q.view(batch_size, len_q, -1, self.dim_head).permute(0, 2, 1, 3)
            h_k = h_k.view(batch_size, len_k, -1, self.dim_head).permute(0, 2, 1, 3)
            h_v = h_v.view(batch_size, len_k, -1, self.dim_head).permute(0, 2, 1, 3)

            if pos_bias_type == "rotary":
                # b h s d
                h_q, h_k = position_bias(h_q, h_k, -2, offset=past_kv[0].size(-2) if past_kv is not None else 0)
            elif pos_bias_type == "chatglm_rotary":
                h_q = apply_chatglm_rotary_pos_emb(h_q, position_bias)
                h_k = apply_chatglm_rotary_pos_emb(h_k, position_bias)

            if past_kv is not None:
                h_k = torch.cat([past_kv[0], h_k], dim=-2)
                h_v = torch.cat([past_kv[1], h_v], dim=-2)
                len_k = h_k.size(-2)

            # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
            # (b, n_kv_h, n_h_groups*len_q, d_h) @ (b, n_kv_h, d_h, len_k) -> (b, n_kv_h, n_h_groups*len_q, len_k) -> (b, n_h, len_q, len_k)
            if self.head_groups == 1:
                score = torch.matmul(h_q, h_k.transpose(-1, -2))  # / math.sqrt(self.dim_head) moved to line 75~76
            else:
                score = torch.matmul(
                    h_q.reshape(batch_size, -1, self.head_groups * len_q, self.dim_head),
                    h_k.transpose(-1, -2),
                ).view(
                    batch_size, -1, len_q, len_k
                )  # / math.sqrt(self.dim_head) moved to line 75~76
            if pos_bias_type == "relative":
                if len_q == 1:  # inference with cache
                    if len(position_bias.size()) == 4:
                        position_bias = position_bias[:, :, -1:, :]
                    else:
                        position_bias = position_bias[:, -1:, :]
                score = score + position_bias
            score = torch.masked_fill(
                score,
                attention_mask.view(batch_size, 1, len_q, len_k) == False,
                torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
            )

            score = self.softmax(score)

            score = torch.masked_fill(
                score,
                attention_mask.view(batch_size, 1, len_q, len_k) == False,
                torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
            )

            if self.dropout is not None:
                score = self.dropout(score)

            # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
            # (b, n_kv_h, n_h_groups*len_q, len_k) @ (b, n_kv_h, len_k, d_h) -> (b, n_kv_h, n_h_groups*len_q, d_h) -> (b, n_h, len_q, d_h)
            score = torch.matmul(score.view(batch_size, -1, self.head_groups * len_q, len_k), h_v).view(
                batch_size, -1, len_q, self.dim_head
            )

            score = score.view(batch_size, -1, len_q, self.dim_head).permute(0, 2, 1, 3)
            score = score.contiguous().view(batch_size, len_q, -1)

        else:
            if attention_mask_bias is not None:
                assert pos_bias_type == "rotary"
                h_q = h_q.view(batch_size, len_q, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_k = h_k.view(batch_size, len_k, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_v = h_v.view(batch_size, len_k, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_q, h_k = position_bias(h_q, h_k, -3)
                score = FlashAttnFunc.apply(h_q, h_k, h_v, attention_mask_bias, False, None)
            else:
                if pos_bias_type == "chatglm_rotary":
                    raise NotImplemented("No FlashAttn version for ChatGLM at present!")
                h_q = h_q.view(batch_size * len_q, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_k = h_k.view(batch_size * len_k, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_v = h_v.view(batch_size * len_k, -1, self.dim_head)  # .permute(0, 2, 1, 3)
                h_q, h_k = position_bias(
                    h_q, h_k, -3, cu_seqlens=cu_seqlens, max_length=max_seqlen, position_ids=position_ids
                )
                # score = flash_attn_varlen_func(
                #     h_q, h_k, h_v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, self.dropout_p, causal=True
                # )
                score = OpFlash.apply(
                    self, not torch.is_grad_enabled(), h_q, h_k, h_v, cu_seqlens, max_seqlen, self.dropout_p, True
                )

            score = score.view(batch_size, len_q, -1)

        score = self.attention_out(score)

        if use_cache:
            return score, (h_k, h_v)
        else:
            return score
