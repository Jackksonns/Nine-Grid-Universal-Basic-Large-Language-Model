from typing import Optional
from typing import Tuple
from typing import Union

import torch

from .attention import Attention
from .feedforward import FeedForward
from .layernorm import LayerNorm
from .position_embedding import RotaryEmbedding
from .position_embedding import RotaryEmbeddingESM


class SelfAttentionBlock(torch.nn.Module):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        num_kv_heads: int,
        dtype=torch.half,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        use_flash_attn: bool = False,
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
            scale=scale,
            use_flash_attn=use_flash_attn,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Union[torch.Tensor, RotaryEmbedding, RotaryEmbeddingESM] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_bias_type: Optional[str] = "relative",
        length_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
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
            attention_mask,
            position_bias,
            use_cache,
            past_key_value,
            pos_bias_type=pos_bias_type,
            length_mask=length_mask,
            context_mask=context_mask,
        )
        if use_cache:
            x, current_key_value = x
        else:
            current_key_value = None

        if self.dropout is not None:
            x = self.dropout(x)

        hidden_states = hidden_states + x

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states


class FFNBlock(torch.nn.Module):
    """The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.half.
        eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str,
        dtype=torch.half,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
        scale: bool = True,
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
            scale=scale,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

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

        hidden_states = hidden_states + x

        return hidden_states


class TransformerBlock(torch.nn.Module):
    """The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and feed-forward block.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
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
        activate_fn: str = "gelu",
        dtype=torch.half,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        mask_att: bool = False,
        mask_ffn: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.mask_att = mask_att
        self.mask_ffn = mask_ffn

        if not self.mask_att:
            self.self_att = SelfAttentionBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim_head=dim_head,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
                scale=scale,
                use_flash_attn=use_flash_attn,
            )

        if not self.mask_ffn:
            self.ffn = FFNBlock(
                dim_model=dim_model,
                dim_ff=dim_ff,
                activate_fn=activate_fn,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
                scale=scale,
            )

    def forward(
        self,
        self_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        self_position_bias: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_bias_type: Optional[str] = "relative",
        length_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
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
        current_key_value = None
        if not self.mask_att:
            hidden_states = self.self_att(
                self_hidden_states,
                attention_mask=self_attention_mask,
                position_bias=self_position_bias,
                use_cache=use_cache,
                past_key_value=past_key_value,
                pos_bias_type=pos_bias_type,
                length_mask=length_mask,
                context_mask=context_mask,
            )
            if use_cache:
                hidden_states, current_key_value = hidden_states
        else:
            hidden_states = self_hidden_states

        # (batch, dim_model, seq_self)
        if not self.mask_ffn:
            hidden_states = self.ffn(hidden_states)

        if use_cache:
            return hidden_states, current_key_value
        else:
            return hidden_states
