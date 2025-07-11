from typing import Optional

import bmtrain as bmt
import torch

from .linear import LastLinear
from .linear import Linear


class DenseGatedACT(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
        activate_fn: str = "gelu",
        scale: bool = True,
        dtype=torch.half,
        tp: int = 0,
    ):
        super().__init__()

        self.w_0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            scale=scale,
            scale_before=False,
            tp=tp,
        )

        self.w_1 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            scale=scale,
            scale_before=False,
            tp=tp,
        )

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
        dtype (optional): Defaults to torch.half.
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
        activate_fn: str = "gelu",
        dtype=torch.half,
        dropout_p: Optional[float] = None,
        scale: bool = True,
        tp: int = 0,
    ):
        super().__init__()

        self.w_in = DenseGatedACT(
            dim_in=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            dtype=dtype,
            scale=scale,
            tp=tp,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.w_out = LastLinear(
            dim_in=dim_ff,
            dim_out=dim_model,
            dtype=dtype,
            scale=scale,
            scale_before=False,
            tp=tp * 2,
        )

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
