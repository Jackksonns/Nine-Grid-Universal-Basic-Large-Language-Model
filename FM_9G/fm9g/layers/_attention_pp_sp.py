import torch
import torch.nn.functional as F
import bmtrain as bmt

def _linear_backward(grad_output, x, weight, bias):
    grad_x = grad_weight = grad_bias = None
    if x.requires_grad:
        grad_x = grad_output.matmul(weight)
    if weight.requires_grad:
        grad_weight = grad_output.reshape(-1,
            grad_output.shape[-1]).t().matmul(x.reshape(-1, x.shape[-1]))
    if bias is not None and bias.requires_grad:
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
    return  grad_x, grad_weight, grad_bias

class OpAttnPipeSP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_w, k_w, v_w, q_b, w_b, v_b, x, cache_kv, cache_kv_inp, cu_seqlens_q, cu_seqlens_k, max_seqlen):
        ctx.save_for_backward(x, q_w, k_w, v_w, q_b, w_b, v_b)
        if cache_kv.numel() = 0:
            q = F.linear(x, q_w, q_b)
            k = F.linear(x, k_w, w_b)
            v = F.linear(x, v_w, v_b)
        else:
            q = F.linear(x, q_w, q_b)
            k = F.linear(x, k_w, w_b)
            v = F.linear(x, v_w, v_b)
            k = torch.cat([cache_kv[0], k], dim=1)
            v = torch.cat([cache_kv[1], v], dim=1)
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen, max_seqlen, 0, causal=True, window_size=(-1,-1), alibi_slopes=None, deterministic=False, return_attn_probs=False
            )
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
        ctx.max_seqlen_q = max_seqlen
        ctx.max_seqlen_k = max_seqlen
        

         
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
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
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            False,
            (-1,-1),
            None,
            False,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]

        d_xq, d_wq, d_bq = _linear_backward(dq, x, q_w, q_b)
        d_xq, d_wq, d_bq = _linear_backward(dq, x, q_w, q_b)
        d_xk, d_wk, d_bk = _linear_backward(dk, x, k_w, k_b)
        


        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None
