"""JVP precision test — compare internal JVP vs finite-difference reference.
Usage: python -m pytest tests/test_jvp_precision.py -s
Based on rCM's rcm/networks/wan2pt1_jvp_test.py
"""
import torch
from ltx_distillation.models.ltx_internal_jvp import (
    _single_input_jvp, _linear_tangent, _rms_norm_with_t,
    _attention_with_t, _feed_forward_with_t, _layer_norm_with_t,
    _unwrap_fsdp,
)

HEAD_DIM = 64
N_HEADS = 8
BATCH = 1
SEQ_LEN_Q = 64
SEQ_LEN_KV = 64
DIM = N_HEADS * HEAD_DIM


def _finite_diff_jvp(fn, x, t_x, eps=5e-2):
    """Reference: central finite difference JVP (bf16-safe, large eps)."""
    with torch.no_grad():
        x_p = (x + eps * t_x).to(x.dtype)
        x_m = (x - eps * t_x).to(x.dtype)
        f_plus = fn(x_p)
        f_minus = fn(x_m)
        return (f_plus.float() - f_minus.float()) / (2 * eps)


def test_linear_tangent():
    """Check _linear_tangent matches finite-diff JVP for Linear layer."""
    linear = torch.nn.Linear(DIM, DIM).cuda().bfloat16()
    x = torch.randn(BATCH, SEQ_LEN_Q, DIM, device='cuda', dtype=torch.bfloat16)
    t_x = torch.randn_like(x)

    def fn(x_in):
        return linear(x_in.to(torch.bfloat16)).float()

    t_manual = _linear_tangent(linear, t_x).float()
    t_fd = _finite_diff_jvp(fn, x.float(), t_x.float())
    cos = torch.nn.functional.cosine_similarity(t_manual.flatten(), t_fd.flatten(), dim=0)

    rel_err = (t_manual - t_fd).abs().mean() / (t_fd.abs().mean() + 1e-8)
    print(f"  Linear tangent: cos={cos:.6f}, rel_err={rel_err*100:.4f}%")
    assert cos > 0.99, f"Linear tangent mismatch (cos={cos:.6f})"


def test_rms_norm_jvp():
    """Check rms norm JVP precision."""
    eps_norm = 1e-6
    x = torch.randn(BATCH, SEQ_LEN_Q, DIM, device='cuda', dtype=torch.bfloat16)
    t_x = torch.randn_like(x)

    def fn(x_in):
        from ltx_core.utils import rms_norm
        return rms_norm(x_in.to(torch.bfloat16), eps=eps_norm).float()

    _, t_manual = _rms_norm_with_t(x, t_x, eps_norm)
    t_fd = _finite_diff_jvp(fn, x.float(), t_x.float())
    cos = torch.nn.functional.cosine_similarity(t_manual.float().flatten(), t_fd.flatten(), dim=0)
    print(f"  RMS norm JVP: cos={cos:.6f}")
    assert cos > 0.99, f"RMS norm JVP mismatch (cos={cos:.6f})"


def test_single_input_jvp():
    """Check _single_input_jvp matches finite diff for a simple non-parametrized fn."""
    x = torch.randn(BATCH, SEQ_LEN_Q, DIM, device='cuda', dtype=torch.bfloat16)
    t_x = torch.randn_like(x)

    def fn(x_in):
        return torch.nn.functional.gelu(x_in.float(), approximate="tanh")

    _, t_manual = _single_input_jvp(fn, x, t_x)
    t_fd = _finite_diff_jvp(fn, x.float(), t_x.float())
    cos = torch.nn.functional.cosine_similarity(t_manual.float().flatten(), t_fd.flatten(), dim=0)
    print(f"  _single_input_jvp (GELU): cos={cos:.6f}")
    assert cos > 0.99, f"_single_input_jvp mismatch (cos={cos:.6f})"


def test_attention_jvp():
    """Check full attention JVP decomposition."""
    from ltx_core.model.transformer.attention import Attention, AttentionFunction

    attn = Attention(query_dim=DIM, context_dim=DIM, heads=N_HEADS, dim_head=HEAD_DIM).cuda().bfloat16()
    attn.attention_function = AttentionFunction.DEFAULT
    attn.eval()

    x = torch.randn(BATCH, SEQ_LEN_Q, DIM, device='cuda', dtype=torch.bfloat16)
    t_x = torch.randn_like(x)

    with torch.no_grad():
        _, t_y_manual = _attention_with_t(attn, x, t_x, context=None, t_context=None,
                                           mask=None, pe=None, k_pe=None)

    def attn_fn(x_in):
        return attn(x_in.to(torch.bfloat16)).float()

    t_y_fd = _finite_diff_jvp(attn_fn, x.float(), t_x.float())

    cos = torch.nn.functional.cosine_similarity(t_y_manual.float().flatten(), t_y_fd.flatten(), dim=0)
    rel_err = (t_y_manual.float() - t_y_fd).abs().mean() / (t_y_fd.abs().mean() + 1e-8)
    print(f"  Attention JVP: cos={cos:.6f}, rel_err={rel_err*100:.4f}%")
    assert cos > 0.99, f"Attention JVP mismatch (cos={cos:.6f}, rel_err={rel_err*100:.2f}%)"


if __name__ == "__main__":
    print("=== JVP Precision Tests ===")
    test_linear_tangent()
    test_rms_norm_jvp()
    test_single_input_jvp()
    test_attention_jvp()
    print("\n[ALL TESTS PASSED]")
