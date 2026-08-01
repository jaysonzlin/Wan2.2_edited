import torch
import torch.nn as nn

from wan.modules.model import WanAttentionBlock


class _ZeroAttention(nn.Module):
    def forward(self, query, *args):
        return torch.zeros_like(query)


def test_wan_block_phase_helpers_preserve_legacy_forward_order():
    block = WanAttentionBlock(dim=4, ffn_dim=8, num_heads=1)
    block.self_attn = _ZeroAttention()
    block.cross_attn = _ZeroAttention()
    x = torch.randn(1, 3, 4)
    e = torch.randn(1, 3, 6, 4)
    kwargs = {
        "seq_lens": torch.tensor([3]),
        "grid_sizes": torch.tensor([[1, 1, 3]]),
        "freqs": torch.zeros(3, 2, dtype=torch.complex64),
        "context": torch.randn(1, 2, 4),
        "context_lens": None,
    }

    expected = block(x, e=e, **kwargs)
    terms = block.modulation_terms(e)
    actual = block.run_self_attention(
        x, terms, kwargs["seq_lens"], kwargs["grid_sizes"], kwargs["freqs"]
    )
    actual = block.run_text_cross_attention(actual, kwargs["context"], None)
    actual = block.run_mlp(actual, terms)

    torch.testing.assert_close(actual, expected)
