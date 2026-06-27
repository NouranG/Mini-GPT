import torch
import torch.nn as nn

from Src.attention.multi_head_attention import MultiHeadAttention

class FeedForward(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(
            embed_dim,
            num_heads,
            block_size
        )

        self.ff = FeedForward(embed_dim)

    def forward(self, x):

        # 1. Attention path
        x = x + self.attn(self.ln1(x))

        # 2. FeedForward path
        x = x + self.ff(self.ln2(x))

        return x
