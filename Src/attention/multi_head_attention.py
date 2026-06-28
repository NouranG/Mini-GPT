# attention.py (Multi-Head Attention)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):


    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V (shared)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask (lower triangular)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        # 1. Linear projections
        q = self.q_proj(x)  
        k = self.k_proj(x)   
        v = self.v_proj(x)   

        # 2. Split into heads
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # 3. Transpose for attention computation
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)  

        # 4. Attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # shape: (B, nh, T, T)

        # 5. Causal mask
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # 6. Softmax
        weights = F.softmax(scores, dim=-1)

        # 7. Weighted sum of values
        out = weights @ v  

        # 8. Merge heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, C)

        # 9. Final projection
        out = self.out_proj(out)

        self.dropout = nn.Dropout(0.1)

        return out