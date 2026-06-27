import torch
import torch.nn as nn
import torch.nn.functional as F

from Src.Embedding.embedding import GPTEmbeddings
from Src.model.transformer_block import TransformerBlock

class GPT(nn.Module):

    def __init__(self,vocab_size,embed_dim,num_heads,block_size,num_layers):
        super().__init__()

        # 1. Token + position embeddings
        self.embeddings = GPTEmbeddings(
            vocab_size,
            embed_dim,
            block_size
        )

        # 2. Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                block_size
            )
            for _ in range(num_layers)
        ])

        # 3. Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # 4. Language modeling head
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):

        # idx: (B, T)
        B, T = idx.shape

        # 1. Embeddings
        x = self.embeddings(idx)  # (B, T, C)

        # 2. Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final normalization
        x = self.ln_f(x)

        # 4. Logits over vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 5. Compute loss if targets exist
        loss = None
        if targets is not None:

            B, T, V = logits.shape

            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

            self.eval()

            for _ in range(max_new_tokens):

                idx_cond = idx[:, -self.block_size:]

                logits, _ = self(idx_cond)

                logits = logits[:, -1, :]

                probs = F.softmax(logits, dim=-1)

                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

                idx = torch.cat(
                    (idx, next_token),
                    dim=1
                )

            return idx