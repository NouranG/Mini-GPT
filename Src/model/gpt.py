import torch
import torch.nn as nn
import torch.nn.functional as F

from Src.Embedding.embedding import GPTEmbeddings
from model.transformer_block import TransformerBlock

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
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # crop context if too long
            idx_cond = idx[:, -self.embeddings.position_embedding.num_embeddings:]

            # forward pass
            logits, _ = self(idx_cond)

            # focus on last token
            logits = logits[:, -1, :]

            # probabilities
            probs = F.softmax(logits, dim=-1)

            # sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # append
            idx = torch.cat((idx, next_token), dim=1)

        return idx
