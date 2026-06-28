# embeddings.py

import torch
import torch.nn as nn

class GPTEmbeddings(nn.Module):

    def __init__(self,vocab_size,embed_dim,block_size):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size,embed_dim)

        # Positional embeddings
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        """
        x shape:
        (batch_size, seq_len)
        """

        batch_size, seq_len = x.shape

        # Create positions:
        # [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(
            seq_len,
            device=x.device
        )

        token_emb = self.token_embedding(x)

        position_emb = self.position_embedding(
            positions
        )

        embeddings = token_emb + position_emb

        return embeddings
