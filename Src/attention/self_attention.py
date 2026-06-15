import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,embed_dim,head_size,block_size):
        super().__init__()

        #query 
        self.query=nn.linear(
            embed_dim,
            head_size,
            bias=False
        )
        #key
        self.key=nn.linear(
            embed_dim,
            head_size,
            bias=False
        )
        #value
        self.value=nn.linear(
            embed_dim,
            head_size,
            bias=False
        )
        #causal mask
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size,block_size))
        )

    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        #compute attention scores
        attention_scores=(q@k.transpose(-2,-1))/(k.shape[-1]**0.5)
        #mask application
        attention_scores=attention_scores.masked_fill(self.tril[:T,:T]==0,float("-inf"))

        attention_scores=F.softmax(attention_scores,dim=-1)

        out=attention_scores@v
        return out