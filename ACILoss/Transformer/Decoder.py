from .Attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from .FeedForward import FeedForwardBlock
from .ResidualAdd import ResidualAdd
import torch.nn as nn
import torch

class TransformerDecoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 768,
                 forward_expansion: int = 4,
                 dropout: float = 0.,
                 num_heads: int = 8
                 ):
        super().__init__()

        block_ff = nn.Sequential(
                #layer norm
                # feed forward block
                nn.LayerNorm(normalized_shape = emb_size),
                FeedForwardBlock(emb_size, forward_expansion, dropout)
            )
        self.lnorm = nn.LayerNorm(normalized_shape=emb_size)
        self.csa = MultiHeadCrossAttention(input_dim=emb_size, num_heads=num_heads, dropout=dropout)
        self.ff = ResidualAdd(block_ff)

        #check how residual connecion are done!! :-)
    def forward(self, x1, x2, mask):
        res = x2
        x2 = self.csa(x1, x2, mask)
        x2 += res
        x2 = self.ff(x2)
        return x2

class Decoder(nn.Module):
    def __init__(self, depth: int = 12,
                 emb_size: int = 768,
                 forward_expansion: int = 4,
                 dropout: float = 0.,
                 num_heads: int = 8):
        
        super().__init__()


        # generate list of transformer blocks
        self.transformer_blocks = torch.nn.ModuleList([TransformerDecoderBlock(emb_size=emb_size,
                                                      forward_expansion=forward_expansion,
                                                      dropout = dropout,
                                                      num_heads = num_heads) for _ in range(depth)])
    def forward(self, x1, x2, mask = None):
        for module in self.transformer_blocks:
            x2 = module(x1, x2, mask=mask)
        return x2