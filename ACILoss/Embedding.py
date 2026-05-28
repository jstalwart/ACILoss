from node2vec import Node2Vec
from ts2vg import NaturalVG
from einops import reduce
import networkx as nx
from torch import nn
import numpy as np
import torch

class TSEmbedding(nn.Module):
    def __init__(self, 
                 in_features: int,
                 emb_size : int):
        super().__init__()
        self.projection = nn.Linear(in_features=in_features, out_features=emb_size)
        #remember that the class token has also an associated positional encoding
        self.positional_enc = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x).unsqueeze(0)
        #concatenate c and x tensors on the seq dimension (use torch.c
        #add posional encodings
        x += self.positional_enc
        return x