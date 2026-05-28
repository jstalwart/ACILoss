from einops.layers.torch import Reduce
import torch.nn as nn

class RegressionHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, out_size: int = 1000):
        super().__init__(
            Reduce('b s e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_size))
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, out_size: int = 1000):
        super().__init__(
            Reduce('b s e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_size),
            nn.Softmax(dim=1))