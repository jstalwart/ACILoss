from einops.layers.torch import Reduce
import torch.nn as nn
import torch

class RegressionHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, out_size: int = 1000):
        super().__init__(
            Reduce('b s e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_size, bias = True))
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, out_size: int = 1000):
        super().__init__(
            Reduce('b s e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_size),
            nn.Softmax(dim=1))
        
class NLLRegression(nn.Module):
    def __init__(self, emb_size: int = 768, out_size: int = 1000):
        super().__init__()
        self.mean = RegressionHead(emb_size = emb_size, out_size = out_size)
        self.var_layer = RegressionHead(emb_size=emb_size, out_size= out_size)
    def forward(self, x):
        mean_output = self.mean(x)
        var = torch.exp(self.var_layer(x)) + 1e-6
        return mean_output, var