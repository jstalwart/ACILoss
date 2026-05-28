import torch.nn as nn
import torch

class FeedForwardBlock(nn.Sequential):
    def __init__(self, 
                 emb_size: int, 
                 expansion: int = 4, 
                 dropout: float = 0.,
                 **kwargs):
        '''
        Initialize the feed forward as a sequential neural network. 

        Args:
        - emb_size (int): embedding size.
        - expansion (int): how many times does the ff expand and compress the embedding. Default is 4.
        - dropout (float): dropout for the module. Default is 0.

        Returns:
        - self
        '''
        super().__init__(
            #put layers here!
            nn.Linear(in_features=emb_size, out_features=expansion*emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=emb_size * expansion, out_features=emb_size)
        )