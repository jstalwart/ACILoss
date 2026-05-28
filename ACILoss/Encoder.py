from .Attention import MultiHeadSelfAttention
from .FeedForward import FeedForwardBlock
from .ResidualAdd import ResidualAdd
import torch.nn as nn
import torch

class EncoderBlock(nn.Sequential):
    '''
    Encoder block class.
    '''
    def __init__(self,
                 emb_size: int,
                 num_heads: int = 8,
                 forward_expansion : int = 4,
                 **kwargs):
        '''
        Initialize the encoder block as a sequential neural network.

        Args:
        - emb_size (int): embedding size. 
        - num_heads (int): number of heads in the self-attention module. Default is 8.
        - non_linearity (str): type of non linearity applied to the encoder. Currently suported ff, KAN and SKAN.
        - forward_expansion (int): how many times does the ff expand and compress the embedding. Default is 4.
        - dropout (float): optional. Dropout for the modules. Default is 0.

        Returns: 
        - self.
        '''

        block_msa = nn.Sequential(
                nn.LayerNorm(normalized_shape = emb_size),
                MultiHeadSelfAttention(input_dim=emb_size, num_heads=num_heads, **kwargs)
        )
        block_ff = nn.Sequential(nn.LayerNorm(normalized_shape = emb_size), 
                                 FeedForwardBlock(emb_size, forward_expansion=forward_expansion, **kwargs))
            
        super().__init__(
            ResidualAdd(block_msa),
            ResidualAdd(block_ff)
            )
        
class Encoder(nn.Sequential):
    '''
    Encoder class. 
    '''
    def __init__(self,
                 emb_size : int, 
                 depth: int = 12,
                 **kwargs):
        '''
        Initialize the encoder class as a sequential neural network. 

        Args:
        - emb_size (int): embedding size. 
        - depth (int): number of encoder blocks applied sequentially. Default is 12. 
        - num_heads (int): optional. Number of heads in the self-attention module. Default is 8.
        - forward expansion (int): optional. How many times does the ff expand and compress the embedding. Default is 4.
        - dropout (float): optional. Dropout for the modules. Default is 0.

        Returns:
        - self.
        '''
        transformer_blocks = [EncoderBlock(emb_size=emb_size, **kwargs) for _ in range(depth)]
        super().__init__(*transformer_blocks)