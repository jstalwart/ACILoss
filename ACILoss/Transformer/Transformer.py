from .Embedding import DataEmbedding
from .Head import NLLRegression
from .Encoder import Encoder
from .Decoder import Decoder
import torch.nn as nn


class EncoderPart(nn.Module):
    '''
    Encoder class. 
    '''
    def __init__(self,
                 input_channels : int, 
                 d_model : int = 168, 
                 depth: int = 8,
                 n_heads = 5,
                 forward_expansion: int = 4,
                 embed_type : str = "fixed",
                 freq : str ="h", 
                 dropout: float = 0.0,
                 **kwargs):
        '''
        Initialize the encoder class as a neural network. 

        Args:
        - input_channels (int): the number of time series to account. 
        - d_model (int): embedding size. 
        - depth (int): number of encoder blocks applied sequentially. Default is 12. 
        - n_heads (int): optional. Number of heads in the self-attention module. Default is 8.
        - forward expansion (int): optional. How many times does the ff expand and compress the embedding. Default is 4.
        - embed_type (str): optional. The type of embedding used. 
        - freq (str): optional. the frequency for timesteps. Default is 'h'.
        - dropout (float): optional. Dropout for the modules. Default is 0.

        Returns:
        - self.
        '''
        super().__init__()
        self.embedding = DataEmbedding(c_in = input_channels, 
                                       d_model = d_model, 
                                       embed_type = embed_type, 
                                       freq = freq, 
                                       dropout = dropout)
        self.encoder_blocks = Encoder(emb_size = d_model, 
                                      depth = depth, 
                                      num_heads = n_heads, 
                                      dropout = dropout, 
                                      forward_expansion = forward_expansion)
        
    def forward(self, x, x_mark, self_mask = None):
        embed = self.embedding(x, x_mark)
        enc_out = self.encoder_blocks(embed)
        return enc_out
    
class DecoderPart(nn.Module):
    '''
    Decoder Part
    '''
    def __init__(self,
                 input_channels:int, 
                 output_size: int, 
                 d_model : int = 168, 
                 depth: int = 8,
                 n_heads_self: int = 5,
                 n_heads_cross: int = 5,
                 forward_expansion: int = 4,
                 embed_type : str = "fixed",
                 freq : str ="h", 
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__()
        self.encoder = EncoderPart(input_channels=input_channels, 
                                   d_model = d_model, 
                                   depth = depth, 
                                   n_heads = n_heads_self, 
                                   forward_expansion=forward_expansion, 
                                   embed_type=embed_type, 
                                   freq=freq, 
                                   dropout=dropout)
        self.decoder = Decoder(depth = depth, 
                               emb_size = d_model, 
                               forward_expansion=forward_expansion,
                               dropout = dropout, 
                               num_heads = n_heads_cross)
        self.head = NLLRegression(emb_size = d_model, 
                                  out_size = output_size)
        
    def forward(self, x, y, y_mark, self_mask = None, cross_mask = None):
        embedding = self.encoder(x=y, 
                                 x_mark=y_mark, 
                                 self_mask=self_mask)
        
        dec_out = self.decoder(x1 = x, x2 = embedding, mask = cross_mask)

        return self.head(dec_out)
    
class Transformer(nn.Module):
    def __init__(self,
                 enc_input_channels : int, 
                 dec_input_channels : int,
                 output_size : int, 
                 d_model : int, 
                 enc_depth: int = 5, 
                 dec_depth: int = 5, 
                 n_heads_self : int = 8, 
                 n_heads_cross: int = 8, 
                 dropout: float = 0.0,
                 forward_expansion: int = 4, 
                 embed_type : str = "fixed", 
                 freq: str = "h"):
        super().__init__()
        self.encoder = EncoderPart(input_channels=enc_input_channels, 
                                   d_model = d_model, 
                                   depth = enc_depth, 
                                   n_heads = n_heads_self, 
                                   forward_expansion=forward_expansion, 
                                   embed_type=embed_type, 
                                   freq=freq, 
                                   dropout=dropout)
        self.decoder = DecoderPart(input_channels=dec_input_channels, 
                                   output_size=output_size, 
                                   d_model = d_model, 
                                   depth = dec_depth, 
                                   n_heads_self = n_heads_self, 
                                   n_heads_cross = n_heads_cross, 
                                   forward_expansion=forward_expansion, 
                                   embed_type=embed_type, 
                                   freq = freq, 
                                   dropout = dropout)
        
    def forward(self, 
                x, 
                x_mark, 
                y, 
                y_mark, 
                enc_mask = None, 
                dec_mask = None, 
                cross_mask = None):
        enc_out = self.encoder(x = x, 
                               x_mark = x_mark)
        out, var = self.decoder(x = enc_out, 
                                y = y, 
                                y_mark = y_mark, 
                                self_mask = dec_mask, 
                                cross_mask = cross_mask)
        return out, var