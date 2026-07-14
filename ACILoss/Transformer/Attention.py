
import torch.nn.functional as F
import torch.nn as nn
import torch

class MultiHeadSelfAttention(nn.Module):
    '''
    MultiHeadSelfAttention class. 

    Attributes:
    - input_dim (int): input dimension for the module.
    - num_heads (int): number of heads in the self attention module.
    - head_dim (int): number of features in each head.
    '''
    def __init__(self, 
                 input_dim:int, 
                 num_heads:int, 
                 dropout:float=0.0,
                 **kwargs):
        '''
        Initialize the multihead self-attention.

        Args:
        - input_dim (int): input dimension for the module.
        - num_heads (int): number of heads in the self-attention module.
        - dropout (float): dropout for the module. Default is 0.

        Returns:
        - self.
        '''
        super().__init__()
        assert input_dim % num_heads == 0, f"Input dimension {input_dim} must be divisible by the number of heads {num_heads}"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.qkv = nn.Linear(input_dim, 3*input_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(input_dim, input_dim)

    def forward(self, 
                x:torch.Tensor, 
                mask:torch.Tensor=None,
                **kwargs):
        '''
        Forward pass for the multihead self-attention mechanism. 

        Args:
        - x: 2D torch.Tensor (batch, seq_len, input_dim).
        - mask: 2D binary torch.Tensor (batch, input_dim). Default is None. 

        Returns:
        - out: 2D torch.Tensor (batch, input_dim).
        '''
        batch_size, seq_len, input_dim = x.size()

        # x_qkv contains queries, keys values for all heads
        x_qkv = self.qkv(x) # only one linear layer
        queries, keys, values = torch.chunk(x_qkv, 3, dim=-1)

        #Reshape to split heads. Output dim is [b_num_heads, seq_len, head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        scaling = self.input_dim ** (1/2)
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / scaling
        if mask is not None:
            # Apply mask if needed
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        att = F.softmax(energy, dim=-1)
        att = self.dropout(att)

        # Apply attention to values
        out = torch.matmul(att, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, input_dim)
        return out
    

class MultiHeadCrossAttention(nn.Module):
    '''
    MultiHeadCrossAttention class. 

    Attributes:
    - input_dim (int): input dimension for the module.
    - num_heads (int): number of heads in the cross-attention module.
    - head_dim (int): number of features in each head.
    '''
    def __init__(self, 
                 input_dim : int, 
                 num_heads : int, 
                 dropout : float = 0.0,
                 **kwargs):
        '''
        Initialize the multihead cross-attention.

        Args:
        - input_dim (int): input dimension for the module.
        - num_heads (int): number of heads in the cross-attention module.
        - dropout (float): dropout for the module. Default is 0.

        Returns:
        - self.
        '''
        super().__init__()
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.q = nn.Linear(input_dim, input_dim)
        self.kv = nn.Linear(input_dim, 2*input_dim)

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(input_dim, input_dim)

    def forward(self, 
                x_enc : torch.Tensor, 
                x_dec : torch.Tensor,
                mask = None,
                **kwargs):
        '''
        Forward pass for the multihead cross-attention mechanism. 

        Args:
        - x_enc: 2D torch.Tensor (batch, input_dim).
        - x_dec: 2D torch.Tensor (batch, input_dim)
        - mask: 2D binary torch.Tensor (batch, input_dim). Default is None.

        Returns:
        - out: 2D torch.Tensor (batch, input_dim).
        '''
        batch_size, seq_len, input_dim = x_enc.size()
        _, seq_len_dec, _ = x_dec.size()

        queries = self.q(x_dec)
        # x_kv contains keys values for all heads
        x_kv = self.kv(x_enc) # only two linear layer
        keys, values = torch.chunk(x_kv, 2, dim=-1)

        #Reshape to split heads. Output dim is [b_num_heads, seq_len, head_dim]
        queries = queries.view(batch_size, seq_len_dec, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        scaling = self.input_dim ** (1/2)
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / scaling
        if mask is not None:
            # Apply mask if needed
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        att = F.softmax(energy, dim=-1)
        att = self.dropout(att)

        # Apply attention to values
        out = torch.matmul(att, values)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len_dec, input_dim)
        return out