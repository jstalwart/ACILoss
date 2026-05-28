import torch.nn as nn
import torch

class ResidualAdd(nn.Module):
    '''
    Residual add class. 
    '''
    def __init__(self, 
                 fn:nn.Module,
                 **kwargs):
        '''
        Initialize the residual add. 

        Parameters:
        - fn: torch nn module that is wrapped.
        '''
        super().__init__()
        self.fn = fn

    def forward(self, 
                x:torch.Tensor, 
                **kwargs):
        '''
        Forward pass for the residual add. 

        Args:
        - x: torch.Tensor

        Returns:
        - y: torch.Tensor, same size as x.
        '''
        y = self.fn(x, **kwargs)
        y += x
        return y