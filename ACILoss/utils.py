from torch import nn
from pywt import cwt 
import numpy as np
import random
import torch
import os


def select_seed(seed):
    """
    Sets the seed for reproducibility across python, numpy, and pytorch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def FFTtransform(x:torch.Tensor, emb_size:int):
    return np.abs(np.fft.fft(x, n=emb_size))

def CWTtransform(x: torch.Tensor, emb_size: int):
    scales = np.geomspace(1, 128, num=emb_size)
    coefficients, frequencies = cwt(np.array(x.squeeze()), scales, 'morl')
    mag = np.abs(coefficients)
    vector=np.mean(mag, axis=1)
    return np.expand_dims(vector, 0)

    