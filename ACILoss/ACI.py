from einops import reduce
from torch import nn
import numpy as np
import random
import torch
import os

class AutoCorrDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        '''
        w_kl = 1/(1+(k-l)^2) -> W in R^{t,t}
        s = rowsum(w)

        Args:
            - x (Tensor): shape batch, emb_size. 

        Returns:
            - dist(x) = (X^2*s)1^T + 1(X^2*s)^T - 2XWX
        '''
        x = x.squeeze(dim=1)
        n, t = x.shape
        device = x.device
        
        # Numerator denominator
        t_idx = torch.arange(t, device=device).float()
        w = 1.0/(1.0+(t_idx.unsqueeze(1)-t_idx.unsqueeze(0))**2)
        s = reduce(w, "t1 t2 -> t1", "sum")
        
        # Numerator
        V = torch.matmul(x**2, s)
        M = x @ w @ x.t()
        numerator = V.unsqueeze(0) + V.unsqueeze(1) - 2 * M

        return numerator#/denominator

class ACILoss(nn.Module):
    def __init__(self, smooth_weight=0.1):
        super().__init__()
        self.autocorr = AutoCorrDistance()
        self.smooth_weight = smooth_weight

    def forward(self, y_pred, y_real):
        y_pred = y_pred.squeeze(dim=1)
        
        # 1. Compute Distances
        current_distances = torch.cdist(y_pred, y_pred, p=2)
        target_distances = self.autocorr(y_real)
        # 2. Masking    
        mask = ~torch.eye(y_pred.size(0), device=y_pred.device).bool()
        # 3. Regularized MSE (Log-scale prevents gradient explosion)
        diff = (torch.log1p(current_distances[mask]) - torch.log1p(target_distances[mask])) ** 2
        base_loss = diff.mean()
        
        # 4. Smoothness Regularization (Temporal/Sequence)
        # Prevents the latent mapping from becoming too erratic
        #smoothness = torch.norm(y_pred[1:] - y_pred[:-1], p=2, dim=1).mean()
        
        return base_loss #+ (self.smooth_weight * smoothness)
    
class SDMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = AutoCorrDistance()
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")

    def forward(self, y_true, y_pred):
        y_true = torch.Tensor(y_true)
        y_pred = torch.Tensor(y_pred)
        
        delta_matrix = self.delta(y_true)
        dist_emb = torch.cdist(y_pred, y_pred, p=2)

        # NEW: Stability Scaling (Z-score normalization of distances)
        # This ensures the Softmax "temperature" is comparable across different models
        delta_matrix = (delta_matrix - delta_matrix.mean()) / (delta_matrix.std() + 1e-6)
        dist_emb = (dist_emb - dist_emb.mean()) / (dist_emb.std() + 1e-6)
        
        # 3. Convert to distributions
        # We use a negative kernel so that small distances = high probability
        p = torch.nn.functional.softmax(-delta_matrix, dim=-1)
        log_q = torch.nn.functional.log_softmax(-dist_emb, dim=-1)
        
        # 4. Compute KLD
        return self.kl(log_q, p)
