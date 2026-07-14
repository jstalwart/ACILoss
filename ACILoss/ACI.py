from einops import reduce
from torch import nn
import numpy as np
import random
import torch
import os

class AutoCorrDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            - x (Tensor): shape (N, S, t) where:
                          N = batch size
                          S = sequence/feature dimension
                          t = time dimension

        Returns:
            - delta (Tensor): shape (N, N) containing the pairwise distances
        """
        N, S, t = x.shape
        device = x.device
        
        # 1. Compute weight matrix w (t, t) and its row sums s (t,)
        t_idx = torch.arange(t, device=device).float()
        w = 1.0 / (1.0 + (t_idx.unsqueeze(1) - t_idx.unsqueeze(0))**2)
        s = reduce(w, "t1 t2 -> t1", "sum")  # shape: (t,)
        
        # 2. Compute V (N, S) and its sum over the S dimension U (N,)
        # V_ik = sum_t (x^2)_ikt * s_t
        V = torch.matmul(x**2, s)  # (N, S, t) x (t,) -> (N, S)
        U = V.sum(dim=1)           # Sum over S -> (N,)
        
        # 3. Compute M of shape (S, N, N) using Einstein Summation
        # i = batch_i, j = batch_j, k = S_dim, l = time_l, m = time_m
        M = torch.einsum('ikl, lm, jkm -> kij', x, w, x)
        
        # Sum M over the S dimension (dim 0) -> (N, N)
        M_sum = M.sum(dim=0)
        
        # 4. Compute delta of shape (N, N)
        # (U.unsqueeze(1) + U.unsqueeze(0)) handles the (V_i + V_j) expansion
        delta = U.unsqueeze(1) + U.unsqueeze(0) - 2 * M_sum

        return delta

class ACILoss(nn.Module):
    def __init__(self, smooth_weight=0.1):
        super().__init__()
        self.autocorr = AutoCorrDistance()
        self.smooth_weight = smooth_weight

    def forward(self, y_pred, y_real):
        # y_pred shape expected: (N, S, features)
        # y_real shape expected: (N, S, t)
        
        N, S, F = y_pred.shape
        
        # 1. Compute Distances
        # Flatten S and feature dimensions for cdist: (N, S * F)
        y_pred_flat = y_pred.reshape(N, -1) 
        
        # Pairwise p-3 distance matrix of shape (N, N)
        current_distances = torch.cdist(y_pred_flat, y_pred_flat, p=3)
        target_distances = self.autocorr(y_real) # Resulting shape: (N, N)
        
        # 2. Masking (exclude diagonal self-distances)   
        mask = ~torch.eye(N, device=y_pred.device).bool()
        
        # 3. Regularized MSE (Log-scale prevents gradient explosion)
        diff = (torch.log1p(current_distances[mask]) - torch.log1p(target_distances[mask])) ** 2
        base_loss = diff.mean()
        
        # 4. Smoothness Regularization (Temporal/Sequence)
        # Regularizes transitions across the sequence dimension (S)
        # Shape of y_pred: (N, S, F) -> difference over sequence axis
        if self.smooth_weight > 0:
            # L2 difference between consecutive steps in sequence S
            smoothness = torch.norm(y_pred[:, 1:, :] - y_pred[:, :-1, :], p=2, dim=-1).mean()
        else:
            smoothness = 0.0
        
        return base_loss + (self.smooth_weight * smoothness)
    
class SDMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = AutoCorrDistance()
        self.kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Args:
            - y_true (Tensor): shape (N, S, t)
            - y_pred (Tensor): shape (N, S, features)
        """
        N, S, F = y_pred.shape
        
        # 1. Compute target distances using AutoCorrDistance
        delta_matrix = self.delta(y_true)  # Shape: (N, N)
        
        # 2. Compute prediction distances
        # Flatten the spatial/feature dimensions to get a flat representation per batch item
        y_pred_flat = y_pred.reshape(N, -1)  # Shape: (N, S * F)
        dist_emb = torch.cdist(y_pred_flat, y_pred_flat, p=2)  # Shape: (N, N)

        # 3. Stability Scaling (Z-score normalization)
        delta_matrix = (delta_matrix - delta_matrix.mean()) / (delta_matrix.std() + 1e-6)
        dist_emb = (dist_emb - dist_emb.mean()) / (dist_emb.std() + 1e-6)
        
        # 4. Convert to distributions
        p = torch.nn.functional.softmax(-delta_matrix, dim=-1)
        log_q = torch.nn.functional.log_softmax(-dist_emb, dim=-1)
        
        # 5. Compute KLD
        return self.kl(log_q, p)
