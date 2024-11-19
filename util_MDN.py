import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from mpl_toolkits.mplot3d import Axes3D

def lstm_layer(lstm_size: int, dropout: float = 0.0) -> nn.Module:
    """Create an LSTM layer with dropout
    
    Args:
        lstm_size: Number of features in hidden state
        dropout: Dropout probability
    Returns:
        LSTM layer with dropout
    """
    return nn.LSTM(
        input_size=lstm_size,
        hidden_size=lstm_size,
        dropout=dropout
    )

def torch_2d_normal(x1: torch.Tensor, 
                   x2: torch.Tensor, 
                   mu1: torch.Tensor,
                   mu2: torch.Tensor,
                   s1: torch.Tensor,
                   s2: torch.Tensor,
                   rho: torch.Tensor) -> torch.Tensor:
    """2D normal distribution
    
    Args:
        x1, x2: Input coordinates
        mu1, mu2: Mean coordinates
        s1, s2: Standard deviations
        rho: Correlation coefficient
    Returns:
        Probability densities
    """
    # Implementation based on eq #24 and 25 of http://arxiv.org/abs/1308.0850
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    
    z = (torch.square(norm1 / s1) + 
         torch.square(norm2 / s2) - 
         2.0 * rho * norm1 * norm2 / s1s2)
    
    neg_rho = 1 - torch.square(rho)
    result = torch.exp(-z / (2.0 * neg_rho))
    denom = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    
    return result / denom

def torch_1d_normal(x3: torch.Tensor,
                   mu3: torch.Tensor,
                   s3: torch.Tensor) -> torch.Tensor:
    """1D normal distribution for third dimension
    
    Args:
        x3: Input coordinate
        mu3: Mean coordinate
        s3: Standard deviation
    Returns:
        Probability densities
    """
    norm3 = x3 - mu3
    z = torch.square(norm3 / s3)
    result = torch.exp(-z / 2.0)
    denom = 2.0 * np.pi * s3
    return result / denom

def plot_traj_mdn_mult(model: nn.Module,
                      batch: torch.Tensor,
                      sl_plot: int = 5,
                      ind: int = -1,
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    """Plot trajectory with multiple mixture components
    
    Args:
        model: PyTorch MDN model
        batch: Batch of sequences [batch_size, coords, seq_len]
        sl_plot: Sequence index to plot distributions for
        ind: Batch index to plot (-1 for random)
        device: PyTorch device
    """
    # Ensure model is in eval mode
    model.eval()
    
    with torch.no_grad():
        # Get model predictions
        outputs = model(batch.to(device))
        mu1, mu2, mu3 = outputs['mu1'], outputs['mu2'], outputs['mu3']
        s1, s2, s3 = outputs['s1'], outputs['s2'], outputs['s3']
        rho, theta = outputs['rho'], outputs['theta']
        
    batch_size, coords, seq_len = batch.shape
    assert ind < batch_size, "Index outside batch size"
    assert sl_plot < seq_len, "Sequence index outside sequence length"
    
    if ind == -1:
        ind = np.random.randint(0, batch_size)
        
    # Grid parameters
    delta = 0.025
    width = 1.0
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Plot 3D trajectory
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    trajectory = batch[ind].cpu().numpy()
    ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], 'r')
    ax.scatter(trajectory[0, sl_plot], trajectory[1, sl_plot], trajectory[2, sl_plot])
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_zlabel('z coordinate')
    
    # Create evaluation grid
    x1 = np.arange(-width, width + 0.1, delta)
    x2 = np.arange(-width, width + 0.2, delta)
    x3 = np.arange(-width, width + 0.3, delta)
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    XX = np.stack((X1, X2, X3), axis=3)
    
    # Compute distributions for each mixture
    mixtures = mu1.shape[1]
    PP = []
    
    for m in range(mixtures):
        mean = torch.zeros(3, device=device)
        mean[0] = mu1[ind, m, sl_plot]
        mean[1] = mu2[ind, m, sl_plot]
        mean[2] = mu3[ind, m, sl_plot]
        
        # Construct covariance matrix
        cov = torch.zeros((3, 3), device=device)
        sigma1 = s1[ind, m, sl_plot]
        sigma2 = s2[ind, m, sl_plot]
        sigma3 = s3[ind, m, sl_plot]
        sigma12 = rho[ind, m, sl_plot] * sigma1 * sigma2
        
        cov[0, 0] = torch.square(sigma1)
        cov[1, 1] = torch.square(sigma2)
        cov[2, 2] = torch.square(sigma3)
        cov[1, 2] = sigma12
        cov[2, 1] = sigma12
        
        # Create distribution and compute PDF
        dist = MultivariateNormal(mean.cpu(), cov.cpu())
        P = dist.log_prob(torch.tensor(XX)).exp().numpy()
        PP.append(P)
        
    # Stack distributions and compute final mixture
    PP = np.stack(PP, axis=3)
    theta_local = theta[ind, :, sl_plot].cpu().numpy()
    ZZ = np.dot(PP, theta_local)
    
    print(f"Mixture weights: {theta_local}")
    
    # Plot marginal distributions
    # X-Y plane
    ax = fig.add_subplot(2, 2, 2)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.sum(ZZ, axis=2)
    CS = ax.contour(X1, X2, Z.T)
    plt.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    
    # X-Z plane
    ax = fig.add_subplot(2, 2, 3)
    X1, X3 = np.meshgrid(x1, x3)
    Z = np.sum(ZZ, axis=1)
    CS = ax.contour(X1, X3, Z.T)
    plt.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('z coordinate')
    
    # Y-Z plane
    ax = fig.add_subplot(2, 2, 4)
    X2, X3 = np.meshgrid(x2, x3)
    Z = np.sum(ZZ, axis=0)
    CS = ax.contour(X2, X3, Z.T)
    plt.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel('y coordinate')
    ax.set_ylabel('z coordinate')
    
    plt.tight_layout()
    plt.show()

class MDNLoss(nn.Module):
    """Loss function for Mixture Density Network"""
    def __init__(self):
        super().__init__()
        
    def forward(self, 
                x: torch.Tensor,
                mu1: torch.Tensor,
                mu2: torch.Tensor,
                mu3: torch.Tensor,
                s1: torch.Tensor,
                s2: torch.Tensor,
                s3: torch.Tensor,
                rho: torch.Tensor,
                theta: torch.Tensor) -> torch.Tensor:
        """Compute MDN loss
        
        Args:
            x: Target coordinates [batch, 3]
            mu1, mu2, mu3: Mixture means
            s1, s2, s3: Mixture standard deviations
            rho: Correlation coefficients
            theta: Mixture weights
        Returns:
            Loss value
        """
        mixture_probabilities = []
        
        for k in range(theta.shape[1]):
            # 2D normal for correlated dimensions
            prob_2d = torch_2d_normal(
                x[:, 0], x[:, 1],
                mu1[:, k], mu2[:, k],
                s1[:, k], s2[:, k],
                rho[:, k]
            )
            
            # 1D normal for independent dimension
            prob_1d = torch_1d_normal(
                x[:, 2],
                mu3[:, k],
                s3[:, k]
            )
            
            # Combined probability
            mixture_probabilities.append(prob_2d * prob_1d * theta[:, k])
            
        # Sum probabilities and take negative log likelihood
        total_probability = torch.stack(mixture_probabilities).sum(dim=0)
        loss = -torch.log(total_probability + 1e-6)
        
        return loss.mean()