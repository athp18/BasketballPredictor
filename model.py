import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class MDNConfig:
    """Configuration for MDN model"""
    num_layers: int
    hidden_size: int
    max_grad_norm: float
    batch_size: int
    sequence_length: int
    mixtures: int
    coords: int
    learning_rate: float
    use_mdn: bool
    dropout: float = 0.1
    
class BasketballMDN(nn.Module):
    """Basketball trajectory prediction model with MDN"""
    
    def __init__(self, config: MDNConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.coords,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True  # Accept input as [batch, seq, features]
        )
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, 2)
        
        # MDN head
        # 8 parameters per mixture: mu1, mu2, mu3, s1, s2, s3, rho, theta
        mdn_output_size = config.mixtures * 8
        self.mdn_head = nn.Linear(config.hidden_size, mdn_output_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
                
    def _process_mdn_params(self, 
                           mdn_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process MDN output into individual parameters
        
        Args:
            mdn_output: Raw output from MDN head [batch, seq, mixtures * 8]
        Returns:
            Dictionary containing processed MDN parameters
        """
        batch_size = mdn_output.size(0)
        seq_len = mdn_output.size(1)
        
        # Reshape to [batch, seq, mixtures, 8]
        params = mdn_output.view(batch_size, seq_len, self.config.mixtures, 8)
        
        # Split into individual parameters
        mu1 = params[..., 0]
        mu2 = params[..., 1]
        mu3 = params[..., 2]
        s1 = torch.exp(params[..., 3])  # Standard deviations must be positive
        s2 = torch.exp(params[..., 4])
        s3 = torch.exp(params[..., 5])
        rho = torch.tanh(params[..., 6])  # Correlation must be between -1 and 1
        
        # Process mixture weights (theta)
        theta = params[..., 7]
        theta = theta - theta.max(dim=2, keepdim=True)[0]  # For numerical stability
        theta = torch.exp(theta)
        theta = theta / theta.sum(dim=2, keepdim=True)  # Normalize to sum to 1
        
        return {
            'mu1': mu1, 'mu2': mu2, 'mu3': mu3,
            's1': s1, 's2': s2, 's3': s3,
            'rho': rho, 'theta': theta
        }
    
    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input tensor [batch, coords, seq]
            hidden: Optional initial hidden state
        Returns:
            Dictionary containing model outputs
        """
        # Transpose to [batch, seq, coords] for LSTM
        x = x.transpose(1, 2)
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Classification head (uses final timestep)
        classification_logits = self.classifier(lstm_out[:, -1])
        
        # MDN head (for all but last timestep)
        mdn_out = self.mdn_head(lstm_out[:, :-1])
        mdn_params = self._process_mdn_params(mdn_out)
        
        return {
            'classification_logits': classification_logits,
            **mdn_params,
            'hidden': hidden
        }

class BasketballMDNWithLoss(BasketballMDN):
    """Basketball MDN with built-in loss computation"""
    
    def compute_loss(self, 
                    outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor,
                    next_coords: torch.Tensor,
                    use_mdn: bool = True
                   ) -> Dict[str, torch.Tensor]:
        """Compute combined loss
        
        Args:
            outputs: Model outputs dictionary
            targets: Classification targets [batch]
            next_coords: Next coordinate targets [batch, 3, seq]
            use_mdn: Whether to include MDN loss
        Returns:
            Dictionary containing losses
        """
        # Classification loss
        classification_loss = nn.functional.cross_entropy(
            outputs['classification_logits'], 
            targets
        )
        
        # MDN loss
        mdn_loss = torch.tensor(0.0, device=targets.device)
        if use_mdn:
            # Convert absolute coordinates to offsets
            next_offsets = next_coords[:, :, 1:] - next_coords[:, :, :-1]
            
            # Split into components
            xn1, xn2, xn3 = next_offsets.split(1, dim=1)
            xn1 = xn1.squeeze(1)
            xn2 = xn2.squeeze(1)
            xn3 = xn3.squeeze(1)
            
            # Compute probability densities
            px1x2 = torch.zeros_like(outputs['theta'])
            px3 = torch.zeros_like(outputs['theta'])
            
            for m in range(self.config.mixtures):
                # 2D normal for correlated dimensions
                norm1 = xn1.unsqueeze(-1) - outputs['mu1'][..., m:m+1]
                norm2 = xn2.unsqueeze(-1) - outputs['mu2'][..., m:m+1]
                s1s2 = outputs['s1'][..., m:m+1] * outputs['s2'][..., m:m+1]
                rho = outputs['rho'][..., m:m+1]
                
                z = (torch.square(norm1 / outputs['s1'][..., m:m+1]) + 
                     torch.square(norm2 / outputs['s2'][..., m:m+1]) -
                     2 * rho * norm1 * norm2 / s1s2)
                
                neg_rho = 1 - torch.square(rho)
                px1x2[..., m:m+1] = torch.exp(-z / (2 * neg_rho)) / (
                    2 * np.pi * s1s2 * torch.sqrt(neg_rho)
                )
                
                # 1D normal for independent dimension
                norm3 = xn3.unsqueeze(-1) - outputs['mu3'][..., m:m+1]
                z3 = torch.square(norm3 / outputs['s3'][..., m:m+1])
                px3[..., m:m+1] = torch.exp(-z3 / 2) / (
                    np.sqrt(2 * np.pi) * outputs['s3'][..., m:m+1]
                )
            
            # Combine probabilities and compute loss
            prob = (px1x2 * px3 * outputs['theta']).sum(dim=-1)
            mdn_loss = -torch.log(torch.clamp(prob, min=1e-20)).mean()
        
        # Combined loss
        total_loss = classification_loss + mdn_loss if use_mdn else classification_loss
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'mdn_loss': mdn_loss
        }
    
    def sample_trajectory(self, 
                         seq: torch.Tensor,
                         sl_pre: int = 4,
                         bias: float = 0.0,
                         device: str = 'cuda'
                        ) -> torch.Tensor:
        """Sample a trajectory from the MDN
        
        Args:
            seq: Input sequence [coords, seq_len]
            sl_pre: Number of predefined timesteps to use
            bias: Bias term for sampling (reduces variance when positive)
            device: PyTorch device
        Returns:
            Sampled trajectory
        """
        self.eval()
        with torch.no_grad():
            # Prepare input
            seq = torch.tensor(seq, dtype=torch.float32, device=device)
            seq_feed = torch.zeros(
                (1, self.config.coords, self.config.sequence_length),
                device=device
            )
            seq_feed[0] = seq
            
            # Sample trajectory
            for t in range(sl_pre, self.config.sequence_length - 1):
                # Get MDN parameters
                outputs = self(seq_feed)
                
                # Sample mixture component
                theta = outputs['theta'][0, t-1].cpu().numpy()
                mixture_idx = np.random.choice(self.config.mixtures, p=theta)
                
                # Construct mean and covariance
                mean = torch.zeros(3, device=device)
                mean[0] = outputs['mu1'][0, t-1, mixture_idx]
                mean[1] = outputs['mu2'][0, t-1, mixture_idx]
                mean[2] = outputs['mu3'][0, t-1, mixture_idx]
                
                cov = torch.zeros((3, 3), device=device)
                s1 = torch.exp(-bias) * outputs['s1'][0, t-1, mixture_idx]
                s2 = torch.exp(-bias) * outputs['s2'][0, t-1, mixture_idx]
                s3 = torch.exp(-bias) * outputs['s3'][0, t-1, mixture_idx]
                rho = outputs['rho'][0, t-1, mixture_idx]
                
                cov[0, 0] = s1 * s1
                cov[1, 1] = s2 * s2
                cov[2, 2] = s3 * s3
                cov[0, 1] = rho * s1 * s2
                cov[1, 0] = cov[0, 1]
                
                # Sample from distribution
                dist = MultivariateNormal(mean, cov)
                offset = dist.sample()
                
                # Update sequence
                seq_feed[0, :3, t+1] = seq_feed[0, :3, t] + offset
            
            return seq_feed[0]

    def plot_sample(self,
                   original_seq: torch.Tensor,
                   sampled_seq: torch.Tensor):
        """Plot original and sampled trajectories
        
        Args:
            original_seq: Original sequence [coords, seq_len]
            sampled_seq: Sampled sequence [coords, seq_len]
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original trajectory
        ax.plot(
            original_seq[0].cpu(), 
            original_seq[1].cpu(), 
            original_seq[2].cpu(), 
            'r', 
            label='Original'
        )
        
        # Plot sampled trajectory
        ax.plot(
            sampled_seq[0].cpu(), 
            sampled_seq[1].cpu(), 
            sampled_seq[2].cpu(), 
            'b', 
            label='Sampled'
        )
        
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        ax.set_zlabel('z coordinate')
        ax.legend()
        plt.show()