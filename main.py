"""
This code comes with the paper
"Applying Deep Learning to Basketball Trajectories"
Originally by Rajiv Shah and Rob Romijnders
Converted to PyTorch version

The script handles training of the basketball trajectory model with optional MDN component
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import time
from pathlib import Path

from dataloader import BasketballDataLoader
from model import BasketballMDNWithLoss, MDNConfig
from util_basket import plot_basket
from util_MDN import plot_traj_mdn_mult

def train_basketball_model(
    data_dir: str = 'data/',
    plot: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[BasketballMDNWithLoss, Dict]:
    """Main training function for basketball trajectory model
    
    Args:
        data_dir: Directory containing the data
        plot: Whether to create visualization plots
        device: PyTorch device to use
    Returns:
        Trained model and performance metrics
    """
    
    # Hyperparameters
    config = MDNConfig(
        num_layers=2,
        hidden_size=64,
        max_grad_norm=1.0,
        batch_size=64,
        sequence_length=12,
        mixtures=3,
        coords=4,
        learning_rate=0.005,
        use_mdn=False,  # Set to True to enable MDN
        dropout=0.3
    )
    
    # Training parameters
    max_iterations = 20000
    plot_every = 100
    early_stopping_patience = 5
    db = 5  # distance to basket to stop trajectories
    
    # Load and prepare data
    center = torch.tensor([5.25, 25.0, 10.0], device=device)
    data_loader = BasketballDataLoader(
        direc=data_dir,
        csv_file='seq_all.csv',
        center=center.cpu().numpy()
    )
    
    # Process data
    data_loader.munge_data(height=11.0, seq_len=config.sequence_length, dist=db)
    data_loader.center_data(center.cpu().numpy())
    
    # Create dataloaders
    train_loader, val_loader = data_loader.create_dataloaders(
        batch_size=config.batch_size,
        train_ratio=0.8,
        shuffle=True
    )
    
    # Calculate epochs
    n_train = len(train_loader.dataset)
    epochs = np.floor(config.batch_size * max_iterations / n_train)
    print(f'Training for approximately {epochs:.0f} epochs')
    
    # Create model and optimizer
    model = BasketballMDNWithLoss(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Performance tracking
    perf_collect = {
        'train_acc': [], 
        'train_cost': [], 
        'train_seq_cost': [],
        'val_acc': [], 
        'val_cost': [], 
        'val_seq_cost': [], 
        'val_auc': []
    }
    
    # Early stopping setup
    best_auc = 0.0
    auc_ma = 0.0
    patience_counter = 0
    
    # Training loop
    step = 0
    iteration = 0
    
    while iteration < max_iterations:
        model.train()
        for batch_data, targets in train_loader:
            if iteration >= max_iterations:
                break
                
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            
            # Performance logging
            if iteration % plot_every == 0:
                model.eval()
                with torch.no_grad():
                    # Training performance
                    outputs = model(batch_data)
                    losses = model.compute_loss(
                        outputs, 
                        targets, 
                        batch_data,
                        use_mdn=config.use_mdn
                    )
                    
                    # Calculate accuracy
                    preds = outputs['classification_logits'].argmax(dim=1)
                    train_acc = (preds == targets).float().mean().item()
                    
                    # Validation performance
                    val_batch_data, val_targets = next(iter(val_loader))
                    val_batch_data = val_batch_data.to(device)
                    val_targets = val_targets.to(device)
                    
                    val_outputs = model(val_batch_data)
                    val_losses = model.compute_loss(
                        val_outputs,
                        val_targets,
                        val_batch_data,
                        use_mdn=config.use_mdn
                    )
                    
                    # Calculate validation metrics
                    val_preds = val_outputs['classification_logits'].argmax(dim=1)
                    val_acc = (val_preds == val_targets).float().mean().item()
                    
                    # Calculate AUC
                    val_probs = torch.softmax(val_outputs['classification_logits'], dim=1)
                    auc = metrics.roc_auc_score(
                        val_targets.cpu().numpy(),
                        val_probs[:, 1].cpu().numpy()
                    )
                    
                    # Update moving average AUC
                    ma_range = 5
                    perf_collect['val_auc'].append(auc)
                    if len(perf_collect['val_auc']) > ma_range:
                        auc_ma = np.mean(perf_collect['val_auc'][-ma_range:])
                    
                    # Early stopping check
                    if auc > best_auc:
                        best_auc = auc
                        patience_counter = 0
                    elif auc_ma < 0.8 * best_auc:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping triggered!")
                        break
                    
                    # Store performances
                    perf_collect['train_acc'].append(train_acc)
                    perf_collect['train_cost'].append(losses['classification_loss'].item())
                    perf_collect['val_acc'].append(val_acc)
                    perf_collect['val_cost'].append(val_losses['classification_loss'].item())
                    
                    if config.use_mdn:
                        perf_collect['train_seq_cost'].append(losses['mdn_loss'].item())
                        perf_collect['val_seq_cost'].append(val_losses['mdn_loss'].item())
                    
                    print(
                        f"Iter {iteration}/{max_iterations} "
                        f"Val acc: {val_acc:.3f} "
                        f"AUC: {auc:.3f}({auc_ma:.3f}) "
                        f"Train loss: {losses['total_loss']:.3f}"
                    )
                    
                    step += 1
                    model.train()
            
            # Training step
            optimizer.zero_grad()
            outputs = model(batch_data)
            losses = model.compute_loss(
                outputs,
                targets,
                batch_data,
                use_mdn=config.use_mdn
            )
            
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            iteration += 1
            
        # Step the learning rate scheduler
        scheduler.step()
    
    # Final validation
    model.eval()
    with torch.no_grad():
        val_batch_data, val_targets = next(iter(val_loader))
        val_batch_data = val_batch_data.to(device)
        val_targets = val_targets.to(device)
        val_outputs = model(val_batch_data)
        val_preds = val_outputs['classification_logits'].argmax(dim=1)
        final_acc = (val_preds == val_targets).float().mean().item()
        
    print(f'Final validation accuracy: {final_acc:.3f}')
    print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # Plotting if requested
    if plot:
        if config.use_mdn:
            # Plot MDN predictions
            with torch.no_grad():
                val_batch_data, _ = next(iter(val_loader))
                val_batch_data = val_batch_data.to(device)
                plot_traj_mdn_mult(model, val_batch_data)
            
            # Sample trajectories
            sl_pre = 5
            seq_pre = val_batch_data[3]
            seq_samp = model.sample_trajectory(seq_pre, sl_pre=sl_pre, bias=2.0)
        
        # Plot performance metrics
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(perf_collect['train_acc'], label='Train accuracy')
        plt.plot(perf_collect['val_acc'], label='Valid accuracy')
        plt.legend()
        plt.title('Model Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(perf_collect['train_cost'], label='Train class cost')
        plt.plot(perf_collect['val_cost'], label='Valid class cost')
        if config.use_mdn:
            plt.plot(perf_collect['train_seq_cost'], label='Train seq cost')
            plt.plot(perf_collect['val_seq_cost'], label='Valid seq cost')
        plt.legend()
        plt.title('Model Loss')
        
        plt.tight_layout()
        plt.show()
    
    return model, perf_collect

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train model
    model, performances = train_basketball_model(
        data_dir='data/',
        plot=True
    )
