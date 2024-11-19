import numpy as np
import pandas as pd
from itertools import groupby
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Tuple, Dict, Optional
from util_basket import plot_basket

def return_large_true(ind_crit: np.ndarray) -> Tuple[int, int]:
    """Calculate the longest consecutive True's in ind_crit
    
    Args:
        ind_crit: Boolean array where criteria are met
    Returns:
        Tuple of (length of longest consecutive True sequence, starting index)
    """
    i = 0
    best_elems = 0
    best_i = 0

    for key, group in groupby(ind_crit, lambda x: x):
        number = next(group)
        elems = len(list(group)) + 1
        if number == 1 and elems > 1:
            if elems > best_elems:
                best_elems = elems
                best_i = i
        i += elems
    return best_elems, best_i

class BasketballDataset(Dataset):
    """PyTorch Dataset for basketball shot trajectories"""
    
    def __init__(self, 
                 data: torch.Tensor,
                 labels: torch.Tensor):
        """
        Args:
            data: Tensor of shape [N, seq_len, features]
            labels: Tensor of shape [N]
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

class BasketballDataLoader:
    def __init__(self, 
                 direc: str,
                 csv_file: str,
                 center: np.ndarray = np.array([5.25, 25.0, 10.0]),
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the basketball data loader
        
        Args:
            direc: Directory containing the data files (must end with '/')
            csv_file: Name of the CSV file (must end with .csv)
            center: Center coordinates for normalizing data
            device: PyTorch device to use
        """
        assert direc[-1] == "/", "Please provide a directory ending with a /"
        assert csv_file[-4:] == ".csv", "Please provide a filename ending with .csv"
        
        self.center = center
        self.csv_loc = direc + csv_file
        self.device = device
        
        # Storage for processed data
        self.data3: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.is_abs = True
        self.data: Dict = {}
        
        # Counters
        self.N = 0
        self.epochs = 0
        self.omit = 0

    def munge_data(self, 
                   height: float = 11.0,
                   seq_len: int = 10,
                   dist: float = 3.0,
                   verbose: bool = True) -> None:
        """Process the raw data into sequences
        
        Args:
            height: Minimum height threshold for trajectories
            seq_len: Desired sequence length
            dist: Minimum distance to basket
            verbose: Whether to print progress information
        """
        if self.data3 is not None:
            print("Data already loaded. Are you calling function twice?")
            return

        # Read and sort data
        df = pd.read_csv(self.csv_loc).sort_values(
            by=["id", "game_clock"], ascending=[1, 0]
        )
        if verbose:
            print(f"The shape of the read data is {df.shape}")

        # Convert to numpy for processing
        df_arr = df[["x", "y", "z", "game_clock", "EVENTMSGTYPE", "rankc"]].values
        processed_sequences = []
        processed_labels = []
        
        start_ind = 0
        N = len(df_arr)
        
        for i in range(1, N):
            if verbose and i % 1000 == 0:
                print(f"At line {i:5.0f} of {N:5.0f}")
                
            if int(df_arr[i, 5]) == 1:
                end_ind = i
                seq = df_arr[start_ind:end_ind, :4]
                
                # Apply criteria
                dist_xyz = np.linalg.norm(seq[:, :3] - self.center, axis=1)
                ind_crit = np.logical_and((seq[:, 2] > height), (dist_xyz > dist))
                
                if np.sum(ind_crit) == 0:
                    continue
                    
                li, i = return_large_true(ind_crit)
                seq = seq[i:i + li, :]
                
                try:
                    seq[:, 3] = seq[:, 3] - np.min(seq[:, 3])
                except:
                    print("A sequence didn't match criteria")
                    continue
                    
                if seq.shape[0] >= seq_len:
                    processed_sequences.append(seq[-seq_len:])
                    processed_labels.append(df_arr[start_ind, 4])
                else:
                    self.omit += 1
                    
                start_ind = end_ind

        # Convert to PyTorch tensors
        try:
            self.data3 = torch.tensor(np.stack(processed_sequences), 
                                    dtype=torch.float32, 
                                    device=self.device)
            self.labels = torch.tensor(np.stack(processed_labels), 
                                     device=self.device)
            
            if torch.min(self.labels) > 0.9:
                self.labels -= 1
                
            self.N = len(self.labels)
            
        except:
            print("Error converting to PyTorch tensors")
            return

        print(f"Lost {self.omit} sequences ({float(self.omit)/self.N:.2f}) that didn't match criteria")

    def center_data(self, center_cent: np.ndarray = np.array([5.25, 25.0, 10.0])) -> None:
        """Center the data around given coordinates
        
        Args:
            center_cent: Coordinates to center around
        """
        assert self.data3 is not None, "First munge the data before centering"
        self.data3[:, :, :3] = self.data3[:, :, :3] - torch.tensor(center_cent, 
                                                                   device=self.device)
        self.center = self.center - center_cent
        print(f"New center: {self.center}")

    def abs_to_off(self) -> None:
        """Convert absolute positions to offset/velocity data"""
        assert self.is_abs, "Data is already offset"
        assert self.data3 is not None, "First munge the data before converting"
        
        off = self.data3[:, 1:, :3] - self.data3[:, :-1, :3]
        time = self.data3[:, 1:, 3].unsqueeze(-1)
        self.data3 = torch.cat((off, time), dim=2)
        self.is_abs = False
        print("Data converted to offset format")

    def create_dataloaders(self, 
                          batch_size: int = 64,
                          train_ratio: float = 0.8,
                          shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for training and validation
        
        Args:
            batch_size: Batch size for DataLoader
            train_ratio: Ratio of data to use for training
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        assert self.data3 is not None, "First munge the data before creating dataloaders"
        assert 0 < train_ratio < 1, "Train ratio must be between 0 and 1"
        
        N = len(self.data3)
        indices = torch.randperm(N)
        train_size = int(train_ratio * N)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets
        train_dataset = BasketballDataset(
            self.data3[train_indices],
            self.labels[train_indices]
        )
        
        val_dataset = BasketballDataset(
            self.data3[val_indices],
            self.labels[val_indices]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        print(f"Created DataLoaders with {train_size} training and {N-train_size} validation samples")
        return train_loader, val_loader

    def plot_basket_traj(self) -> None:
        """Plot basketball trajectories using the util_basket function"""
        assert self.data3 is not None, "No data to plot"
        plot_basket(self.data3.cpu().numpy(), self.labels.cpu().numpy())

    def plot_traj_2d(self, Nplot: int, extra_title: str = " ") -> None:
        """Plot N trajectories in 2D plane (XY vs Z)
        
        Args:
            Nplot: Number of trajectories to plot
            extra_title: Additional title text
        """
        assert self.data3 is not None, "No data to plot"
        
        fig = plt.figure()
        data2 = torch.norm(self.data3[:, :, :2], dim=2)
        data2 = torch.stack((data2, self.data3[:, :, 2]), dim=2)
        
        N = len(data2)
        for i in range(Nplot):
            ind = torch.randint(0, N, (1,)).item()
            if self.labels[ind] == 1:
                plt.plot(data2[ind, :, 0].cpu(), data2[ind, :, 1].cpu(), 'r', label='miss')
            if self.labels[ind] == 0:
                plt.plot(data2[ind, :, 0].cpu(), data2[ind, :, 1].cpu(), 'b', label='hit')
                
        plt.title("Example trajectories " + extra_title)
        plt.xlabel("Distance to basket (feet)")
        plt.ylabel("Height (feet)")
        
        handles, labels = plt.gca().get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
        plt.legend(newHandles, newLabels)
        plt.show()