"""
Models for fairness experiments.

Implements MLPs designed to be "borderline fair" for testing numerical effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class FairMLPClassifier(nn.Module):
    """
    Multi-layer perceptron for binary classification with fairness considerations.
    
    Designed to produce predictions that can be borderline fair (DPG ≈ 0.05-0.10)
    to stress-test numerical effects.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 use_batchnorm: bool = False,
                 dropout_rate: float = 0.0,
                 activation: str = 'relu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())  # Output probability
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_layer_dims(self) -> List[int]:
        """Get dimensions for error tracking"""
        return [self.input_dim] + self.hidden_dims + [1]
    
    def get_activations(self) -> List[str]:
        """Get activation functions for error tracking"""
        activations = [self.activation_name] * len(self.hidden_dims)
        activations.append('sigmoid')  # Output activation
        return activations


def train_fair_classifier(model: nn.Module,
                          train_data: torch.Tensor,
                          train_labels: torch.Tensor,
                          train_groups: torch.Tensor,
                          n_epochs: int = 100,
                          lr: float = 0.001,
                          fairness_weight: float = 0.0,
                          device: str = 'cpu',
                          verbose: bool = True) -> List[Dict]:
    """
    Train a classifier with optional fairness regularization.
    
    Args:
        model: The model to train
        train_data: Training features
        train_labels: Training labels
        train_groups: Group membership (for fairness)
        n_epochs: Number of training epochs
        lr: Learning rate
        fairness_weight: Weight for demographic parity regularization
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        List of training history dicts
    """
    model = model.to(device)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    train_groups = train_groups.to(device)
    
    # Convert labels and groups to match model's dtype for BCE loss
    model_dtype = next(model.parameters()).dtype
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    history = []
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(train_data).squeeze()
        
        # Classification loss (convert labels to model's dtype)
        loss = criterion(predictions, train_labels.to(model_dtype))
        
        # Define group masks  
        group_0_mask = train_groups == 0
        group_1_mask = train_groups == 1
        
        # Fairness regularization (demographic parity)
        if fairness_weight > 0:
            if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
                pos_rate_0 = predictions[group_0_mask].mean()
                pos_rate_1 = predictions[group_1_mask].mean()
                fairness_loss = (pos_rate_0 - pos_rate_1).abs()
                loss = loss + fairness_weight * fairness_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        model.eval()
        with torch.no_grad():
            pred_labels = (predictions > 0.5).to(model_dtype)
            accuracy = (pred_labels == train_labels.to(model_dtype)).to(model_dtype).mean().item()
            
            # Compute demographic parity gap
            if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
                dpg = abs(
                    (predictions[group_0_mask] > 0.5).to(model_dtype).mean().item() -
                    (predictions[group_1_mask] > 0.5).to(model_dtype).mean().item()
                )
            else:
                dpg = 0.0
        
        history.append({
            'epoch': epoch,
            'loss': loss.item(),
            'accuracy': accuracy,
            'dpg': dpg
        })
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {accuracy:.4f}, "
                  f"DPG: {dpg:.4f}")
    
    return history


def create_borderline_fair_model(input_dim: int,
                                 hidden_dims: List[int] = [32, 16],
                                 seed: Optional[int] = None) -> FairMLPClassifier:
    """
    Create a model initialized to be borderline fair.
    
    Uses specific initialization to encourage borderline fairness metrics.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    model = FairMLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        use_batchnorm=False,
        dropout_rate=0.0,
        activation='relu'
    )
    
    # Custom initialization: small weights to keep predictions near 0.5
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.05)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    return model
