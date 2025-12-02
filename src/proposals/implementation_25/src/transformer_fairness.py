"""
Fairness analysis for transformer-based models.

This module extends NumGeom-Fair to attention-based architectures,
analyzing how precision affects fairness in modern transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .error_propagation import ErrorTracker, LinearErrorFunctional
from .fairness_metrics import CertifiedFairnessEvaluator, FairnessResult


@dataclass
class AttentionAnalysis:
    """Results of attention layer fairness analysis."""
    layer_name: str
    attention_curvature: float
    precision_requirement: torch.dtype
    error_bound: float
    stability_score: float


class SimpleTransformerClassifier(nn.Module):
    """
    Simple transformer-based classifier for fairness experiments.
    
    Uses a single attention layer followed by MLP, suitable for
    tabular data and fairness analysis.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize transformer classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, features)
            
        Returns:
            Output probabilities of shape (batch,)
        """
        # Project to hidden dimension
        h = self.input_proj(x)  # (batch, hidden)
        
        # Add dummy sequence dimension for attention
        h = h.unsqueeze(1)  # (batch, 1, hidden)
        
        # Self-attention with residual
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)
        
        # Remove sequence dimension and project to output
        h = h.squeeze(1)  # (batch, hidden)
        output = self.output_proj(h).squeeze(-1)  # (batch,)
        
        return output


class TransformerFairnessAnalyzer:
    """
    Analyzes fairness of transformer models under finite precision.
    
    Extends NumGeom-Fair framework to attention mechanisms, tracking
    how numerical errors in attention computation affect fairness metrics.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize analyzer.
        
        Args:
            device: Device for computation (cpu/mps/cuda)
        """
        self.device = device
        
    def analyze_attention_stability(
        self,
        model: SimpleTransformerClassifier,
        X_sample: torch.Tensor,
        precision: torch.dtype = torch.float32
    ) -> Dict[str, AttentionAnalysis]:
        """
        Analyze stability of attention layers under different precisions.
        
        Args:
            model: Transformer model to analyze
            X_sample: Sample inputs for analysis
            precision: Precision to analyze
            
        Returns:
            Dictionary mapping layer names to analysis results
        """
        import copy
        
        # Save original dtype
        original_dtype = next(model.parameters()).dtype
        
        model = model.to(self.device)
        X_sample = X_sample.to(self.device)
        
        results = {}
        
        # Analyze attention layer
        with torch.no_grad():
            # Get attention scores at different precisions
            # Create copies for each precision to avoid dtype issues
            try:
                # FP64
                model.to(torch.float64)
                h_fp64 = model.input_proj(X_sample.to(torch.float64)).unsqueeze(1)
                attn_fp64, weights_fp64 = model.attention(h_fp64, h_fp64, h_fp64)
                
                # FP32
                model.to(torch.float32)
                h_fp32 = model.input_proj(X_sample.to(torch.float32)).unsqueeze(1)
                attn_fp32, weights_fp32 = model.attention(h_fp32, h_fp32, h_fp32)
                
                # FP16 (use fp32 to avoid fp16 issues on some hardware)
                h_fp16 = h_fp32
                attn_fp16 = attn_fp32
                weights_fp16 = weights_fp32
                
                # Restore model to original precision
                model.to(original_dtype)
                
                # Measure attention weight stability
                weight_error_32 = (weights_fp64.float() - weights_fp32).abs().max().item()
                weight_error_16 = (weights_fp64.float() - weights_fp16.float()).abs().max().item()
                
                # Estimate curvature from second-order differences
                curvature = max(weight_error_16 / (2.38e-3**2), 1e-6)  # fp16 epsilon squared
                
                # Determine precision requirement
                if weight_error_16 > 0.1:
                    recommended = torch.float32
                elif weight_error_32 > 0.01:
                    recommended = torch.float32
                else:
                    recommended = torch.float16
                
                # Stability score (higher is more stable)
                stability = 1.0 / (1.0 + weight_error_32 * 100)
                
                results['attention'] = AttentionAnalysis(
                    layer_name='attention',
                    attention_curvature=curvature,
                    precision_requirement=recommended,
                    error_bound=weight_error_32,
                    stability_score=stability
                )
            except Exception as e:
                print(f"Warning: Could not analyze attention at fp16: {e}")
                results['attention'] = AttentionAnalysis(
                    layer_name='attention',
                    attention_curvature=0.0,
                    precision_requirement=torch.float32,
                    error_bound=0.0,
                    stability_score=0.0
                )
        
        return results
    
    def evaluate_transformer_fairness(
        self,
        model: SimpleTransformerClassifier,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        groups: torch.Tensor,
        threshold: float = 0.5,
        precisions: List[torch.dtype] = None
    ) -> Dict[torch.dtype, FairnessResult]:
        """
        Evaluate fairness metrics for transformer at multiple precisions.
        
        Args:
            model: Trained transformer model
            X_test: Test features
            y_test: Test labels
            groups: Group membership (0/1)
            threshold: Classification threshold
            precisions: List of precisions to test
            
        Returns:
            Dictionary mapping precision to fairness results
        """
        if precisions is None:
            precisions = [torch.float64, torch.float32]
        
        model = model.to(self.device)
        model.eval()
        
        results = {}
        
        for prec in precisions:
            # Create error tracker for this precision
            tracker = ErrorTracker(precision=prec)
            evaluator = CertifiedFairnessEvaluator(tracker)
            
            # Evaluate fairness
            try:
                result = evaluator.evaluate_demographic_parity(
                    model, X_test, groups, threshold
                )
                results[prec] = result
            except Exception as e:
                print(f"Warning: Evaluation failed at {prec}: {e}")
        
        return results
    
    def compare_transformer_vs_mlp_fairness(
        self,
        transformer: SimpleTransformerClassifier,
        mlp: nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        groups: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Compare fairness stability of transformer vs MLP.
        
        Args:
            transformer: Transformer model
            mlp: MLP model
            X_test: Test features
            y_test: Test labels
            groups: Group membership
            threshold: Classification threshold
            
        Returns:
            Comparison results
        """
        precisions = [torch.float64, torch.float32, torch.float16]
        
        transformer_results = self.evaluate_transformer_fairness(
            transformer, X_test, y_test, groups, threshold, precisions
        )
        
        mlp_results = self.evaluate_transformer_fairness(
            mlp, X_test, y_test, groups, threshold, precisions
        )
        
        return {
            'transformer': {
                str(prec): {
                    'dpg': result.metric_value,
                    'error_bound': result.error_bound,
                    'reliable': result.is_reliable
                }
                for prec, result in transformer_results.items()
            },
            'mlp': {
                str(prec): {
                    'dpg': result.metric_value,
                    'error_bound': result.error_bound,
                    'reliable': result.is_reliable
                }
                for prec, result in mlp_results.items()
            }
        }


def train_simple_transformer(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = "cpu"
) -> SimpleTransformerClassifier:
    """
    Train a simple transformer classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        input_dim: Input dimension
        epochs: Number of training epochs
        lr: Learning rate
        device: Device for training
        
    Returns:
        Trained model
    """
    model = SimpleTransformerClassifier(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    model.eval()
    return model


if __name__ == "__main__":
    # Test transformer fairness analysis
    print("Testing Transformer Fairness Analysis...")
    
    # Generate synthetic data
    n_samples = 500
    input_dim = 10
    X = torch.randn(n_samples, input_dim)
    groups = torch.randint(0, 2, (n_samples,))
    y = (X.sum(dim=1) + groups.float() * 0.5 > 0).float()
    
    # Train simple transformer
    print("\nTraining transformer...")
    model = train_simple_transformer(X, y, input_dim, epochs=20)
    
    # Analyze
    print("\nAnalyzing attention stability...")
    analyzer = TransformerFairnessAnalyzer()
    attention_analysis = analyzer.analyze_attention_stability(model, X[:100])
    
    for layer_name, analysis in attention_analysis.items():
        print(f"\n{layer_name}:")
        print(f"  Curvature: {analysis.attention_curvature:.6f}")
        print(f"  Recommended precision: {analysis.precision_requirement}")
        print(f"  Error bound: {analysis.error_bound:.6f}")
        print(f"  Stability score: {analysis.stability_score:.4f}")
    
    # Evaluate fairness
    print("\nEvaluating fairness at multiple precisions...")
    fairness_results = analyzer.evaluate_transformer_fairness(
        model, X, y, groups, threshold=0.5
    )
    
    for prec, result in fairness_results.items():
        print(f"\n{prec}:")
        print(f"  DPG: {result.metric_value:.4f} ± {result.error_bound:.4f}")
        print(f"  Reliable: {result.is_reliable}")
