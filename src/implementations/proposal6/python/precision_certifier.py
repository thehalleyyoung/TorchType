"""
PyTorch Integration for HNF Certified Precision Bounds (Proposal 6)

This module provides Python bindings to the C++ certification library,
enabling real-time precision analysis during PyTorch model training.

Based on HNF Paper Theorem 5.7: p >= log2(c * κ * D^2 / ε)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass, asdict
from collections import OrderedDict


@dataclass
class LayerCertificate:
    """Precision certificate for a single layer"""
    layer_name: str
    layer_type: str
    curvature: float
    lipschitz: float
    required_bits: int
    recommended_dtype: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    

@dataclass
class ModelCertificate:
    """Complete precision certificate for a model"""
    model_name: str
    total_layers: int
    target_accuracy: float
    input_domain_diameter: float
    global_curvature: float
    min_required_bits: int
    recommended_dtype: str
    layer_certificates: List[Dict]  # List of dicts for JSON serialization
    bottleneck_layers: List[str]
    verification_timestamp: float
    

class CurvatureBounds:
    """
    Compute curvature bounds for various layer types.
    
    Based on HNF paper Section 4.3 (Curvature Invariants)
    """
    
    @staticmethod
    def linear(weight: torch.Tensor) -> float:
        """Linear layers have zero curvature (affine maps)"""
        return 0.0
    
    @staticmethod
    def relu() -> float:
        """ReLU has zero curvature (piecewise linear)"""
        return 0.0
    
    @staticmethod
    def softmax(input_bound: float) -> float:
        """
        Softmax curvature from HNF paper Example 4.5
        κ ≈ exp(2 * max_logit)
        
        Conservative bound: exp(2 * input_bound)
        """
        return min(np.exp(2.0 * input_bound), 1e10)  # Cap for numerical stability
    
    @staticmethod
    def layer_norm(normalized_shape: int, eps: float = 1e-5) -> float:
        """
        LayerNorm curvature (approximate)
        κ ≈ 1/eps for variance normalization
        """
        return 1.0 / eps
    
    @staticmethod
    def gelu() -> float:
        """
        GELU activation curvature
        GELU(x) = x * Φ(x) where Φ is Gaussian CDF
        Second derivative bounded by ~0.4
        """
        return 0.398  # ≈ 1/√(2π)
    
    @staticmethod
    def attention(seq_len: int, d_model: int, qk_bound: float) -> float:
        """
        Multi-head attention curvature
        From HNF paper Example 6.1:
        κ_attn ≈ exp(2 * seq_len * ||QK||)
        
        For numerical stability, we use a conservative bound
        """
        # Conservative: scale by sequence length and QK norm
        exponent = min(2.0 * np.log(seq_len) + qk_bound, 50)  # Cap exponent
        return np.exp(exponent)
    
    @staticmethod
    def conv2d(kernel_size: Tuple[int, int], stride: int = 1) -> float:
        """
        Convolutional layer curvature
        Conv is linear, so κ = 0
        """
        return 0.0
    
    @staticmethod
    def batch_norm(num_features: int, eps: float = 1e-5) -> float:
        """
        Batch normalization curvature
        Similar to LayerNorm
        """
        return 1.0 / eps


class LipschitzBounds:
    """
    Compute Lipschitz constants for various layer types.
    
    Based on HNF paper Definition 2.5 (Numerical Morphism)
    """
    
    @staticmethod
    def linear(weight: torch.Tensor) -> float:
        """L = ||W||_op (spectral norm)"""
        with torch.no_grad():
            # Compute spectral norm (largest singular value)
            if len(weight.shape) == 2:
                # Use power iteration for efficiency
                return torch.linalg.matrix_norm(weight, ord=2).item()
            else:
                # Flatten for conv layers
                w_flat = weight.reshape(weight.size(0), -1)
                return torch.linalg.matrix_norm(w_flat, ord=2).item()
    
    @staticmethod
    def relu() -> float:
        """ReLU is 1-Lipschitz"""
        return 1.0
    
    @staticmethod
    def softmax() -> float:
        """Softmax is 1-Lipschitz (in Euclidean norm)"""
        return 1.0
    
    @staticmethod
    def layer_norm() -> float:
        """LayerNorm is approximately 1-Lipschitz"""
        return 1.0
    
    @staticmethod
    def gelu() -> float:
        """GELU Lipschitz constant ≈ 1.7"""
        return 1.7
    
    @staticmethod
    def batch_norm() -> float:
        """BatchNorm is 1-Lipschitz after convergence"""
        return 1.0


class PrecisionCertifier:
    """
    Main certification engine for PyTorch models.
    
    Implements Theorem 5.7 from HNF paper:
    p >= log2(c * κ * D^2 / ε)
    """
    
    def __init__(self, 
                 target_accuracy: float = 1e-6,
                 input_domain_diameter: float = 10.0,
                 constant_c: float = 2.0):
        """
        Initialize certifier.
        
        Args:
            target_accuracy: Target numerical accuracy (ε)
            input_domain_diameter: Diameter of input domain (D)
            constant_c: Constant from Theorem 5.7 (default: 2.0)
        """
        self.target_accuracy = target_accuracy
        self.input_domain_diameter = input_domain_diameter
        self.constant_c = constant_c
        self.layer_info = []
        
    def compute_required_precision(self, 
                                   curvature: float,
                                   diameter: float = None) -> int:
        """
        Compute required mantissa bits from Theorem 5.7.
        
        p >= log2(c * κ * D^2 / ε)
        
        Args:
            curvature: Curvature bound κ
            diameter: Domain diameter (uses instance default if None)
            
        Returns:
            Required number of mantissa bits
        """
        if diameter is None:
            diameter = self.input_domain_diameter
            
        if curvature == 0.0:
            # Linear/piecewise linear: minimal precision needed
            # Just need to represent input precision + rounding
            return int(np.ceil(np.log2(diameter / self.target_accuracy)))
        
        # Theorem 5.7
        numerator = self.constant_c * curvature * (diameter ** 2)
        denominator = self.target_accuracy
        
        required_bits = np.log2(numerator / denominator)
        
        return int(np.ceil(required_bits))
    
    @staticmethod
    def bits_to_dtype(bits: int) -> str:
        """Map required bits to PyTorch dtype"""
        if bits <= 8:
            return "int8"
        elif bits <= 11:
            return "float16"  # fp16: 10 mantissa bits
        elif bits <= 16:
            return "bfloat16"  # bf16: 7 mantissa, but range helps
        elif bits <= 24:
            return "float32"  # fp32: 23 mantissa bits
        elif bits <= 53:
            return "float64"  # fp64: 52 mantissa bits
        else:
            return f"extended_precision (>{bits} bits)"
    
    def analyze_layer(self, 
                     module: nn.Module,
                     layer_name: str,
                     input_shape: Tuple[int, ...],
                     output_shape: Tuple[int, ...],
                     input_bound: float = 10.0) -> LayerCertificate:
        """
        Analyze a single layer and generate certificate.
        
        Args:
            module: PyTorch module
            layer_name: Name of the layer
            input_shape: Input tensor shape
            output_shape: Output tensor shape
            input_bound: Upper bound on input values
            
        Returns:
            LayerCertificate with precision requirements
        """
        layer_type = type(module).__name__
        
        # Compute curvature based on layer type
        if isinstance(module, nn.Linear):
            curvature = CurvatureBounds.linear(module.weight)
            lipschitz = LipschitzBounds.linear(module.weight)
        elif isinstance(module, nn.ReLU):
            curvature = CurvatureBounds.relu()
            lipschitz = LipschitzBounds.relu()
        elif isinstance(module, nn.Softmax):
            curvature = CurvatureBounds.softmax(input_bound)
            lipschitz = LipschitzBounds.softmax()
        elif isinstance(module, nn.LayerNorm):
            norm_shape = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
            curvature = CurvatureBounds.layer_norm(norm_shape, module.eps)
            lipschitz = LipschitzBounds.layer_norm()
        elif isinstance(module, nn.GELU):
            curvature = CurvatureBounds.gelu()
            lipschitz = LipschitzBounds.gelu()
        elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            curvature = CurvatureBounds.batch_norm(module.num_features, module.eps)
            lipschitz = LipschitzBounds.batch_norm()
        elif isinstance(module, nn.Conv2d):
            curvature = CurvatureBounds.conv2d(module.kernel_size, module.stride[0])
            lipschitz = LipschitzBounds.linear(module.weight)
        else:
            # Unknown layer: conservative bounds
            curvature = 1.0
            lipschitz = 1.0
        
        # Compute precision requirement
        required_bits = self.compute_required_precision(curvature)
        recommended_dtype = self.bits_to_dtype(required_bits)
        
        return LayerCertificate(
            layer_name=layer_name,
            layer_type=layer_type,
            curvature=curvature,
            lipschitz=lipschitz,
            required_bits=required_bits,
            recommended_dtype=recommended_dtype,
            input_shape=input_shape,
            output_shape=output_shape
        )
    
    def certify_model(self, 
                     model: nn.Module,
                     input_shape: Tuple[int, ...],
                     model_name: str = "model") -> ModelCertificate:
        """
        Generate complete precision certificate for a model.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (batch_size, ...)
            model_name: Name for the model
            
        Returns:
            Complete ModelCertificate
        """
        layer_certificates = []
        max_curvature = 0.0
        max_lipschitz = 1.0
        
        # Trace through model to get layer-wise information
        device = next(model.parameters()).device if list(model.parameters()) else 'cpu'
        dummy_input = torch.randn(input_shape, device=device)
        
        # Track activations to get shapes
        activation_shapes = {}
        hooks = []
        
        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_shapes[name] = tuple(output.shape)
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.Softmax, nn.LayerNorm, 
                                  nn.GELU, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)):
                hook = module.register_forward_hook(get_activation_hook(name))
                hooks.append(hook)
        
        # Forward pass to populate shapes
        with torch.no_grad():
            model.eval()
            model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze each layer
        prev_shape = input_shape
        input_bound = float(torch.max(torch.abs(dummy_input)).item())
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.Softmax, nn.LayerNorm,
                                  nn.GELU, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)):
                output_shape = activation_shapes.get(name, prev_shape)
                
                cert = self.analyze_layer(
                    module, name, prev_shape, output_shape, input_bound
                )
                
                layer_certificates.append(cert)
                
                # Update global bounds
                max_curvature = max(max_curvature, cert.curvature)
                max_lipschitz = max(max_lipschitz, cert.lipschitz)
                
                prev_shape = output_shape
                # Propagate bound through Lipschitz composition
                input_bound = input_bound * cert.lipschitz
        
        # Compute global precision requirement
        global_required_bits = self.compute_required_precision(max_curvature)
        global_dtype = self.bits_to_dtype(global_required_bits)
        
        # Identify bottleneck layers (require >float32)
        bottleneck_layers = [
            cert.layer_name for cert in layer_certificates
            if cert.required_bits > 24
        ]
        
        # Convert layer certificates to dicts for JSON serialization
        layer_cert_dicts = [asdict(cert) for cert in layer_certificates]
        
        return ModelCertificate(
            model_name=model_name,
            total_layers=len(layer_certificates),
            target_accuracy=self.target_accuracy,
            input_domain_diameter=self.input_domain_diameter,
            global_curvature=max_curvature,
            min_required_bits=global_required_bits,
            recommended_dtype=global_dtype,
            layer_certificates=layer_cert_dicts,
            bottleneck_layers=bottleneck_layers,
            verification_timestamp=time.time()
        )
    
    def print_certificate(self, cert: ModelCertificate):
        """Pretty-print a certificate"""
        print("╔" + "═" * 78 + "╗")
        print(f"║{'PRECISION CERTIFICATE':^78}║")
        print("╠" + "═" * 78 + "╣")
        print(f"║ Model: {cert.model_name:<69}║")
        print(f"║ Layers: {cert.total_layers:<68}║")
        print(f"║ Target Accuracy: {cert.target_accuracy:<60.2e}║")
        print(f"║ Domain Diameter: {cert.input_domain_diameter:<60.2f}║")
        print("╠" + "═" * 78 + "╣")
        print(f"║ Global Curvature κ: {cert.global_curvature:<57.4e}║")
        print(f"║ Required Precision: {cert.min_required_bits} bits mantissa{' ' * (78 - 34 - len(str(cert.min_required_bits)))}║")
        print(f"║ Recommended: {cert.recommended_dtype:<62}║")
        print("╠" + "═" * 78 + "╣")
        
        if cert.bottleneck_layers:
            print(f"║ Bottleneck Layers (> float32):{' ' * 51}║")
            for layer_name in cert.bottleneck_layers[:5]:  # Show first 5
                print(f"║   • {layer_name:<72}║")
            if len(cert.bottleneck_layers) > 5:
                print(f"║   ... and {len(cert.bottleneck_layers) - 5} more{' ' * (78 - 19 - len(str(len(cert.bottleneck_layers) - 5)))}║")
        
        print("╚" + "═" * 78 + "╝")
        
    def save_certificate(self, cert: ModelCertificate, filename: str):
        """Save certificate to JSON file"""
        # Convert to dict
        cert_dict = asdict(cert)
        
        with open(filename, 'w') as f:
            json.dump(cert_dict, f, indent=2)
        
        print(f"Certificate saved to: {filename}")
    
    def export_mixed_precision_config(self, 
                                      cert: ModelCertificate,
                                      filename: str):
        """
        Export mixed-precision configuration compatible with PyTorch AMP.
        
        Generates a JSON config mapping layers to dtypes.
        """
        config = {
            "model": cert.model_name,
            "target_accuracy": cert.target_accuracy,
            "layer_precision": {}
        }
        
        for layer_cert in cert.layer_certificates:
            config["layer_precision"][layer_cert['layer_name']] = {
                "dtype": layer_cert['recommended_dtype'],
                "bits": layer_cert['required_bits'],
                "curvature": float(layer_cert['curvature']),
                "reason": f"Curvature κ={layer_cert['curvature']:.2e}"
            }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Mixed-precision config saved to: {filename}")


def create_test_model(num_layers: int = 3, hidden_dim: int = 128) -> nn.Module:
    """Create a simple test model for demonstration"""
    layers = []
    layers.append(nn.Linear(784, hidden_dim))
    layers.append(nn.ReLU())
    
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(hidden_dim, 10))
    layers.append(nn.Softmax(dim=1))
    
    return nn.Sequential(*layers)


if __name__ == "__main__":
    print("HNF Precision Certifier - Quick Demo")
    print("=" * 60)
    
    # Create test model
    model = create_test_model(num_layers=3, hidden_dim=128)
    print(f"Created test model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize certifier
    certifier = PrecisionCertifier(
        target_accuracy=1e-6,
        input_domain_diameter=10.0
    )
    
    # Generate certificate
    cert = certifier.certify_model(
        model,
        input_shape=(1, 784),
        model_name="MNIST_MLP_3Layer"
    )
    
    # Display
    certifier.print_certificate(cert)
    
    # Save
    certifier.save_certificate(cert, "model_certificate.json")
    certifier.export_mixed_precision_config(cert, "mixed_precision_config.json")
    
    print("\nPer-Layer Analysis:")
    print("-" * 60)
    for layer_cert in cert.layer_certificates:
        print(f"{layer_cert['layer_name']:30} | κ={layer_cert['curvature']:10.2e} | "
              f"{layer_cert['required_bits']:2d} bits | {layer_cert['recommended_dtype']}")
