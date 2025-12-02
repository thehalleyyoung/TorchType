"""
Dataset loaders and generators for fairness experiments.

Implements Adult Income subset, synthetic COMPAS-style data, and generic tabular datasets.
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


def load_adult_income(n_samples: int = 5000,
                     fairness_gap: float = 0.06,
                     seed: Optional[int] = None,
                     test_split: float = 0.2) -> Tuple:
    """
    Load Adult Income dataset subset.
    
    Since we don't have the full dataset, we'll generate synthetic data
    with similar statistical properties.
    
    Args:
        n_samples: Number of samples
        fairness_gap: Target demographic parity gap (not used currently, for API consistency)
        seed: Random seed
        test_split: Fraction for test set
    
    Returns:
        (train_data, train_labels, train_groups, test_data, test_labels, test_groups)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate synthetic tabular data similar to Adult Income
    # Features: age, education-num, hours-per-week, etc. (simplified to ~10 features)
    n_features = 10
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=7,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        flip_y=0.1,
        class_sep=0.8,
        random_state=seed
    )
    
    # Generate groups (gender: 0=female, 1=male)
    # Introduce correlation between features and group membership
    groups = np.random.binomial(1, 0.5, n_samples)
    
    # Make the classification slightly easier for one group to create fairness issues
    for i in range(n_samples):
        if groups[i] == 1 and np.random.random() < 0.15:
            # Slightly bias toward positive class for group 1
            y[i] = 1
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train/test split
    n_train = int(n_samples * (1 - test_split))
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return (
        torch.FloatTensor(X[train_idx]),
        torch.LongTensor(y[train_idx]),
        torch.LongTensor(groups[train_idx]),
        torch.FloatTensor(X[test_idx]),
        torch.LongTensor(y[test_idx]),
        torch.LongTensor(groups[test_idx])
    )


def generate_synthetic_compas(n_samples: int = 2000,
                             fairness_gap: float = 0.10,
                             seed: Optional[int] = None,
                             test_split: float = 0.2) -> Tuple:
    """
    Generate synthetic COMPAS-style recidivism prediction data.
    
    Args:
        n_samples: Number of samples
        fairness_gap: Target demographic parity gap
        seed: Random seed
        test_split: Fraction for test set
    
    Returns:
        (train_data, train_labels, train_groups, test_data, test_labels, test_groups)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Features: age, priors_count, charge_degree, etc. (simplified to ~8 features)
    n_features = 8
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        flip_y=0.15,
        class_sep=0.7,
        random_state=seed
    )
    
    # Generate groups (race: 0=non-protected, 1=protected)
    groups = np.random.binomial(1, 0.4, n_samples)
    
    # Calibrate to achieve target fairness gap
    pos_rate_0 = y[groups == 0].mean()
    pos_rate_1 = y[groups == 1].mean()
    current_gap = abs(pos_rate_0 - pos_rate_1)
    
    # Adjust to match target gap
    if current_gap < fairness_gap:
        n_to_flip = int((fairness_gap - current_gap) * n_samples / 2)
        
        if pos_rate_0 < pos_rate_1:
            candidates = np.where((groups == 0) & (y == 0))[0]
        else:
            candidates = np.where((groups == 1) & (y == 0))[0]
        
        if len(candidates) >= n_to_flip:
            flip_idx = np.random.choice(candidates, n_to_flip, replace=False)
            y[flip_idx] = 1
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train/test split
    n_train = int(n_samples * (1 - test_split))
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return (
        torch.FloatTensor(X[train_idx]),
        torch.LongTensor(y[train_idx]),
        torch.LongTensor(groups[train_idx]),
        torch.FloatTensor(X[test_idx]),
        torch.LongTensor(y[test_idx]),
        torch.LongTensor(groups[test_idx])
    )


def generate_synthetic_tabular(n_samples: int = 3000,
                               n_features: int = 12,
                               fairness_gap: float = 0.08,
                               seed: Optional[int] = None,
                               test_split: float = 0.2) -> Tuple:
    """
    Generate synthetic tabular data with controlled fairness gap.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        fairness_gap: Target demographic parity gap
        seed: Random seed
        test_split: Fraction of data for testing
    
    Returns:
        (train_data, train_labels, train_groups, test_data, test_labels, test_groups)
    """
    if seed is not None:
        np.random.seed(seed)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_repeated=0,
        n_classes=2,
        flip_y=0.12,
        class_sep=0.75,
        random_state=seed
    )
    
    # Generate groups (A vs B)
    groups = np.random.binomial(1, 0.5, n_samples)
    
    # Calibrate to achieve target fairness gap
    # Compute current positive rates
    pos_rate_0 = y[groups == 0].mean()
    pos_rate_1 = y[groups == 1].mean()
    current_gap = abs(pos_rate_0 - pos_rate_1)
    
    # Adjust to match target gap
    if current_gap < fairness_gap:
        # Need to increase gap
        n_to_flip = int((fairness_gap - current_gap) * n_samples / 2)
        
        # Flip some labels in group with lower positive rate
        if pos_rate_0 < pos_rate_1:
            # Flip negatives to positives in group 0
            candidates = np.where((groups == 0) & (y == 0))[0]
        else:
            # Flip negatives to positives in group 1
            candidates = np.where((groups == 1) & (y == 0))[0]
        
        if len(candidates) >= n_to_flip:
            flip_idx = np.random.choice(candidates, n_to_flip, replace=False)
            y[flip_idx] = 1
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train/test split
    n_train = int(n_samples * (1 - test_split))
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return (
        torch.FloatTensor(X[train_idx]),
        torch.LongTensor(y[train_idx]),
        torch.LongTensor(groups[train_idx]),
        torch.FloatTensor(X[test_idx]),
        torch.LongTensor(y[test_idx]),
        torch.LongTensor(groups[test_idx])
    )


def create_borderline_fair_dataset(n_samples: int = 2000,
                                   n_features: int = 10,
                                   target_dpg: float = 0.05,
                                   seed: Optional[int] = None) -> Tuple:
    """
    Create a dataset where a trained model will have DPG ≈ target_dpg.
    
    This creates challenging cases for numerical precision analysis.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate base data
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels with a simple decision boundary
    decision_values = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.3
    y = (decision_values > 0).astype(int)
    
    # Generate groups
    groups = np.random.binomial(1, 0.5, n_samples)
    
    # Calibrate for target DPG by adjusting positive rates
    pos_rate_target_0 = 0.5
    pos_rate_target_1 = 0.5 + target_dpg
    
    # Adjust group 1 to have higher positive rate
    n_group_1 = (groups == 1).sum()
    n_pos_needed = int(pos_rate_target_1 * n_group_1)
    n_pos_current = y[groups == 1].sum()
    
    if n_pos_current < n_pos_needed:
        # Need more positives in group 1
        candidates = np.where((groups == 1) & (y == 0))[0]
        n_to_flip = min(n_pos_needed - n_pos_current, len(candidates))
        if n_to_flip > 0:
            flip_idx = np.random.choice(candidates, n_to_flip, replace=False)
            y[flip_idx] = 1
    elif n_pos_current > n_pos_needed:
        # Need fewer positives in group 1
        candidates = np.where((groups == 1) & (y == 1))[0]
        n_to_flip = min(n_pos_current - n_pos_needed, len(candidates))
        if n_to_flip > 0:
            flip_idx = np.random.choice(candidates, n_to_flip, replace=False)
            y[flip_idx] = 0
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return (
        torch.FloatTensor(X),
        torch.LongTensor(y),
        torch.LongTensor(groups)
    )
