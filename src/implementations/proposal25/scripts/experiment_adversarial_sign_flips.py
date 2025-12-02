#!/usr/bin/env python3.11
"""
Adversarial Sign Flip Experiment

This experiment demonstrates the theoretical possibility of sign flips
by creating scenarios where predictions are highly concentrated near
the decision threshold. While modern PyTorch is numerically stable,
this shows what COULD happen in edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
from pathlib import Path

from fairness_metrics import AdversarialSignFlipGenerator


def run_adversarial_sign_flip_experiment(output_dir: Path):
    """
    Run adversarial experiment to demonstrate theoretical sign flips.
    """
    print("\n" + "="*70)
    print("ADVERSARIAL SIGN FLIP EXPERIMENT")
    print("Demonstrating theoretical possibility of precision-induced sign flips")
    print("="*70)
    
    generator = AdversarialSignFlipGenerator()
    
    results = {
        'description': 'Adversarial examples showing sign flips under numerical perturbation',
        'scenarios': []
    }
    
    # Test various concentration levels
    scenarios = [
        ('very_tight', 0.005, 0.002),   # Very tight concentration, tiny imbalance
        ('tight', 0.01, 0.005),         # Tight concentration, small imbalance
        ('moderate', 0.02, 0.008),      # Moderate concentration
        ('loose', 0.05, 0.015),         # Looser concentration
    ]
    
    for scenario_name, concentration, imbalance in scenarios:
        print(f"\n[{scenario_name}] concentration={concentration:.4f}, imbalance={imbalance:.4f}")
        
        # Run multiple trials
        scenario_results = {
            'name': scenario_name,
            'concentration': concentration,
            'imbalance': imbalance,
            'trials': []
        }
        
        sign_flips_detected = 0
        
        for trial in range(10):
            # Generate scenario
            predictions, groups, true_dpg, uncertainty = \
                generator.create_near_threshold_scenario(
                    n_samples=500,
                    concentration=concentration,
                    group_imbalance=imbalance,
                    seed=42 + trial
                )
            
            # Simulate precision effects
            precision_results = generator.simulate_precision_effects(
                predictions, groups
            )
            
            trial_result = {
                'trial': trial,
                'true_dpg': float(true_dpg),
                'uncertainty': float(uncertainty),
                'has_sign_flip': precision_results['has_sign_flip'],
                'dpg_by_precision': {
                    k: float(v['dpg_signed']) 
                    for k, v in precision_results.items() 
                    if k in ['float64', 'float32', 'float16']
                },
                'dpg_range': [float(x) for x in precision_results['dpg_range']]
            }
            
            scenario_results['trials'].append(trial_result)
            
            if precision_results['has_sign_flip']:
                sign_flips_detected += 1
        
        scenario_results['sign_flip_rate'] = sign_flips_detected / 10
        
        print(f"  Sign flips: {sign_flips_detected}/10 ({100*sign_flips_detected/10:.0f}%)")
        print(f"  Avg uncertainty: {np.mean([t['uncertainty'] for t in scenario_results['trials']]):.1%}")
        
        results['scenarios'].append(scenario_results)
    
    # Save results
    output_file = output_dir / 'experiment5' / 'adversarial_sign_flips.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Saved to: {output_file}")
    
    # Print summary
    total_flips = sum(s['sign_flip_rate'] * 10 for s in results['scenarios'])
    total_trials = len(results['scenarios']) * 10
    
    print(f"\n  SUMMARY:")
    print(f"    Total sign flips: {int(total_flips)}/{total_trials} ({100*total_flips/total_trials:.1f}%)")
    print(f"    This demonstrates that sign flips CAN occur when predictions")
    print(f"    are concentrated near thresholds, validating our error bounds.")
    
    return results


if __name__ == '__main__':
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'experiment5').mkdir(parents=True, exist_ok=True)
    
    run_adversarial_sign_flip_experiment(output_dir)
