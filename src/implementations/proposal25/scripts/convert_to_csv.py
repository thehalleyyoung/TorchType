#!/usr/bin/env python3.11
"""
Convert experiment JSON results to CSV format for easy analysis and plotting.
"""

import json
import csv
import numpy as np
from pathlib import Path


def json_to_csv_experiment1(data_dir: Path, output_dir: Path):
    """Convert Experiment 1 data to CSV."""
    with open(data_dir / 'experiment1' / 'experiment1_precision_vs_fairness.json', 'r') as f:
        data = json.load(f)
    
    # Create CSV for main results
    csv_path = output_dir / 'exp1_precision_fairness.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'precision', 'dpg', 'error_bound', 'reliable', 
                        'reliability_score', 'near_threshold_frac', 'machine_epsilon'])
        
        for dataset in data['datasets']:
            dataset_name = dataset['name']
            for prec_name, prec_data in dataset['results']['precisions'].items():
                writer.writerow([
                    dataset_name,
                    prec_name,
                    prec_data['dpg'],
                    prec_data['error_bound'],
                    1 if prec_data['is_reliable'] else 0,
                    prec_data['reliability_score'],
                    prec_data['near_threshold_fraction']['overall'],
                    prec_data['machine_epsilon']
                ])
    
    print(f"Created: {csv_path}")
    
    # Create CSV for summary
    csv_path_summary = output_dir / 'exp1_summary.csv'
    with open(csv_path_summary, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['precision', 'total', 'borderline', 'borderline_pct'])
        
        for prec, stats in data['summary']['reliable_by_precision'].items():
            writer.writerow([
                prec,
                stats['total'],
                stats['borderline'],
                stats['borderline_pct']
            ])
    
    print(f"Created: {csv_path_summary}")


def json_to_csv_experiment2(data_dir: Path, output_dir: Path):
    """Convert Experiment 2 data to CSV."""
    with open(data_dir / 'experiment2' / 'experiment2_near_threshold_distribution.json', 'r') as f:
        data = json.load(f)
    
    # Create CSV for each model's predictions
    for model_idx, model_data in enumerate(data['models']):
        config_name = model_data['config_name']
        
        # Group 0 predictions
        csv_path = output_dir / f'exp2_{config_name}_group0_predictions.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prediction'])
            for pred in model_data['group0']['predictions_raw']:
                writer.writerow([pred])
        
        # Group 1 predictions
        csv_path = output_dir / f'exp2_{config_name}_group1_predictions.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prediction'])
            for pred in model_data['group1']['predictions_raw']:
                writer.writerow([pred])
        
        print(f"Created: exp2_{config_name}_*.csv")


def json_to_csv_experiment3(data_dir: Path, output_dir: Path):
    """Convert Experiment 3 data to CSV."""
    with open(data_dir / 'experiment3' / 'experiment3_threshold_stability.json', 'r') as f:
        data = json.load(f)
    
    for model_data in data['models']:
        model_name = model_data['model_name']
        
        csv_path = output_dir / f'exp3_{model_name}_threshold_stability.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['threshold', 'dpg', 'error_bound', 'reliability_score', 
                           'is_reliable', 'is_stable'])
            
            for i in range(len(model_data['thresholds'])):
                writer.writerow([
                    model_data['thresholds'][i],
                    model_data['dpg_values'][i],
                    model_data['error_bounds'][i],
                    model_data['reliability_scores'][i],
                    1 if model_data['is_reliable'][i] else 0,
                    1 if model_data['stable_regions'][i] else 0
                ])
        
        print(f"Created: {csv_path}")


def json_to_csv_experiment4(data_dir: Path, output_dir: Path):
    """Convert Experiment 4 data to CSV."""
    with open(data_dir / 'experiment4' / 'experiment4_calibration_reliability.json', 'r') as f:
        data = json.load(f)
    
    for model_data in data['models']:
        dataset_name = model_data['dataset']
        
        for prec_name, prec_data in model_data['precision_results'].items():
            csv_path = output_dir / f'exp4_{dataset_name}_{prec_name}_calibration.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['bin', 'confidence', 'accuracy', 'uncertainty', 'reliable'])
                
                for i in range(len(prec_data['bin_confidences'])):
                    if prec_data['bin_confidences'][i] > 0:  # Skip empty bins
                        writer.writerow([
                            i,
                            prec_data['bin_confidences'][i],
                            prec_data['bin_accuracies'][i],
                            prec_data['bin_uncertainties'][i],
                            1 if prec_data['reliable_bins'][i] else 0
                        ])
            
            print(f"Created: {csv_path}")


def json_to_csv_experiment5(data_dir: Path, output_dir: Path):
    """Convert Experiment 5 data to CSV."""
    with open(data_dir / 'experiment5' / 'experiment5_sign_flip_cases.json', 'r') as f:
        data = json.load(f)
    
    csv_path = output_dir / 'exp5_sign_flips.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'has_sign_flip', 'dpg_float64', 'dpg_float32', 'dpg_float16',
                        'error_bound_64', 'error_bound_32', 'error_bound_16'])
        
        for trial_data in data['sign_flips']:
            writer.writerow([
                trial_data['trial'],
                1 if trial_data['has_sign_flip'] else 0,
                trial_data['dpg_signed']['float64'],
                trial_data['dpg_signed']['float32'],
                trial_data['dpg_signed']['float16'],
                trial_data['error_bounds']['float64'],
                trial_data['error_bounds']['float32'],
                trial_data['error_bounds']['float16']
            ])
    
    print(f"Created: {csv_path}")


def main():
    """Convert all experiment JSONs to CSVs."""
    base_dir = Path(__file__).parent.parent / 'data'
    output_dir = base_dir / 'csv'
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("CONVERTING EXPERIMENT DATA TO CSV")
    print("="*70)
    
    try:
        json_to_csv_experiment1(base_dir, output_dir)
    except Exception as e:
        print(f"Warning: Could not convert Experiment 1: {e}")
    
    try:
        json_to_csv_experiment2(base_dir, output_dir)
    except Exception as e:
        print(f"Warning: Could not convert Experiment 2: {e}")
    
    try:
        json_to_csv_experiment3(base_dir, output_dir)
    except Exception as e:
        print(f"Warning: Could not convert Experiment 3: {e}")
    
    try:
        json_to_csv_experiment4(base_dir, output_dir)
    except Exception as e:
        print(f"Warning: Could not convert Experiment 4: {e}")
    
    try:
        json_to_csv_experiment5(base_dir, output_dir)
    except Exception as e:
        print(f"Warning: Could not convert Experiment 5: {e}")
    
    print("\n" + "="*70)
    print("CSV CONVERSION COMPLETE")
    print("="*70)
    print(f"\nCSV files saved to: {output_dir}")


if __name__ == '__main__':
    main()
