"""
Compare HRV patterns across PhysioNet and WESAD datasets.
Standardizes metrics and evaluates stress response consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_datasets():
    """Load both cleaned HRV datasets."""
    project_root = Path(__file__).parent.parent
    
    physionet_path = project_root / 'data' / 'cleaned' / 'physionet_hrv.csv'
    wesad_path = project_root / 'data' / 'cleaned' / 'wesad_hrv.csv'
    
    physionet_df = pd.read_csv(physionet_path)
    wesad_df = pd.read_csv(wesad_path)
    
    return physionet_df, wesad_df


def standardize_metrics(df, metrics=['rmssd', 'sdnn']):
    """
    Z-score normalize metrics within each dataset.
    
    Args:
        df: DataFrame with HRV metrics
        metrics: List of metric columns to standardize
        
    Returns:
        DataFrame with standardized metrics added
    """
    df_std = df.copy()
    
    for metric in metrics:
        if metric in df_std.columns:
            mean_val = df_std[metric].mean()
            std_val = df_std[metric].std()
            if std_val > 0:
                df_std[f'{metric}_z'] = (df_std[metric] - mean_val) / std_val
            else:
                df_std[f'{metric}_z'] = 0
    
    return df_std


def compare_direction(physionet_df, wesad_df):
    """
    Check if HRV decreases in stress in both datasets.
    
    Returns:
        Dictionary with direction comparison results
    """
    results = {}
    
    # PhysioNet: STRESS vs baseline (AEROBIC as baseline)
    physionet_stress = physionet_df[physionet_df['workload_label'] == 'STRESS']
    physionet_baseline = physionet_df[physionet_df['workload_label'] == 'AEROBIC']
    
    results['physionet'] = {
        'stress_mean_rmssd': physionet_stress['rmssd'].mean(),
        'baseline_mean_rmssd': physionet_baseline['rmssd'].mean(),
        'stress_mean_sdnn': physionet_stress['sdnn'].mean(),
        'baseline_mean_sdnn': physionet_baseline['sdnn'].mean(),
        'stress_mean_hr': physionet_stress['mean_hr'].mean(),
        'baseline_mean_hr': physionet_baseline['mean_hr'].mean(),
        'hrv_decreases_in_stress': physionet_stress['rmssd'].mean() < physionet_baseline['rmssd'].mean(),
        'hr_increases_in_stress': physionet_stress['mean_hr'].mean() > physionet_baseline['mean_hr'].mean()
    }
    
    # WESAD: stress vs baseline
    wesad_stress = wesad_df[wesad_df['state'] == 'stress']
    wesad_baseline = wesad_df[wesad_df['state'] == 'baseline']
    
    results['wesad'] = {
        'stress_mean_rmssd': wesad_stress['rmssd'].mean(),
        'baseline_mean_rmssd': wesad_baseline['rmssd'].mean(),
        'stress_mean_sdnn': wesad_stress['sdnn'].mean(),
        'baseline_mean_sdnn': wesad_baseline['sdnn'].mean(),
        'stress_mean_hr': wesad_stress['mean_hr'].mean(),
        'baseline_mean_hr': wesad_baseline['mean_hr'].mean(),
        'hrv_decreases_in_stress': wesad_stress['rmssd'].mean() < wesad_baseline['rmssd'].mean(),
        'hr_increases_in_stress': wesad_stress['mean_hr'].mean() > wesad_baseline['mean_hr'].mean()
    }
    
    # Consistency check
    results['consistent_direction'] = (
        results['physionet']['hrv_decreases_in_stress'] == results['wesad']['hrv_decreases_in_stress']
    )
    
    return results


def compute_effect_size(group1, group2, metric):
    """
    Compute Cohen's d effect size between two groups.
    
    Args:
        group1: First group values
        group2: Second group values
        metric: Metric name
        
    Returns:
        Effect size (Cohen's d)
    """
    n1 = len(group1)
    n2 = len(group2)
    
    if n1 < 2 or n2 < 2:
        return np.nan
    
    mean1 = group1.mean()
    mean2 = group2.mean()
    
    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d


def compare_metrics(physionet_df, wesad_df):
    """
    Compare HR vs HRV effectiveness for stress detection.
    
    Returns:
        Dictionary with metric comparison results
    """
    results = {}
    
    # PhysioNet effect sizes
    physionet_stress = physionet_df[physionet_df['workload_label'] == 'STRESS']
    physionet_baseline = physionet_df[physionet_df['workload_label'] == 'AEROBIC']
    
    results['physionet'] = {
        'rmssd_effect_size': compute_effect_size(physionet_baseline['rmssd'], physionet_stress['rmssd'], 'rmssd'),
        'sdnn_effect_size': compute_effect_size(physionet_baseline['sdnn'], physionet_stress['sdnn'], 'sdnn'),
        'hr_effect_size': compute_effect_size(physionet_baseline['mean_hr'], physionet_stress['mean_hr'], 'mean_hr')
    }
    
    # WESAD effect sizes
    wesad_stress = wesad_df[wesad_df['state'] == 'stress']
    wesad_baseline = wesad_df[wesad_df['state'] == 'baseline']
    
    results['wesad'] = {
        'rmssd_effect_size': compute_effect_size(wesad_baseline['rmssd'], wesad_stress['rmssd'], 'rmssd'),
        'sdnn_effect_size': compute_effect_size(wesad_baseline['sdnn'], wesad_stress['sdnn'], 'sdnn'),
        'hr_effect_size': compute_effect_size(wesad_baseline['mean_hr'], wesad_stress['mean_hr'], 'mean_hr')
    }
    
    return results


def interpret_results(direction_results, metric_results):
    """
    Provide final interpretation of comparison results.
    
    Returns:
        Dictionary with interpretation
    """
    interpretation = {}
    
    # Which dataset shows clearer separation?
    physionet_max_effect = max(
        abs(metric_results['physionet']['rmssd_effect_size']),
        abs(metric_results['physionet']['sdnn_effect_size']),
        abs(metric_results['physionet']['hr_effect_size'])
    )
    
    wesad_max_effect = max(
        abs(metric_results['wesad']['rmssd_effect_size']),
        abs(metric_results['wesad']['sdnn_effect_size']),
        abs(metric_results['wesad']['hr_effect_size'])
    )
    
    if physionet_max_effect > wesad_max_effect:
        interpretation['clearest_separation'] = 'PhysioNet'
        interpretation['separation_magnitude'] = f"PhysioNet: {physionet_max_effect:.3f} vs WESAD: {wesad_max_effect:.3f}"
    else:
        interpretation['clearest_separation'] = 'WESAD'
        interpretation['separation_magnitude'] = f"WESAD: {wesad_max_effect:.3f} vs PhysioNet: {physionet_max_effect:.3f}"
    
    # Which metric is most reliable overall?
    physionet_best_metric = max(
        ('RMSSD', abs(metric_results['physionet']['rmssd_effect_size'])),
        ('SDNN', abs(metric_results['physionet']['sdnn_effect_size'])),
        ('HR', abs(metric_results['physionet']['hr_effect_size'])),
        key=lambda x: x[1]
    )
    
    wesad_best_metric = max(
        ('RMSSD', abs(metric_results['wesad']['rmssd_effect_size'])),
        ('SDNN', abs(metric_results['wesad']['sdnn_effect_size'])),
        ('HR', abs(metric_results['wesad']['hr_effect_size'])),
        key=lambda x: x[1]
    )
    
    interpretation['physionet_best_metric'] = f"{physionet_best_metric[0]} (d={physionet_best_metric[1]:.3f})"
    interpretation['wesad_best_metric'] = f"{wesad_best_metric[0]} (d={wesad_best_metric[1]:.3f})"
    
    # Check consistency across datasets
    rmssd_consistent = (
        metric_results['physionet']['rmssd_effect_size'] < 0 and
        metric_results['wesad']['rmssd_effect_size'] < 0
    ) or (
        metric_results['physionet']['rmssd_effect_size'] > 0 and
        metric_results['wesad']['rmssd_effect_size'] > 0
    )
    
    hr_consistent = (
        metric_results['physionet']['hr_effect_size'] < 0 and
        metric_results['wesad']['hr_effect_size'] < 0
    ) or (
        metric_results['physionet']['hr_effect_size'] > 0 and
        metric_results['wesad']['hr_effect_size'] > 0
    )
    
    interpretation['rmssd_direction_consistent'] = rmssd_consistent
    interpretation['hr_direction_consistent'] = hr_consistent
    interpretation['overall_consistency'] = direction_results['consistent_direction']
    
    return interpretation


def print_results(direction_results, metric_results, interpretation):
    """Print formatted comparison results."""
    print("\n" + "="*70)
    print("HRV DATASET COMPARISON")
    print("="*70)
    
    print("\n--- STEP 1: DIRECTION COMPARISON ---")
    print("\nPhysioNet (STRESS vs AEROBIC baseline):")
    print(f"  RMSSD: {direction_results['physionet']['stress_mean_rmssd']:.2f} vs {direction_results['physionet']['baseline_mean_rmssd']:.2f}")
    print(f"  SDNN: {direction_results['physionet']['stress_mean_sdnn']:.2f} vs {direction_results['physionet']['baseline_mean_sdnn']:.2f}")
    print(f"  HR: {direction_results['physionet']['stress_mean_hr']:.2f} vs {direction_results['physionet']['baseline_mean_hr']:.2f}")
    print(f"  HRV decreases in stress: {direction_results['physionet']['hrv_decreases_in_stress']}")
    print(f"  HR increases in stress: {direction_results['physionet']['hr_increases_in_stress']}")
    
    print("\nWESAD (stress vs baseline):")
    print(f"  RMSSD: {direction_results['wesad']['stress_mean_rmssd']:.2f} vs {direction_results['wesad']['baseline_mean_rmssd']:.2f}")
    print(f"  SDNN: {direction_results['wesad']['stress_mean_sdnn']:.2f} vs {direction_results['wesad']['baseline_mean_sdnn']:.2f}")
    print(f"  HR: {direction_results['wesad']['stress_mean_hr']:.2f} vs {direction_results['wesad']['baseline_mean_hr']:.2f}")
    print(f"  HRV decreases in stress: {direction_results['wesad']['hrv_decreases_in_stress']}")
    print(f"  HR increases in stress: {direction_results['wesad']['hr_increases_in_stress']}")
    
    print(f"\nDirection consistent across datasets: {direction_results['consistent_direction']}")
    
    print("\n--- STEP 2: METRIC EFFECT SIZES (Cohen's d) ---")
    print("\nPhysioNet:")
    print(f"  RMSSD: d = {metric_results['physionet']['rmssd_effect_size']:.3f}")
    print(f"  SDNN: d = {metric_results['physionet']['sdnn_effect_size']:.3f}")
    print(f"  HR: d = {metric_results['physionet']['hr_effect_size']:.3f}")
    
    print("\nWESAD:")
    print(f"  RMSSD: d = {metric_results['wesad']['rmssd_effect_size']:.3f}")
    print(f"  SDNN: d = {metric_results['wesad']['sdnn_effect_size']:.3f}")
    print(f"  HR: d = {metric_results['wesad']['hr_effect_size']:.3f}")
    
    print("\n--- STEP 3: FINAL INTERPRETATION ---")
    print(f"\nClearer separation: {interpretation['clearest_separation']}")
    print(f"  ({interpretation['separation_magnitude']})")
    
    print(f"\nBest metric by dataset:")
    print(f"  PhysioNet: {interpretation['physionet_best_metric']}")
    print(f"  WESAD: {interpretation['wesad_best_metric']}")
    
    print(f"\nConsistency across contexts:")
    print(f"  RMSSD direction consistent: {interpretation['rmssd_direction_consistent']}")
    print(f"  HR direction consistent: {interpretation['hr_direction_consistent']}")
    print(f"  Overall consistency: {interpretation['overall_consistency']}")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    print("Loading datasets...")
    physionet_df, wesad_df = load_datasets()
    
    print(f"PhysioNet: {len(physionet_df)} observations")
    print(f"WESAD: {len(wesad_df)} observations")
    
    # Step 1: Standardize metrics
    print("\nStandardizing metrics...")
    physionet_std = standardize_metrics(physionet_df)
    wesad_std = standardize_metrics(wesad_df)
    
    # Step 2: Compare direction
    print("Comparing direction...")
    direction_results = compare_direction(physionet_df, wesad_df)
    
    # Step 3: Compare metrics
    print("Comparing metrics...")
    metric_results = compare_metrics(physionet_df, wesad_df)
    
    # Step 4: Interpret results
    print("Interpreting results...")
    interpretation = interpret_results(direction_results, metric_results)
    
    # Print results
    print_results(direction_results, metric_results, interpretation)


if __name__ == '__main__':
    main()
