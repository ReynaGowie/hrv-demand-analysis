"""
Analyze WESAD HRV data.
Computes group summaries, statistical tests, and effect sizes.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """Load cleaned HRV data."""
    return pd.read_csv(file_path)


def compute_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute group summary statistics for each state.
    
    Args:
        df: HRV dataset
        
    Returns:
        DataFrame with summary statistics by state
    """
    metrics = ['rmssd', 'sdnn', 'mean_hr']
    
    summary = df.groupby('state').agg({
        'rmssd': ['count', 'mean', 'std'],
        'sdnn': ['mean', 'std'],
        'mean_hr': ['mean', 'std']
    })
    
    # Flatten multi-index columns
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    
    return summary


def perform_t_tests(df: pd.DataFrame, metric: str, group1: str, group2: str) -> Dict:
    """
    Perform t-test between two groups for a given metric.
    Uses non-parametric test if assumptions are violated.
    
    Args:
        df: HRV dataset
        metric: Metric to test
        group1: First state
        group2: Second state
        
    Returns:
        Dictionary with test results
    """
    # Extract data for each group
    data1 = df[df['state'] == group1][metric].dropna()
    data2 = df[df['state'] == group2][metric].dropna()
    
    # Check normality using Shapiro-Wilk test
    _, p1 = stats.shapiro(data1)
    _, p2 = stats.shapiro(data2)
    
    # Use Mann-Whitney U if normality is violated (p < 0.05)
    if p1 < 0.05 or p2 < 0.05:
        # Non-parametric test
        stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        test_type = 'Mann-Whitney U'
    else:
        # Check homogeneity of variances using Levene's test
        _, p_levene = stats.levene(data1, data2)
        
        if p_levene < 0.05:
            # Use Welch's t-test (unequal variances)
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            test_type = "Welch's t-test"
        else:
            # Use standard t-test (equal variances)
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)
            test_type = "Student's t-test"
    
    return {
        'metric': metric,
        'group1': group1,
        'group2': group2,
        'test_type': test_type,
        'statistic': stat,
        'p_value': p_value,
        'mean_diff': data1.mean() - data2.mean(),
        'n1': len(data1),
        'n2': len(data2)
    }


def perform_mann_whitney_robustness(df: pd.DataFrame, metric: str, 
                                     group1: str, group2: str) -> Dict:
    """
    Perform Mann-Whitney U test as robustness check.
    
    Args:
        df: HRV dataset
        metric: Metric to test
        group1: First state
        group2: Second state
        
    Returns:
        Dictionary with test results
    """
    data1 = df[df['state'] == group1][metric].dropna()
    data2 = df[df['state'] == group2][metric].dropna()
    
    stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    
    return {
        'metric': metric,
        'group1': group1,
        'group2': group2,
        'test_type': 'Mann-Whitney U (robustness)',
        'statistic': stat,
        'p_value': p_value,
        'mean_diff': data1.mean() - data2.mean(),
        'n1': len(data1),
        'n2': len(data2)
    }


def compute_cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        data1: First group data
        data2: Second group data
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(data1) - np.mean(data2)) / pooled_std
    
    return d


def run_all_tests(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all statistical tests and effect size calculations.
    
    Args:
        df: HRV dataset
        
    Returns:
        Tuple of (test_results, effect_sizes)
    """
    comparisons = [
        ('stress', 'baseline'),
        ('stress', 'amusement'),
        ('baseline', 'amusement')
    ]
    
    metrics = ['rmssd', 'sdnn', 'mean_hr']
    
    test_results = []
    effect_sizes = []
    
    for group1, group2 in comparisons:
        for metric in metrics:
            # Perform primary test (t-test or Mann-Whitney based on assumptions)
            test_result = perform_t_tests(df, metric, group1, group2)
            test_results.append(test_result)
            
            # Perform Mann-Whitney U as robustness check
            robust_result = perform_mann_whitney_robustness(df, metric, group1, group2)
            test_results.append(robust_result)
            
            # Compute Cohen's d
            data1 = df[df['state'] == group1][metric].dropna()
            data2 = df[df['state'] == group2][metric].dropna()
            
            d = compute_cohens_d(data1, data2)
            
            effect_sizes.append({
                'metric': metric,
                'group1': group1,
                'group2': group2,
                'cohens_d': d
            })
    
    return pd.DataFrame(test_results), pd.DataFrame(effect_sizes)


def save_results(summary: pd.DataFrame, test_results: pd.DataFrame, 
                effect_sizes: pd.DataFrame, output_path: str):
    """
    Save all analysis results to CSV.
    
    Args:
        summary: Group summary statistics
        test_results: Statistical test results
        effect_sizes: Effect size results
        output_path: Path to save results
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write group summary
        f.write("=== GROUP SUMMARY ===\n")
        summary.to_csv(f, index=False)
        f.write("\n\n")
        
        # Write test results
        f.write("=== STATISTICAL TESTS ===\n")
        test_results.to_csv(f, index=False)
        f.write("\n\n")
        
        # Write effect sizes
        f.write("=== EFFECT SIZES (Cohen's d) ===\n")
        effect_sizes.to_csv(f, index=False)
    
    print(f"Saved results to {output_path}")


def save_group_summary(summary: pd.DataFrame, output_path: str):
    """Save group summary to separate CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Saved group summary to {output_path}")


def save_pairwise_results(test_results: pd.DataFrame, effect_sizes: pd.DataFrame, 
                         output_path: str):
    """Save pairwise test results to separate CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=== STATISTICAL TESTS ===\n")
        test_results.to_csv(f, index=False)
        f.write("\n\n")
        f.write("=== EFFECT SIZES (Cohen's d) ===\n")
        effect_sizes.to_csv(f, index=False)
    
    print(f"Saved pairwise results to {output_path}")


def interpret_results(summary: pd.DataFrame, test_results: pd.DataFrame, 
                      effect_sizes: pd.DataFrame):
    """
    Provide interpretation of results.
    
    Args:
        summary: Group summary statistics
        test_results: Statistical test results
        effect_sizes: Effect size results
    """
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Extract key values
    stress_rmssd = summary[summary['state'] == 'stress']['rmssd_mean'].values[0]
    baseline_rmssd = summary[summary['state'] == 'baseline']['rmssd_mean'].values[0]
    amusement_rmssd = summary[summary['state'] == 'amusement']['rmssd_mean'].values[0]
    
    stress_sdnn = summary[summary['state'] == 'stress']['sdnn_mean'].values[0]
    baseline_sdnn = summary[summary['state'] == 'baseline']['sdnn_mean'].values[0]
    amusement_sdnn = summary[summary['state'] == 'amusement']['sdnn_mean'].values[0]
    
    stress_hr = summary[summary['state'] == 'stress']['mean_hr_mean'].values[0]
    baseline_hr = summary[summary['state'] == 'baseline']['mean_hr_mean'].values[0]
    amusement_hr = summary[summary['state'] == 'amusement']['mean_hr_mean'].values[0]
    
    print("\nHRV Values by State:")
    print(f"  STRESS:    RMSSD={stress_rmssd:.4f}, SDNN={stress_sdnn:.4f}, HR={stress_hr:.2f}")
    print(f"  BASELINE:  RMSSD={baseline_rmssd:.4f}, SDNN={baseline_sdnn:.4f}, HR={baseline_hr:.2f}")
    print(f"  AMUSEMENT: RMSSD={amusement_rmssd:.4f}, SDNN={amusement_sdnn:.4f}, HR={amusement_hr:.2f}")
    
    print("\n" + "-"*70)
    print("1. Does HRV clearly distinguish stress from non-stress states?")
    print("-"*70)
    
    # Check stress vs baseline significance
    stress_baseline_rmssd = test_results[
        (test_results['group1'] == 'stress') & 
        (test_results['group2'] == 'baseline') & 
        (test_results['metric'] == 'rmssd') &
        (test_results['test_type'] != 'Mann-Whitney U (robustness)')
    ]
    
    stress_baseline_sdnn = test_results[
        (test_results['group1'] == 'stress') & 
        (test_results['group2'] == 'baseline') & 
        (test_results['metric'] == 'sdnn') &
        (test_results['test_type'] != 'Mann-Whitney U (robustness)')
    ]
    
    stress_baseline_hr = test_results[
        (test_results['group1'] == 'stress') & 
        (test_results['group2'] == 'baseline') & 
        (test_results['metric'] == 'mean_hr') &
        (test_results['test_type'] != 'Mann-Whitney U (robustness)')
    ]
    
    if len(stress_baseline_rmssd) > 0:
        p_rmssd = stress_baseline_rmssd['p_value'].values[0]
        p_sdnn = stress_baseline_sdnn['p_value'].values[0]
        p_hr = stress_baseline_hr['p_value'].values[0]
        
        significant_count = sum([p_rmssd < 0.05, p_sdnn < 0.05, p_hr < 0.05])
        
        if significant_count >= 2:
            print(f"  YES: {significant_count}/3 metrics show significant differences (p < 0.05).")
            print(f"    RMSSD: p={p_rmssd:.4f}")
            print(f"    SDNN: p={p_sdnn:.4f}")
            print(f"    HR: p={p_hr:.4f}")
        else:
            print(f"  PARTIAL: {significant_count}/3 metrics show significant differences (p < 0.05).")
    
    print("\n" + "-"*70)
    print("2. Is the effect stronger than in the PhysioNet dataset?")
    print("-"*70)
    
    # Get effect sizes for stress vs baseline
    stress_baseline_d_rmssd = effect_sizes[
        (effect_sizes['group1'] == 'stress') & 
        (effect_sizes['group2'] == 'baseline') & 
        (effect_sizes['metric'] == 'rmssd')
    ]['cohens_d'].values[0]
    
    stress_baseline_d_sdnn = effect_sizes[
        (effect_sizes['group1'] == 'stress') & 
        (effect_sizes['group2'] == 'baseline') & 
        (effect_sizes['metric'] == 'sdnn')
    ]['cohens_d'].values[0]
    
    print(f"  WESAD stress vs baseline effect sizes:")
    print(f"    RMSSD: Cohen's d = {abs(stress_baseline_d_rmssd):.3f}")
    print(f"    SDNN: Cohen's d = {abs(stress_baseline_d_sdnn):.3f}")
    print(f"  Note: Comparison with PhysioNet requires loading PhysioNet results.")
    print(f"  Based on typical effect sizes, large effects (d > 0.8) are strong.")
    
    if abs(stress_baseline_d_rmssd) > 0.8 or abs(stress_baseline_d_sdnn) > 0.8:
        print(f"  WESAD shows at least one large effect size.")
    elif abs(stress_baseline_d_rmssd) > 0.5 or abs(stress_baseline_d_sdnn) > 0.5:
        print(f"  WESAD shows medium effect sizes.")
    else:
        print(f"  WESAD shows small effect sizes.")
    
    print("\n" + "-"*70)
    print("3. Which metric appears most sensitive?")
    print("-"*70)
    
    # Compare effect sizes across metrics
    metrics_sensitivity = []
    for metric in ['rmssd', 'sdnn', 'mean_hr']:
        d_stress_baseline = effect_sizes[
            (effect_sizes['group1'] == 'stress') & 
            (effect_sizes['group2'] == 'baseline') & 
            (effect_sizes['metric'] == metric)
        ]['cohens_d'].values[0]
        
        d_stress_amusement = effect_sizes[
            (effect_sizes['group1'] == 'stress') & 
            (effect_sizes['group2'] == 'amusement') & 
            (effect_sizes['metric'] == metric)
        ]['cohens_d'].values[0]
        
        avg_d = (abs(d_stress_baseline) + abs(d_stress_amusement)) / 2
        metrics_sensitivity.append((metric, avg_d))
    
    metrics_sensitivity.sort(key=lambda x: x[1], reverse=True)
    
    for metric, avg_d in metrics_sensitivity:
        print(f"  {metric.upper()}: average Cohen's d = {avg_d:.3f}")
    
    most_sensitive = metrics_sensitivity[0][0]
    print(f"\n  Most sensitive metric: {most_sensitive.upper()}")
    
    print("\n" + "-"*70)
    print("4. Does amusement look more like recovery than stress?")
    print("-"*70)
    
    # Compare amusement to baseline vs stress
    # Handle both orderings of group comparisons
    amusement_baseline_d_rmssd = effect_sizes[
        (((effect_sizes['group1'] == 'amusement') & (effect_sizes['group2'] == 'baseline')) |
         ((effect_sizes['group1'] == 'baseline') & (effect_sizes['group2'] == 'amusement'))) & 
        (effect_sizes['metric'] == 'rmssd')
    ]['cohens_d'].values[0]
    
    amusement_stress_d_rmssd = effect_sizes[
        (((effect_sizes['group1'] == 'amusement') & (effect_sizes['group2'] == 'stress')) |
         ((effect_sizes['group1'] == 'stress') & (effect_sizes['group2'] == 'amusement'))) & 
        (effect_sizes['metric'] == 'rmssd')
    ]['cohens_d'].values[0]
    
    amusement_hr = summary[summary['state'] == 'amusement']['mean_hr_mean'].values[0]
    
    print(f"  Amusement HRV (RMSSD={amusement_rmssd:.4f}) vs Baseline (RMSSD={baseline_rmssd:.4f})")
    print(f"    Cohen's d = {abs(amusement_baseline_d_rmssd):.3f}")
    print(f"  Amusement HRV (RMSSD={amusement_rmssd:.4f}) vs Stress (RMSSD={stress_rmssd:.4f})")
    print(f"    Cohen's d = {abs(amusement_stress_d_rmssd):.3f}")
    print(f"  Amusement HR ({amusement_hr:.2f}) vs Baseline HR ({baseline_hr:.2f})")
    print(f"  Amusement HR ({amusement_hr:.2f}) vs Stress HR ({stress_hr:.2f})")
    
    if abs(amusement_stress_d_rmssd) > abs(amusement_baseline_d_rmssd):
        print(f"\n  Amusement differs more from stress than from baseline.")
        print(f"  This suggests amusement may be more like recovery (baseline) than stress.")
    else:
        print(f"\n  Amusement differs more from baseline than from stress.")
        print(f"  This pattern is less consistent with a pure recovery state.")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data' / 'cleaned' / 'wesad_hrv.csv'
    group_summary_path = project_root / 'outputs' / 'tables' / 'wesad_group_summary.csv'
    pairwise_results_path = project_root / 'outputs' / 'tables' / 'wesad_pairwise_results.csv'
    
    print("Loading data...")
    df = load_data(str(input_path))
    print(f"Loaded {len(df)} observations")
    
    print("\nComputing group summary...")
    summary = compute_group_summary(df)
    print(summary)
    
    print("\nRunning statistical tests...")
    test_results, effect_sizes = run_all_tests(df)
    print("Primary test results:")
    print(test_results[test_results['test_type'] != 'Mann-Whitney U (robustness)'])
    print("\nEffect sizes:")
    print(effect_sizes)
    
    print("\nSaving results...")
    save_group_summary(summary, str(group_summary_path))
    save_pairwise_results(test_results, effect_sizes, str(pairwise_results_path))
    
    print("\nInterpreting results...")
    interpret_results(summary, test_results, effect_sizes)


if __name__ == '__main__':
    main()
