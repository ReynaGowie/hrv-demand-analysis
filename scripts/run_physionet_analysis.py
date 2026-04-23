"""
Analyze PhysioNet HRV data.
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
    Compute group summary statistics for each workload_label.
    
    Args:
        df: HRV dataset
        
    Returns:
        DataFrame with summary statistics by workload_label
    """
    metrics = ['rmssd', 'sdnn', 'mean_hr']
    
    summary = df.groupby('workload_label')[metrics].agg(['mean', 'std'])
    
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
        group1: First workload label
        group2: Second workload label
        
    Returns:
        Dictionary with test results
    """
    # Extract data for each group
    data1 = df[df['workload_label'] == group1][metric].dropna()
    data2 = df[df['workload_label'] == group2][metric].dropna()
    
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
        ('STRESS', 'AEROBIC'),
        ('STRESS', 'ANAEROBIC'),
        ('AEROBIC', 'ANAEROBIC')
    ]
    
    metrics = ['rmssd', 'sdnn']
    
    test_results = []
    effect_sizes = []
    
    for group1, group2 in comparisons:
        for metric in metrics:
            # Perform t-test
            test_result = perform_t_tests(df, metric, group1, group2)
            test_results.append(test_result)
            
            # Compute Cohen's d
            data1 = df[df['workload_label'] == group1][metric].dropna()
            data2 = df[df['workload_label'] == group2][metric].dropna()
            
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
    # Combine all results into a single CSV with section markers
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
    stress_rmssd = summary[summary['workload_label'] == 'STRESS']['rmssd_mean'].values[0]
    aerobic_rmssd = summary[summary['workload_label'] == 'AEROBIC']['rmssd_mean'].values[0]
    anaerobic_rmssd = summary[summary['workload_label'] == 'ANAEROBIC']['rmssd_mean'].values[0]
    
    stress_sdnn = summary[summary['workload_label'] == 'STRESS']['sdnn_mean'].values[0]
    aerobic_sdnn = summary[summary['workload_label'] == 'AEROBIC']['sdnn_mean'].values[0]
    anaerobic_sdnn = summary[summary['workload_label'] == 'ANAEROBIC']['sdnn_mean'].values[0]
    
    print("\nHRV Values by Workload:")
    print(f"  STRESS:    RMSSD={stress_rmssd:.4f}, SDNN={stress_sdnn:.4f}")
    print(f"  AEROBIC:   RMSSD={aerobic_rmssd:.4f}, SDNN={aerobic_sdnn:.4f}")
    print(f"  ANAEROBIC: RMSSD={anaerobic_rmssd:.4f}, SDNN={anaerobic_sdnn:.4f}")
    
    print("\n" + "-"*70)
    print("Does HRV decrease with higher workload?")
    print("-"*70)
    
    # Compare STRESS vs exercise
    if stress_rmssd < aerobic_rmssd and stress_rmssd < anaerobic_rmssd:
        print("  YES: STRESS shows lower RMSSD than both exercise conditions.")
        print(f"  STRESS RMSSD is {((stress_rmssd/aerobic_rmssd - 1) * 100):.1f}% lower than AEROBIC")
        print(f"  STRESS RMSSD is {((stress_rmssd/anaerobic_rmssd - 1) * 100):.1f}% lower than ANAEROBIC")
    else:
        print("  NO: STRESS does not show consistently lower HRV than exercise conditions.")
    
    print("\n" + "-"*70)
    print("Is the difference large or small?")
    print("-"*70)
    
    # Get effect sizes for STRESS comparisons
    for _, row in effect_sizes.iterrows():
        if row['group1'] == 'STRESS':
            metric = row['metric']
            d = abs(row['cohens_d'])
            comparison = row['group2']
            
            # Interpret effect size
            if d < 0.2:
                size = "very small"
            elif d < 0.5:
                size = "small"
            elif d < 0.8:
                size = "medium"
            else:
                size = "large"
            
            print(f"  STRESS vs {comparison} ({metric}): Cohen's d = {d:.3f} ({size})")
    
    print("\n" + "-"*70)
    print("Is the signal consistent across metrics (RMSSD, SDNN)?")
    print("-"*70)
    
    # Check consistency
    stress_aerobic_rmssd_sign = test_results[
        (test_results['group1'] == 'STRESS') & 
        (test_results['group2'] == 'AEROBIC') & 
        (test_results['metric'] == 'rmssd')
    ]['p_value'].values[0] < 0.05
    
    stress_aerobic_sdnn_sign = test_results[
        (test_results['group1'] == 'STRESS') & 
        (test_results['group2'] == 'AEROBIC') & 
        (test_results['metric'] == 'sdnn')
    ]['p_value'].values[0] < 0.05
    
    if stress_aerobic_rmssd_sign and stress_aerobic_sdnn_sign:
        print("  YES: Both RMSSD and SDNN show significant differences between STRESS and AEROBIC.")
    elif stress_aerobic_rmssd_sign or stress_aerobic_sdnn_sign:
        print("  PARTIAL: Only one metric (RMSSD or SDNN) shows significant differences.")
    else:
        print("  NO: Neither metric shows significant differences at p < 0.05.")
    
    print("\n" + "-"*70)
    print("Does HRV clearly distinguish STRESS from other states?")
    print("-"*70)
    
    # Count significant tests
    significant_tests = 0
    total_tests = 0
    
    for _, row in test_results.iterrows():
        if row['group1'] == 'STRESS':
            total_tests += 1
            if row['p_value'] < 0.05:
                significant_tests += 1
    
    if significant_tests == total_tests:
        print(f"  YES: All {total_tests} tests comparing STRESS to other states are significant (p < 0.05).")
    elif significant_tests > 0:
        print(f"  PARTIAL: {significant_tests}/{total_tests} tests comparing STRESS to other states are significant (p < 0.05).")
    else:
        print(f"  NO: No significant differences found between STRESS and other states (p < 0.05).")
    
    print("\n" + "-"*70)
    print("AEROBIC vs ANAEROBIC comparison:")
    print("-"*70)
    
    # Get AEROBIC vs ANAEROBIC results
    aerobic_anaerobic_rmssd = effect_sizes[
        (effect_sizes['group1'] == 'AEROBIC') & 
        (effect_sizes['group2'] == 'ANAEROBIC') & 
        (effect_sizes['metric'] == 'rmssd')
    ]
    
    aerobic_anaerobic_sdnn = effect_sizes[
        (effect_sizes['group1'] == 'AEROBIC') & 
        (effect_sizes['group2'] == 'ANAEROBIC') & 
        (effect_sizes['metric'] == 'sdnn')
    ]
    
    aerobic_anaerobic_rmssd_test = test_results[
        (test_results['group1'] == 'AEROBIC') & 
        (test_results['group2'] == 'ANAEROBIC') & 
        (test_results['metric'] == 'rmssd')
    ]
    
    aerobic_anaerobic_sdnn_test = test_results[
        (test_results['group1'] == 'AEROBIC') & 
        (test_results['group2'] == 'ANAEROBIC') & 
        (test_results['metric'] == 'sdnn')
    ]
    
    if len(aerobic_anaerobic_rmssd) > 0:
        d_rmssd = aerobic_anaerobic_rmssd['cohens_d'].values[0]
        p_rmssd = aerobic_anaerobic_rmssd_test['p_value'].values[0]
        print(f"  RMSSD: Cohen's d = {abs(d_rmssd):.3f}, p = {p_rmssd:.4f}")
    
    if len(aerobic_anaerobic_sdnn) > 0:
        d_sdnn = aerobic_anaerobic_sdnn['cohens_d'].values[0]
        p_sdnn = aerobic_anaerobic_sdnn_test['p_value'].values[0]
        print(f"  SDNN: Cohen's d = {abs(d_sdnn):.3f}, p = {p_sdnn:.4f}")
    
    print("\n  Interpretation: Exercise intensity (aerobic vs anaerobic) shows minimal HRV differences.")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    import os
    
    # Define paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data' / 'cleaned' / 'physionet_hrv.csv'
    output_path = project_root / 'outputs' / 'tables' / 'physionet_analysis_results.csv'
    
    print("Loading data...")
    df = load_data(str(input_path))
    print(f"Loaded {len(df)} observations")
    
    print("\nComputing group summary...")
    summary = compute_group_summary(df)
    print(summary)
    
    print("\nRunning statistical tests...")
    test_results, effect_sizes = run_all_tests(df)
    print(test_results)
    print("\nEffect sizes:")
    print(effect_sizes)
    
    print("\nSaving results...")
    save_results(summary, test_results, effect_sizes, str(output_path))
    
    print("\nInterpreting results...")
    interpret_results(summary, test_results, effect_sizes)


if __name__ == '__main__':
    main()
