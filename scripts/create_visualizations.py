"""
Create publication-style visualizations for HRV analysis.
Design system: burgundy (#640520), white background, minimalist editorial style.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats


# Design system constants
COLORS = {
    'background': '#FFFFFF',
    'primary': '#640520',
    'text': '#111111',
    'secondary': '#EAEAEA',
    'light_gray': '#F5F5F5'
}

FONTS = {
    'title': {'family': 'serif', 'weight': 'bold', 'size': 14},
    'subtitle': {'family': 'sans-serif', 'weight': 'normal', 'size': 11},
    'label': {'family': 'sans-serif', 'weight': 'normal', 'size': 10},
    'tick': {'family': 'sans-serif', 'weight': 'normal', 'size': 9}
}


def setup_figure(figsize=(8, 5)):
    """Setup figure with design system styling."""
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['text'])
    ax.spines['bottom'].set_color(COLORS['text'])
    
    return fig, ax


def add_title_subtitle(fig, ax, title, subtitle):
    """Add title and subtitle with design system styling."""
    fig.suptitle(title, **FONTS['title'], color=COLORS['text'], y=0.98, x=0.5, ha='center')
    ax.text(0.5, 0.93, subtitle, **FONTS['subtitle'], color=COLORS['text'], 
            ha='center', transform=ax.transAxes)


def style_axis_labels(ax, xlabel, ylabel):
    """Style axis labels."""
    ax.set_xlabel(xlabel, **FONTS['label'], color=COLORS['text'])
    ax.set_ylabel(ylabel, **FONTS['label'], color=COLORS['text'])
    ax.tick_params(axis='both', colors=COLORS['text'], labelsize=FONTS['tick']['size'])


def load_datasets():
    """Load both cleaned HRV datasets."""
    project_root = Path(__file__).parent.parent
    
    physionet_path = project_root / 'data' / 'cleaned' / 'physionet_hrv.csv'
    wesad_path = project_root / 'data' / 'cleaned' / 'wesad_hrv.csv'
    
    physionet_df = pd.read_csv(physionet_path)
    wesad_df = pd.read_csv(wesad_path)
    
    return physionet_df, wesad_df


def chart1_wesad_hrv_by_state(wesad_df, output_dir):
    """Chart 1: WESAD HRV by state boxplot."""
    fig, ax = setup_figure(figsize=(8, 5))
    
    # Filter to relevant states
    states = ['baseline', 'stress', 'amusement']
    plot_data = wesad_df[wesad_df['state'].isin(states)].copy()
    
    # Order states
    plot_data['state'] = pd.Categorical(plot_data['state'], categories=states, ordered=True)
    plot_data = plot_data.sort_values('state')

    # RMSSD is stored in seconds; convert to milliseconds (the HRV convention)
    plot_data['rmssd_ms'] = plot_data['rmssd'] * 1000

    # Create boxplot
    box_colors = [COLORS['primary'] if s == 'stress' else COLORS['secondary'] for s in states]
    bp = ax.boxplot([plot_data[plot_data['state'] == s]['rmssd_ms'].values for s in states],
                    tick_labels=states, patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor=COLORS['secondary'], edgecolor=COLORS['primary'], linewidth=1.5),
                    medianprops=dict(color=COLORS['primary'], linewidth=2),
                    whiskerprops=dict(color=COLORS['text'], linewidth=1),
                    capprops=dict(color=COLORS['text'], linewidth=1))
    
    # Color stress box differently
    for patch, state in zip(bp['boxes'], states):
        if state == 'stress':
            patch.set_facecolor(COLORS['primary'])
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(COLORS['secondary'])
            patch.set_alpha(0.5)
    
    # Add mean markers
    for i, state in enumerate(states, 1):
        state_data = plot_data[plot_data['state'] == state]['rmssd_ms']
        mean_val = state_data.mean()
        ax.scatter(i, mean_val, color=COLORS['text'], s=50, zorder=5, marker='D')
    
    add_title_subtitle(fig, ax, "HRV Responds to Psychological State", 
                      "Clearer separation under controlled stress conditions (WESAD)")
    style_axis_labels(ax, "State", "RMSSD (ms)")
    
    plt.tight_layout()
    output_path = output_dir / 'wesad_hrv_by_state.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"Saved: {output_path}")


def chart2_physionet_hrv_by_workload(physionet_df, output_dir):
    """Chart 2: PhysioNet HRV by workload boxplot."""
    fig, ax = setup_figure(figsize=(8, 5))
    
    # Filter to relevant workloads
    workloads = ['AEROBIC', 'ANAEROBIC', 'STRESS']
    plot_data = physionet_df[physionet_df['workload_label'].isin(workloads)].copy()
    
    # Order workloads
    plot_data['workload_label'] = pd.Categorical(plot_data['workload_label'],
                                                  categories=workloads, ordered=True)
    plot_data = plot_data.sort_values('workload_label')

    # RMSSD is stored in seconds; convert to milliseconds (the HRV convention)
    plot_data['rmssd_ms'] = plot_data['rmssd'] * 1000

    # Create boxplot
    bp = ax.boxplot([plot_data[plot_data['workload_label'] == w]['rmssd_ms'].values for w in workloads],
                    tick_labels=workloads, patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor=COLORS['secondary'], edgecolor=COLORS['primary'], linewidth=1.5),
                    medianprops=dict(color=COLORS['primary'], linewidth=2),
                    whiskerprops=dict(color=COLORS['text'], linewidth=1),
                    capprops=dict(color=COLORS['text'], linewidth=1))
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['secondary'])
        patch.set_alpha(0.5)
    
    # Add mean markers
    for i, workload in enumerate(workloads, 1):
        workload_data = plot_data[plot_data['workload_label'] == workload]['rmssd_ms']
        mean_val = workload_data.mean()
        ax.scatter(i, mean_val, color=COLORS['text'], s=50, zorder=5, marker='D')
    
    add_title_subtitle(fig, ax, "HRV Shows Weaker Separation in Workload Contexts", 
                      "Limited differentiation across physical demand conditions")
    style_axis_labels(ax, "Workload", "RMSSD (ms)")
    
    plt.tight_layout()
    output_path = output_dir / 'physionet_hrv_by_workload.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"Saved: {output_path}")


def chart3_hr_comparison(physionet_df, wesad_df, output_dir):
    """Chart 3: Heart rate comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.patch.set_facecolor(COLORS['background'])
    
    for ax in [ax1, ax2]:
        ax.set_facecolor(COLORS['background'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['text'])
        ax.spines['bottom'].set_color(COLORS['text'])
    
    # WESAD panel
    wesad_states = ['baseline', 'stress', 'amusement']
    wesad_means = []
    wesad_stds = []
    
    for state in wesad_states:
        state_data = wesad_df[wesad_df['state'] == state]['mean_hr']
        wesad_means.append(state_data.mean())
        wesad_stds.append(state_data.std())
    
    x_pos = np.arange(len(wesad_states))
    bars1 = ax1.bar(x_pos, wesad_means, yerr=wesad_stds, capsize=5,
                   color=COLORS['primary'], alpha=0.7, edgecolor=COLORS['primary'], linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(wesad_states, **FONTS['tick'])
    ax1.set_ylabel('Mean HR (bpm)', **FONTS['label'], color=COLORS['text'])
    ax1.set_title('WESAD', **FONTS['title'], color=COLORS['text'])
    ax1.tick_params(axis='both', colors=COLORS['text'], labelsize=FONTS['tick']['size'])
    
    # PhysioNet panel
    physionet_workloads = ['AEROBIC', 'ANAEROBIC', 'STRESS']
    physionet_means = []
    physionet_stds = []
    
    for workload in physionet_workloads:
        workload_data = physionet_df[physionet_df['workload_label'] == workload]['mean_hr']
        physionet_means.append(workload_data.mean())
        physionet_stds.append(workload_data.std())
    
    x_pos = np.arange(len(physionet_workloads))
    bars2 = ax2.bar(x_pos, physionet_means, yerr=physionet_stds, capsize=5,
                   color=COLORS['primary'], alpha=0.7, edgecolor=COLORS['primary'], linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(physionet_workloads, **FONTS['tick'])
    ax2.set_ylabel('Mean HR (bpm)', **FONTS['label'], color=COLORS['text'])
    ax2.set_title('PhysioNet', **FONTS['title'], color=COLORS['text'])
    ax2.tick_params(axis='both', colors=COLORS['text'], labelsize=FONTS['tick']['size'])
    
    fig.suptitle("Heart Rate Is Stronger but Context-Dependent", **FONTS['title'], 
                 color=COLORS['text'], y=0.98)
    fig.text(0.5, 0.93, "Direction of response differs across datasets", **FONTS['subtitle'], 
             color=COLORS['text'], ha='center')
    
    plt.tight_layout()
    output_path = output_dir / 'hr_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"Saved: {output_path}")


def compute_effect_sizes(physionet_df, wesad_df):
    """Compute Cohen's d effect sizes for both datasets."""
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return np.nan
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return np.nan
        return (group1.mean() - group2.mean()) / pooled_std
    
    # PhysioNet: STRESS vs AEROBIC
    physionet_stress = physionet_df[physionet_df['workload_label'] == 'STRESS']
    physionet_baseline = physionet_df[physionet_df['workload_label'] == 'AEROBIC']
    
    physionet_effects = {
        'RMSSD': cohens_d(physionet_baseline['rmssd'], physionet_stress['rmssd']),
        'SDNN': cohens_d(physionet_baseline['sdnn'], physionet_stress['sdnn']),
        'HR': cohens_d(physionet_baseline['mean_hr'], physionet_stress['mean_hr'])
    }
    
    # WESAD: stress vs baseline
    wesad_stress = wesad_df[wesad_df['state'] == 'stress']
    wesad_baseline = wesad_df[wesad_df['state'] == 'baseline']
    
    wesad_effects = {
        'RMSSD': cohens_d(wesad_baseline['rmssd'], wesad_stress['rmssd']),
        'SDNN': cohens_d(wesad_baseline['sdnn'], wesad_stress['sdnn']),
        'HR': cohens_d(wesad_baseline['mean_hr'], wesad_stress['mean_hr'])
    }
    
    return physionet_effects, wesad_effects


def chart4_effect_size_comparison(physionet_df, wesad_df, output_dir):
    """Chart 4: Effect size comparison bar chart (MOST IMPORTANT)."""
    physionet_effects, wesad_effects = compute_effect_sizes(physionet_df, wesad_df)
    
    fig, ax = setup_figure(figsize=(8, 5))
    
    metrics = ['RMSSD', 'SDNN', 'HR']
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    physionet_values = [physionet_effects[m] for m in metrics]
    wesad_values = [wesad_effects[m] for m in metrics]
    
    bars1 = ax.bar(x_pos - width/2, physionet_values, width, 
                   label='PhysioNet', color=COLORS['primary'], alpha=0.8,
                   edgecolor=COLORS['primary'], linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, wesad_values, width,
                   label='WESAD', color=COLORS['secondary'], alpha=0.8,
                   edgecolor=COLORS['primary'], linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, **FONTS['tick'])
    ax.set_ylabel("Cohen's d", **FONTS['label'], color=COLORS['text'])
    ax.axhline(y=0, color=COLORS['text'], linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add legend
    legend = ax.legend(loc='upper right', frameon=False, fontsize=FONTS['tick']['size'])
    for text in legend.get_texts():
        text.set_color(COLORS['text'])
    
    add_title_subtitle(fig, ax, "No Single Metric Consistently Captures Demand", 
                      "Effect sizes vary across datasets and signals")
    style_axis_labels(ax, "Metric", "Cohen's d")
    
    plt.tight_layout()
    output_path = output_dir / 'effect_size_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"Saved: {output_path}")


def chart5_summary_visual(physionet_df, wesad_df, output_dir):
    """Chart 5: Summary visual of signal consistency."""
    physionet_effects, wesad_effects = compute_effect_sizes(physionet_df, wesad_df)
    
    fig, ax = setup_figure(figsize=(8, 4))
    
    # Calculate average effect sizes and direction consistency
    metrics = ['RMSSD', 'SDNN', 'HR']
    
    # Average effect size across datasets
    avg_effects = []
    for m in metrics:
        p_val = physionet_effects[m]
        w_val = wesad_effects[m]
        avg_effects.append(abs((p_val + w_val) / 2))
    
    # Direction consistency (True if same sign)
    direction_consistency = []
    for m in metrics:
        p_val = physionet_effects[m]
        w_val = wesad_effects[m]
        consistent = (p_val > 0 and w_val > 0) or (p_val < 0 and w_val < 0)
        direction_consistency.append(1 if consistent else 0)
    
    # Create scatter plot
    x_pos = np.arange(len(metrics))
    
    # Plot effect size as bar height
    bars = ax.bar(x_pos, avg_effects, color=COLORS['primary'], alpha=0.7,
                  edgecolor=COLORS['primary'], linewidth=1.5, width=0.6)
    
    # Add direction consistency markers
    for i, (x, eff, cons) in enumerate(zip(x_pos, avg_effects, direction_consistency)):
        if cons:
            ax.scatter(x, eff + 0.1, color='green', s=100, marker='o', zorder=5, label='Consistent' if i == 0 else "")
        else:
            ax.scatter(x, eff + 0.1, color='red', s=100, marker='x', zorder=5, label='Inconsistent' if i == 1 else "")
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, **FONTS['tick'])
    ax.set_ylabel("Average |Cohen's d|", **FONTS['label'], color=COLORS['text'])
    
    # Add legend
    legend = ax.legend(loc='upper right', frameon=False, fontsize=FONTS['tick']['size'])
    for text in legend.get_texts():
        text.set_color(COLORS['text'])
    
    add_title_subtitle(fig, ax, "Physiological Signals Are Context-Dependent", 
                      "Capacity cannot be captured by a single biomarker")
    style_axis_labels(ax, "Metric", "Average Effect Size")
    
    plt.tight_layout()
    output_path = output_dir / 'signal_consistency_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main execution function."""
    print("Creating publication-style visualizations...")
    
    # Load data
    physionet_df, wesad_df = load_datasets()
    print(f"Loaded PhysioNet: {len(physionet_df)} observations")
    print(f"Loaded WESAD: {len(wesad_df)} observations")
    
    # Setup output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate charts
    print("\nGenerating Chart 1: WESAD HRV by state...")
    chart1_wesad_hrv_by_state(wesad_df, output_dir)
    
    print("\nGenerating Chart 2: PhysioNet HRV by workload...")
    chart2_physionet_hrv_by_workload(physionet_df, output_dir)
    
    print("\nGenerating Chart 3: Heart rate comparison...")
    chart3_hr_comparison(physionet_df, wesad_df, output_dir)
    
    print("\nGenerating Chart 4: Effect size comparison...")
    chart4_effect_size_comparison(physionet_df, wesad_df, output_dir)
    
    print("\nGenerating Chart 5: Summary visual...")
    chart5_summary_visual(physionet_df, wesad_df, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
