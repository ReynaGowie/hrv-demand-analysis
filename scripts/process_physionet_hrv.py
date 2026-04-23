"""
Process PhysioNet HRV data from IBI files.
Builds a subject-condition HRV dataset using IBI.csv files.
"""

import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def find_ibi_files(root_dir: str) -> List[Dict]:
    """
    Recursively locate all IBI.csv files and extract metadata.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        List of dictionaries with file path and metadata
    """
    root_path = Path(root_dir)
    ibi_files = []
    
    for ibi_file in root_path.rglob("IBI.csv"):
        # Extract workload label from parent folder name (AEROBIC, ANAEROBIC, STRESS)
        # Path structure: .../Wearable_Dataset/{ACTIVITY}/{SUBJECT}/IBI.csv
        parts = ibi_file.parts
        try:
            wearable_idx = parts.index("Wearable_Dataset")
            workload_label = parts[wearable_idx + 1]  # Activity folder
            subject_id = parts[wearable_idx + 2]  # Subject folder
        except (ValueError, IndexError):
            print(f"Warning: Unexpected path structure for {ibi_file}, skipping")
            continue
        
        ibi_files.append({
            'path': str(ibi_file),
            'subject_id': subject_id,
            'workload_label': workload_label
        })
    
    return ibi_files


def load_ibi_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load IBI data from CSV file.
    
    Args:
        file_path: Path to IBI.csv file
        
    Returns:
        DataFrame with IBI values, or None if file is empty/invalid
    """
    # Check if file is empty
    if os.path.getsize(file_path) == 0:
        return None
    
    try:
        # Read CSV - skip header row (first row has timestamps)
        # Data rows have two columns: index/timestamp, IBI value
        df = pd.read_csv(file_path, skiprows=1, header=None)
        
        # Column 1 (index 1) contains IBI values in seconds
        if df.shape[1] < 2:
            print(f"Warning: {file_path} has insufficient columns, skipping")
            return None
        
        # Extract IBI values (second column)
        ibi_values = df[1].values
        
        # Convert to numeric, handling any conversion errors
        ibi_values = pd.to_numeric(ibi_values, errors='coerce')
        
        # Remove NaN values
        ibi_values = ibi_values[~np.isnan(ibi_values)]
        
        if len(ibi_values) == 0:
            print(f"Warning: {file_path} has no valid IBI values, skipping")
            return None
        
        return pd.DataFrame({'ibi': ibi_values})
        
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}, skipping")
        return None


def compute_hrv_features(ibi_values: np.ndarray) -> Dict:
    """
    Compute HRV features from IBI values.
    
    Args:
        ibi_values: Array of inter-beat intervals in seconds
        
    Returns:
        Dictionary with HRV features
    """
    n_intervals = len(ibi_values)
    
    # Mean IBI
    mean_ibi = np.mean(ibi_values)
    
    # SDNN: standard deviation of IBI values
    sdnn = np.std(ibi_values, ddof=1)
    
    # RMSSD: root mean square of successive differences
    successive_diffs = np.diff(ibi_values)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    
    # Mean HR: derived from mean IBI (HR = 60 / IBI)
    mean_hr = 60.0 / mean_ibi
    
    return {
        'mean_ibi': mean_ibi,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'mean_hr': mean_hr,
        'n_intervals': n_intervals
    }


def process_physionet_hrv(root_dir: str, min_intervals: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all PhysioNet IBI files and build HRV dataset.
    
    Args:
        root_dir: Root directory of PhysioNet data
        min_intervals: Minimum number of valid IBI intervals required
        
    Returns:
        Tuple of (cleaned_dataset, processing_summary)
    """
    # Find all IBI files
    ibi_files = find_ibi_files(root_dir)
    print(f"Found {len(ibi_files)} IBI.csv files")
    
    # Process each file
    results = []
    excluded = []
    
    for file_info in ibi_files:
        file_path = file_info['path']
        subject_id = file_info['subject_id']
        workload_label = file_info['workload_label']
        
        # Load IBI data
        ibi_df = load_ibi_data(file_path)
        
        if ibi_df is None:
            excluded.append({
                'subject_id': subject_id,
                'workload_label': workload_label,
                'file_path': file_path,
                'reason': 'empty_or_invalid'
            })
            continue
        
        ibi_values = ibi_df['ibi'].values
        
        # Check minimum interval requirement
        if len(ibi_values) < min_intervals:
            excluded.append({
                'subject_id': subject_id,
                'workload_label': workload_label,
                'file_path': file_path,
                'reason': f'insufficient_intervals ({len(ibi_values)} < {min_intervals})'
            })
            continue
        
        # Compute HRV features
        features = compute_hrv_features(ibi_values)
        
        results.append({
            'subject_id': subject_id,
            'workload_label': workload_label,
            **features
        })
    
    # Create cleaned dataset
    cleaned_df = pd.DataFrame(results)
    
    # Create processing summary
    excluded_df = pd.DataFrame(excluded)
    
    return cleaned_df, excluded_df


def save_outputs(cleaned_df: pd.DataFrame, excluded_df: pd.DataFrame, 
                 cleaned_path: str, summary_path: str):
    """
    Save cleaned dataset and processing summary.
    
    Args:
        cleaned_df: Cleaned HRV dataset
        excluded_df: Excluded files summary
        cleaned_path: Path to save cleaned dataset
        summary_path: Path to save processing summary
    """
    # Create directories if needed
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    # Save cleaned dataset
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_path}")
    
    # Save processing summary
    excluded_df.to_csv(summary_path, index=False)
    print(f"Saved processing summary to {summary_path}")


def print_summary(cleaned_df: pd.DataFrame, excluded_df: pd.DataFrame):
    """
    Print summary statistics.
    
    Args:
        cleaned_df: Cleaned HRV dataset
        excluded_df: Excluded files summary
    """
    print("\n" + "="*70)
    print("PHYSIONET HRV PROCESSING SUMMARY")
    print("="*70)
    
    print(f"\nTotal subject-condition observations: {len(cleaned_df)}")
    
    print(f"\nFiles excluded: {len(excluded_df)}")
    if len(excluded_df) > 0:
        print("  Exclusion reasons:")
        for reason, count in excluded_df['reason'].value_counts().items():
            print(f"    - {reason}: {count}")
    
    print(f"\nCounts by workload_label:")
    print(cleaned_df['workload_label'].value_counts().to_string())
    
    print(f"\nSummary statistics by workload_label:")
    hrv_metrics = ['mean_ibi', 'sdnn', 'rmssd', 'mean_hr', 'n_intervals']
    summary = cleaned_df.groupby('workload_label')[hrv_metrics].agg(['mean', 'std', 'min', 'max'])
    print(summary)
    
    # Check for anomalies
    print(f"\nAnomaly checks:")
    print(f"  - Negative IBI values: {(cleaned_df['mean_ibi'] < 0).sum()}")
    print(f"  - Extremely high HR (>200 bpm): {(cleaned_df['mean_hr'] > 200).sum()}")
    print(f"  - Extremely low HR (<30 bpm): {(cleaned_df['mean_hr'] < 30).sum()}")
    print(f"  - Zero SDNN: {(cleaned_df['sdnn'] == 0).sum()}")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    root_dir = project_root / 'data' / 'raw' / 'physionet'
    cleaned_path = project_root / 'data' / 'cleaned' / 'physionet_hrv.csv'
    summary_path = project_root / 'outputs' / 'tables' / 'physionet_hrv_processing_summary.csv'
    
    print("Processing PhysioNet HRV data...")
    print(f"Input directory: {root_dir}")
    
    # Process data
    cleaned_df, excluded_df = process_physionet_hrv(str(root_dir), min_intervals=50)
    
    # Save outputs
    save_outputs(cleaned_df, excluded_df, str(cleaned_path), str(summary_path))
    
    # Print summary
    print_summary(cleaned_df, excluded_df)


if __name__ == '__main__':
    main()
