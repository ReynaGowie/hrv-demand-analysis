"""
Process WESAD ECG data to extract HRV features aligned with state labels.
Extracts RMSSD, SDNN, and mean HR from segmented ECG data.
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import signal
from scipy.signal import find_peaks


def load_subject_data(pkl_path: str) -> Dict:
    """
    Load a subject's .pkl file.
    
    Args:
        pkl_path: Path to the .pkl file
        
    Returns:
        Dictionary containing signal, label, and subject info
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def detect_r_peaks(ecg_signal: np.ndarray, fs: float = 700.0) -> np.ndarray:
    """
    Detect R-peaks in ECG signal using scipy's find_peaks.
    
    Args:
        ecg_signal: ECG signal array (1D)
        fs: Sampling frequency in Hz (WESAD chest ECG is 700 Hz)
        
    Returns:
        Array of R-peak indices
    """
    # Simple peak detection - can be improved with more sophisticated methods
    # Normalize signal
    ecg_normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    
    # Find peaks with minimum distance based on heart rate (assume max 200 bpm = 3.33 Hz)
    min_distance = int(fs / 3.33)  # Minimum samples between peaks
    
    # Find peaks with prominence threshold
    peaks, _ = find_peaks(ecg_normalized, distance=min_distance, prominence=0.5)
    
    return peaks


def compute_rr_intervals(r_peaks: np.ndarray, fs: float = 700.0) -> np.ndarray:
    """
    Compute RR intervals from R-peak indices.
    
    Args:
        r_peaks: Array of R-peak indices
        fs: Sampling frequency in Hz
        
    Returns:
        Array of RR intervals in seconds
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    rr_samples = np.diff(r_peaks)
    rr_seconds = rr_samples / fs
    
    return rr_seconds


def compute_hrv_features(rr_intervals: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute HRV features from RR intervals.
    
    Args:
        rr_intervals: Array of RR intervals in seconds
        
    Returns:
        Tuple of (rmssd, sdnn, mean_hr)
    """
    if len(rr_intervals) < 2:
        return np.nan, np.nan, np.nan
    
    # RMSSD: Root Mean Square of Successive Differences
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    # SDNN: Standard Deviation of NN intervals
    sdnn = np.std(rr_intervals)
    
    # Mean HR: Mean Heart Rate (bpm)
    mean_rr = np.mean(rr_intervals)
    mean_hr = 60.0 / mean_rr if mean_rr > 0 else np.nan
    
    return rmssd, sdnn, mean_hr


def map_labels(label_array: np.ndarray) -> np.ndarray:
    """
    Map WESAD labels to state names.
    
    WESAD label mapping:
    - 0: not defined / transition
    - 1: baseline (low demand)
    - 2: stress (high demand)
    - 3: amusement (recovery)
    - 4: meditation
    - 5: not defined
    - 6: not defined
    - 7: not defined
    
    Args:
        label_array: Array of integer labels
        
    Returns:
        Array of state names (or None for unmapped labels)
    """
    label_map = {
        0: None,
        1: 'baseline',
        2: 'stress',
        3: 'amusement',
        4: None,
        5: None,
        6: None,
        7: None
    }
    
    state_array = np.array([label_map.get(int(l), None) for l in label_array])
    
    return state_array


def segment_ecg_by_state(ecg_signal: np.ndarray, state_array: np.ndarray, 
                         min_segment_length: int = 1000) -> List[Dict]:
    """
    Segment ECG signal by state labels.
    
    Args:
        ecg_signal: ECG signal array
        state_array: Array of state names
        min_segment_length: Minimum samples required for a segment
        
    Returns:
        List of segments with metadata
    """
    segments = []
    
    # Find contiguous segments of the same state
    current_state = None
    start_idx = 0
    
    for i, state in enumerate(state_array):
        if state is None:
            # Skip undefined states
            if current_state is not None:
                segment_length = i - start_idx
                if segment_length >= min_segment_length:
                    segments.append({
                        'state': current_state,
                        'start_idx': start_idx,
                        'end_idx': i,
                        'length': segment_length
                    })
                current_state = None
            continue
        
        if current_state is None:
            # Start new segment
            current_state = state
            start_idx = i
        elif state != current_state:
            # End current segment
            segment_length = i - start_idx
            if segment_length >= min_segment_length:
                segments.append({
                    'state': current_state,
                    'start_idx': start_idx,
                    'end_idx': i,
                    'length': segment_length
                })
            # Start new segment
            current_state = state
            start_idx = i
    
    # Handle last segment
    if current_state is not None:
        segment_length = len(state_array) - start_idx
        if segment_length >= min_segment_length:
            segments.append({
                'state': current_state,
                'start_idx': start_idx,
                'end_idx': len(state_array),
                'length': segment_length
            })
    
    return segments


def process_subject(pkl_path: str, fs: float = 700.0, 
                   min_segment_length: int = 30000) -> List[Dict]:
    """
    Process a single subject's data to extract HRV features.
    
    Args:
        pkl_path: Path to the .pkl file
        fs: Sampling frequency in Hz
        min_segment_length: Minimum samples required for a segment
        
    Returns:
        List of HRV feature records
    """
    # Load data
    data = load_subject_data(pkl_path)
    subject_id = data['subject']
    
    # Extract ECG signal (chest)
    ecg_signal = data['signal']['chest']['ECG'].flatten()
    labels = data['label']
    
    # Map labels to states
    state_array = map_labels(labels)
    
    # Segment ECG by state
    segments = segment_ecg_by_state(ecg_signal, state_array, min_segment_length)
    
    records = []
    
    for segment in segments:
        state = segment['state']
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        segment_length = segment['length']
        
        # Extract ECG segment
        ecg_segment = ecg_signal[start_idx:end_idx]
        
        # Detect R-peaks
        r_peaks = detect_r_peaks(ecg_segment, fs)
        
        # Compute RR intervals
        rr_intervals = compute_rr_intervals(r_peaks, fs)
        
        # Compute HRV features
        rmssd, sdnn, mean_hr = compute_hrv_features(rr_intervals)
        
        # Create record
        record = {
            'subject_id': subject_id,
            'state': state,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'mean_hr': mean_hr,
            'segment_length': segment_length,
            'num_rr_intervals': len(rr_intervals)
        }
        
        records.append(record)
    
    return records


def process_all_subjects(wesad_path: str, fs: float = 700.0,
                        min_segment_length: int = 30000) -> pd.DataFrame:
    """
    Process all subjects in the WESAD dataset.
    
    Args:
        wesad_path: Path to WESAD raw data directory
        fs: Sampling frequency in Hz
        min_segment_length: Minimum samples required for a segment
        
    Returns:
        DataFrame with all HRV features
    """
    wesad_path = Path(wesad_path)
    
    all_records = []
    
    # Find all subject directories
    subject_dirs = [d for d in wesad_path.iterdir() if d.is_dir() and d.name.startswith('S')]
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.name
        pkl_path = subject_dir / f"{subject_id}.pkl"
        
        if not pkl_path.exists():
            print(f"Warning: {pkl_path} not found, skipping")
            continue
        
        print(f"Processing {subject_id}...")
        
        try:
            records = process_subject(str(pkl_path), fs, min_segment_length)
            all_records.extend(records)
            print(f"  Extracted {len(records)} segments")
        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
    
    df = pd.DataFrame(all_records)
    
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """
    Save cleaned HRV data to CSV.
    
    Args:
        df: DataFrame with HRV features
        output_path: Path to save the CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


def print_summary_stats(df: pd.DataFrame):
    """
    Print summary statistics for the dataset.
    
    Args:
        df: DataFrame with HRV features
    """
    print("\n" + "="*70)
    print("WESAD HRV DATASET SUMMARY")
    print("="*70)
    
    # Counts per state
    print("\n--- Counts per State ---")
    state_counts = df['state'].value_counts()
    for state, count in state_counts.items():
        print(f"  {state}: {count} segments")
    
    # Summary statistics per state
    print("\n--- Summary Statistics per State ---")
    metrics = ['rmssd', 'sdnn', 'mean_hr']
    
    for state in df['state'].unique():
        print(f"\n{state.upper()}:")
        state_df = df[df['state'] == state]
        
        for metric in metrics:
            valid_data = state_df[metric].dropna()
            if len(valid_data) > 0:
                print(f"  {metric}:")
                print(f"    Mean: {valid_data.mean():.4f}")
                print(f"    Std: {valid_data.std():.4f}")
                print(f"    Min: {valid_data.min():.4f}")
                print(f"    Max: {valid_data.max():.4f}")
                print(f"    N: {len(valid_data)}")
            else:
                print(f"  {metric}: No valid data")
    
    # Overall statistics
    print("\n--- Overall Statistics ---")
    print(f"Total segments: {len(df)}")
    print(f"Total subjects: {df['subject_id'].nunique()}")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    wesad_path = project_root / 'data' / 'raw' / 'wesad'
    output_path = project_root / 'data' / 'cleaned' / 'wesad_hrv.csv'
    
    # Parameters
    fs = 700.0  # WESAD chest ECG sampling frequency
    min_segment_length = 30000  # Minimum ~43 seconds of data (at 700 Hz)
    
    print("Processing WESAD HRV data...")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Minimum segment length: {min_segment_length} samples")
    
    # Process all subjects
    df = process_all_subjects(str(wesad_path), fs, min_segment_length)
    
    print(f"\nTotal records extracted: {len(df)}")
    
    # Save cleaned data
    save_cleaned_data(df, str(output_path))
    
    # Print summary statistics
    print_summary_stats(df)


if __name__ == '__main__':
    main()
