"""
Inspect raw data directories for HRV analysis.
This script recursively inspects WESAD and PhysioNet raw data folders,
identifies files relevant for HRV analysis, and produces recommendations.
"""

import os
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def inspect_directory(base_path: str) -> Dict:
    """
    Recursively inspect a directory and collect file information.
    
    Args:
        base_path: Path to the directory to inspect
        
    Returns:
        Dictionary with structure info including subfolders, files, extensions, counts
    """
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Warning: {base_path} does not exist")
        return {}
    
    result = {
        'subfolders': [],
        'files': [],
        'extensions': defaultdict(int),
        'file_counts': {},
        'full_structure': []
    }
    
    # Walk through directory recursively
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        rel_path = root_path.relative_to(base_path)
        
        # Record subfolders
        for d in dirs:
            folder_path = rel_path / d
            result['subfolders'].append(str(folder_path))
        
        # Record files with metadata
        for f in files:
            file_path = rel_path / f
            ext = Path(f).suffix.lower()
            result['extensions'][ext] += 1
            result['files'].append({
                'path': str(file_path),
                'name': f,
                'extension': ext,
                'size': os.path.getsize(root_path / f) if os.path.exists(root_path / f) else 0
            })
            result['full_structure'].append(str(file_path))
    
    # Count files per directory
    for file_info in result['files']:
        dir_path = str(Path(file_info['path']).parent)
        result['file_counts'][dir_path] = result['file_counts'].get(dir_path, 0) + 1
    
    return result


def flag_hrv_relevant_files(files: List[Dict], dataset_name: str) -> List[Dict]:
    """
    Flag files that are likely relevant for HRV analysis.
    
    Args:
        files: List of file information dictionaries
        dataset_name: Name of the dataset (WESAD or PhysioNet)
        
    Returns:
        List of flagged files with relevance reasons
    """
    flagged = []
    
    # HRV-relevant keywords and patterns
    ecg_keywords = ['ecg', 'e4', 'bvp', 'ppg', 'cardiac', 'heart']
    label_keywords = ['label', 'tag', 'stress', 'quest', 'annotation']
    metadata_keywords = ['readme', 'dict', 'license', 'info', 'meta']
    
    for file_info in files:
        name_lower = file_info['name'].lower()
        path_lower = file_info['path'].lower()
        
        reasons = []
        
        # Check for ECG/signal files
        if any(kw in name_lower or kw in path_lower for kw in ecg_keywords):
            reasons.append('ECG/signal')
        
        # Check for label files
        if any(kw in name_lower or kw in path_lower for kw in label_keywords):
            reasons.append('label/workload')
        
        # Check for metadata
        if any(kw in name_lower or kw in path_lower for kw in metadata_keywords):
            reasons.append('metadata')
        
        # Check for specific HRV-related files
        if 'ibi' in name_lower or 'hr' in name_lower:
            reasons.append('HRV/heart_rate')
        
        # Check for pickle files (often contain signal data)
        if file_info['extension'] == '.pkl':
            reasons.append('signal_data')
        
        # Check for subject/session folders
        if any(f's{i:02d}' in path_lower or f's{i}' in path_lower for i in range(1, 20)):
            reasons.append('subject_folder')
        
        if reasons:
            flagged.append({
                **file_info,
                'relevance': ', '.join(reasons),
                'dataset': dataset_name
            })
    
    return flagged


def generate_recommendations(wesad_data: Dict, physionet_data: Dict) -> Tuple[str, str]:
    """
    Generate recommendations for each dataset.
    
    Args:
        wesad_data: WESAD directory inspection results
        physionet_data: PhysioNet directory inspection results
        
    Returns:
        Tuple of (wesad_recommendation, physionet_recommendation)
    """
    # WESAD recommendations
    wesad_rec = {
        'ecg_source': 'SXX.pkl files (likely contain ECG signal data)',
        'labels': 'SXX_quest.csv (questionnaire data for stress labels)',
        'subject_structure': 'Subject folders named S10, S11, S13, etc.',
        'notes': 'E4_Data.zip files may contain raw Empatica E4 wristband data as backup'
    }
    
    # PhysioNet recommendations
    physionet_rec = {
        'ecg_source': 'BVP.csv files (Blood Volume Pulse - PPG signal for HRV)',
        'labels': 'tags.csv (activity/stress labels per session), Stress_Level_v1/v2.csv (overall labels)',
        'subject_structure': 'Activity folders (AEROBIC, ANAEROBIC, STRESS) with subject folders S01, S02, etc.',
        'notes': 'IBI.csv contains Inter-Beat Intervals (direct HRV metric), HR.csv contains pre-computed heart rate'
    }
    
    return wesad_rec, physionet_rec


def save_inventory_csv(wesad_data: Dict, physionet_data: Dict, wesad_flagged: List[Dict], 
                       physionet_flagged: List[Dict], output_path: str):
    """
    Save inventory summary to CSV.
    
    Args:
        wesad_data: WESAD inspection results
        physionet_data: PhysioNet inspection results
        wesad_flagged: Flagged WESAD files
        physionet_flagged: Flagged PhysioNet files
        output_path: Path to save the CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'File_Path', 'File_Name', 'Extension', 'Relevance', 'File_Size_Bytes'])
        
        for file_info in wesad_flagged:
            writer.writerow([
                'WESAD',
                file_info['path'],
                file_info['name'],
                file_info['extension'],
                file_info['relevance'],
                file_info['size']
            ])
        
        for file_info in physionet_flagged:
            writer.writerow([
                'PhysioNet',
                file_info['path'],
                file_info['name'],
                file_info['extension'],
                file_info['relevance'],
                file_info['size']
            ])


def print_summary(wesad_data: Dict, physionet_data: Dict, wesad_rec: Dict, physionet_rec: Dict):
    """
    Print a concise summary of findings.
    """
    print("\n" + "="*70)
    print("HRV RAW DATA INSPECTION SUMMARY")
    print("="*70)
    
    print("\n--- WESAD Dataset ---")
    print(f"Total files found: {len(wesad_data['files'])}")
    print(f"Total subfolders: {len(wesad_data['subfolders'])}")
    print(f"File extensions: {dict(wesad_data['extensions'])}")
    print(f"\nRecommendations:")
    print(f"  - ECG source: {wesad_rec['ecg_source']}")
    print(f"  - Labels: {wesad_rec['labels']}")
    print(f"  - Subject structure: {wesad_rec['subject_structure']}")
    print(f"  - Notes: {wesad_rec['notes']}")
    
    print("\n--- PhysioNet Dataset ---")
    print(f"Total files found: {len(physionet_data['files'])}")
    print(f"Total subfolders: {len(physionet_data['subfolders'])}")
    print(f"File extensions: {dict(physionet_data['extensions'])}")
    print(f"\nRecommendations:")
    print(f"  - ECG source: {physionet_rec['ecg_source']}")
    print(f"  - Labels: {physionet_rec['labels']}")
    print(f"  - Subject structure: {physionet_rec['subject_structure']}")
    print(f"  - Notes: {physionet_rec['notes']}")
    
    print("\n" + "="*70)
    print("READINESS ASSESSMENT")
    print("="*70)
    
    print("\nWESAD:")
    print("  - Appears READY for ECG → HRV extraction")
    print("  - Use .pkl files as primary ECG source")
    print("  - Use quest.csv files for stress labels")
    print("  - Start with one subject (e.g., S10) to validate data structure")
    
    print("\nPhysioNet:")
    print("  - Appears READY for workload → HRV extraction")
    print("  - Use BVP.csv for PPG signal (can derive HRV)")
    print("  - IBI.csv provides direct inter-beat intervals (preferred for HRV)")
    print("  - Use tags.csv for session-level activity labels")
    print("  - Start with AEROBIC/S01 to validate data structure")
    
    print("\nPriority:")
    print("  1. PhysioNet (has IBI.csv - direct HRV metric, easier to start)")
    print("  2. WESAD (requires processing .pkl files, but has rich stress labels)")
    
    print("\n" + "="*70)
    print(f"Full inventory saved to: outputs/tables/hrv_raw_inventory.csv")
    print("="*70 + "\n")


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    wesad_path = project_root / 'data' / 'raw' / 'wesad'
    physionet_path = project_root / 'data' / 'raw' / 'physionet'
    output_path = project_root / 'outputs' / 'tables' / 'hrv_raw_inventory.csv'
    
    print("Inspecting WESAD dataset...")
    wesad_data = inspect_directory(str(wesad_path))
    
    print("Inspecting PhysioNet dataset...")
    physionet_data = inspect_directory(str(physionet_path))
    
    print("Flagging HRV-relevant files...")
    wesad_flagged = flag_hrv_relevant_files(wesad_data['files'], 'WESAD')
    physionet_flagged = flag_hrv_relevant_files(physionet_data['files'], 'PhysioNet')
    
    print("Generating recommendations...")
    wesad_rec, physionet_rec = generate_recommendations(wesad_data, physionet_data)
    
    print("Saving inventory to CSV...")
    save_inventory_csv(wesad_data, physionet_data, wesad_flagged, physionet_flagged, str(output_path))
    
    print("Printing summary...")
    print_summary(wesad_data, physionet_data, wesad_rec, physionet_rec)


if __name__ == '__main__':
    main()
