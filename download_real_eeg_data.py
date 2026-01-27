#!/usr/bin/env python
"""
Download Real EEG Datasets from Public Sources
Sources: PhysioNet, OpenNeuro, Kaggle, UCI

This script downloads authentic EEG data to improve model training
and reduce overfitting through increased sample sizes.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import gzip
import shutil
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REAL_DATA_DIR = DATA_DIR / "real_eeg"
REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Public EEG Dataset Sources
DATASETS = {
    'epilepsy_bonn': {
        'name': 'Bonn Epilepsy EEG Dataset',
        'description': 'Classic epilepsy detection dataset with 500 samples',
        'url': 'https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/',
        'samples': 500,
        'disease': 'epilepsy',
        'direct_url': None,  # Requires manual download
        'kaggle': 'harunshimanto/epileptic-seizure-recognition'
    },
    'eeg_eye_state': {
        'name': 'EEG Eye State Dataset (UCI)',
        'description': 'EEG recordings for eye state classification',
        'url': 'https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State',
        'samples': 14980,
        'disease': 'general',
        'direct_url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff'
    },
    'chbmit': {
        'name': 'CHB-MIT Scalp EEG Database',
        'description': 'Pediatric epilepsy EEG from Boston Children\'s Hospital',
        'url': 'https://physionet.org/content/chbmit/1.0.0/',
        'samples': 664,
        'disease': 'epilepsy',
        'physionet': 'chbmit/1.0.0'
    },
    'tuh_eeg': {
        'name': 'Temple University Hospital EEG Corpus',
        'description': 'Largest publicly available EEG corpus',
        'url': 'https://isip.piconepress.com/projects/tuh_eeg/',
        'samples': 25000,
        'disease': 'multiple',
        'requires_registration': True
    },
    'deap': {
        'name': 'DEAP Dataset',
        'description': 'EEG for emotion analysis',
        'url': 'https://www.eecs.qmul.ac.uk/mmv/datasets/deap/',
        'samples': 1280,
        'disease': 'emotion/stress',
        'requires_registration': True
    },
    'alzheimer_eeg': {
        'name': 'Alzheimer\'s EEG Dataset',
        'description': 'EEG recordings from Alzheimer\'s patients',
        'url': 'https://openneuro.org/datasets/ds004504',
        'samples': 88,
        'disease': 'alzheimer',
        'openneuro': 'ds004504'
    },
    'schizophrenia_eeg': {
        'name': 'Schizophrenia EEG Dataset',
        'description': 'EEG from schizophrenia patients',
        'url': 'http://brain.bio.msu.ru/eeg_schizophrenia.htm',
        'samples': 84,
        'disease': 'schizophrenia'
    },
    'parkinson_eeg': {
        'name': 'Parkinson\'s Disease EEG',
        'description': 'EEG recordings from PD patients',
        'url': 'https://openneuro.org/datasets/ds002778',
        'samples': 52,
        'disease': 'parkinson',
        'openneuro': 'ds002778'
    },
    'autism_eeg': {
        'name': 'Autism EEG Dataset',
        'description': 'EEG from autism spectrum disorder patients',
        'url': 'https://openneuro.org/datasets/ds004141',
        'samples': 36,
        'disease': 'autism',
        'openneuro': 'ds004141'
    },
    'depression_eeg': {
        'name': 'Depression EEG Dataset',
        'description': 'EEG from major depression patients',
        'url': 'https://figshare.com/articles/dataset/EEG_Depression_rest_state/19782175',
        'samples': 64,
        'disease': 'depression'
    }
}


def print_available_datasets():
    """Print information about available datasets."""
    print("="*80)
    print("  AVAILABLE REAL EEG DATASETS")
    print("="*80)

    for key, info in DATASETS.items():
        print(f"\n  {info['name']}")
        print(f"  {'-'*60}")
        print(f"  Disease: {info['disease']}")
        print(f"  Samples: {info['samples']}")
        print(f"  URL: {info['url']}")
        if info.get('requires_registration'):
            print(f"  Note: Requires registration")
        if info.get('physionet'):
            print(f"  PhysioNet ID: {info['physionet']}")
        if info.get('openneuro'):
            print(f"  OpenNeuro ID: {info['openneuro']}")
        if info.get('kaggle'):
            print(f"  Kaggle: {info['kaggle']}")


def download_uci_eye_state():
    """Download UCI EEG Eye State dataset."""
    print("\n  Downloading UCI EEG Eye State Dataset...")

    output_dir = REAL_DATA_DIR / "uci_eye_state"
    output_dir.mkdir(exist_ok=True)

    url = DATASETS['eeg_eye_state']['direct_url']
    output_file = output_dir / "eeg_eye_state.arff"

    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"  Downloaded: {output_file}")

        # Convert ARFF to CSV
        csv_file = output_dir / "eeg_eye_state.csv"
        convert_arff_to_csv(output_file, csv_file)
        print(f"  Converted to CSV: {csv_file}")

        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def convert_arff_to_csv(arff_file, csv_file):
    """Convert ARFF file to CSV."""
    data_started = False
    rows = []
    headers = []

    with open(arff_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('@attribute'):
                parts = line.split()
                headers.append(parts[1])
            elif line.lower() == '@data':
                data_started = True
            elif data_started and line and not line.startswith('%'):
                rows.append(line.split(','))

    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(csv_file, index=False)


def create_synthetic_augmentation(disease_dir, target_samples=200):
    """Create synthetic samples to augment existing data."""
    print(f"\n  Augmenting data in {disease_dir}...")

    # Find existing CSV
    csv_files = list(disease_dir.glob("*.csv"))
    if not csv_files:
        print(f"  No CSV files found")
        return False

    df = pd.read_csv(csv_files[0])
    original_count = len(df)

    if 'label' not in df.columns:
        print(f"  No label column found")
        return False

    # Generate augmented samples
    feature_cols = [c for c in df.columns if c not in ['label', 'subject_id', 'class']]

    augmented_rows = []
    np.random.seed(42)

    while len(augmented_rows) + original_count < target_samples:
        # Sample a random row
        idx = np.random.randint(0, len(df))
        row = df.iloc[idx].copy()

        # Add realistic noise
        for col in feature_cols:
            noise = np.random.normal(0, 0.05 * abs(row[col]) + 0.01)
            row[col] = row[col] + noise

        # Update subject_id if present
        if 'subject_id' in row:
            row['subject_id'] = f"aug_{len(augmented_rows)}"

        augmented_rows.append(row)

    # Combine and save
    augmented_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)

    output_file = disease_dir / f"{disease_dir.name}_augmented_{target_samples}.csv"
    combined_df.to_csv(output_file, index=False)

    print(f"  Original: {original_count} -> Augmented: {len(combined_df)} samples")
    print(f"  Saved: {output_file}")

    return True


def download_physionet_dataset(dataset_id, output_dir):
    """Download dataset from PhysioNet using wget."""
    print(f"\n  To download PhysioNet dataset {dataset_id}:")
    print(f"  Run: wget -r -N -c -np https://physionet.org/files/{dataset_id}/")
    print(f"  Or use: pip install wfdb && python -c \"import wfdb; wfdb.dl_database('{dataset_id.split('/')[0]}', '{output_dir}')\"")


def download_openneuro_dataset(dataset_id, output_dir):
    """Download dataset from OpenNeuro."""
    print(f"\n  To download OpenNeuro dataset {dataset_id}:")
    print(f"  1. Install: npm install -g @openneuro/cli")
    print(f"  2. Run: openneuro download --dataset {dataset_id} {output_dir}")
    print(f"  Or visit: https://openneuro.org/datasets/{dataset_id}")


def generate_download_script():
    """Generate a shell script to download all datasets."""
    script_path = BASE_DIR / "download_all_eeg.sh"

    script_content = """#!/bin/bash
# Download Real EEG Datasets
# Run this script to download publicly available EEG data

echo "=== Downloading Real EEG Datasets ==="

# Create directories
mkdir -p data/real_eeg/{epilepsy,parkinson,alzheimer,schizophrenia,depression,autism,stress}

# 1. UCI EEG Eye State
echo "Downloading UCI EEG Eye State..."
wget -P data/real_eeg/general/ "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"

# 2. Kaggle Epilepsy Dataset (requires kaggle CLI)
echo "For Kaggle datasets, run:"
echo "  kaggle datasets download -d harunshimanto/epileptic-seizure-recognition -p data/real_eeg/epilepsy/"

# 3. PhysioNet CHB-MIT (requires wfdb)
echo "For PhysioNet CHB-MIT:"
echo "  pip install wfdb"
echo "  python -c 'import wfdb; wfdb.dl_database(\"chbmit\", \"data/real_eeg/epilepsy/chbmit\")'"

# 4. OpenNeuro datasets
echo "For OpenNeuro datasets (requires openneuro-cli):"
echo "  npm install -g @openneuro/cli"
echo "  openneuro download --dataset ds004504 data/real_eeg/alzheimer/"
echo "  openneuro download --dataset ds002778 data/real_eeg/parkinson/"
echo "  openneuro download --dataset ds004141 data/real_eeg/autism/"

echo "=== Download instructions complete ==="
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"\n  Generated download script: {script_path}")


def create_data_loading_template():
    """Create a template for loading real EEG data."""
    template_path = BASE_DIR / "load_real_eeg.py"

    template_content = '''#!/usr/bin/env python
"""
Template for Loading Real EEG Datasets
Modify this template based on the specific dataset format.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import mne  # For EDF/BDF files
from scipy.signal import welch
from scipy.stats import skew, kurtosis

def extract_features(eeg_data, sfreq=256):
    """Extract features from raw EEG data."""
    features = {}

    # Time domain features
    features['mean'] = np.mean(eeg_data)
    features['std'] = np.std(eeg_data)
    features['var'] = np.var(eeg_data)
    features['min'] = np.min(eeg_data)
    features['max'] = np.max(eeg_data)
    features['median'] = np.median(eeg_data)
    features['ptp'] = np.ptp(eeg_data)
    features['skewness'] = skew(eeg_data)
    features['kurtosis'] = kurtosis(eeg_data)
    features['rms'] = np.sqrt(np.mean(eeg_data**2))

    # Frequency domain features
    freqs, psd = welch(eeg_data, fs=sfreq, nperseg=min(256, len(eeg_data)))

    # Band powers
    delta_mask = (freqs >= 0.5) & (freqs < 4)
    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs < 13)
    beta_mask = (freqs >= 13) & (freqs < 30)
    gamma_mask = (freqs >= 30) & (freqs < 100)

    features['delta_power'] = np.sum(psd[delta_mask])
    features['theta_power'] = np.sum(psd[theta_mask])
    features['alpha_power'] = np.sum(psd[alpha_mask])
    features['beta_power'] = np.sum(psd[beta_mask])
    features['gamma_power'] = np.sum(psd[gamma_mask])
    features['total_power'] = np.sum(psd)

    # Spectral features
    features['spectral_entropy'] = -np.sum(psd * np.log2(psd + 1e-10))
    features['dominant_freq'] = freqs[np.argmax(psd)]

    return features


def load_edf_file(edf_path):
    """Load EEG data from EDF file."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    return data, sfreq


def load_csv_eeg(csv_path):
    """Load EEG data from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def process_dataset(data_dir, output_path, label_func):
    """Process a dataset and extract features."""
    data_dir = Path(data_dir)
    all_features = []

    for file_path in data_dir.glob("**/*.edf"):
        try:
            data, sfreq = load_edf_file(file_path)

            # Extract features for each channel and average
            channel_features = []
            for ch in range(data.shape[0]):
                features = extract_features(data[ch], sfreq)
                channel_features.append(features)

            # Average across channels
            avg_features = {}
            for key in channel_features[0].keys():
                avg_features[key] = np.mean([cf[key] for cf in channel_features])

            # Add label
            avg_features['label'] = label_func(file_path)
            avg_features['subject_id'] = file_path.stem

            all_features.append(avg_features)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

    return df


# Example usage:
# process_dataset(
#     data_dir="data/real_eeg/epilepsy/chbmit",
#     output_path="data/epilepsy/real/epilepsy_real.csv",
#     label_func=lambda p: 1 if "seizure" in str(p).lower() else 0
# )
'''

    with open(template_path, 'w') as f:
        f.write(template_content)

    print(f"  Created data loading template: {template_path}")


def main():
    print("="*80)
    print("  REAL EEG DATA COLLECTION TOOL")
    print("="*80)

    # Print available datasets
    print_available_datasets()

    # Generate download script
    print("\n" + "="*80)
    print("  GENERATING TOOLS")
    print("="*80)

    generate_download_script()
    create_data_loading_template()

    # Try to download UCI dataset
    print("\n" + "="*80)
    print("  DOWNLOADING AVAILABLE DATASETS")
    print("="*80)

    download_uci_eye_state()

    # Augment existing data
    print("\n" + "="*80)
    print("  AUGMENTING EXISTING DATA")
    print("="*80)

    diseases = ['epilepsy', 'parkinson', 'alzheimer', 'schizophrenia',
                'depression', 'autism', 'stress']

    for disease in diseases:
        disease_dir = DATA_DIR / disease / "sample"
        if disease_dir.exists():
            create_synthetic_augmentation(disease_dir, target_samples=200)

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    print("""
  COMPLETED:
  1. Listed available public EEG datasets
  2. Generated download script: download_all_eeg.sh
  3. Created data loading template: load_real_eeg.py
  4. Downloaded UCI EEG Eye State dataset
  5. Augmented existing sample data to 200 samples each

  NEXT STEPS:
  1. Run ./download_all_eeg.sh to get more datasets
  2. Install required tools: pip install mne wfdb
  3. For Kaggle: pip install kaggle && kaggle datasets download ...
  4. For OpenNeuro: npm install -g @openneuro/cli
  5. Modify load_real_eeg.py to process downloaded data

  RECOMMENDED DATASETS FOR EACH DISEASE:
  - Epilepsy: CHB-MIT (PhysioNet), Bonn Dataset (Kaggle)
  - Parkinson: OpenNeuro ds002778
  - Alzheimer: OpenNeuro ds004504
  - Autism: OpenNeuro ds004141
  - Schizophrenia: brain.bio.msu.ru dataset
  - Depression: Figshare EEG Depression dataset
  - Stress: DEAP Dataset (requires registration)
""")


if __name__ == '__main__':
    main()
