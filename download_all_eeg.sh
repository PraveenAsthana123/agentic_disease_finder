#!/bin/bash
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
echo "  python -c 'import wfdb; wfdb.dl_database("chbmit", "data/real_eeg/epilepsy/chbmit")'"

# 4. OpenNeuro datasets
echo "For OpenNeuro datasets (requires openneuro-cli):"
echo "  npm install -g @openneuro/cli"
echo "  openneuro download --dataset ds004504 data/real_eeg/alzheimer/"
echo "  openneuro download --dataset ds002778 data/real_eeg/parkinson/"
echo "  openneuro download --dataset ds004141 data/real_eeg/autism/"

echo "=== Download instructions complete ==="
