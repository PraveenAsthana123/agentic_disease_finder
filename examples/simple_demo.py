#!/usr/bin/env python3
"""
AgenticFinder Simple Demo
==========================

A quick 30-second demo showing the core functionality.

Usage:
    python examples/simple_demo.py
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("=" * 60)
    print(" AgenticFinder - Quick Demo (30 seconds)")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/4] Generating synthetic EEG data...")
    from generate_sample_data import EEGDataGenerator, extract_simple_features

    generator = EEGDataGenerator(sampling_rate=256, seed=42)
    data = generator.generate_dataset(
        disease='parkinson',
        n_subjects=10,
        samples_per_subject=5,
        duration=2.0
    )

    print(f"      Generated {len(data['signals'])} samples")

    # Step 2: Extract features
    print("[2/4] Extracting 47 EEG features...")
    X = extract_simple_features(data['signals'], 256)
    y = data['labels']
    print(f"      Feature shape: {X.shape}")

    # Step 3: Train (simple model for speed)
    print("[3/4] Training classifier...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("      Training complete!")

    # Step 4: Evaluate
    print("[4/4] Evaluating...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f" Results: Accuracy = {accuracy:.2%}")
    print(f"{'='*60}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Control', 'Parkinson']))

    print("\nDemo complete! For full Ultra Stacking Ensemble, run:")
    print("  python examples/quickstart.py")


if __name__ == '__main__':
    main()
