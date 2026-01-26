#!/usr/bin/env python3
"""
Train and Evaluate Models for ALL Available Disease Datasets
============================================================
This script loads REAL EEG data for each disease and trains classifiers.
"""

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
from ui_app import RealDataLoader

def main():
    print("="*70)
    print("  TRAINING MODELS FOR ALL DISEASES WITH REAL DATA")
    print("="*70)
    print()

    loader = RealDataLoader(base_path='./datasets')
    results_summary = {}

    # 1. SCHIZOPHRENIA
    print("-"*70)
    print("  DISEASE: SCHIZOPHRENIA")
    print("-"*70)
    try:
        X, y, metadata = loader.load_schizophrenia_data()
        if len(X) > 0:
            print(f"  Dataset: MHRC Russia EEG")
            print(f"  Subjects: {len(y)} (Healthy: {metadata['healthy_count']}, Patients: {metadata['patient_count']})")
            print(f"  Shape: {X.shape}")
            print("  Training classifier...")
            model, res = loader.train_classifier(X, y)
            results_summary['Schizophrenia'] = {
                'accuracy': res['accuracy'],
                'f1': res['f1'],
                'subjects': len(y),
                'cv_mean': res['cv_mean']
            }
            print(f"\n  RESULT: {res['accuracy']:.1%} accuracy, F1: {res['f1']:.1%}")
        else:
            print("  ERROR: No data loaded")
            results_summary['Schizophrenia'] = {'error': 'No data'}
    except Exception as e:
        print(f"  ERROR: {e}")
        results_summary['Schizophrenia'] = {'error': str(e)}
    print()

    # 2. DEPRESSION
    print("-"*70)
    print("  DISEASE: DEPRESSION")
    print("-"*70)
    try:
        X, y, metadata = loader.load_depression_data()
        if len(X) > 0:
            print(f"  Dataset: ds003478 (BIDS format)")
            print(f"  Subjects: {len(y)} (Healthy: {metadata['healthy_count']}, Patients: {metadata['patient_count']})")
            print(f"  Shape: {X.shape}")
            print("  Training classifier...")
            model, res = loader.train_classifier(X, y)
            results_summary['Depression'] = {
                'accuracy': res['accuracy'],
                'f1': res['f1'],
                'subjects': len(y),
                'cv_mean': res['cv_mean']
            }
            print(f"\n  RESULT: {res['accuracy']:.1%} accuracy, F1: {res['f1']:.1%}")
        else:
            print("  ERROR: No data loaded (check if .set/.fdt files are valid)")
            results_summary['Depression'] = {'error': 'No data loaded'}
    except Exception as e:
        print(f"  ERROR: {e}")
        results_summary['Depression'] = {'error': str(e)}
    print()

    # 3. EPILEPSY
    print("-"*70)
    print("  DISEASE: EPILEPSY")
    print("-"*70)
    try:
        X, y, metadata = loader.load_epilepsy_data()
        if len(X) > 0:
            print(f"  Dataset: CHB-MIT")
            print(f"  Subjects: {len(y)} (Normal: {metadata['normal_count']}, Seizure: {metadata['seizure_count']})")
            print(f"  Shape: {X.shape}")
            print("  Training classifier...")
            model, res = loader.train_classifier(X, y)
            results_summary['Epilepsy'] = {
                'accuracy': res['accuracy'],
                'f1': res['f1'],
                'subjects': len(y),
                'cv_mean': res['cv_mean']
            }
            print(f"\n  RESULT: {res['accuracy']:.1%} accuracy, F1: {res['f1']:.1%}")
        else:
            print("  ERROR: No data loaded")
            results_summary['Epilepsy'] = {'error': 'No data'}
    except Exception as e:
        print(f"  ERROR: {e}")
        results_summary['Epilepsy'] = {'error': str(e)}
    print()

    # FINAL SUMMARY
    print("="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    print()
    print(f"  {'Disease':<20} {'Subjects':<12} {'Accuracy':<12} {'F1 Score':<12} {'CV Mean':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for disease, result in results_summary.items():
        if 'error' in result:
            print(f"  {disease:<20} {'N/A':<12} {'ERROR':<12} {'-':<12} {'-':<12}")
        else:
            print(f"  {disease:<20} {result['subjects']:<12} {result['accuracy']:.1%}{'':7} {result['f1']:.1%}{'':7} {result['cv_mean']:.1%}")

    print()
    print("="*70)

    # Save results to JSON
    import json
    with open('disease_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=float)
    print("  Results saved to: disease_results.json")
    print("="*70)


if __name__ == "__main__":
    main()
