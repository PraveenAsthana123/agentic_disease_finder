#!/usr/bin/env python3
"""
Dataset Configuration Loader
AgenticFinder - Multi-Disease EEG Detection System

Usage:
    from config.dataset_loader import DatasetConfig

    config = DatasetConfig()

    # Get all datasets for a disease
    scz_datasets = config.get_disease_datasets('schizophrenia')

    # Get specific dataset path
    path = config.get_dataset_path('schizophrenia', 'RepOD')

    # Get all available (downloaded) datasets
    available = config.get_available_datasets()
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

class DatasetConfig:
    """Dataset configuration manager for AgenticFinder"""

    CONFIG_DIR = Path("/media/praveen/Asthana3/rajveer/agenticfinder/config")
    DB_PATH = CONFIG_DIR / "datasets.db"
    JSON_PATH = CONFIG_DIR / "datasets.json"

    def __init__(self):
        """Initialize dataset configuration"""
        self._load_json_config()

    def _load_json_config(self):
        """Load JSON configuration"""
        with open(self.JSON_PATH, 'r') as f:
            self._config = json.load(f)

    def _get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.DB_PATH)

    # =========================================================================
    # Disease Methods
    # =========================================================================

    def get_all_diseases(self) -> List[Dict]:
        """Get all diseases with their status"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT name, status, total_subjects, accuracy FROM diseases')
        diseases = [
            {'name': name, 'status': status, 'subjects': subjects, 'accuracy': accuracy}
            for name, status, subjects, accuracy in cursor.fetchall()
        ]
        conn.close()
        return diseases

    def get_disease_info(self, disease: str) -> Optional[Dict]:
        """Get information about a specific disease"""
        disease = disease.lower()
        if disease in self._config['diseases']:
            return self._config['diseases'][disease]
        return None

    def get_disease_datasets(self, disease: str) -> List[Dict]:
        """Get all datasets for a disease"""
        info = self.get_disease_info(disease)
        if info:
            return info.get('datasets', [])
        return []

    # =========================================================================
    # Dataset Methods
    # =========================================================================

    def get_dataset_path(self, disease: str, dataset_name: str) -> Optional[str]:
        """Get the path to a specific dataset"""
        datasets = self.get_disease_datasets(disease)
        for ds in datasets:
            if ds['name'].lower() == dataset_name.lower():
                return ds['path']
        return None

    def get_available_datasets(self) -> List[Dict]:
        """Get all downloaded/available datasets"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT d.name as disease, ds.name, ds.path, ds.subjects, ds.format
            FROM datasets ds
            JOIN diseases d ON ds.disease_id = d.id
            WHERE ds.is_downloaded = 1
        ''')
        datasets = [
            {'disease': row[0], 'name': row[1], 'path': row[2], 'subjects': row[3], 'format': row[4]}
            for row in cursor.fetchall()
        ]
        conn.close()
        return datasets

    def get_pending_datasets(self) -> List[Dict]:
        """Get all pending (not downloaded) datasets"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT d.name as disease, ds.name, ds.download_url, ds.requires_registration
            FROM datasets ds
            JOIN diseases d ON ds.disease_id = d.id
            WHERE ds.is_downloaded = 0
        ''')
        datasets = [
            {'disease': row[0], 'name': row[1], 'url': row[2], 'requires_registration': bool(row[3])}
            for row in cursor.fetchall()
        ]
        conn.close()
        return datasets

    # =========================================================================
    # Path Methods
    # =========================================================================

    def get_base_path(self, name: str) -> Optional[str]:
        """Get a base path by name"""
        if name in self._config['base_paths']:
            return self._config['base_paths'][name]['path']
        return None

    def get_all_base_paths(self) -> Dict[str, str]:
        """Get all base paths"""
        return {name: info['path'] for name, info in self._config['base_paths'].items()}

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def get_summary(self) -> Dict:
        """Get configuration summary"""
        diseases = self.get_all_diseases()
        return {
            'total_diseases': len(diseases),
            'real_data': [d['name'] for d in diseases if d['status'] == 'REAL_DATA'],
            'downloading': [d['name'] for d in diseases if d['status'] == 'DOWNLOADING'],
            'pending': [d['name'] for d in diseases if d['status'] == 'PENDING'],
            'total_subjects': sum(d['subjects'] for d in diseases),
            'available_datasets': len(self.get_available_datasets()),
            'pending_datasets': len(self.get_pending_datasets())
        }

    def print_summary(self):
        """Print configuration summary"""
        summary = self.get_summary()
        print("=" * 60)
        print("AGENTICFINDER DATASET CONFIGURATION")
        print("=" * 60)
        print(f"\nTotal Diseases: {summary['total_diseases']}")
        print(f"Total Subjects: {summary['total_subjects']}")
        print(f"\n✅ Real Data Available: {', '.join(summary['real_data'])}")
        print(f"🔄 Downloading: {', '.join(summary['downloading'])}")
        print(f"⏳ Pending: {', '.join(summary['pending'])}")
        print(f"\nAvailable Datasets: {summary['available_datasets']}")
        print(f"Pending Downloads: {summary['pending_datasets']}")


# CLI usage
if __name__ == "__main__":
    config = DatasetConfig()
    config.print_summary()

    print("\n" + "=" * 60)
    print("AVAILABLE DATASETS")
    print("=" * 60)
    for ds in config.get_available_datasets():
        print(f"  [{ds['disease']}] {ds['name']}: {ds['subjects']} subjects ({ds['format']})")
        print(f"      Path: {ds['path']}")
