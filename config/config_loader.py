#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Loader for NeuroMCP-Agent
Cross-platform compatible (Windows, Linux, macOS)
Python 3.6+ compatible

Usage:
    from config.config_loader import load_config, get_dataset_info

    config = load_config()
    epilepsy_datasets = get_dataset_info('epilepsy')
"""

from __future__ import print_function, division, absolute_import
import os
import sys
import json
from pathlib import Path

# Try to import yaml (optional)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def get_config_dir():
    """Get the config directory path (cross-platform)."""
    return Path(__file__).parent


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_type='data', format='auto'):
    """
    Load configuration file.

    Args:
        config_type: Type of config ('data', 'default', 'datasets')
        format: 'yaml', 'json', or 'auto' (tries both)

    Returns:
        dict: Configuration dictionary
    """
    config_dir = get_config_dir()

    config_files = {
        'data': ['data_config.yaml', 'data_config.json'],
        'default': ['../configs/default_config.yaml'],
        'datasets': ['datasets_config.yaml', 'datasets.json']
    }

    files = config_files.get(config_type, ['{}.yaml'.format(config_type), '{}.json'.format(config_type)])

    for filename in files:
        filepath = config_dir / filename
        if filepath.exists():
            return load_file(filepath)

    raise FileNotFoundError("Config file not found for type: {}".format(config_type))


def load_file(filepath):
    """Load a config file (YAML or JSON)."""
    filepath = Path(filepath)

    if filepath.suffix in ['.yaml', '.yml']:
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    elif filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    else:
        raise ValueError("Unsupported config format: {}".format(filepath.suffix))


def get_dataset_info(disease, dataset_id=None):
    """
    Get dataset information for a disease.

    Args:
        disease: Disease name (epilepsy, parkinson, etc.)
        dataset_id: Optional specific dataset ID

    Returns:
        dict: Dataset information
    """
    config = load_config('data')

    # Try YAML structure
    if disease in config:
        disease_config = config[disease]
        if dataset_id:
            if 'primary' in disease_config and dataset_id in disease_config['primary']:
                return disease_config['primary'][dataset_id]
            if 'supplementary' in disease_config and dataset_id in disease_config['supplementary']:
                return disease_config['supplementary'][dataset_id]
        return disease_config

    # Try JSON structure (diseases array)
    if 'diseases' in config:
        for d in config['diseases']:
            if d['name'] == disease:
                if dataset_id:
                    for ds in d.get('primary_datasets', []):
                        if ds['id'] == dataset_id:
                            return ds
                return d

    return None


def get_validation_datasets():
    """Get validation dataset information."""
    config = load_config('data')
    return config.get('validation_datasets', {})


def get_all_datasets():
    """Get list of all available datasets."""
    config = load_config('data')
    datasets = []

    # From YAML structure
    for disease in ['epilepsy', 'parkinson', 'alzheimer', 'schizophrenia',
                    'autism', 'depression', 'stress']:
        if disease in config:
            disease_config = config[disease]
            if 'primary' in disease_config:
                for key, info in disease_config['primary'].items():
                    datasets.append({
                        'disease': disease,
                        'id': key,
                        'type': 'primary',
                        **info
                    })
            if 'supplementary' in disease_config:
                for key, info in disease_config['supplementary'].items():
                    datasets.append({
                        'disease': disease,
                        'id': key,
                        'type': 'supplementary',
                        **info
                    })

    return datasets


def get_download_urls(source='all'):
    """Get download URLs for datasets."""
    config = load_config('data')

    if 'download_urls' in config:
        if source == 'all':
            return config['download_urls']
        return config['download_urls'].get(source, {})

    return {}


def get_feature_config():
    """Get feature extraction configuration."""
    config = load_config('data')
    return config.get('features', {})


def get_preprocessing_config():
    """Get preprocessing configuration."""
    config = load_config('data')
    return config.get('preprocessing', {})


def get_model_config(model_type='ultra_stacking'):
    """Get model configuration."""
    config = load_config('data')
    models = config.get('models', {})
    return models.get(model_type, {})


def get_paths():
    """Get configured paths."""
    config = load_config('data')
    paths = config.get('paths', {})

    # Resolve environment variables and relative paths
    project_root = str(get_project_root())
    resolved = {}

    for key, value in paths.items():
        if isinstance(value, str):
            # Replace ${PROJECT_ROOT} with actual path
            resolved[key] = value.replace('${PROJECT_ROOT}', project_root)
        else:
            resolved[key] = value

    return resolved


def list_diseases():
    """List all supported diseases."""
    return ['epilepsy', 'parkinson', 'alzheimer', 'schizophrenia',
            'autism', 'depression', 'stress']


def print_config_summary():
    """Print configuration summary."""
    config = load_config('data')

    print("=" * 60)
    print("  NeuroMCP-Agent Configuration Summary")
    print("=" * 60)
    print("")
    print("Version: {}".format(config.get('version', 'N/A')))
    print("Total Datasets: {}".format(config.get('total_datasets', 0)))
    print("Total Diseases: {}".format(config.get('total_diseases', 0)))
    print("")
    print("Diseases:")
    for disease in list_diseases():
        info = get_dataset_info(disease)
        if info:
            count = info.get('datasets_count', len(info.get('primary', {})) + len(info.get('supplementary', {})))
            print("  - {}: {} datasets".format(disease.capitalize(), count))
    print("")
    print("Validation Datasets:")
    for name, info in get_validation_datasets().items():
        status = info.get('status', 'unknown')
        size = info.get('size_mb', 0)
        print("  - {}: {} ({} MB)".format(name, status, size))


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Configuration Loader for NeuroMCP-Agent')
    parser.add_argument('--summary', action='store_true', help='Print config summary')
    parser.add_argument('--disease', type=str, help='Get info for specific disease')
    parser.add_argument('--dataset', type=str, help='Get info for specific dataset')
    parser.add_argument('--list', action='store_true', help='List all datasets')
    parser.add_argument('--paths', action='store_true', help='Show configured paths')
    parser.add_argument('--urls', action='store_true', help='Show download URLs')

    args = parser.parse_args()

    if args.summary or not any([args.disease, args.list, args.paths, args.urls]):
        print_config_summary()

    if args.disease:
        info = get_dataset_info(args.disease, args.dataset)
        if info:
            print(json.dumps(info, indent=2, default=str))
        else:
            print("Disease not found: {}".format(args.disease))

    if args.list:
        datasets = get_all_datasets()
        for ds in datasets:
            print("{}/{}: {} ({})".format(
                ds['disease'], ds['id'], ds.get('name', 'N/A'), ds['type']))

    if args.paths:
        paths = get_paths()
        for key, value in paths.items():
            print("{}: {}".format(key, value))

    if args.urls:
        urls = get_download_urls()
        print(json.dumps(urls, indent=2))
