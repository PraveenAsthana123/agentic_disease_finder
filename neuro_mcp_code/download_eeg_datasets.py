#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEG Dataset Downloader for NeuroMCP-Agent
Downloads open-source EEG datasets for neurological disease detection research.

COMPATIBILITY:
    - Python 3.6+ (backward compatible)
    - Windows, Linux, macOS (cross-platform)
    - No wget required (uses Python urllib as fallback)

Usage:
    python download_eeg_datasets.py --all           # Download all available datasets
    python download_eeg_datasets.py --epilepsy      # Download epilepsy datasets
    python download_eeg_datasets.py --parkinson     # Download Parkinson's datasets
    python download_eeg_datasets.py --alzheimer     # Download Alzheimer's datasets
    python download_eeg_datasets.py --list          # List all datasets
    python download_eeg_datasets.py --validation    # Download 3 validation datasets

Note: Some datasets require registration. This script will download freely available
datasets and provide instructions for those requiring Data Use Agreements (DUA).

Author: NeuroMCP-Agent Team
License: MIT
"""

from __future__ import print_function, division, absolute_import
import os
import sys
import argparse
import subprocess
import platform
import zipfile
import tarfile
import shutil
from pathlib import Path
import json

# Python 2/3 compatibility for urllib
try:
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
except ImportError:
    from urllib2 import urlopen, Request, URLError, HTTPError

# Type hints (Python 3.5+, but optional)
try:
    from typing import Dict, List, Optional
except ImportError:
    pass  # Type hints are optional

# Base directory for datasets
BASE_DIR = Path(__file__).parent / "data" / "eeg_datasets"

# Dataset configurations
DATASETS = {
    # =====================
    # EPILEPSY DATASETS
    # =====================
    "epilepsy": {
        "chb_mit": {
            "name": "CHB-MIT Scalp EEG Database",
            "url": "https://physionet.org/content/chbmit/1.0.0/",
            "download_cmd": "wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/",
            "license": "ODC-BY (Open)",
            "subjects": 23,
            "format": "EDF",
            "description": "Pediatric seizure recordings from Boston Children's Hospital",
            "auto_download": True
        },
        "bonn_university": {
            "name": "Bonn University Epilepsy Dataset",
            "url": "https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/",
            "download_cmd": None,
            "license": "Research Use",
            "subjects": 500,
            "format": "ASCII/TXT",
            "description": "5-class seizure classification benchmark dataset",
            "auto_download": False,
            "instructions": "Visit URL and download manually. Dataset includes 5 sets (A-E) with 100 segments each."
        },
        "siena_scalp": {
            "name": "SIENA Scalp EEG Database",
            "url": "https://physionet.org/content/siena-scalp-eeg/1.0.0/",
            "download_cmd": "wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/",
            "license": "ODC-BY (Open)",
            "subjects": 14,
            "format": "EDF",
            "description": "Long-term epilepsy monitoring EEG recordings",
            "auto_download": True
        },
        "tuh_seizure": {
            "name": "TUH EEG Seizure Corpus",
            "url": "https://isip.piconepress.com/projects/tuh_eeg/",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 642,
            "format": "EDF",
            "description": "Largest clinical seizure corpus - requires registration",
            "auto_download": False,
            "instructions": "1. Visit https://isip.piconepress.com/projects/tuh_eeg/\n2. Create account\n3. Sign DUA\n4. Download via provided scripts"
        },
        "epilepsy_ieeg": {
            "name": "Epilepsy iEEG Dataset",
            "url": "https://openneuro.org/datasets/ds003029",
            "download_cmd": "openneuro-py download --dataset ds003029",
            "license": "CC0 (Public Domain)",
            "subjects": 16,
            "format": "BIDS",
            "description": "Intracranial EEG recordings for epilepsy research",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "zenodo_epilepsy": {
            "name": "Zenodo Epilepsy EEG",
            "url": "https://zenodo.org/record/4940267",
            "download_cmd": "wget https://zenodo.org/record/4940267/files/EEG_Epilepsy_Dataset.zip",
            "license": "CC-BY",
            "subjects": 24,
            "format": "EDF",
            "description": "Open epilepsy EEG dataset from Zenodo",
            "auto_download": True
        }
    },

    # =====================
    # PARKINSON'S DATASETS
    # =====================
    "parkinson": {
        "ppmi": {
            "name": "PPMI - Parkinson's Progression Markers Initiative",
            "url": "https://www.ppmi-info.org/access-data-specimens/download-data/",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 423,
            "format": "EDF/CSV",
            "description": "Multi-site Parkinson's disease initiative - requires registration",
            "auto_download": False,
            "instructions": "1. Visit https://www.ppmi-info.org/\n2. Register for account\n3. Sign DUA\n4. Download from LONI portal"
        },
        "ucsd_pd": {
            "name": "UC San Diego Parkinson's EEG",
            "url": "https://openneuro.org/datasets/ds003490",
            "download_cmd": "openneuro-py download --dataset ds003490",
            "license": "CC0 (Public Domain)",
            "subjects": 31,
            "format": "BIDS",
            "description": "Resting-state EEG from Parkinson's patients",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "physionet_gait_pd": {
            "name": "PhysioNet Gait in Parkinson's Disease",
            "url": "https://physionet.org/content/gaitpdb/1.0.0/",
            "download_cmd": "wget -r -N -c -np https://physionet.org/files/gaitpdb/1.0.0/",
            "license": "ODC-BY (Open)",
            "subjects": 93,
            "format": "TXT",
            "description": "Gait signals from PD patients with vertical ground reaction force",
            "auto_download": True
        },
        "motor_imagery": {
            "name": "EEG Motor Movement/Imagery Dataset",
            "url": "https://physionet.org/content/eegmmidb/1.0.0/",
            "download_cmd": "wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/",
            "license": "ODC-BY (Open)",
            "subjects": 109,
            "format": "EDF",
            "description": "Motor movement and imagery EEG recordings",
            "auto_download": True
        },
        "openneuro_ds002778": {
            "name": "OpenNeuro Parkinson's ds002778",
            "url": "https://openneuro.org/datasets/ds002778",
            "download_cmd": "openneuro-py download --dataset ds002778",
            "license": "CC0 (Public Domain)",
            "subjects": 54,
            "format": "BIDS",
            "description": "Parkinson's disease neuroimaging dataset",
            "auto_download": True,
            "requires": ["openneuro-py"]
        }
    },

    # =====================
    # ALZHEIMER'S DATASETS
    # =====================
    "alzheimer": {
        "adni": {
            "name": "ADNI - Alzheimer's Disease Neuroimaging Initiative",
            "url": "https://adni.loni.usc.edu/data-samples/access-data/",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 1200,
            "format": "EDF",
            "description": "Multi-center Alzheimer's study - requires registration",
            "auto_download": False,
            "instructions": "1. Visit https://adni.loni.usc.edu/\n2. Apply for access\n3. Sign DUA\n4. Download from IDA portal"
        },
        "openneuro_ds004504": {
            "name": "OpenNeuro Alzheimer's EEG ds004504",
            "url": "https://openneuro.org/datasets/ds004504",
            "download_cmd": "openneuro-py download --dataset ds004504",
            "license": "CC0 (Public Domain)",
            "subjects": 88,
            "format": "BIDS",
            "description": "AD/MCI/Healthy control EEG dataset",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "eeg_ad_kaggle": {
            "name": "EEG Alzheimer's Dataset (Kaggle)",
            "url": "https://www.kaggle.com/datasets/gaborvecsei/eeg-alzheimers",
            "download_cmd": "kaggle datasets download -d gaborvecsei/eeg-alzheimers",
            "license": "CC0 (Public Domain)",
            "subjects": 36,
            "format": "CSV",
            "description": "Clinical Alzheimer's EEG recordings",
            "auto_download": True,
            "requires": ["kaggle"]
        },
        "oasis3": {
            "name": "OASIS-3 Brain Database",
            "url": "https://www.oasis-brains.org/",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 1098,
            "format": "NIfTI",
            "description": "Longitudinal neuroimaging and clinical data",
            "auto_download": False,
            "instructions": "1. Visit https://www.oasis-brains.org/\n2. Register\n3. Request access\n4. Download via XNAT"
        },
        "openneuro_ds003507": {
            "name": "OpenNeuro Alzheimer's ds003507",
            "url": "https://openneuro.org/datasets/ds003507",
            "download_cmd": "openneuro-py download --dataset ds003507",
            "license": "CC0 (Public Domain)",
            "subjects": 29,
            "format": "BIDS",
            "description": "Alzheimer's disease EEG recordings",
            "auto_download": True,
            "requires": ["openneuro-py"]
        }
    },

    # =====================
    # SCHIZOPHRENIA DATASETS
    # =====================
    "schizophrenia": {
        "cobre": {
            "name": "COBRE - Center for Biomedical Research Excellence",
            "url": "http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 145,
            "format": "NIfTI/EDF",
            "description": "Multi-site schizophrenia neuroimaging",
            "auto_download": False,
            "instructions": "1. Visit NITRC website\n2. Create account\n3. Request data access\n4. Download via AWS S3"
        },
        "ucla_cnp": {
            "name": "UCLA Consortium for Neuropsychiatric Phenomics",
            "url": "https://openneuro.org/datasets/ds000030",
            "download_cmd": "openneuro-py download --dataset ds000030",
            "license": "CC0 (Public Domain)",
            "subjects": 130,
            "format": "BIDS",
            "description": "Schizophrenia, bipolar, and ADHD neuroimaging",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "openneuro_ds002761": {
            "name": "OpenNeuro Schizophrenia ds002761",
            "url": "https://openneuro.org/datasets/ds002761",
            "download_cmd": "openneuro-py download --dataset ds002761",
            "license": "CC0 (Public Domain)",
            "subjects": 36,
            "format": "BIDS",
            "description": "Schizophrenia EEG/fMRI dataset",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "kaggle_sz": {
            "name": "Kaggle Schizophrenia EEG",
            "url": "https://www.kaggle.com/datasets/broach/button-tone-sz",
            "download_cmd": "kaggle datasets download -d broach/button-tone-sz",
            "license": "CC0 (Public Domain)",
            "subjects": 84,
            "format": "CSV",
            "description": "Button-tone schizophrenia EEG task",
            "auto_download": True,
            "requires": ["kaggle"]
        }
    },

    # =====================
    # AUTISM DATASETS
    # =====================
    "autism": {
        "abide_i": {
            "name": "ABIDE I - Autism Brain Imaging Data Exchange",
            "url": "http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 539,
            "format": "NIfTI",
            "description": "Multi-site autism neuroimaging consortium",
            "auto_download": False,
            "instructions": "1. Visit ABIDE website\n2. Review DUA\n3. Download via provided scripts or AWS S3"
        },
        "abide_ii": {
            "name": "ABIDE II - Autism Brain Imaging Data Exchange",
            "url": "http://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 521,
            "format": "NIfTI",
            "description": "Extended autism neuroimaging dataset",
            "auto_download": False,
            "instructions": "1. Visit ABIDE II website\n2. Review DUA\n3. Download via provided scripts"
        },
        "openneuro_ds004186": {
            "name": "OpenNeuro Autism EEG ds004186",
            "url": "https://openneuro.org/datasets/ds004186",
            "download_cmd": "openneuro-py download --dataset ds004186",
            "license": "CC0 (Public Domain)",
            "subjects": 36,
            "format": "BIDS",
            "description": "High-density autism EEG recordings",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "openneuro_ds002843": {
            "name": "OpenNeuro Autism ds002843",
            "url": "https://openneuro.org/datasets/ds002843",
            "download_cmd": "openneuro-py download --dataset ds002843",
            "license": "CC0 (Public Domain)",
            "subjects": 50,
            "format": "BIDS",
            "description": "Autism spectrum disorder EEG",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "adhd_200": {
            "name": "ADHD-200 Dataset",
            "url": "http://fcon_1000.projects.nitrc.org/indi/adhd200/",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 973,
            "format": "NIfTI",
            "description": "ADHD neuroimaging consortium dataset",
            "auto_download": False,
            "instructions": "1. Visit ADHD-200 website\n2. Review DUA\n3. Download via Athena or NITRC"
        }
    },

    # =====================
    # DEPRESSION DATASETS
    # =====================
    "depression": {
        "modma": {
            "name": "MODMA - Multi-modal Open Dataset for Mental-disorder Analysis",
            "url": "http://modma.lzu.edu.cn/data/index/",
            "download_cmd": None,
            "license": "Research Use",
            "subjects": 53,
            "format": "EDF/MAT",
            "description": "Multi-modal depression dataset from Lanzhou University",
            "auto_download": False,
            "instructions": "1. Visit MODMA website\n2. Register for account\n3. Request data access"
        },
        "openneuro_ds003478": {
            "name": "OpenNeuro Depression EEG ds003478",
            "url": "https://openneuro.org/datasets/ds003478",
            "download_cmd": "openneuro-py download --dataset ds003478",
            "license": "CC0 (Public Domain)",
            "subjects": 122,
            "format": "BIDS",
            "description": "Resting-state major depression EEG",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "openneuro_ds002748": {
            "name": "OpenNeuro MDD REST ds002748",
            "url": "https://openneuro.org/datasets/ds002748",
            "download_cmd": "openneuro-py download --dataset ds002748",
            "license": "CC0 (Public Domain)",
            "subjects": 384,
            "format": "BIDS",
            "description": "Major depressive disorder resting-state",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "openneuro_ds003653": {
            "name": "OpenNeuro Depression ds003653",
            "url": "https://openneuro.org/datasets/ds003653",
            "download_cmd": "openneuro-py download --dataset ds003653",
            "license": "CC0 (Public Domain)",
            "subjects": 56,
            "format": "BIDS",
            "description": "Depression EEG dataset",
            "auto_download": True,
            "requires": ["openneuro-py"]
        },
        "tdbrain": {
            "name": "TDBRAIN - Treatment-resistant Depression",
            "url": "https://brainclinics.com/resources/",
            "download_cmd": None,
            "license": "Research Use",
            "subjects": 1274,
            "format": "EDF",
            "description": "Large treatment-resistant depression EEG database",
            "auto_download": False,
            "instructions": "1. Visit Brainclinics website\n2. Request access\n3. Sign agreement"
        }
    },

    # =====================
    # STRESS/EMOTION DATASETS
    # =====================
    "stress": {
        "deap": {
            "name": "DEAP - Database for Emotion Analysis using Physiological Signals",
            "url": "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 32,
            "format": "BDF/MAT",
            "description": "Multimodal emotion and stress analysis",
            "auto_download": False,
            "instructions": "1. Visit DEAP website\n2. Request access via email\n3. Sign EULA\n4. Receive download link"
        },
        "wesad": {
            "name": "WESAD - Wearable Stress and Affect Detection",
            "url": "https://archive.ics.uci.edu/ml/datasets/WESAD",
            "download_cmd": "wget https://archive.ics.uci.edu/ml/machine-learning-databases/00465/WESAD.zip",
            "license": "CC-BY",
            "subjects": 15,
            "format": "CSV/PKL",
            "description": "Multimodal stress detection with wearables",
            "auto_download": True
        },
        "dreamer": {
            "name": "DREAMER - Affect Recognition Database",
            "url": "https://zenodo.org/record/546113",
            "download_cmd": "wget https://zenodo.org/record/546113/files/DREAMER.mat",
            "license": "CC-BY",
            "subjects": 23,
            "format": "MAT",
            "description": "EEG-based affect recognition during film watching",
            "auto_download": True
        },
        "seed": {
            "name": "SEED - SJTU Emotion EEG Dataset",
            "url": "https://bcmi.sjtu.edu.cn/home/seed/",
            "download_cmd": None,
            "license": "Research Use",
            "subjects": 15,
            "format": "MAT",
            "description": "Emotion recognition EEG dataset",
            "auto_download": False,
            "instructions": "1. Visit BCMI Lab website\n2. Request access\n3. Sign agreement"
        },
        "seed_iv": {
            "name": "SEED-IV - Multi-session Emotion Dataset",
            "url": "https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html",
            "download_cmd": None,
            "license": "Research Use",
            "subjects": 15,
            "format": "MAT",
            "description": "Multi-session emotion EEG with 4 emotions",
            "auto_download": False,
            "instructions": "1. Visit SEED-IV page\n2. Request access\n3. Sign agreement"
        },
        "mahnob_hci": {
            "name": "MAHNOB-HCI Database",
            "url": "https://mahnob-db.eu/hci-tagging/",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 27,
            "format": "BDF",
            "description": "Multimodal emotion tagging database",
            "auto_download": False,
            "instructions": "1. Visit MAHNOB-HCI website\n2. Register\n3. Request access"
        },
        "amigos": {
            "name": "AMIGOS - Affect, Personality and Mood Research",
            "url": "http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 40,
            "format": "MAT",
            "description": "Multimodal affect and personality dataset",
            "auto_download": False,
            "instructions": "1. Visit AMIGOS website\n2. Request access\n3. Sign EULA"
        }
    },

    # =====================
    # SLEEP DATASETS
    # =====================
    "sleep": {
        "sleep_edf": {
            "name": "Sleep-EDF Database Expanded",
            "url": "https://physionet.org/content/sleep-edfx/1.0.0/",
            "download_cmd": "wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/",
            "license": "ODC-BY (Open)",
            "subjects": 197,
            "format": "EDF",
            "description": "Polysomnographic sleep recordings",
            "auto_download": True
        },
        "shhs": {
            "name": "SHHS - Sleep Heart Health Study",
            "url": "https://sleepdata.org/datasets/shhs",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 5804,
            "format": "EDF",
            "description": "Large-scale sleep and cardiovascular health study",
            "auto_download": False,
            "instructions": "1. Visit sleepdata.org\n2. Create account\n3. Request access to SHHS"
        },
        "isruc_sleep": {
            "name": "ISRUC-Sleep Dataset",
            "url": "https://sleeptight.isr.uc.pt/",
            "download_cmd": None,
            "license": "Research Use",
            "subjects": 100,
            "format": "EDF",
            "description": "Sleep staging and disorder detection",
            "auto_download": False,
            "instructions": "1. Visit ISRUC website\n2. Request access"
        },
        "physionet_fatigue": {
            "name": "PhysioNet Driver Drowsiness EEG",
            "url": "https://physionet.org/content/driving-drowsiness/1.0.0/",
            "download_cmd": "wget -r -N -c -np https://physionet.org/files/driving-drowsiness/1.0.0/",
            "license": "ODC-BY (Open)",
            "subjects": 12,
            "format": "EDF",
            "description": "EEG during simulated driving for drowsiness detection",
            "auto_download": True
        }
    },

    # =====================
    # BCI DATASETS
    # =====================
    "bci": {
        "bci_competition_iv": {
            "name": "BCI Competition IV",
            "url": "https://www.bbci.de/competition/iv/",
            "download_cmd": None,
            "license": "Research Use",
            "subjects": 9,
            "format": "GDF/MAT",
            "description": "Brain-computer interface benchmark datasets",
            "auto_download": False,
            "instructions": "1. Visit BCI Competition website\n2. Download individual datasets"
        },
        "grasp_lift": {
            "name": "Grasp and Lift EEG Detection",
            "url": "https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data",
            "download_cmd": "kaggle competitions download -c grasp-and-lift-eeg-detection",
            "license": "CC0 (Public Domain)",
            "subjects": 12,
            "format": "CSV",
            "description": "EEG during grasp and lift movements",
            "auto_download": True,
            "requires": ["kaggle"]
        },
        "stew_workload": {
            "name": "STEW - Simultaneous Task EEG Workload",
            "url": "https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload",
            "download_cmd": None,
            "license": "CC-BY",
            "subjects": 48,
            "format": "MAT",
            "description": "Mental workload during multitasking",
            "auto_download": False,
            "instructions": "1. Visit IEEE DataPort\n2. Download dataset"
        }
    },

    # =====================
    # GENERAL EEG DATASETS
    # =====================
    "general": {
        "tuh_eeg_corpus": {
            "name": "TUH EEG Corpus",
            "url": "https://isip.piconepress.com/projects/tuh_eeg/",
            "download_cmd": None,
            "license": "DUA Required",
            "subjects": 30000,
            "format": "EDF",
            "description": "Largest clinical EEG corpus worldwide",
            "auto_download": False,
            "instructions": "1. Visit TUH EEG website\n2. Create account\n3. Sign DUA\n4. Use rsync for download"
        }
    }
}


def print_header():
    """Print script header."""
    print("=" * 70)
    print("  EEG Dataset Downloader for NeuroMCP-Agent")
    print("  Neurological Disease Detection Research")
    print("=" * 70)
    print()


def install_dependencies():
    """Install required Python packages."""
    packages = ["openneuro-py", "kaggle", "requests", "tqdm"]
    print("Installing dependencies...")
    for pkg in packages:
        try:
            # Python 3.6 compatible (no capture_output)
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"],
                         check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("  [OK] {}".format(pkg))
        except Exception:
            print("  [SKIP] {} (install manually if needed)".format(pkg))


def create_directories(categories: List[str]):
    """Create directory structure for datasets."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for category in categories:
        (BASE_DIR / category).mkdir(exist_ok=True)
    print(f"Created directory structure at: {BASE_DIR}")


def list_datasets():
    """List all available datasets."""
    print_header()
    print("AVAILABLE EEG DATASETS")
    print("=" * 70)

    total_subjects = 0
    total_datasets = 0

    for category, datasets in DATASETS.items():
        print(f"\n[{category.upper()}]")
        print("-" * 40)
        for key, info in datasets.items():
            auto = "[AUTO]" if info.get("auto_download") else "[MANUAL]"
            print(f"  {auto} {info['name']}")
            print(f"         Subjects: {info['subjects']} | Format: {info['format']} | License: {info['license']}")
            total_subjects += info['subjects'] if isinstance(info['subjects'], int) else 0
            total_datasets += 1

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_datasets} datasets | ~{total_subjects:,}+ subjects")
    print("[AUTO] = Can be downloaded automatically")
    print("[MANUAL] = Requires registration/DUA")


def download_dataset(category: str, key: str, info: dict, output_dir: Path):
    """Download a single dataset."""
    print(f"\n{'='*50}")
    print(f"Downloading: {info['name']}")
    print(f"{'='*50}")

    dataset_dir = output_dir / category / key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "name": info['name'],
        "url": info['url'],
        "license": info['license'],
        "subjects": info['subjects'],
        "format": info['format'],
        "description": info['description']
    }
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if not info.get("auto_download"):
        print(f"  [!] Manual download required")
        print(f"  URL: {info['url']}")
        if "instructions" in info:
            print(f"  Instructions:")
            for line in info['instructions'].split('\n'):
                print(f"    {line}")

        # Create instructions file
        with open(dataset_dir / "DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
            f.write(f"Dataset: {info['name']}\n")
            f.write(f"URL: {info['url']}\n")
            f.write(f"License: {info['license']}\n\n")
            if "instructions" in info:
                f.write("Instructions:\n")
                f.write(info['instructions'])
        return False

    # Check dependencies
    if "requires" in info:
        for req in info["requires"]:
            if req == "openneuro-py":
                try:
                    import openneuro
                except ImportError:
                    print(f"  Installing {req}...")
                    subprocess.run([sys.executable, "-m", "pip", "install", req, "-q"])
            elif req == "kaggle":
                # Check for kaggle credentials
                kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
                if not kaggle_json.exists():
                    print(f"  [!] Kaggle credentials not found")
                    print(f"  Please setup ~/.kaggle/kaggle.json")
                    return False

    # Execute download command
    cmd = info['download_cmd']
    if cmd:
        print("  Running: {}".format(cmd))
        try:
            # Check if wget is available on Windows
            if platform.system() == 'Windows' and 'wget' in cmd:
                # Convert wget to Python download
                return download_with_python(cmd, dataset_dir)

            os.chdir(dataset_dir)
            # Python 3.6 compatible (no capture_output, no text)
            result = subprocess.run(cmd, shell=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                print("  [OK] Download complete")
                return True
            else:
                error_msg = result.stderr.decode('utf-8', errors='ignore')[:200]
                print("  [ERROR] {}".format(error_msg))
                return False
        except Exception as e:
            print("  [ERROR] {}".format(str(e)))
            return False

    return False


def download_with_python(wget_cmd, output_dir):
    """
    Convert wget command to Python download (Windows compatibility).
    Handles: wget -r -N -c -np <url>
    """
    import re

    # Extract URL from wget command
    url_match = re.search(r'https?://[^\s]+', wget_cmd)
    if not url_match:
        print("  [ERROR] Could not parse URL from command")
        return False

    url = url_match.group(0)
    print("  [INFO] Using Python download (Windows compatible)")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; NeuroMCP-Agent/1.0)'}
        request = Request(url, headers=headers)
        response = urlopen(request, timeout=60)

        # Determine filename from URL
        filename = os.path.basename(url.rstrip('/'))
        if not filename:
            filename = 'downloaded_data'

        output_path = os.path.join(str(output_dir), filename)

        # Download file
        with open(output_path, 'wb') as f:
            while True:
                chunk = response.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)

        print("  [OK] Downloaded: {}".format(output_path))
        return True

    except Exception as e:
        print("  [ERROR] Download failed: {}".format(str(e)))
        return False


def download_category(category: str, output_dir: Path):
    """Download all datasets in a category."""
    if category not in DATASETS:
        print(f"Unknown category: {category}")
        print(f"Available: {list(DATASETS.keys())}")
        return

    print(f"\nDownloading {category.upper()} datasets...")
    datasets = DATASETS[category]

    success = 0
    manual = 0
    failed = 0

    for key, info in datasets.items():
        result = download_dataset(category, key, info, output_dir)
        if result:
            success += 1
        elif not info.get("auto_download"):
            manual += 1
        else:
            failed += 1

    print(f"\n{category.upper()} Summary:")
    print(f"  Downloaded: {success}")
    print(f"  Manual required: {manual}")
    print(f"  Failed: {failed}")


def download_all(output_dir: Path):
    """Download all available datasets."""
    print_header()
    print("Downloading ALL datasets...")
    print(f"Output directory: {output_dir}")

    create_directories(list(DATASETS.keys()))

    total_success = 0
    total_manual = 0
    total_failed = 0

    for category in DATASETS.keys():
        for key, info in DATASETS[category].items():
            result = download_dataset(category, key, info, output_dir)
            if result:
                total_success += 1
            elif not info.get("auto_download"):
                total_manual += 1
            else:
                total_failed += 1

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  Successfully downloaded: {total_success}")
    print(f"  Manual download required: {total_manual}")
    print(f"  Failed: {total_failed}")
    print(f"\nDatasets saved to: {output_dir}")


def create_download_script():
    """Create shell scripts for batch downloading."""
    script_dir = BASE_DIR / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    # PhysioNet download script
    physionet_script = """#!/bin/bash
# PhysioNet EEG Datasets Download Script
# Requires: wget

echo "Downloading PhysioNet EEG Datasets..."

# CHB-MIT Epilepsy
echo "1. CHB-MIT Scalp EEG Database"
wget -r -N -c -np -P ./epilepsy/chb_mit https://physionet.org/files/chbmit/1.0.0/

# SIENA Epilepsy
echo "2. SIENA Scalp EEG Database"
wget -r -N -c -np -P ./epilepsy/siena https://physionet.org/files/siena-scalp-eeg/1.0.0/

# Sleep-EDF
echo "3. Sleep-EDF Database"
wget -r -N -c -np -P ./sleep/sleep_edf https://physionet.org/files/sleep-edfx/1.0.0/

# Motor Imagery
echo "4. Motor Movement/Imagery Dataset"
wget -r -N -c -np -P ./bci/motor_imagery https://physionet.org/files/eegmmidb/1.0.0/

# Gait PD
echo "5. Gait in Parkinson's Disease"
wget -r -N -c -np -P ./parkinson/gait_pd https://physionet.org/files/gaitpdb/1.0.0/

# Driver Drowsiness
echo "6. Driver Drowsiness EEG"
wget -r -N -c -np -P ./sleep/drowsiness https://physionet.org/files/driving-drowsiness/1.0.0/

echo "PhysioNet downloads complete!"
"""

    with open(script_dir / "download_physionet.sh", "w") as f:
        f.write(physionet_script)

    # OpenNeuro download script
    openneuro_script = """#!/bin/bash
# OpenNeuro EEG Datasets Download Script
# Requires: pip install openneuro-py

echo "Downloading OpenNeuro EEG Datasets..."

# Install openneuro-py if not present
pip install openneuro-py -q

# Parkinson's
echo "1. UC San Diego Parkinson's (ds003490)"
openneuro-py download --dataset ds003490 -o ./parkinson/ucsd_pd

# Alzheimer's
echo "2. Alzheimer's EEG (ds004504)"
openneuro-py download --dataset ds004504 -o ./alzheimer/ds004504

echo "3. Alzheimer's EEG (ds003507)"
openneuro-py download --dataset ds003507 -o ./alzheimer/ds003507

# Schizophrenia
echo "4. UCLA CNP (ds000030)"
openneuro-py download --dataset ds000030 -o ./schizophrenia/ucla_cnp

echo "5. Schizophrenia EEG (ds002761)"
openneuro-py download --dataset ds002761 -o ./schizophrenia/ds002761

# Autism
echo "6. Autism EEG (ds004186)"
openneuro-py download --dataset ds004186 -o ./autism/ds004186

echo "7. Autism EEG (ds002843)"
openneuro-py download --dataset ds002843 -o ./autism/ds002843

# Depression
echo "8. Depression EEG (ds003478)"
openneuro-py download --dataset ds003478 -o ./depression/ds003478

echo "9. MDD REST (ds002748)"
openneuro-py download --dataset ds002748 -o ./depression/ds002748

echo "10. Depression (ds003653)"
openneuro-py download --dataset ds003653 -o ./depression/ds003653

# Epilepsy
echo "11. Epilepsy iEEG (ds003029)"
openneuro-py download --dataset ds003029 -o ./epilepsy/ds003029

echo "OpenNeuro downloads complete!"
"""

    with open(script_dir / "download_openneuro.sh", "w") as f:
        f.write(openneuro_script)

    # Make scripts executable
    os.chmod(script_dir / "download_physionet.sh", 0o755)
    os.chmod(script_dir / "download_openneuro.sh", 0o755)

    print(f"Download scripts created in: {script_dir}")
    print("  - download_physionet.sh")
    print("  - download_openneuro.sh")


def main():
    parser = argparse.ArgumentParser(
        description="Download open-source EEG datasets for neurological disease detection"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--epilepsy", action="store_true", help="Download epilepsy datasets")
    parser.add_argument("--parkinson", action="store_true", help="Download Parkinson's datasets")
    parser.add_argument("--alzheimer", action="store_true", help="Download Alzheimer's datasets")
    parser.add_argument("--schizophrenia", action="store_true", help="Download schizophrenia datasets")
    parser.add_argument("--autism", action="store_true", help="Download autism datasets")
    parser.add_argument("--depression", action="store_true", help="Download depression datasets")
    parser.add_argument("--stress", action="store_true", help="Download stress/emotion datasets")
    parser.add_argument("--sleep", action="store_true", help="Download sleep datasets")
    parser.add_argument("--bci", action="store_true", help="Download BCI datasets")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--scripts", action="store_true", help="Create download shell scripts")
    parser.add_argument("--output", type=str, default=str(BASE_DIR), help="Output directory")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.install_deps:
        install_dependencies()
        return

    if args.list:
        list_datasets()
        return

    if args.scripts:
        create_download_script()
        return

    if args.all:
        download_all(output_dir)
        return

    # Download specific categories
    categories = []
    if args.epilepsy:
        categories.append("epilepsy")
    if args.parkinson:
        categories.append("parkinson")
    if args.alzheimer:
        categories.append("alzheimer")
    if args.schizophrenia:
        categories.append("schizophrenia")
    if args.autism:
        categories.append("autism")
    if args.depression:
        categories.append("depression")
    if args.stress:
        categories.append("stress")
    if args.sleep:
        categories.append("sleep")
    if args.bci:
        categories.append("bci")

    if categories:
        print_header()
        create_directories(categories)
        for cat in categories:
            download_category(cat, output_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python download_eeg_datasets.py --list")
        print("  python download_eeg_datasets.py --epilepsy --parkinson")
        print("  python download_eeg_datasets.py --all")
        print("  python download_eeg_datasets.py --scripts")


if __name__ == "__main__":
    main()
