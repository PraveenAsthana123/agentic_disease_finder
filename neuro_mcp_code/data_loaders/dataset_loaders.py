"""
Dataset Loaders for Neurological Disease Detection
===================================================
Provides data loading utilities for three major neurological disease datasets:
1. ADNI (Alzheimer's Disease Neuroimaging Initiative) - Alzheimer's
2. PPMI (Parkinson's Progression Markers Initiative) - Parkinson's
3. COBRE (Center for Biomedical Research Excellence) - Schizophrenia

Author: Research Team
Project: Neurological Disease Detection using Agentic AI
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Generator
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import json
from pathlib import Path

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logger = logging.getLogger(__name__)


@dataclass
class Subject:
    """Data structure for a single subject"""
    subject_id: str
    diagnosis: str
    age: float
    sex: str
    data: Dict[str, np.ndarray]
    clinical: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class DatasetInfo:
    """Dataset metadata"""
    name: str
    disease: str
    description: str
    n_subjects: int
    modalities: List[str]
    labels: List[str]
    source_url: str


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders

    Provides common interface for loading neuroimaging datasets.
    """

    def __init__(self, data_path: str, cache_dir: str = None):
        """
        Initialize dataset loader

        Parameters
        ----------
        data_path : str
            Path to dataset directory
        cache_dir : str
            Directory for caching processed data
        """
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path / 'cache'

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Dataset info
        self.info: DatasetInfo = None
        self.subjects: List[Subject] = []

        # Preprocessing flags
        self.preprocessed = False

    @abstractmethod
    def load(self) -> List[Subject]:
        """Load dataset"""
        pass

    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Get dataset information"""
        pass

    def get_data_matrix(self, modality: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data as numpy arrays

        Parameters
        ----------
        modality : str
            Specific modality to extract

        Returns
        -------
        X : ndarray
            Feature matrix (n_subjects, features)
        y : ndarray
            Labels (n_subjects,)
        """
        if not self.subjects:
            self.load()

        X_list = []
        y_list = []
        label_map = {label: i for i, label in enumerate(self.info.labels)}

        for subject in self.subjects:
            if modality and modality in subject.data:
                X_list.append(subject.data[modality].flatten())
            elif 'features' in subject.data:
                X_list.append(subject.data['features'].flatten())
            else:
                # Combine all modalities
                features = []
                for key, data in subject.data.items():
                    features.extend(data.flatten().tolist())
                X_list.append(np.array(features))

            y_list.append(label_map.get(subject.diagnosis, -1))

        return np.array(X_list), np.array(y_list)

    def get_batches(self, batch_size: int = 32,
                    shuffle: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of data

        Parameters
        ----------
        batch_size : int
            Batch size
        shuffle : bool
            Whether to shuffle data

        Yields
        ------
        X_batch, y_batch : tuple
            Batch of data and labels
        """
        X, y = self.get_data_matrix()

        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch_indices = indices[start:end]
            yield X[batch_indices], y[batch_indices]

    def split_data(self, test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train/val/test sets

        Returns
        -------
        splits : dict
            Dictionary with 'train', 'val', 'test' keys
        """
        X, y = self.get_data_matrix()

        np.random.seed(random_state)
        indices = np.random.permutation(len(X))

        n_test = int(len(X) * test_size)
        n_val = int(len(X) * val_size)

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        return {
            'train': (X[train_idx], y[train_idx]),
            'val': (X[val_idx], y[val_idx]),
            'test': (X[test_idx], y[test_idx])
        }

    def save_cache(self, filename: str, data: Any):
        """Save processed data to cache"""
        cache_path = self.cache_dir / filename
        np.save(cache_path, data, allow_pickle=True)

    def load_cache(self, filename: str) -> Optional[Any]:
        """Load data from cache"""
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=True)
        return None


class ADNIDatasetLoader(BaseDatasetLoader):
    """
    ADNI (Alzheimer's Disease Neuroimaging Initiative) Dataset Loader

    Supports loading:
    - Structural MRI (T1-weighted)
    - PET scans (FDG, Amyloid)
    - Clinical assessments (MMSE, CDR, ADAS-Cog)
    - Demographics

    Labels: CN (Cognitively Normal), MCI (Mild Cognitive Impairment), AD (Alzheimer's Disease)

    Dataset URL: https://adni.loni.usc.edu/
    """

    def __init__(self, data_path: str, cache_dir: str = None):
        super().__init__(data_path, cache_dir)

        self.info = DatasetInfo(
            name="ADNI",
            disease="Alzheimer's Disease",
            description="Alzheimer's Disease Neuroimaging Initiative dataset with MRI, PET, and clinical data",
            n_subjects=0,
            modalities=['mri_t1', 'pet_fdg', 'pet_amyloid', 'clinical'],
            labels=['CN', 'MCI', 'AD'],
            source_url="https://adni.loni.usc.edu/"
        )

    def load(self) -> List[Subject]:
        """Load ADNI dataset"""
        logger.info("Loading ADNI dataset...")

        # Check for cached data
        cached = self.load_cache('adni_subjects.npy')
        if cached is not None:
            self.subjects = list(cached)
            self.info.n_subjects = len(self.subjects)
            logger.info(f"Loaded {len(self.subjects)} subjects from cache")
            return self.subjects

        # Try to load from actual data path
        if self.data_path.exists():
            self.subjects = self._load_from_directory()
        else:
            # Generate synthetic data for demo
            logger.info("Generating synthetic ADNI data for demonstration...")
            self.subjects = self._generate_synthetic_data()

        self.info.n_subjects = len(self.subjects)
        self.save_cache('adni_subjects.npy', np.array(self.subjects))

        return self.subjects

    def _load_from_directory(self) -> List[Subject]:
        """Load data from ADNI directory structure"""
        subjects = []

        # Look for subject directories
        subject_dirs = list(self.data_path.glob('ADNI_*')) + \
                       list(self.data_path.glob('sub-*'))

        for subj_dir in subject_dirs:
            try:
                subject = self._load_subject(subj_dir)
                if subject:
                    subjects.append(subject)
            except Exception as e:
                logger.warning(f"Error loading {subj_dir}: {e}")

        # Also check for tabular data
        csv_files = list(self.data_path.glob('*.csv'))
        for csv_file in csv_files:
            if 'demographics' in csv_file.name.lower() or 'clinical' in csv_file.name.lower():
                self._load_clinical_data(csv_file, subjects)

        return subjects

    def _load_subject(self, subj_dir: Path) -> Optional[Subject]:
        """Load single subject data"""
        subject_id = subj_dir.name
        data = {}
        metadata = {}

        # Load MRI if available
        mri_files = list(subj_dir.glob('**/*T1*.nii*')) + \
                    list(subj_dir.glob('**/*t1*.nii*'))
        if mri_files and HAS_NIBABEL:
            img = nib.load(str(mri_files[0]))
            data['mri_t1'] = img.get_fdata()
            metadata['mri_affine'] = img.affine

        # Load clinical data
        clinical_files = list(subj_dir.glob('**/*.json')) + \
                         list(subj_dir.glob('**/*clinical*.csv'))

        clinical = {}
        for cf in clinical_files:
            if cf.suffix == '.json':
                with open(cf) as f:
                    clinical.update(json.load(f))
            elif cf.suffix == '.csv':
                df = pd.read_csv(cf)
                if len(df) > 0:
                    clinical.update(df.iloc[0].to_dict())

        # Determine diagnosis
        diagnosis = clinical.get('DX', clinical.get('diagnosis', 'Unknown'))

        return Subject(
            subject_id=subject_id,
            diagnosis=diagnosis,
            age=clinical.get('AGE', 0),
            sex=clinical.get('SEX', 'Unknown'),
            data=data,
            clinical=clinical,
            metadata=metadata
        )

    def _load_clinical_data(self, csv_file: Path, subjects: List[Subject]):
        """Load clinical data from CSV"""
        df = pd.read_csv(csv_file)

        # Map clinical data to subjects
        for subject in subjects:
            mask = df['SUBJECT_ID'] == subject.subject_id
            if mask.any():
                row = df[mask].iloc[0]
                subject.clinical.update(row.to_dict())

    def _generate_synthetic_data(self, n_subjects: int = 300) -> List[Subject]:
        """Generate synthetic ADNI-like data for demonstration"""
        subjects = []

        diagnoses = ['CN', 'MCI', 'AD']
        diagnosis_weights = [0.35, 0.35, 0.30]

        for i in range(n_subjects):
            diagnosis = np.random.choice(diagnoses, p=diagnosis_weights)

            # Age distribution varies by diagnosis
            if diagnosis == 'CN':
                age = np.random.normal(72, 8)
            elif diagnosis == 'MCI':
                age = np.random.normal(74, 7)
            else:  # AD
                age = np.random.normal(76, 6)

            age = max(55, min(95, age))

            # Generate features based on diagnosis
            features = self._generate_features_for_diagnosis(diagnosis)

            # Clinical scores
            clinical = {
                'MMSE': self._generate_mmse(diagnosis),
                'CDR': self._generate_cdr(diagnosis),
                'ADAS_COG': self._generate_adas(diagnosis),
                'APOE4': np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]),
                'education_years': np.random.randint(8, 20)
            }

            subject = Subject(
                subject_id=f"ADNI_S{i+1:04d}",
                diagnosis=diagnosis,
                age=age,
                sex=np.random.choice(['M', 'F']),
                data={
                    'features': features,
                    'brain_volumes': self._generate_brain_volumes(diagnosis),
                    'cortical_thickness': self._generate_cortical_thickness(diagnosis)
                },
                clinical=clinical,
                metadata={'synthetic': True}
            )
            subjects.append(subject)

        return subjects

    def _generate_features_for_diagnosis(self, diagnosis: str) -> np.ndarray:
        """Generate features based on diagnosis"""
        n_features = 100

        if diagnosis == 'CN':
            features = np.random.randn(n_features) * 0.5
        elif diagnosis == 'MCI':
            features = np.random.randn(n_features) * 0.6 + 0.3
        else:  # AD
            features = np.random.randn(n_features) * 0.7 + 0.6

        return features

    def _generate_brain_volumes(self, diagnosis: str) -> np.ndarray:
        """Generate brain region volumes"""
        regions = ['hippocampus', 'entorhinal', 'temporal', 'parietal', 'frontal']
        volumes = []

        for region in regions:
            if diagnosis == 'CN':
                vol = np.random.normal(1.0, 0.1)
            elif diagnosis == 'MCI':
                vol = np.random.normal(0.9, 0.12)
            else:
                vol = np.random.normal(0.75, 0.15)
            volumes.append(max(0.3, vol))

        return np.array(volumes)

    def _generate_cortical_thickness(self, diagnosis: str) -> np.ndarray:
        """Generate cortical thickness values"""
        n_regions = 68

        if diagnosis == 'CN':
            thickness = np.random.normal(2.5, 0.3, n_regions)
        elif diagnosis == 'MCI':
            thickness = np.random.normal(2.3, 0.35, n_regions)
        else:
            thickness = np.random.normal(2.0, 0.4, n_regions)

        return np.clip(thickness, 1.0, 4.0)

    def _generate_mmse(self, diagnosis: str) -> int:
        """Generate MMSE score"""
        if diagnosis == 'CN':
            return min(30, max(24, int(np.random.normal(29, 1))))
        elif diagnosis == 'MCI':
            return min(30, max(18, int(np.random.normal(26, 2))))
        else:
            return min(26, max(0, int(np.random.normal(18, 5))))

    def _generate_cdr(self, diagnosis: str) -> float:
        """Generate CDR score"""
        if diagnosis == 'CN':
            return 0.0
        elif diagnosis == 'MCI':
            return np.random.choice([0.5], p=[1.0])
        else:
            return np.random.choice([0.5, 1.0, 2.0], p=[0.2, 0.5, 0.3])

    def _generate_adas(self, diagnosis: str) -> float:
        """Generate ADAS-Cog score"""
        if diagnosis == 'CN':
            return max(0, np.random.normal(8, 3))
        elif diagnosis == 'MCI':
            return max(0, np.random.normal(15, 5))
        else:
            return max(0, np.random.normal(30, 10))

    def get_info(self) -> DatasetInfo:
        return self.info


class PPMIDatasetLoader(BaseDatasetLoader):
    """
    PPMI (Parkinson's Progression Markers Initiative) Dataset Loader

    Supports loading:
    - DaTscan SPECT imaging
    - MRI data
    - Motor assessments (UPDRS)
    - Non-motor assessments
    - Biospecimen data

    Labels: HC (Healthy Control), PD (Parkinson's Disease), SWEDD, Prodromal

    Dataset URL: https://www.ppmi-info.org/
    """

    def __init__(self, data_path: str, cache_dir: str = None):
        super().__init__(data_path, cache_dir)

        self.info = DatasetInfo(
            name="PPMI",
            disease="Parkinson's Disease",
            description="Parkinson's Progression Markers Initiative with imaging and clinical assessments",
            n_subjects=0,
            modalities=['datscan', 'mri_t1', 'motor_assessment', 'clinical'],
            labels=['HC', 'PD', 'SWEDD', 'Prodromal'],
            source_url="https://www.ppmi-info.org/"
        )

    def load(self) -> List[Subject]:
        """Load PPMI dataset"""
        logger.info("Loading PPMI dataset...")

        cached = self.load_cache('ppmi_subjects.npy')
        if cached is not None:
            self.subjects = list(cached)
            self.info.n_subjects = len(self.subjects)
            return self.subjects

        if self.data_path.exists():
            self.subjects = self._load_from_directory()
        else:
            logger.info("Generating synthetic PPMI data for demonstration...")
            self.subjects = self._generate_synthetic_data()

        self.info.n_subjects = len(self.subjects)
        self.save_cache('ppmi_subjects.npy', np.array(self.subjects))

        return self.subjects

    def _load_from_directory(self) -> List[Subject]:
        """Load data from PPMI directory"""
        subjects = []
        # Implementation for actual PPMI data loading
        return subjects

    def _generate_synthetic_data(self, n_subjects: int = 400) -> List[Subject]:
        """Generate synthetic PPMI-like data"""
        subjects = []

        diagnoses = ['HC', 'PD', 'SWEDD', 'Prodromal']
        weights = [0.25, 0.50, 0.10, 0.15]

        for i in range(n_subjects):
            diagnosis = np.random.choice(diagnoses, p=weights)

            age = np.random.normal(62, 10)
            age = max(40, min(85, age))

            # Disease duration (only for PD)
            duration = np.random.exponential(5) if diagnosis == 'PD' else 0

            # Generate motor assessment features
            motor_features = self._generate_motor_features(diagnosis)

            # Generate voice features
            voice_features = self._generate_voice_features(diagnosis)

            # Generate DaTscan features (SBR values)
            datscan_features = self._generate_datscan_features(diagnosis)

            # UPDRS scores
            updrs = self._generate_updrs(diagnosis)

            clinical = {
                'UPDRS_I': updrs['part1'],
                'UPDRS_II': updrs['part2'],
                'UPDRS_III': updrs['part3'],
                'UPDRS_TOTAL': updrs['total'],
                'HOEHN_YAHR': self._generate_hy(diagnosis),
                'MoCA': self._generate_moca(diagnosis),
                'disease_duration': duration,
                'LED': np.random.uniform(0, 1500) if diagnosis == 'PD' else 0
            }

            subject = Subject(
                subject_id=f"PPMI_S{i+1:04d}",
                diagnosis=diagnosis,
                age=age,
                sex=np.random.choice(['M', 'F'], p=[0.6, 0.4]),
                data={
                    'motor_features': motor_features,
                    'voice_features': voice_features,
                    'datscan_sbr': datscan_features,
                    'features': np.concatenate([motor_features, voice_features, datscan_features])
                },
                clinical=clinical,
                metadata={'synthetic': True}
            )
            subjects.append(subject)

        return subjects

    def _generate_motor_features(self, diagnosis: str) -> np.ndarray:
        """Generate motor assessment features"""
        n_features = 20

        if diagnosis == 'HC':
            features = np.random.exponential(0.5, n_features)
        elif diagnosis == 'PD':
            features = np.random.exponential(2.0, n_features) + 1
        elif diagnosis == 'SWEDD':
            features = np.random.exponential(1.0, n_features) + 0.5
        else:  # Prodromal
            features = np.random.exponential(0.8, n_features) + 0.3

        return np.clip(features, 0, 10)

    def _generate_voice_features(self, diagnosis: str) -> np.ndarray:
        """Generate voice analysis features"""
        features = {}

        if diagnosis in ['PD', 'Prodromal']:
            features['jitter'] = np.random.uniform(0.5, 2.5)
            features['shimmer'] = np.random.uniform(3, 12)
            features['hnr'] = np.random.uniform(15, 25)
            features['dfa'] = np.random.uniform(0.55, 0.75)
        else:
            features['jitter'] = np.random.uniform(0.1, 0.8)
            features['shimmer'] = np.random.uniform(1, 5)
            features['hnr'] = np.random.uniform(22, 35)
            features['dfa'] = np.random.uniform(0.5, 0.65)

        return np.array(list(features.values()))

    def _generate_datscan_features(self, diagnosis: str) -> np.ndarray:
        """Generate DaTscan SBR values"""
        # Striatum binding ratios (left/right putamen, caudate)
        if diagnosis == 'HC':
            left_putamen = np.random.normal(2.5, 0.3)
            right_putamen = np.random.normal(2.5, 0.3)
            left_caudate = np.random.normal(3.0, 0.3)
            right_caudate = np.random.normal(3.0, 0.3)
        elif diagnosis == 'PD':
            left_putamen = np.random.normal(1.2, 0.4)
            right_putamen = np.random.normal(1.5, 0.4)
            left_caudate = np.random.normal(2.0, 0.4)
            right_caudate = np.random.normal(2.2, 0.4)
        elif diagnosis == 'SWEDD':
            left_putamen = np.random.normal(2.3, 0.3)
            right_putamen = np.random.normal(2.3, 0.3)
            left_caudate = np.random.normal(2.8, 0.3)
            right_caudate = np.random.normal(2.8, 0.3)
        else:  # Prodromal
            left_putamen = np.random.normal(1.8, 0.4)
            right_putamen = np.random.normal(2.0, 0.4)
            left_caudate = np.random.normal(2.5, 0.4)
            right_caudate = np.random.normal(2.6, 0.4)

        return np.array([left_putamen, right_putamen, left_caudate, right_caudate])

    def _generate_updrs(self, diagnosis: str) -> Dict:
        """Generate UPDRS scores"""
        if diagnosis == 'HC':
            part1 = np.random.randint(0, 5)
            part2 = np.random.randint(0, 5)
            part3 = np.random.randint(0, 8)
        elif diagnosis == 'PD':
            part1 = np.random.randint(3, 15)
            part2 = np.random.randint(5, 25)
            part3 = np.random.randint(15, 60)
        else:
            part1 = np.random.randint(1, 8)
            part2 = np.random.randint(2, 12)
            part3 = np.random.randint(5, 25)

        return {
            'part1': part1,
            'part2': part2,
            'part3': part3,
            'total': part1 + part2 + part3
        }

    def _generate_hy(self, diagnosis: str) -> float:
        """Generate Hoehn & Yahr stage"""
        if diagnosis == 'HC':
            return 0
        elif diagnosis == 'PD':
            return np.random.choice([1, 1.5, 2, 2.5, 3, 4], p=[0.1, 0.15, 0.35, 0.2, 0.15, 0.05])
        else:
            return np.random.choice([0, 1, 1.5], p=[0.5, 0.3, 0.2])

    def _generate_moca(self, diagnosis: str) -> int:
        """Generate MoCA score"""
        if diagnosis == 'HC':
            return min(30, max(22, int(np.random.normal(27, 2))))
        elif diagnosis == 'PD':
            return min(30, max(15, int(np.random.normal(24, 4))))
        else:
            return min(30, max(20, int(np.random.normal(26, 3))))

    def get_info(self) -> DatasetInfo:
        return self.info


class COBREDatasetLoader(BaseDatasetLoader):
    """
    COBRE (Center for Biomedical Research Excellence) Dataset Loader

    Supports loading:
    - Resting-state fMRI
    - Structural MRI
    - Clinical assessments (PANSS)
    - Demographics

    Labels: HC (Healthy Control), SZ (Schizophrenia)

    Dataset URL: http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html
    """

    def __init__(self, data_path: str, cache_dir: str = None):
        super().__init__(data_path, cache_dir)

        self.info = DatasetInfo(
            name="COBRE",
            disease="Schizophrenia",
            description="COBRE schizophrenia dataset with fMRI and clinical assessments",
            n_subjects=0,
            modalities=['fmri', 'mri_t1', 'connectivity', 'clinical'],
            labels=['HC', 'SZ'],
            source_url="http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html"
        )

    def load(self) -> List[Subject]:
        """Load COBRE dataset"""
        logger.info("Loading COBRE dataset...")

        cached = self.load_cache('cobre_subjects.npy')
        if cached is not None:
            self.subjects = list(cached)
            self.info.n_subjects = len(self.subjects)
            return self.subjects

        if self.data_path.exists():
            self.subjects = self._load_from_directory()
        else:
            logger.info("Generating synthetic COBRE data for demonstration...")
            self.subjects = self._generate_synthetic_data()

        self.info.n_subjects = len(self.subjects)
        self.save_cache('cobre_subjects.npy', np.array(self.subjects))

        return self.subjects

    def _load_from_directory(self) -> List[Subject]:
        """Load from COBRE directory"""
        subjects = []
        return subjects

    def _generate_synthetic_data(self, n_subjects: int = 150) -> List[Subject]:
        """Generate synthetic COBRE-like data"""
        subjects = []

        diagnoses = ['HC', 'SZ']
        weights = [0.45, 0.55]

        for i in range(n_subjects):
            diagnosis = np.random.choice(diagnoses, p=weights)

            age = np.random.normal(35, 12)
            age = max(18, min(65, age))

            # Generate connectivity features
            connectivity_features = self._generate_connectivity_features(diagnosis)

            # Generate EEG features
            eeg_features = self._generate_eeg_features(diagnosis)

            # PANSS scores
            panss = self._generate_panss(diagnosis)

            clinical = {
                'PANSS_positive': panss['positive'],
                'PANSS_negative': panss['negative'],
                'PANSS_general': panss['general'],
                'PANSS_total': panss['total'],
                'age_onset': np.random.normal(22, 5) if diagnosis == 'SZ' else None,
                'duration_illness': np.random.exponential(8) if diagnosis == 'SZ' else 0,
                'medication_status': np.random.choice([0, 1], p=[0.1, 0.9]) if diagnosis == 'SZ' else 0
            }

            subject = Subject(
                subject_id=f"COBRE_S{i+1:04d}",
                diagnosis=diagnosis,
                age=age,
                sex=np.random.choice(['M', 'F'], p=[0.65, 0.35]),
                data={
                    'connectivity': connectivity_features,
                    'eeg_features': eeg_features,
                    'features': np.concatenate([connectivity_features.flatten(), eeg_features])
                },
                clinical=clinical,
                metadata={'synthetic': True}
            )
            subjects.append(subject)

        return subjects

    def _generate_connectivity_features(self, diagnosis: str) -> np.ndarray:
        """Generate functional connectivity matrix"""
        n_rois = 116  # AAL atlas regions

        # Generate correlation matrix
        if diagnosis == 'HC':
            # More organized connectivity
            base_connectivity = np.random.uniform(0.2, 0.6, (n_rois, n_rois))
        else:
            # Disrupted connectivity in SZ
            base_connectivity = np.random.uniform(0.1, 0.5, (n_rois, n_rois))

        # Make symmetric
        connectivity = (base_connectivity + base_connectivity.T) / 2

        # Set diagonal to 1
        np.fill_diagonal(connectivity, 1.0)

        # Extract upper triangle (unique connections)
        upper_triangle = connectivity[np.triu_indices(n_rois, k=1)]

        return upper_triangle

    def _generate_eeg_features(self, diagnosis: str) -> np.ndarray:
        """Generate EEG-derived features"""
        features = []

        # Power spectral features
        if diagnosis == 'HC':
            features.extend([
                np.random.uniform(0.1, 0.25),   # Delta
                np.random.uniform(0.1, 0.2),    # Theta
                np.random.uniform(0.2, 0.35),   # Alpha
                np.random.uniform(0.15, 0.25),  # Beta
                np.random.uniform(0.05, 0.15)   # Gamma
            ])
            features.append(np.random.uniform(0.6, 0.9))  # Gamma phase-locking
            features.append(np.random.uniform(8, 15))     # P300 amplitude
            features.append(np.random.uniform(4, 10))     # MMN amplitude
        else:
            features.extend([
                np.random.uniform(0.15, 0.3),   # Delta (increased)
                np.random.uniform(0.12, 0.25),  # Theta (increased)
                np.random.uniform(0.12, 0.25),  # Alpha (decreased)
                np.random.uniform(0.1, 0.2),    # Beta
                np.random.uniform(0.02, 0.1)    # Gamma (decreased)
            ])
            features.append(np.random.uniform(0.3, 0.6))  # Gamma phase-locking (reduced)
            features.append(np.random.uniform(3, 10))     # P300 (reduced)
            features.append(np.random.uniform(2, 6))      # MMN (reduced)

        # Complexity measures
        features.append(np.random.uniform(0.5, 1.2))  # Sample entropy
        features.append(np.random.uniform(0.3, 0.7))  # Lempel-Ziv complexity

        return np.array(features)

    def _generate_panss(self, diagnosis: str) -> Dict:
        """Generate PANSS scores"""
        if diagnosis == 'HC':
            return {
                'positive': 7,
                'negative': 7,
                'general': 16,
                'total': 30
            }
        else:
            positive = np.random.randint(10, 35)
            negative = np.random.randint(10, 40)
            general = np.random.randint(20, 60)
            return {
                'positive': positive,
                'negative': negative,
                'general': general,
                'total': positive + negative + general
            }

    def get_info(self) -> DatasetInfo:
        return self.info


class UnifiedDataLoader:
    """
    Unified Data Loader for All Datasets

    Provides a single interface for loading any supported dataset.
    """

    DATASET_CLASSES = {
        'adni': ADNIDatasetLoader,
        'ppmi': PPMIDatasetLoader,
        'cobre': COBREDatasetLoader
    }

    def __init__(self, base_path: str = './data'):
        self.base_path = Path(base_path)
        self.loaders: Dict[str, BaseDatasetLoader] = {}

    def load_dataset(self, name: str, data_path: str = None) -> BaseDatasetLoader:
        """
        Load a specific dataset

        Parameters
        ----------
        name : str
            Dataset name ('adni', 'ppmi', or 'cobre')
        data_path : str
            Optional custom data path

        Returns
        -------
        loader : BaseDatasetLoader
            Dataset loader instance
        """
        name = name.lower()

        if name not in self.DATASET_CLASSES:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.DATASET_CLASSES.keys())}")

        if data_path is None:
            data_path = self.base_path / name

        loader_class = self.DATASET_CLASSES[name]
        loader = loader_class(str(data_path))
        loader.load()

        self.loaders[name] = loader
        return loader

    def load_all(self) -> Dict[str, BaseDatasetLoader]:
        """Load all available datasets"""
        for name in self.DATASET_CLASSES:
            self.load_dataset(name)
        return self.loaders

    def get_combined_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get combined data from all loaded datasets

        Returns
        -------
        X : ndarray
            Feature matrix
        y : ndarray
            Disease labels (0: Alzheimer, 1: Parkinson, 2: Schizophrenia, 3: Healthy)
        dataset_ids : ndarray
            Dataset identifier for each sample
        """
        X_list = []
        y_list = []
        dataset_list = []

        disease_map = {
            'alzheimer': {'CN': 3, 'MCI': 0, 'AD': 0},
            'parkinson': {'HC': 3, 'PD': 1, 'SWEDD': 3, 'Prodromal': 1},
            'schizophrenia': {'HC': 3, 'SZ': 2}
        }

        for dataset_name, loader in self.loaders.items():
            X, _ = loader.get_data_matrix()

            disease = loader.info.disease.lower().split()[0]
            mapping = disease_map.get(disease, {})

            for subject, features in zip(loader.subjects, X):
                X_list.append(features)
                y_list.append(mapping.get(subject.diagnosis, 3))
                dataset_list.append(dataset_name)

        return np.array(X_list), np.array(y_list), np.array(dataset_list)


if __name__ == "__main__":
    print("Dataset Loaders Demo")
    print("=" * 50)

    # Create unified loader
    loader = UnifiedDataLoader()

    # Load each dataset
    for name in ['adni', 'ppmi', 'cobre']:
        print(f"\nLoading {name.upper()} dataset...")
        dataset = loader.load_dataset(name)
        info = dataset.get_info()

        print(f"  Name: {info.name}")
        print(f"  Disease: {info.disease}")
        print(f"  Subjects: {info.n_subjects}")
        print(f"  Labels: {info.labels}")

        # Get data matrix
        X, y = dataset.get_data_matrix()
        print(f"  Data shape: {X.shape}")
        print(f"  Label distribution: {np.bincount(y[y >= 0])}")

    # Get combined data
    print("\n\nCombined Data:")
    X, y, datasets = loader.get_combined_data()
    print(f"  Total samples: {len(X)}")
    print(f"  Disease distribution: {np.bincount(y)}")

    print("\nDemo complete!")
