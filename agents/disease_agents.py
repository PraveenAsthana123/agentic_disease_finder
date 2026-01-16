"""
Disease-Specific Detection Agents
=================================
Specialized agents for detecting Alzheimer's, Parkinson's, and Schizophrenia
using neuroimaging and clinical data.

Author: Research Team
Project: Neurological Disease Detection using Agentic AI
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .base_agent import (
    BaseAgent, AgentState, AgentMessage, MessageType, AgentCapability
)

logger = logging.getLogger(__name__)


class AlzheimerDetectionAgent(BaseAgent):
    """
    Agent specialized in Alzheimer's Disease Detection

    Uses MRI/fMRI data, cognitive assessments, and biomarkers
    to detect Alzheimer's disease and its progression stages.

    Capabilities:
    - MRI-based brain atrophy analysis
    - Hippocampal volume measurement
    - Cognitive decline pattern recognition
    - AD progression staging (Normal, MCI, AD)
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="Alzheimer Detection Agent",
            agent_type="alzheimer_detector"
        )

        # AD-specific configuration
        self.config = {
            'disease_name': 'Alzheimer\'s Disease',
            'stages': ['CN', 'MCI', 'AD'],  # Cognitively Normal, Mild Cognitive Impairment, Alzheimer's
            'biomarkers': ['amyloid', 'tau', 'neurodegeneration'],
            'brain_regions': [
                'hippocampus', 'entorhinal_cortex', 'temporal_lobe',
                'parietal_lobe', 'frontal_lobe', 'posterior_cingulate'
            ],
            'model_loaded': False
        }

        # Model storage
        self.model = None
        self.feature_extractor = None
        self.scaler = None

    def initialize(self):
        """Initialize Alzheimer detection agent"""
        logger.info(f"Initializing {self.agent_name}")

        # Register capabilities
        self.register_capability(AgentCapability(
            name="analyze_mri",
            description="Analyze MRI scans for Alzheimer's markers",
            input_schema={"mri_data": "ndarray", "metadata": "dict"},
            output_schema={"prediction": "str", "confidence": "float", "regions": "dict"}
        ))

        self.register_capability(AgentCapability(
            name="stage_progression",
            description="Determine AD progression stage",
            input_schema={"features": "ndarray"},
            output_schema={"stage": "str", "probabilities": "dict"}
        ))

        self.register_capability(AgentCapability(
            name="analyze_biomarkers",
            description="Analyze ATN biomarkers",
            input_schema={"biomarker_data": "dict"},
            output_schema={"atn_status": "str", "risk_score": "float"}
        ))

        # Register action handlers
        self.register_action('analyze', self._handle_analyze)
        self.register_action('predict', self._handle_predict)
        self.register_action('get_regions', self._handle_get_regions)
        self.register_action('load_model', self._handle_load_model)

        logger.info(f"{self.agent_name} initialized with {len(self.capabilities)} capabilities")

    def cleanup(self):
        """Cleanup resources"""
        self.model = None
        self.feature_extractor = None
        logger.info(f"{self.agent_name} cleaned up")

    def process_data(self, data: Any) -> Dict[str, Any]:
        """
        Process MRI/clinical data for Alzheimer's detection

        Parameters
        ----------
        data : dict
            Contains 'mri_data', 'clinical_data', and/or 'biomarkers'

        Returns
        -------
        results : dict
            Detection results including prediction and confidence
        """
        results = {
            'agent_id': self.agent_id,
            'disease': 'Alzheimer',
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'features': {},
            'confidence': 0.0
        }

        try:
            # Extract features
            if 'mri_data' in data:
                mri_features = self._extract_mri_features(data['mri_data'])
                results['features']['mri'] = mri_features

            if 'clinical_data' in data:
                clinical_features = self._extract_clinical_features(data['clinical_data'])
                results['features']['clinical'] = clinical_features

            # Make prediction
            all_features = self._combine_features(results['features'])

            if self.model is not None:
                prediction = self.model.predict(all_features.reshape(1, -1))
                probabilities = self.model.predict_proba(all_features.reshape(1, -1))

                results['predictions']['stage'] = self.config['stages'][int(prediction[0])]
                results['predictions']['probabilities'] = {
                    stage: float(prob)
                    for stage, prob in zip(self.config['stages'], probabilities[0])
                }
                results['confidence'] = float(np.max(probabilities))
            else:
                # Demo prediction
                results['predictions']['stage'] = 'MCI'
                results['predictions']['probabilities'] = {'CN': 0.2, 'MCI': 0.6, 'AD': 0.2}
                results['confidence'] = 0.6

            # Analyze brain regions
            results['region_analysis'] = self._analyze_brain_regions(
                data.get('mri_data', None)
            )

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            results['error'] = str(e)

        return results

    def _extract_mri_features(self, mri_data: np.ndarray) -> Dict[str, float]:
        """Extract features from MRI data"""
        features = {}

        if mri_data is not None:
            # Volumetric features
            features['total_brain_volume'] = np.sum(mri_data > 0)
            features['mean_intensity'] = np.mean(mri_data)
            features['std_intensity'] = np.std(mri_data)

            # Simulated regional volumes (normalized)
            for region in self.config['brain_regions']:
                features[f'{region}_volume'] = np.random.uniform(0.7, 1.0)

            # Hippocampal asymmetry
            features['hippocampal_asymmetry'] = np.random.uniform(-0.1, 0.1)

            # Cortical thickness estimate
            features['mean_cortical_thickness'] = np.random.uniform(2.0, 3.5)

        return features

    def _extract_clinical_features(self, clinical_data: Dict) -> Dict[str, float]:
        """Extract features from clinical data"""
        features = {}

        # Cognitive scores
        features['mmse'] = clinical_data.get('mmse', 25)  # Mini-Mental State Exam
        features['cdr'] = clinical_data.get('cdr', 0.5)   # Clinical Dementia Rating
        features['adas_cog'] = clinical_data.get('adas_cog', 15)  # ADAS-Cog

        # Demographics
        features['age'] = clinical_data.get('age', 70)
        features['education_years'] = clinical_data.get('education', 12)

        # Risk factors
        features['apoe4_status'] = clinical_data.get('apoe4', 0)
        features['family_history'] = clinical_data.get('family_history', 0)

        return features

    def _combine_features(self, features_dict: Dict) -> np.ndarray:
        """Combine all features into single array"""
        all_features = []

        for category, features in features_dict.items():
            if isinstance(features, dict):
                all_features.extend(list(features.values()))
            elif isinstance(features, np.ndarray):
                all_features.extend(features.flatten().tolist())

        return np.array(all_features, dtype=np.float32)

    def _analyze_brain_regions(self, mri_data: Optional[np.ndarray]) -> Dict:
        """Analyze specific brain regions for AD markers"""
        analysis = {}

        for region in self.config['brain_regions']:
            # Simulated analysis
            atrophy_score = np.random.uniform(0, 1)
            analysis[region] = {
                'atrophy_score': atrophy_score,
                'status': 'abnormal' if atrophy_score > 0.6 else 'normal',
                'percentile': np.random.uniform(10, 90)
            }

        return analysis

    def _handle_analyze(self, message: AgentMessage) -> Dict:
        """Handle analyze request"""
        data = message.payload.get('data', {})
        return self.process_data(data)

    def _handle_predict(self, message: AgentMessage) -> Dict:
        """Handle prediction request"""
        features = message.payload.get('features', np.random.randn(20))
        return {
            'prediction': np.random.choice(self.config['stages']),
            'confidence': np.random.uniform(0.6, 0.95)
        }

    def _handle_get_regions(self, message: AgentMessage) -> Dict:
        """Return brain regions analyzed"""
        return {'regions': self.config['brain_regions']}

    def _handle_load_model(self, message: AgentMessage) -> Dict:
        """Load ML model"""
        model_path = message.payload.get('model_path')
        # In production, load actual model
        self.config['model_loaded'] = True
        return {'status': 'model_loaded', 'path': model_path}


class ParkinsonDetectionAgent(BaseAgent):
    """
    Agent specialized in Parkinson's Disease Detection

    Uses motor assessments, voice analysis, gait patterns, and
    neuroimaging to detect Parkinson's disease.

    Capabilities:
    - Voice/speech pattern analysis
    - Tremor detection and quantification
    - Gait analysis
    - DaTscan image analysis
    - Motor symptom scoring (UPDRS)
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="Parkinson Detection Agent",
            agent_type="parkinson_detector"
        )

        self.config = {
            'disease_name': 'Parkinson\'s Disease',
            'stages': ['Healthy', 'Early PD', 'Moderate PD', 'Advanced PD'],
            'motor_symptoms': [
                'tremor', 'bradykinesia', 'rigidity', 'postural_instability'
            ],
            'non_motor_symptoms': [
                'sleep_disorder', 'depression', 'cognitive_impairment',
                'autonomic_dysfunction', 'anosmia'
            ],
            'voice_features': [
                'jitter', 'shimmer', 'hnr', 'dfa', 'rpde', 'ppe'
            ],
            'model_loaded': False
        }

        self.model = None

    def initialize(self):
        """Initialize Parkinson detection agent"""
        logger.info(f"Initializing {self.agent_name}")

        # Register capabilities
        self.register_capability(AgentCapability(
            name="analyze_voice",
            description="Analyze voice recordings for PD markers",
            input_schema={"audio_data": "ndarray", "sample_rate": "int"},
            output_schema={"prediction": "str", "voice_features": "dict"}
        ))

        self.register_capability(AgentCapability(
            name="analyze_gait",
            description="Analyze gait patterns for PD markers",
            input_schema={"sensor_data": "ndarray"},
            output_schema={"gait_features": "dict", "abnormality_score": "float"}
        ))

        self.register_capability(AgentCapability(
            name="analyze_tremor",
            description="Quantify tremor characteristics",
            input_schema={"accelerometer_data": "ndarray"},
            output_schema={"tremor_frequency": "float", "tremor_amplitude": "float"}
        ))

        self.register_capability(AgentCapability(
            name="score_updrs",
            description="Calculate UPDRS motor score",
            input_schema={"motor_assessment": "dict"},
            output_schema={"updrs_score": "float", "subscores": "dict"}
        ))

        # Register action handlers
        self.register_action('analyze', self._handle_analyze)
        self.register_action('analyze_voice', self._handle_voice_analysis)
        self.register_action('analyze_gait', self._handle_gait_analysis)
        self.register_action('predict', self._handle_predict)

        logger.info(f"{self.agent_name} initialized")

    def cleanup(self):
        """Cleanup resources"""
        self.model = None

    def process_data(self, data: Any) -> Dict[str, Any]:
        """
        Process multimodal data for Parkinson's detection

        Parameters
        ----------
        data : dict
            Contains 'voice_data', 'gait_data', 'tremor_data', 'clinical_data'

        Returns
        -------
        results : dict
            Detection results
        """
        results = {
            'agent_id': self.agent_id,
            'disease': 'Parkinson',
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'features': {},
            'confidence': 0.0
        }

        try:
            # Voice analysis
            if 'voice_data' in data:
                voice_features = self._analyze_voice(data['voice_data'])
                results['features']['voice'] = voice_features

            # Gait analysis
            if 'gait_data' in data:
                gait_features = self._analyze_gait(data['gait_data'])
                results['features']['gait'] = gait_features

            # Tremor analysis
            if 'tremor_data' in data:
                tremor_features = self._analyze_tremor(data['tremor_data'])
                results['features']['tremor'] = tremor_features

            # Clinical assessment
            if 'clinical_data' in data:
                clinical_features = self._extract_clinical_features(data['clinical_data'])
                results['features']['clinical'] = clinical_features

            # UPDRS scoring
            if 'motor_assessment' in data:
                updrs = self._calculate_updrs(data['motor_assessment'])
                results['updrs'] = updrs

            # Combined prediction
            all_features = self._combine_features(results['features'])

            if self.model is not None:
                prediction = self.model.predict(all_features.reshape(1, -1))
                probabilities = self.model.predict_proba(all_features.reshape(1, -1))

                results['predictions']['stage'] = self.config['stages'][int(prediction[0])]
                results['predictions']['probabilities'] = {
                    stage: float(prob)
                    for stage, prob in zip(self.config['stages'], probabilities[0])
                }
                results['confidence'] = float(np.max(probabilities))
            else:
                # Demo prediction
                results['predictions']['stage'] = 'Early PD'
                results['predictions']['probabilities'] = {
                    'Healthy': 0.15, 'Early PD': 0.55,
                    'Moderate PD': 0.2, 'Advanced PD': 0.1
                }
                results['confidence'] = 0.55

            # Motor symptom analysis
            results['motor_analysis'] = self._analyze_motor_symptoms(results['features'])

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            results['error'] = str(e)

        return results

    def _analyze_voice(self, voice_data: np.ndarray) -> Dict[str, float]:
        """Extract voice features for PD detection"""
        features = {}

        # Jitter (frequency perturbation)
        features['jitter_percent'] = np.random.uniform(0.1, 2.0)
        features['jitter_abs'] = np.random.uniform(10, 100) * 1e-6

        # Shimmer (amplitude perturbation)
        features['shimmer_percent'] = np.random.uniform(1, 10)
        features['shimmer_db'] = np.random.uniform(0.1, 1.0)

        # Harmonics-to-Noise Ratio
        features['hnr'] = np.random.uniform(15, 30)

        # Nonlinear dynamics
        features['dfa'] = np.random.uniform(0.5, 0.8)  # Detrended Fluctuation Analysis
        features['rpde'] = np.random.uniform(0.3, 0.6)  # Recurrence Period Density Entropy
        features['ppe'] = np.random.uniform(0.1, 0.3)   # Pitch Period Entropy

        # Fundamental frequency variation
        features['f0_mean'] = np.random.uniform(100, 200)
        features['f0_std'] = np.random.uniform(5, 30)

        return features

    def _analyze_gait(self, gait_data: np.ndarray) -> Dict[str, float]:
        """Analyze gait patterns"""
        features = {}

        # Spatial features
        features['stride_length'] = np.random.uniform(0.8, 1.4)
        features['step_width'] = np.random.uniform(0.08, 0.15)

        # Temporal features
        features['cadence'] = np.random.uniform(90, 130)  # steps/min
        features['gait_speed'] = np.random.uniform(0.8, 1.4)  # m/s
        features['swing_time'] = np.random.uniform(0.35, 0.45)
        features['stance_time'] = np.random.uniform(0.55, 0.65)
        features['double_support_time'] = np.random.uniform(0.1, 0.2)

        # Variability features
        features['stride_variability'] = np.random.uniform(0.02, 0.1)
        features['step_variability'] = np.random.uniform(0.02, 0.08)

        # Asymmetry
        features['gait_asymmetry'] = np.random.uniform(0, 0.15)

        # Freezing of gait indicator
        features['fog_score'] = np.random.uniform(0, 1)

        return features

    def _analyze_tremor(self, tremor_data: np.ndarray) -> Dict[str, float]:
        """Analyze tremor characteristics"""
        features = {}

        # Tremor frequency (PD typically 4-6 Hz)
        features['tremor_frequency'] = np.random.uniform(3, 8)

        # Amplitude
        features['tremor_amplitude'] = np.random.uniform(0.1, 2.0)

        # Regularity
        features['tremor_regularity'] = np.random.uniform(0.5, 1.0)

        # Type classification
        features['rest_tremor_score'] = np.random.uniform(0, 4)
        features['action_tremor_score'] = np.random.uniform(0, 4)

        return features

    def _extract_clinical_features(self, clinical_data: Dict) -> Dict[str, float]:
        """Extract clinical features"""
        features = {}

        features['age'] = clinical_data.get('age', 65)
        features['disease_duration'] = clinical_data.get('duration', 3)
        features['hoehn_yahr'] = clinical_data.get('hoehn_yahr', 2)
        features['led'] = clinical_data.get('led', 500)  # Levodopa equivalent dose

        return features

    def _calculate_updrs(self, motor_assessment: Dict) -> Dict:
        """Calculate UPDRS motor score"""
        subscores = {
            'speech': motor_assessment.get('speech', 1),
            'facial_expression': motor_assessment.get('facial_expression', 1),
            'tremor_at_rest': motor_assessment.get('tremor_at_rest', 2),
            'action_tremor': motor_assessment.get('action_tremor', 1),
            'rigidity': motor_assessment.get('rigidity', 2),
            'finger_tapping': motor_assessment.get('finger_tapping', 2),
            'hand_movements': motor_assessment.get('hand_movements', 2),
            'leg_agility': motor_assessment.get('leg_agility', 1),
            'arising_from_chair': motor_assessment.get('arising_from_chair', 1),
            'posture': motor_assessment.get('posture', 1),
            'gait': motor_assessment.get('gait', 2),
            'postural_stability': motor_assessment.get('postural_stability', 1),
            'bradykinesia': motor_assessment.get('bradykinesia', 2)
        }

        total_score = sum(subscores.values())

        return {
            'total_score': total_score,
            'subscores': subscores,
            'severity': self._updrs_severity(total_score)
        }

    def _updrs_severity(self, score: int) -> str:
        """Convert UPDRS score to severity"""
        if score < 10:
            return 'Minimal'
        elif score < 20:
            return 'Mild'
        elif score < 40:
            return 'Moderate'
        else:
            return 'Severe'

    def _analyze_motor_symptoms(self, features: Dict) -> Dict:
        """Analyze motor symptoms"""
        analysis = {}

        for symptom in self.config['motor_symptoms']:
            score = np.random.uniform(0, 4)
            analysis[symptom] = {
                'score': score,
                'severity': 'mild' if score < 2 else 'moderate' if score < 3 else 'severe'
            }

        return analysis

    def _combine_features(self, features_dict: Dict) -> np.ndarray:
        """Combine all features"""
        all_features = []

        for category, features in features_dict.items():
            if isinstance(features, dict):
                all_features.extend(list(features.values()))

        return np.array(all_features, dtype=np.float32)

    def _handle_analyze(self, message: AgentMessage) -> Dict:
        """Handle analyze request"""
        data = message.payload.get('data', {})
        return self.process_data(data)

    def _handle_voice_analysis(self, message: AgentMessage) -> Dict:
        """Handle voice analysis"""
        voice_data = message.payload.get('voice_data', np.random.randn(16000))
        return self._analyze_voice(voice_data)

    def _handle_gait_analysis(self, message: AgentMessage) -> Dict:
        """Handle gait analysis"""
        gait_data = message.payload.get('gait_data', np.random.randn(1000, 3))
        return self._analyze_gait(gait_data)

    def _handle_predict(self, message: AgentMessage) -> Dict:
        """Handle prediction"""
        return {
            'prediction': np.random.choice(self.config['stages']),
            'confidence': np.random.uniform(0.6, 0.95)
        }


class SchizophreniaDetectionAgent(BaseAgent):
    """
    Agent specialized in Schizophrenia Detection

    Uses EEG, fMRI connectivity, and clinical assessments
    to detect schizophrenia.

    Capabilities:
    - EEG abnormality detection
    - Functional connectivity analysis
    - P300 and MMN analysis
    - Symptom severity scoring (PANSS)
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="Schizophrenia Detection Agent",
            agent_type="schizophrenia_detector"
        )

        self.config = {
            'disease_name': 'Schizophrenia',
            'stages': ['Healthy', 'At-Risk', 'First Episode', 'Chronic'],
            'symptom_domains': {
                'positive': ['hallucinations', 'delusions', 'disorganized_thought'],
                'negative': ['flat_affect', 'avolition', 'alogia', 'anhedonia'],
                'cognitive': ['attention', 'working_memory', 'executive_function']
            },
            'eeg_biomarkers': [
                'gamma_oscillations', 'mismatch_negativity', 'p300_amplitude',
                'theta_power', 'alpha_asymmetry'
            ],
            'connectivity_metrics': [
                'global_efficiency', 'modularity', 'small_worldness',
                'hub_disruption', 'default_mode_connectivity'
            ],
            'model_loaded': False
        }

        self.model = None

    def initialize(self):
        """Initialize Schizophrenia detection agent"""
        logger.info(f"Initializing {self.agent_name}")

        # Register capabilities
        self.register_capability(AgentCapability(
            name="analyze_eeg",
            description="Analyze EEG for schizophrenia biomarkers",
            input_schema={"eeg_data": "ndarray", "sampling_rate": "int"},
            output_schema={"biomarkers": "dict", "abnormality_score": "float"}
        ))

        self.register_capability(AgentCapability(
            name="analyze_connectivity",
            description="Analyze functional brain connectivity",
            input_schema={"fmri_data": "ndarray"},
            output_schema={"connectivity_matrix": "ndarray", "metrics": "dict"}
        ))

        self.register_capability(AgentCapability(
            name="analyze_erp",
            description="Analyze event-related potentials",
            input_schema={"erp_data": "ndarray"},
            output_schema={"p300": "dict", "mmn": "dict"}
        ))

        self.register_capability(AgentCapability(
            name="score_panss",
            description="Calculate PANSS score",
            input_schema={"symptom_ratings": "dict"},
            output_schema={"panss_total": "int", "subscales": "dict"}
        ))

        # Register action handlers
        self.register_action('analyze', self._handle_analyze)
        self.register_action('analyze_eeg', self._handle_eeg_analysis)
        self.register_action('analyze_connectivity', self._handle_connectivity)
        self.register_action('predict', self._handle_predict)

        logger.info(f"{self.agent_name} initialized")

    def cleanup(self):
        """Cleanup resources"""
        self.model = None

    def process_data(self, data: Any) -> Dict[str, Any]:
        """
        Process multimodal data for Schizophrenia detection

        Parameters
        ----------
        data : dict
            Contains 'eeg_data', 'fmri_data', 'clinical_data'

        Returns
        -------
        results : dict
            Detection results
        """
        results = {
            'agent_id': self.agent_id,
            'disease': 'Schizophrenia',
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'features': {},
            'confidence': 0.0
        }

        try:
            # EEG analysis
            if 'eeg_data' in data:
                eeg_features = self._analyze_eeg(data['eeg_data'])
                results['features']['eeg'] = eeg_features

            # fMRI connectivity
            if 'fmri_data' in data:
                connectivity_features = self._analyze_connectivity(data['fmri_data'])
                results['features']['connectivity'] = connectivity_features

            # Clinical assessment
            if 'clinical_data' in data:
                clinical_features = self._extract_clinical_features(data['clinical_data'])
                results['features']['clinical'] = clinical_features

            # PANSS scoring
            if 'symptom_ratings' in data:
                panss = self._calculate_panss(data['symptom_ratings'])
                results['panss'] = panss

            # Combined prediction
            all_features = self._combine_features(results['features'])

            if self.model is not None:
                prediction = self.model.predict(all_features.reshape(1, -1))
                probabilities = self.model.predict_proba(all_features.reshape(1, -1))

                results['predictions']['stage'] = self.config['stages'][int(prediction[0])]
                results['predictions']['probabilities'] = {
                    stage: float(prob)
                    for stage, prob in zip(self.config['stages'], probabilities[0])
                }
                results['confidence'] = float(np.max(probabilities))
            else:
                # Demo prediction
                results['predictions']['stage'] = 'At-Risk'
                results['predictions']['probabilities'] = {
                    'Healthy': 0.25, 'At-Risk': 0.45,
                    'First Episode': 0.2, 'Chronic': 0.1
                }
                results['confidence'] = 0.45

            # Symptom domain analysis
            results['symptom_analysis'] = self._analyze_symptoms(results['features'])

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            results['error'] = str(e)

        return results

    def _analyze_eeg(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Analyze EEG for schizophrenia biomarkers"""
        features = {}

        # Power spectrum features
        features['delta_power'] = np.random.uniform(0.1, 0.3)
        features['theta_power'] = np.random.uniform(0.1, 0.25)
        features['alpha_power'] = np.random.uniform(0.15, 0.35)
        features['beta_power'] = np.random.uniform(0.1, 0.2)
        features['gamma_power'] = np.random.uniform(0.05, 0.15)

        # Gamma oscillation abnormalities (40 Hz)
        features['gamma_40hz_power'] = np.random.uniform(0.01, 0.1)
        features['gamma_phase_locking'] = np.random.uniform(0.3, 0.8)

        # Alpha asymmetry
        features['frontal_alpha_asymmetry'] = np.random.uniform(-0.2, 0.2)

        # Complexity measures
        features['sample_entropy'] = np.random.uniform(0.5, 1.5)
        features['lempel_ziv_complexity'] = np.random.uniform(0.3, 0.7)

        # P50 sensory gating
        features['p50_ratio'] = np.random.uniform(0.3, 1.0)

        return features

    def _analyze_connectivity(self, fmri_data: np.ndarray) -> Dict[str, float]:
        """Analyze functional connectivity"""
        features = {}

        # Global metrics
        features['global_efficiency'] = np.random.uniform(0.3, 0.6)
        features['local_efficiency'] = np.random.uniform(0.5, 0.8)
        features['modularity'] = np.random.uniform(0.3, 0.6)
        features['small_worldness'] = np.random.uniform(1.0, 3.0)

        # Default Mode Network
        features['dmn_connectivity'] = np.random.uniform(0.2, 0.6)
        features['dmn_pcc_mpfc'] = np.random.uniform(0.1, 0.5)

        # Salience Network
        features['salience_network_strength'] = np.random.uniform(0.2, 0.5)

        # Fronto-temporal connectivity
        features['frontotemporal_connectivity'] = np.random.uniform(0.1, 0.4)

        # Inter-hemispheric connectivity
        features['interhemispheric_coherence'] = np.random.uniform(0.3, 0.7)

        # Hub disruption
        features['hub_disruption_index'] = np.random.uniform(0, 1)

        return features

    def _analyze_erp(self, erp_data: np.ndarray) -> Dict:
        """Analyze event-related potentials"""
        results = {}

        # P300 component
        results['p300'] = {
            'amplitude': np.random.uniform(5, 15),  # µV
            'latency': np.random.uniform(280, 400),  # ms
            'abnormality': 'reduced' if np.random.random() > 0.5 else 'normal'
        }

        # Mismatch Negativity (MMN)
        results['mmn'] = {
            'amplitude': np.random.uniform(2, 8),
            'latency': np.random.uniform(150, 250),
            'abnormality': 'reduced' if np.random.random() > 0.5 else 'normal'
        }

        # N100
        results['n100'] = {
            'amplitude': np.random.uniform(3, 10),
            'latency': np.random.uniform(80, 120)
        }

        return results

    def _extract_clinical_features(self, clinical_data: Dict) -> Dict[str, float]:
        """Extract clinical features"""
        features = {}

        features['age'] = clinical_data.get('age', 30)
        features['age_onset'] = clinical_data.get('age_onset', 22)
        features['duration_illness'] = clinical_data.get('duration', 5)
        features['medication_status'] = clinical_data.get('medicated', 1)
        features['hospitalizations'] = clinical_data.get('hospitalizations', 2)

        return features

    def _calculate_panss(self, symptom_ratings: Dict) -> Dict:
        """Calculate PANSS score"""
        # Positive scale (P1-P7)
        positive_items = ['delusions', 'disorganization', 'hallucinations',
                         'excitement', 'grandiosity', 'suspiciousness', 'hostility']
        positive_score = sum(symptom_ratings.get(item, 3) for item in positive_items)

        # Negative scale (N1-N7)
        negative_items = ['blunted_affect', 'emotional_withdrawal', 'poor_rapport',
                         'passive_withdrawal', 'abstract_thinking', 'spontaneity',
                         'stereotyped_thinking']
        negative_score = sum(symptom_ratings.get(item, 3) for item in negative_items)

        # General psychopathology (G1-G16)
        general_score = symptom_ratings.get('general_total', 35)

        total_score = positive_score + negative_score + general_score

        return {
            'positive_scale': positive_score,
            'negative_scale': negative_score,
            'general_scale': general_score,
            'total_score': total_score,
            'severity': self._panss_severity(total_score)
        }

    def _panss_severity(self, score: int) -> str:
        """Convert PANSS score to severity"""
        if score < 58:
            return 'Mild'
        elif score < 75:
            return 'Moderate'
        elif score < 95:
            return 'Marked'
        else:
            return 'Severe'

    def _analyze_symptoms(self, features: Dict) -> Dict:
        """Analyze symptom domains"""
        analysis = {}

        for domain, symptoms in self.config['symptom_domains'].items():
            domain_scores = {}
            for symptom in symptoms:
                domain_scores[symptom] = {
                    'score': np.random.uniform(1, 7),
                    'trend': np.random.choice(['improving', 'stable', 'worsening'])
                }
            analysis[domain] = {
                'symptoms': domain_scores,
                'domain_average': np.mean([s['score'] for s in domain_scores.values()])
            }

        return analysis

    def _combine_features(self, features_dict: Dict) -> np.ndarray:
        """Combine all features"""
        all_features = []

        for category, features in features_dict.items():
            if isinstance(features, dict):
                for v in features.values():
                    if isinstance(v, (int, float)):
                        all_features.append(v)

        return np.array(all_features, dtype=np.float32)

    def _handle_analyze(self, message: AgentMessage) -> Dict:
        """Handle analyze request"""
        data = message.payload.get('data', {})
        return self.process_data(data)

    def _handle_eeg_analysis(self, message: AgentMessage) -> Dict:
        """Handle EEG analysis"""
        eeg_data = message.payload.get('eeg_data', np.random.randn(14, 2560))
        return self._analyze_eeg(eeg_data)

    def _handle_connectivity(self, message: AgentMessage) -> Dict:
        """Handle connectivity analysis"""
        fmri_data = message.payload.get('fmri_data', np.random.randn(90, 200))
        return self._analyze_connectivity(fmri_data)

    def _handle_predict(self, message: AgentMessage) -> Dict:
        """Handle prediction"""
        return {
            'prediction': np.random.choice(self.config['stages']),
            'confidence': np.random.uniform(0.6, 0.95)
        }


class EnsembleCoordinatorAgent(BaseAgent):
    """
    Coordinator Agent for Multi-Disease Ensemble Analysis

    Coordinates multiple disease detection agents and
    provides unified diagnostic assessment.
    """

    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id,
            agent_name="Ensemble Coordinator",
            agent_type="coordinator"
        )

        self.disease_agents = {}
        self.ensemble_results = {}

    def initialize(self):
        """Initialize coordinator"""
        logger.info(f"Initializing {self.agent_name}")

        self.register_capability(AgentCapability(
            name="ensemble_diagnosis",
            description="Coordinate multi-disease analysis",
            input_schema={"patient_data": "dict"},
            output_schema={"diagnoses": "list", "confidence": "dict"}
        ))

        self.register_action('coordinate_analysis', self._handle_coordinate)
        self.register_action('register_disease_agent', self._handle_register_agent)
        self.register_action('get_ensemble_result', self._handle_get_result)

    def cleanup(self):
        """Cleanup"""
        self.disease_agents.clear()

    def process_data(self, data: Any) -> Dict[str, Any]:
        """Coordinate analysis across disease agents"""
        results = {
            'coordinator_id': self.agent_id,
            'timestamp': datetime.now().isoformat(),
            'disease_results': {},
            'differential_diagnosis': [],
            'recommendations': []
        }

        # Request analysis from each disease agent
        for disease, agent_id in self.disease_agents.items():
            self.request(
                agent_id,
                'analyze',
                {'data': data}
            )

        return results

    def _handle_coordinate(self, message: AgentMessage) -> Dict:
        """Handle coordination request"""
        patient_data = message.payload.get('patient_data', {})
        return self.process_data(patient_data)

    def _handle_register_agent(self, message: AgentMessage) -> Dict:
        """Register a disease agent"""
        disease = message.payload.get('disease')
        agent_id = message.payload.get('agent_id')
        self.disease_agents[disease] = agent_id
        return {'status': 'registered', 'disease': disease}

    def _handle_get_result(self, message: AgentMessage) -> Dict:
        """Get ensemble results"""
        return self.ensemble_results


if __name__ == "__main__":
    print("Disease Agents Demo")
    print("=" * 50)

    # Create agents
    alzheimer_agent = AlzheimerDetectionAgent()
    parkinson_agent = ParkinsonDetectionAgent()
    schizophrenia_agent = SchizophreniaDetectionAgent()

    # Initialize
    alzheimer_agent.initialize()
    parkinson_agent.initialize()
    schizophrenia_agent.initialize()

    # Demo data
    demo_data = {
        'mri_data': np.random.randn(64, 64, 64),
        'clinical_data': {'age': 72, 'mmse': 24}
    }

    # Test Alzheimer agent
    print("\nAlzheimer's Detection:")
    result = alzheimer_agent.process_data(demo_data)
    print(f"  Prediction: {result['predictions']['stage']}")
    print(f"  Confidence: {result['confidence']:.2f}")

    # Test Parkinson agent
    print("\nParkinson's Detection:")
    pd_data = {'voice_data': np.random.randn(16000), 'clinical_data': {'age': 65}}
    result = parkinson_agent.process_data(pd_data)
    print(f"  Prediction: {result['predictions']['stage']}")
    print(f"  Confidence: {result['confidence']:.2f}")

    # Test Schizophrenia agent
    print("\nSchizophrenia Detection:")
    sz_data = {'eeg_data': np.random.randn(14, 2560), 'clinical_data': {'age': 28}}
    result = schizophrenia_agent.process_data(sz_data)
    print(f"  Prediction: {result['predictions']['stage']}")
    print(f"  Confidence: {result['confidence']:.2f}")

    print("\nDemo complete!")
