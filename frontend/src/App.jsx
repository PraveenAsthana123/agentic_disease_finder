import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, AreaChart, Area
} from 'recharts'

// Import new components
import PipelineManager from './components/PipelineManager'
import JobScheduler from './components/JobScheduler'
import InferenceDashboard from './components/InferenceDashboard'
import IntegrationHub from './components/IntegrationHub'
import MonitoringDashboard from './components/MonitoringDashboard'
import AnalysisUI from './components/AnalysisUI'
import MetricsDashboard from './components/MetricsDashboard'
import InfographicsDashboard from './components/InfographicsDashboard'
import DepartmentsDashboard, { DEPARTMENTS } from './components/DepartmentsDashboard'
import EntropyDashboard from './components/EntropyDashboard'
import TopomapDashboard from './components/TopomapDashboard'
import ExpertDashboard from './components/ExpertDashboard'
import DataCleaningDashboard from './components/DataCleaningDashboard'
import ICLabelDashboard from './components/ICLabelDashboard'
import SeizureTimelineDashboard from './components/SeizureTimelineDashboard'
import SynchrosqueezingDashboard from './components/SynchrosqueezingDashboard'
import XAIDashboard from './components/XAIDashboard'
import GreatExpectationsDashboard from './components/GreatExpectationsDashboard'
import DataSharingDashboard from './components/DataSharingDashboard'
import DataGovernanceDashboard from './components/DataGovernanceDashboard'
import TorchMetricsDashboard from './components/TorchMetricsDashboard'
import DeepchecksDashboard from './components/DeepchecksDashboard'
import AIF360Dashboard from './components/AIF360Dashboard'
import TorchEEGDashboard from './components/TorchEEGDashboard'
import ILAEClassificationDashboard from './components/ILAEClassificationDashboard'
import AnnotationDashboard from './components/AnnotationDashboard'
import AICostDashboard from './components/AICostDashboard'
import InferenceGPUDashboard from './components/InferenceGPUDashboard'
import SpikeOverlayDashboard from './components/SpikeOverlayDashboard'
import EpilepsyNurseDashboard from './components/EpilepsyNurseDashboard'
import PharmacistDashboard from './components/PharmacistDashboard'
import NeuropsychologistDashboard from './components/NeuropsychologistDashboard'
import RadiologistDashboard from './components/RadiologistDashboard'
import EmbeddingDriftDashboard from './components/EmbeddingDriftDashboard'
import SLPDashboard from './components/SLPDashboard'
import PsychologistDashboard from './components/PsychologistDashboard'
import PsychiatristDashboard from './components/PsychiatristDashboard'
import DietitianDashboard from './components/DietitianDashboard'
import SocialWorkerDashboard from './components/SocialWorkerDashboard'
import MedicationDashboard from './components/MedicationDashboard'
import ExecutiveScorecardDashboard from './components/ExecutiveScorecardDashboard'
import AIUsageDashboard from './components/AIUsageDashboard'
import TherapyDashboard from './components/TherapyDashboard'
import NotificationDashboard from './components/NotificationDashboard'
import AlertsDashboard from './components/AlertsDashboard'
import ToolExecutionDashboard from './components/ToolExecutionDashboard'
import ReportsDashboard from './components/ReportsDashboard'
import DatabaseOpsDashboard from './components/DatabaseOpsDashboard'
import CampaignsDashboard from './components/CampaignsDashboard'
import AIRiskDashboard from './components/AIRiskDashboard'
import ChunkingDashboard from './components/ChunkingDashboard'
import HallucinationDashboard from './components/HallucinationDashboard'
import DevOpsDashboard from './components/DevOpsDashboard'
import ContentFreshnessDashboard from './components/ContentFreshnessDashboard'
import AIComplianceDashboard from './components/AIComplianceDashboard'
import ResponseQualityDashboard from './components/ResponseQualityDashboard'
import RetrievalEvalDashboard from './components/RetrievalEvalDashboard'
import AgentLoopDashboard from './components/AgentLoopDashboard'
import ExecutiveAIDashboard from './components/ExecutiveAIDashboard'
import EventQueueDashboard from './components/EventQueueDashboard'
import RoutingDashboard from './components/RoutingDashboard'
import CitationDashboard from './components/CitationDashboard'
import AgentMemoryDashboard from './components/AgentMemoryDashboard'
import MCPFederationDashboard from './components/MCPFederationDashboard'
import MCPOverviewDashboard from './components/MCPOverviewDashboard'
import ReleaseDashboard from './components/ReleaseDashboard'
import RetrievalDashboard from './components/RetrievalDashboard'
import AgentEvaluationDashboard from './components/AgentEvaluationDashboard'
import IntegrationDashboard from './components/IntegrationDashboard'
import ResponsibleAIDashboard from './components/ResponsibleAIDashboard'
import AppointmentsDashboard from './components/AppointmentsDashboard'
import BillingClaimsDashboard from './components/BillingClaimsDashboard'
import FinOpsDashboard from './components/FinOpsDashboard'
import CSSRSDashboard from './components/CSSRSDashboard'
import PHQ9Dashboard from './components/PHQ9Dashboard'
import ICANoiseCleaningDashboard from './components/ICANoiseCleaningDashboard'
import FIMDashboard from './components/FIMDashboard'
import COPMDashboard from './components/COPMDashboard'
import WAISDashboard from './components/WAISDashboard'
import DigitSpanDashboard from './components/DigitSpanDashboard'
import AMPSDashboard from './components/AMPSDashboard'
import VideoEEGDashboard from './components/VideoEEGDashboard'
import NCVDashboard from './components/NCVDashboard'
import BlinkReflexDashboard from './components/BlinkReflexDashboard'
import SSEPDashboard from './components/SSEPDashboard'
import VEPDashboard from './components/VEPDashboard'
import EMGDashboard from './components/EMGDashboard'
import RNSDashboard from './components/RNSDashboard'
import BeraDashboard from './components/BeraDashboard'
import SSRDashboard from './components/SSRDashboard'
import ABPMHolterDashboard from './components/ABPMHolterDashboard'
import FeatureEvaluationDashboard from './components/FeatureEvaluationDashboard'
import FeatureSelectionDashboard from './components/FeatureSelectionDashboard'
import AutomaticPipelinesDashboard from './components/AutomaticPipelinesDashboard'
import TransferLearningDashboard from './components/TransferLearningDashboard'
import CrossPatientDashboard from './components/CrossPatientDashboard'
import AutonomicDashboard from './components/AutonomicDashboard'
import HRVDashboard from './components/HRVDashboard'
import PNESDifferentialDashboard from './components/PNESDifferentialDashboard'
import EEGMRIConcordanceDashboard from './components/EEGMRIConcordanceDashboard'
import DataVersioningDashboard from './components/DataVersioningDashboard'
import ModelOpsDashboard from './components/ModelOpsDashboard'
import MLOpsDashboard from './components/MLOpsDashboard'
import LLMOpsDashboard from './components/LLMOpsDashboard'
import DataAugmentationDashboard from './components/DataAugmentationDashboard'
import SeizurePredictionDashboard from './components/SeizurePredictionDashboard'
import DataStewardDashboard from './components/DataStewardDashboard'
import HybridPipelineDashboard from './components/HybridPipelineDashboard'
import ConnectivityDashboard from './components/ConnectivityDashboard'
import TrustAIDashboard from './components/TrustAIDashboard'
import EthicalAIDashboard from './components/EthicalAIDashboard'
import DataDriftDashboard from './components/DataDriftDashboard'
import ModelDriftDashboard from './components/ModelDriftDashboard'
import FeatureDriftDashboard from './components/FeatureDriftDashboard'
import OutputDriftDashboard from './components/OutputDriftDashboard'
import ExerciseDashboard from './components/ExerciseDashboard'
import KnowledgeGraphDashboard from './components/KnowledgeGraphDashboard'
import PromptDriftDashboard from './components/PromptDriftDashboard'
import AnomalyDetectionDashboard from './components/AnomalyDetectionDashboard'
import CausalAIDashboard from './components/CausalAIDashboard'
import DeepLearningDashboard from './components/DeepLearningDashboard'
import BiasDetectionDashboard from './components/BiasDetectionDashboard'
import DigitalTwinDashboard from './components/DigitalTwinDashboard'
import AIObservabilityDashboard from './components/AIObservabilityDashboard'
import ModelMonitoringDashboard from './components/ModelMonitoringDashboard'
import ContinuousMonitoringDashboard from './components/ContinuousMonitoringDashboard'
import AIControlTowerDashboard from './components/AIControlTowerDashboard'
import GenerativeAIDashboard from './components/GenerativeAIDashboard'
import HumanEvaluationDashboard from './components/HumanEvaluationDashboard'
import ModelGovernanceDashboard from './components/ModelGovernanceDashboard'
import MultimodalAIDashboard from './components/MultimodalAIDashboard'
import DriftDetectionDashboard from './components/DriftDetectionDashboard'
import ExplainableAIDashboard from './components/ExplainableAIDashboard'
import CommunicationAIDashboard from './components/CommunicationAIDashboard'
import FoundationModelsDashboard from './components/FoundationModelsDashboard'
import NeonatalEEGDashboard from './components/NeonatalEEGDashboard'
import AnalyticsAIDashboard from './components/AnalyticsAIDashboard'
import InterpretableAIDashboard from './components/InterpretableAIDashboard'
import AgenticRAGDashboard from './components/AgenticRAGDashboard'
import VisitsDashboard from './components/VisitsDashboard'
import PrescriptionsDashboard from './components/PrescriptionsDashboard'
import ADLDashboard from './components/ADLDashboard'
import ClinicalTasksDashboard from './components/ClinicalTasksDashboard'
import PatientSeenDashboard from './components/PatientSeenDashboard'
import PatientDashboardPanel from './components/PatientDashboardPanel'
import DataLineageDashboard from './components/DataLineageDashboard'
import AISecurityDashboard from './components/AISecurityDashboard'
import DataAcquisitionDashboard from './components/DataAcquisitionDashboard'
import DataPrivacyDashboard from './components/DataPrivacyDashboard'
import DataQualityDashboard from './components/DataQualityDashboard'
import ContinuousLearningDashboard from './components/ContinuousLearningDashboard'
import EmbeddingDashboard from './components/EmbeddingDashboard'
import AILifecycleDashboard from './components/AILifecycleDashboard'
import MCPGovernanceDashboard from './components/MCPGovernanceDashboard'
import AgentCouncilDashboard from './components/AgentCouncilDashboard'
import GroundingDashboard from './components/GroundingDashboard'
import AIRedTeamDashboard from './components/AIRedTeamDashboard'
import KnowledgeManagementDashboard from './components/KnowledgeManagementDashboard'
import FineTuningDashboard from './components/FineTuningDashboard'
import VectorDBDashboard from './components/VectorDBDashboard'
import ImageSegmentationDashboard from './components/ImageSegmentationDashboard'
import ObjectDetectionDashboard from './components/ObjectDetectionDashboard'
import YOLODetectionDashboard from './components/YOLODetectionDashboard'
import SpeechAIDashboard from './components/SpeechAIDashboard'
import VoiceAIDashboard from './components/VoiceAIDashboard'
import TextToAudioDashboard from './components/TextToAudioDashboard'
import TextToVideoDashboard from './components/TextToVideoDashboard'
import ConversationalAIDashboard from './components/ConversationalAIDashboard'
import CognitiveProfileDashboard from './components/CognitiveProfileDashboard'
import TimeSeriesAIDashboard from './components/TimeSeriesAIDashboard'
import MedicationInteractionDashboard from './components/MedicationInteractionDashboard'
import PatientReportingDashboard from './components/PatientReportingDashboard'
import ResearchCoordinatorDashboard from './components/ResearchCoordinatorDashboard'
import NeurologistDashboard from './components/NeurologistDashboard'
import NeurosurgeonDashboard from './components/NeurosurgeonDashboard'
import CNNResNetDashboard from './components/CNNResNetDashboard'
import NeurophysiologistDashboard from './components/NeurophysiologistDashboard'
import RNNLSTMDashboard from './components/RNNLSTMDashboard'
import OccupationalTherapistDashboard from './components/OccupationalTherapistDashboard'
import EEGTechnicianDashboard from './components/EEGTechnicianDashboard'
import IRBEthicsDashboard from './components/IRBEthicsDashboard'
import ClinicalDataManagerDashboard from './components/ClinicalDataManagerDashboard'
import PatientCaregiverDashboard from './components/PatientCaregiverDashboard'
import ClinicalPsychologistDashboard from './components/ClinicalPsychologistDashboard'
import IoTEngineerDashboard from './components/IoTEngineerDashboard'
import AIFederationDashboard from './components/AIFederationDashboard'
import ISSopDashboard from './components/ISSopDashboard'
import TriggerTrackingDashboard from './components/TriggerTrackingDashboard'
import EmergencyCaregiverDashboard from './components/EmergencyCaregiverDashboard'
import CaregiverReadinessDashboard from './components/CaregiverReadinessDashboard'
import MedicationManagementDashboard from './components/MedicationManagementDashboard'
import PROOutcomesDashboard from './components/PROOutcomesDashboard'
import DemographicsDashboard from './components/DemographicsDashboard'
import WearablesDigitalTwinDashboard from './components/WearablesDigitalTwinDashboard'
import SelfServiceDashboard from './components/SelfServiceDashboard'
import QATestSuiteDashboard from './components/QATestSuiteDashboard'
import ProductManagerDashboard from './components/ProductManagerDashboard'
import AdminDashboard from './components/AdminDashboard'
import FunctionalBADashboard from './components/FunctionalBADashboard'
import IntegrationRoleDashboard from './components/IntegrationRoleDashboard'
import DatasetCoverageDashboard from './components/DatasetCoverageDashboard'
import AIDarkFactoryDashboard from './components/AIDarkFactoryDashboard'
import SeizureRiskForecastingDashboard from './components/SeizureRiskForecastingDashboard'
import CloudOpsDashboard from './components/CloudOpsDashboard'
import ObservabilityDashboard from './components/ObservabilityDashboard'
import SeizureSeverityDashboard from './components/SeizureSeverityDashboard'
import SeizureDiaryDashboard from './components/SeizureDiaryDashboard'
import AbpmDashboard from './components/AbpmDashboard'
import ChangeManagementDashboard from './components/ChangeManagementDashboard'
import ScalogramDashboard from './components/ScalogramDashboard'
import SaliencyAttentionDashboard from './components/SaliencyAttentionDashboard'
import GuardrailsDashboard from './components/GuardrailsDashboard'
import SpwvdDashboard from './components/SpwvdDashboard'
import PatientFacingReportDashboard from './components/PatientFacingReportDashboard'
import RLHFTrainingDashboard from './components/RLHFTrainingDashboard'
import FederatedLearningDashboard from './components/FederatedLearningDashboard'
import GNNElectrodeConnectivityDashboard from './components/GNNElectrodeConnectivityDashboard'
import PatientEducationDashboard from './components/PatientEducationDashboard'
import AudioConverterDashboard from './components/AudioConverterDashboard'
import PACDashboard from './components/PACDashboard'
import BodyMovementDashboard from './components/BodyMovementDashboard'
import VideoConverterDashboard from './components/VideoConverterDashboard'
import SurveyLinkDashboard from './components/SurveyLinkDashboard'
import TokenCostDashboard from './components/TokenCostDashboard'
import ShadowAIDashboard from './components/ShadowAIDashboard'
import NoiseCleaningDashboard from './components/NoiseCleaningDashboard'
import MoCAAutoscoringDashboard from './components/MoCAAutoscoringDashboard'
import EdgeDeployDashboard from './components/EdgeDeployDashboard'
import ClosedLoopDashboard from './components/ClosedLoopDashboard'
import BandHeatmapDashboard from './components/BandHeatmapDashboard'
import XAIGroundTruthDashboard from './components/XAIGroundTruthDashboard'
import DeviceTelemetryDashboard from './components/DeviceTelemetryDashboard'
import TelehealthDashboard from './components/TelehealthDashboard'
import WorkflowDashboard from './components/WorkflowDashboard'
import ClinicalFlowchartsDashboard from './components/ClinicalFlowchartsDashboard'
import FunctionalRecoveryDashboard from './components/FunctionalRecoveryDashboard'
import IncidentManagementDashboard from './components/IncidentManagementDashboard'
import SegmentationDashboard from './components/SegmentationDashboard'
import AssessmentDashboard from './components/AssessmentDashboard'
import EpilepsyBoardDashboard from './components/EpilepsyBoardDashboard'
import ConsentManagementDashboard from './components/ConsentManagementDashboard'
import ReferralTriageDashboard from './components/ReferralTriageDashboard'
import RAGMetadataFilterDashboard from './components/RAGMetadataFilterDashboard'
import RecoveryTrajectoryDashboard from './components/RecoveryTrajectoryDashboard'
import ArtifactDetectionDashboard from './components/ArtifactDetectionDashboard'
import CognitiveDeclineDashboard from './components/CognitiveDeclineDashboard'
import MRIReviewDashboard from './components/MRIReviewDashboard'
import GoalAttainmentDashboard from './components/GoalAttainmentDashboard'
import AutonomicAnalysisDashboard from './components/AutonomicAnalysisDashboard'
import GuidedAssessmentDashboard from './components/GuidedAssessmentDashboard'
import ModelRetirementDashboard from './components/ModelRetirementDashboard'
import AIROIDashboard from './components/AIROIDashboard'
import DailyCarePlanDashboard from './components/DailyCarePlanDashboard'
import PatientReportDashboard from './components/PatientReportDashboard'
import UserManagementDashboard from './components/UserManagementDashboard'
import BenchmarkValidationDashboard from './components/BenchmarkValidationDashboard'
import GroupsTeamsDashboard from './components/GroupsTeamsDashboard'
import RehabPlanDashboard from './components/RehabPlanDashboard'
import MedicationAdherenceDashboard from './components/MedicationAdherenceDashboard'
import MultimodalFusionDashboard from './components/MultimodalFusionDashboard'
import PnesScreeningDashboard from './components/PnesScreeningDashboard'
import SnnNeuromorphicDashboard from './components/SnnNeuromorphicDashboard'
import PatientPortalDashboard from './components/PatientPortalDashboard'
import MCPServerDashboard from './components/MCPServerDashboard'
import NeuroLabReadinessDashboard from './components/NeuroLabReadinessDashboard'
import ComorbidityAnalysisDashboard from './components/ComorbidityAnalysisDashboard'
import SleepStagingDashboard from './components/SleepStagingDashboard'
import SemiologyClassifierDashboard from './components/SemiologyClassifierDashboard'
import AIGovernanceDashboard from './components/AIGovernanceDashboard'
import EpworthDashboard from './components/EpworthDashboard'
import RealtimeEEGQCDashboard from './components/RealtimeEEGQCDashboard'
import PatientVideoDashboard from './components/PatientVideoDashboard'
import RAGReportGenDashboard from './components/RAGReportGenDashboard'
import SubtleSeizureDashboard from './components/SubtleSeizureDashboard'
import APIResilienceDashboard from './components/APIResilienceDashboard'
import OTelLLMDashboard from './components/OTelLLMDashboard'
import MobileAlertsDashboard from './components/MobileAlertsDashboard'
import ResourceMonitorDashboard from './components/ResourceMonitorDashboard'
import ConfigDriftDashboard from './components/ConfigDriftDashboard'
import AlertFatigueDashboard from './components/AlertFatigueDashboard'
import DataCompletenessDashboard from './components/DataCompletenessDashboard'
import TreatmentEfficacyDashboard from './components/TreatmentEfficacyDashboard'
import StructuredReportingDashboard from './components/StructuredReportingDashboard'
import ReinforcementLearningDashboard from './components/ReinforcementLearningDashboard'
import ICD10CodingDashboard from './components/ICD10CodingDashboard'
import AIIncidentDashboard from './components/AIIncidentDashboard'
import PreSurgicalEvaluationDashboard from './components/PreSurgicalEvaluationDashboard'
import MedicationRefillDashboard from './components/MedicationRefillDashboard'
import SecureMessagingDashboard from './components/SecureMessagingDashboard'
import PatientDocumentsDashboard from './components/PatientDocumentsDashboard'
import ConsentDashboard from './components/ConsentDashboard'
import BmadDashboard from './components/BmadDashboard'
import CrossPatientBenchmarkDashboard from './components/CrossPatientBenchmarkDashboard'
import ClinicalPharmacistDashboard from './components/ClinicalPharmacistDashboard'
import DecisionAiDashboard from './components/DecisionAiDashboard'
import PatientsSeenDashboard from './components/PatientsSeenDashboard'
import DataOpsDashboard from './components/DataOpsDashboard'
import PopulationHealthDashboard from './components/PopulationHealthDashboard'
import PharmacogenomicsDashboard from './components/PharmacogenomicsDashboard'
import SurgicalOutcomeDashboard from './components/SurgicalOutcomeDashboard'
import DatasetRequirementsDashboard from './components/DatasetRequirementsDashboard'

// API Base URL
const API_URL = '/api'

// Colors
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

// Disease Options
const DISEASES = [
  { id: 'alzheimer', name: "Alzheimer's Disease" },
  { id: 'parkinson', name: "Parkinson's Disease" },
  { id: 'schizophrenia', name: 'Schizophrenia' },
  { id: 'epilepsy', name: 'Epilepsy' },
  { id: 'autism', name: 'Autism Spectrum Disorder' },
  { id: 'stress', name: 'Chronic Stress' },
  { id: 'depression', name: 'Depression' }
]

// Channel Configurations
const CHANNEL_CONFIGS = [
  { channels: 2, name: 'Custom 2-ch' },
  { channels: 4, name: 'Custom 4-ch' },
  { channels: 5, name: 'Emotiv Insight' },
  { channels: 8, name: 'Custom 8-ch' },
  { channels: 14, name: 'Emotiv EPOC X' },
  { channels: 16, name: 'Custom 16-ch' },
  { channels: 22, name: 'Standard 22-ch' },
  { channels: 24, name: 'Custom 24-ch' },
  { channels: 32, name: 'Emotiv EPOC Flex' },
  { channels: 64, name: 'Custom 64-ch' }
]

// Main App Component
function App() {
  // State
  const [activeTab, setActiveTab] = useState('departments')
  const [activeDept, setActiveDept] = useState(DEPARTMENTS[0].id)
  const [selectedDisease, setSelectedDisease] = useState('depression')
  const [modality, setModality] = useState('eeg')
  const [channelConfig, setChannelConfig] = useState(22)
  const [analysisOptions, setAnalysisOptions] = useState({
    asis: true,
    social: true,
    tobeManual: true,
    tobeAuto: true,
    statistical: true,
    clinical: true
  })
  const [targetAccuracy, setTargetAccuracy] = useState(99.9)
  const [trainingSamples, setTrainingSamples] = useState(300)
  const [epochs, setEpochs] = useState(30)

  // Global REAL-error capture — surfaces server 5xx, network failures, and JS errors on the UI.
  const [globalErrors, setGlobalErrors] = useState([])
  const pushErr = useCallback((source, message) => {
    setGlobalErrors(prev => [{ t: new Date().toLocaleTimeString(), source, message: String(message).slice(0, 300) },
      ...prev.filter(e => e.message !== String(message).slice(0, 300))].slice(0, 8))
  }, [])
  useEffect(() => {
    // axios interceptor — only REAL errors (5xx server errors + network/timeouts), not expected 4xx
    const id = axios.interceptors.response.use(r => r, (err) => {
      const url = err.config?.url || ''
      if (err.response) {
        if (err.response.status >= 500) pushErr(`API ${err.response.status}`, `${url} → ${err.response.data?.detail || err.response.data?.message || 'server error'}`)
      } else if (err.code === 'ECONNABORTED') {
        pushErr('API timeout', `${url} timed out`)
      } else {
        pushErr('No backend', `${url} → request failed (backend down on :8010 or wrong port — use :3003)`)
      }
      return Promise.reject(err)
    })
    const onErr = (e) => pushErr('JS error', e.message || e.error?.message || 'script error')
    const onRej = (e) => pushErr('Unhandled', e.reason?.message || String(e.reason))
    window.addEventListener('error', onErr)
    window.addEventListener('unhandledrejection', onRej)
    return () => { axios.interceptors.response.eject(id); window.removeEventListener('error', onErr); window.removeEventListener('unhandledrejection', onRej) }
  }, [pushErr])

  // Results state
  const [isLoading, setIsLoading] = useState(false)
  const [classificationResult, setClassificationResult] = useState(null)
  const [asisAnalysis, setAsisAnalysis] = useState(null)
  const [socialAnalysis, setSocialAnalysis] = useState(null)
  const [tobeAnalysis, setTobeAnalysis] = useState(null)
  const [statisticalData, setStatisticalData] = useState(null)
  const [clinicalData, setClinicalData] = useState(null)
  const [error, setError] = useState(null)

  // Tab options
  const tabs = [
    { id: 'departments', label: 'Departments' },
    { id: 'analysis', label: 'AI Analysis' },
    { id: 'metrics', label: 'Metrics Dashboard' },
    { id: 'asis', label: 'AS-IS Analysis' },
    { id: 'social', label: 'Social Analysis' },
    { id: 'tobe', label: 'To-Be Analysis' },
    { id: 'statistical', label: 'Statistical' },
    { id: 'clinical', label: 'Clinical' },
    { id: 'monitoring', label: 'RAG Monitoring' },
    { id: 'pipelines', label: 'Pipelines' },
    { id: 'jobs', label: 'Jobs' },
    { id: 'inference', label: 'Inference Testing' },
    { id: 'integrations', label: 'Integrations' },
    { id: 'infographics', label: 'Infographics' },
    { id: 'entropy', label: 'Entropy Analysis' },
    { id: 'topomap', label: 'Topographic Maps' },
    { id: 'expert', label: 'Expert Dashboards' },
    { id: 'datacleaning', label: 'Data Cleaning' },
    { id: 'icalabel', label: 'ICLabel QC' },
    { id: 'seizuretimeline', label: 'Seizure Timeline' },
    { id: 'synchrosqueezing', label: 'Synchrosqueezing' },
    { id: 'xai', label: 'Explainable AI' },
    { id: 'great-expectations', label: 'Data Quality (GE)' },
    { id: 'datasharing', label: 'Data Sharing' },
    { id: 'datagovernance', label: 'Data Governance' },
    { id: 'torchmetrics', label: 'TorchMetrics' },
    { id: 'deepchecks', label: 'Deepchecks' },
    { id: 'aif360', label: 'AIF360 Bias' },
    { id: 'torcheeg', label: 'TorchEEG' },
    { id: 'ilae-classification', label: 'ILAE Classification' },
    { id: 'annotation', label: 'Annotation QC' },
    { id: 'ai-cost', label: 'AI Cost' },
    { id: 'token-cost', label: 'Token / Cost' },
    { id: 'shadow-ai', label: 'Shadow AI' },
    { id: 'noise-cleaning', label: 'Noise Cleaning' },
    { id: 'moca-autoscoring', label: 'MoCA Auto-Scoring' },
    { id: 'inference-gpu', label: 'Inference/GPU' },
    { id: 'spike-overlay', label: 'Spike Overlay' },
    { id: 'epilepsy-nurse', label: 'Epilepsy Nurse' },
    { id: 'pharmacist', label: 'Pharmacist' },
    { id: 'embedding-drift', label: 'Embedding Drift' },
    { id: 'slp', label: 'SLP' },
    { id: 'psychologist', label: 'Psychologist' },
    { id: 'dietitian', label: 'Dietitian' },
    { id: 'social-worker', label: 'Social Worker' },
    { id: 'medication', label: 'Medication' },
    { id: 'executive-scorecard', label: 'Executive Scorecard' },
    { id: 'ai-usage', label: 'AI Usage' },
    { id: 'therapy', label: 'Therapy' },
    { id: 'notifications', label: 'Notifications' },
    { id: 'alerts', label: 'Alerts' },
    { id: 'tool-execution', label: 'Tool Execution' },
    { id: 'reports', label: 'My Reports' },
    { id: 'database-ops', label: 'Database Ops' },
    { id: 'campaigns', label: 'Campaigns' },
    { id: 'ai-risk', label: 'AI Risk Mgmt' },
    { id: 'chunking', label: 'Chunking' },
    { id: 'hallucination', label: 'Hallucination' },
    { id: 'devops', label: 'DevOps / CI-CD' },
    { id: 'content-freshness', label: 'Content Freshness' },
    { id: 'ai-compliance', label: 'AI Compliance' },
    { id: 'ai-red-team', label: 'AI Red Team' },
    { id: 'knowledge-mgmt', label: 'Knowledge Management' },
    { id: 'fine-tuning', label: 'Fine-Tuning Pipeline' },
    { id: 'response-quality', label: 'Response Quality' },
    { id: 'retrieval', label: 'Retrieval' },
    { id: 'retrieval-eval', label: 'Retrieval Evaluation' },
    { id: 'agent-loop', label: 'Agent Loop / Goal-Drift' },
    { id: 'executive-ai', label: 'Executive AI' },
    { id: 'event-queue', label: 'Event / Queue' },
    { id: 'routing', label: 'Routing' },
    { id: 'citation', label: 'Citation' },
    { id: 'agent-memory', label: 'Agent Memory' },
    { id: 'mcp-overview', label: 'MCP Overview' },
    { id: 'mcp-federation', label: 'MCP Federation' },
    { id: 'release', label: 'Release Mgmt' },
    { id: 'integration-dash', label: 'Integration' },
    { id: 'responsible-ai', label: 'Responsible AI' },
    { id: 'appointments', label: 'Appointments' },
    { id: 'billing-claims', label: 'Billing & Claims' },
    { id: 'visits', label: 'True Visits' },
    { id: 'finops', label: 'FinOps' },
    { id: 'cssrs', label: 'C-SSRS' },
    { id: 'phq9', label: 'PHQ-9' },
    { id: 'ica-noise-cleaning', label: 'ICA Cleaning' },
    { id: 'fim', label: 'FIM' },
    { id: 'copm', label: 'COPM' },
    { id: 'wais', label: 'WAIS (IQ)' },
    { id: 'digit-span', label: 'Digit Span' },
    { id: 'amps', label: 'AMPS' },
    { id: 'video-eeg', label: 'Video EEG' },
    { id: 'ncv', label: 'NCV' },
    { id: 'blink-reflex', label: 'Blink Reflex' },
    { id: 'ssep', label: 'SSEP' },
    { id: 'vep', label: 'VEP' },
    { id: 'emg', label: 'EMG' },
    { id: 'rns', label: 'RNS' },
    { id: 'bera', label: 'BERA' },
    { id: 'ssr', label: 'SSR' },
    { id: 'abpm-holter', label: 'ABPM / Holter' },
    { id: 'feature-evaluation', label: 'Feature Evaluation' },
    { id: 'feature-selection', label: 'Feature Selection' },
    { id: 'autonomic', label: 'Autonomic' },
    { id: 'hrv', label: 'HRV / RR Variation' },
    { id: 'pnes-differential', label: 'PNES Differential' },
    { id: 'eeg-mri-concordance', label: 'EEG-MRI Concordance' },
    { id: 'data-versioning', label: 'Data Versioning' },
    { id: 'model-ops', label: 'Model Ops' },
    { id: 'mlops', label: 'MLOps' },
    { id: 'llmops', label: 'LLMOps' },
    { id: 'trust-ai', label: 'Trust AI' },
    { id: 'ethical-ai', label: 'Ethical AI' },
    { id: 'data-drift', label: 'Data Drift' },
    { id: 'feature-drift', label: 'Feature Drift' },
    { id: 'model-drift', label: 'Model Drift' },
    { id: 'output-drift', label: 'Output/RAG Drift' },
    { id: 'prompt-drift', label: 'Prompt Drift' },
    { id: 'exercise', label: 'Exercise / Rehab' },
    { id: 'knowledge-graph', label: 'Knowledge Graph' },
    { id: 'anomaly-detection', label: 'Anomaly Detection' },
    { id: 'causal-ai', label: 'Causal AI' },
    { id: 'deep-learning', label: 'Deep Learning' },
    { id: 'bias-detection', label: 'Bias Detection' },
    { id: 'digital-twin', label: 'Digital Twin' },
    { id: 'ai-observability', label: 'AI Observability' },
    { id: 'model-monitoring', label: 'Model Monitoring' },
    { id: 'continuous-monitoring', label: 'Continuous Monitoring' },
    { id: 'ai-control-tower', label: 'AI Control Tower' },
    { id: 'generative-ai', label: 'Generative AI' },
    { id: 'human-evaluation', label: 'Human Evaluation' },
    { id: 'model-governance', label: 'Model Governance' },
    { id: 'continuous-learning', label: 'Continuous Learning' },
    { id: 'multimodal-ai', label: 'Multimodal AI' },
    { id: 'drift-detection', label: 'Drift Detection' },
    { id: 'explainable-ai', label: 'Explainable AI' },
    { id: 'xai-groundtruth', label: 'XAI Ground-Truth' },
    { id: 'communication-ai', label: 'Communication AI' },
    { id: 'foundation-models', label: 'Foundation Models' },
    { id: 'neonatal-eeg', label: 'Neonatal EEG' },
    { id: 'analytics-ai', label: 'Analytics AI' },
    { id: 'interpretable-ai', label: 'Interpretable AI' },
    { id: 'agentic-rag', label: 'Agentic RAG' },
    { id: 'grounding', label: 'Grounding' },
    { id: 'prescriptions', label: 'Prescriptions' },
    { id: 'adl', label: 'ADL' },
    { id: 'clinical-tasks', label: 'Clinical Tasks' },
    { id: 'patients-seen', label: 'Patients Seen' },
    { id: 'patient-dashboard', label: 'Patient Dashboard' },
    { id: 'data-lineage', label: 'Data Lineage' },
    { id: 'ai-security', label: 'AI Security' },
    { id: 'data-acquisition', label: 'Data Acquisition' },
    { id: 'data-privacy', label: 'Data Privacy' },
    { id: 'data-quality', label: 'Data Quality' },
    { id: 'embedding', label: 'Embedding & Features' },
    { id: 'ai-lifecycle', label: 'AI Lifecycle Mgmt' },
    { id: 'mcp-governance', label: 'MCP Governance' },
    { id: 'agent-council', label: 'Agent Council' },
    { id: 'vector-db', label: 'Vector DB' },
    { id: 'image-segmentation', label: 'Image Segmentation' },
    { id: 'object-detection', label: 'Object Detection' },
    { id: 'yolo-detection', label: 'YOLO Detection' },
    { id: 'speech-ai', label: 'Speech AI' },
    { id: 'voice-ai', label: 'Voice AI' },
    { id: 'text-to-audio', label: 'Text-to-Audio AI' },
    { id: 'text-to-video', label: 'Text-to-Video AI' },
    { id: 'cognitive-profile', label: 'Cognitive Profile Summary' },
    { id: 'time-series-ai', label: 'Time-Series AI' },
    { id: 'medication-interaction', label: 'Medication Interaction Checker' },
    { id: 'conversational-ai', label: 'Conversational AI' },
    { id: 'patient-reporting', label: 'Patient Reporting' },
    { id: 'research-coordinator', label: 'Research Coordinator' },
    { id: 'neurologist', label: 'Neurologist' },
    { id: 'neurosurgeon', label: 'Neurosurgeon / Epilepsy Surgery' },
    { id: 'cnn-resnet', label: 'CNN/ResNet Spectrogram' },
    { id: 'neurophysiologist', label: 'Neurophysiologist / EEG Reviewer' },
    { id: 'rnn-lstm', label: 'RNN/LSTM Temporal Model' },
    { id: 'neuropsychologist', label: 'Neuropsychologist' },
    { id: 'radiologist', label: 'Radiologist' },
    { id: 'psychiatrist', label: 'Psychiatrist' },
    { id: 'occupational-therapist', label: 'Occupational Therapist' },
    { id: 'eeg-technician', label: 'EEG Technician QC' },
    { id: 'irb-ethics', label: 'IRB / Ethics Officer' },
    { id: 'clinical-data-manager', label: 'Clinical Data Manager' },
    { id: 'patient-caregiver', label: 'Patient / Caregiver' },
    { id: 'clinical-psychologist', label: 'Clinical Psychologist' },
    { id: 'iot-engineer', label: 'IoT Engineer' },
    { id: 'ai-federation', label: 'AI Federation' },
    { id: 'is-sop', label: 'IS SOP' },
    { id: 'trigger-tracking', label: 'Trigger Tracking' },
    { id: 'emergency-caregiver', label: 'Emergency / Caregiver' },
    { id: 'caregiver-readiness', label: 'Caregiver Readiness' },
    { id: 'medication-management', label: 'Medication Management' },
    { id: 'pro-outcomes', label: 'PRO Outcomes' },
    { id: 'demographics', label: 'Demographics' },
    { id: 'wearables-digital-twin', label: 'Wearables & Digital Twin' },
    { id: 'self-service', label: 'Self-Service Portal' },
    { id: 'qa-test-suite', label: 'QA Test Suite' },
    { id: 'product-manager', label: 'Product Manager' },
    { id: 'admin-panel', label: 'Admin Panel' },
    { id: 'functional-ba', label: 'Functional / BA' },
    { id: 'integration-role', label: 'Integration' },
    { id: 'dataset-coverage', label: 'Dataset Coverage' },
    { id: 'dark-factory', label: 'AI Dark Factory' },
    { id: 'seizure-risk-forecast', label: 'Seizure Risk Forecasting' },
    { id: 'cloud-ops', label: 'Cloud Ops' },
    { id: 'observability', label: 'Observability' },
    { id: 'seizure-severity', label: 'Seizure Severity' },
    { id: 'seizure-diary', label: 'Seizure Diary' },
    { id: 'abpm-holter', label: 'ABPM / Holter' },
    { id: 'feature-evaluation', label: 'Feature Evaluation' },
    { id: 'feature-selection', label: 'Feature Selection' },
    { id: 'automatic-pipelines', label: 'Automatic Pipelines' },
    { id: 'transfer-learning', label: 'Transfer Learning' },
    { id: 'cross-patient-benchmark', label: 'Cross-Patient Benchmark' },
    { id: 'data-augmentation', label: 'Data Augmentation' },
    { id: 'seizure-prediction', label: 'Seizure Prediction' },
    { id: 'data-steward', label: 'Data Steward' },
    { id: 'hybrid-pipeline', label: 'Hybrid Pipeline' },
    { id: 'connectivity', label: 'Connectivity Analysis' },
    { id: 'change-management', label: 'Change Management' },
    { id: 'scalogram', label: 'Scalogram (CWT)' },
    { id: 'saliency-attention', label: 'Saliency & Attention' },
    { id: 'guardrails', label: 'NeMo Guardrails' },
    { id: 'spwvd', label: 'SPWVD' },
    { id: 'patient-facing-report', label: 'Patient Report' },
    { id: 'rlhf-training', label: 'RLHF Training' },
    { id: 'federated-learning', label: 'Federated Learning' },
    { id: 'gnn-electrode-connectivity', label: 'GNN Connectivity' },
    { id: 'patient-education', label: 'Patient Education' },
    { id: 'audio-converter', label: 'Audio Converter' },
    { id: 'pac', label: 'PAC Analysis' },
    { id: 'body-movement', label: 'Body Movement' },
    { id: 'video-converter', label: 'Video Converter' },
    { id: 'survey-link', label: 'Survey Link' },
    { id: 'edge-deploy', label: 'Edge Deployment' },
    { id: 'closed-loop', label: 'Closed-Loop Neurostim' },
    { id: 'band-heatmap', label: 'Band Heatmap' },
    { id: 'device-telemetry', label: 'Device Telemetry' },
    { id: 'telehealth', label: 'Telehealth' },
    { id: 'workflow', label: 'Workflow' },
    { id: 'clinical-flowcharts', label: 'Clinical Flowcharts' },
    { id: 'functional-recovery', label: 'Functional Recovery' },
    { id: 'incident-management', label: 'Incident Management' },
    { id: 'segmentation', label: 'EEG Segmentation' },
    { id: 'assessment-analytics', label: 'Assessment Analytics' },
    { id: 'epilepsy-board', label: 'Epilepsy Board' },
    { id: 'consent-management', label: 'Consent Management' },
    { id: 'referral-triage', label: 'Referral Triage' },
    { id: 'rag-metadata-filter', label: 'RAG Metadata Filter' },
    { id: 'recovery-trajectory', label: 'Recovery Trajectory' },
    { id: 'artifact-detection', label: 'Artifact Detection' },
    { id: 'cognitive-decline', label: 'Cognitive Decline' },
    { id: 'mri-review', label: 'MRI Review' },
    { id: 'goal-attainment', label: 'Goal Attainment (GAS)' },
    { id: 'autonomic-analysis', label: 'Autonomic Analysis' },
    { id: 'guided-assessment', label: 'Guided Assessment Flow' },
    { id: 'model-retirement', label: 'Model Retirement' },
    { id: 'ai-roi', label: 'AI ROI' },
    { id: 'daily-care-plan', label: 'Daily Care Plan' },
    { id: 'patient-report', label: 'Patient Report' },
    { id: 'user-management', label: 'User Management' },
    { id: 'benchmark-validation', label: 'Benchmark Validation' },
    { id: 'groups-teams', label: 'Groups & Teams' },
    { id: 'rehab-plan', label: 'Rehab Plan (OT)' },
    { id: 'medication-adherence', label: 'Medication Adherence' },
    { id: 'multimodal-fusion', label: 'Multimodal Fusion' },
    { id: 'pnes-screening', label: 'PNES Screening' },
    { id: 'snn-neuromorphic', label: 'SNN Neuromorphic' },
    { id: 'patient-portal', label: 'Patient Portal' },
    { id: 'mcp-server', label: 'MCP Server' },
    { id: 'neurolab-readiness', label: 'NeuroLab Readiness' },
    { id: 'comorbidity-analysis', label: 'Comorbidity Analysis' },
    { id: 'sleep-staging', label: 'Sleep Staging' },
    { id: 'semiology-classifier', label: 'Semiology Classifier' },
    { id: 'ai-governance', label: 'AI Governance' },
    { id: 'epworth', label: 'Epworth Sleepiness' },
    { id: 'realtime-eeg-qc', label: 'Real-Time EEG QC' },
    { id: 'patient-video', label: 'Patient Video Analysis' },
    { id: 'rag-report-gen', label: 'RAG Report Generation' },
    { id: 'subtle-seizure', label: 'Subtle Seizure Detection' },
    { id: 'api-resilience', label: 'API Resilience' },
    { id: 'otel-llm', label: 'OTel LLM Observability' },
    { id: 'mobile-alerts', label: 'Mobile Alerts / SOS' },
    { id: 'resource-monitor', label: 'Resource Monitor' },
    { id: 'config-drift', label: 'Config Drift Monitor' },
    { id: 'alert-fatigue', label: 'Alert Fatigue Monitor' },
    { id: 'data-completeness', label: 'Data Completeness' },
    { id: 'treatment-efficacy', label: 'Treatment Efficacy' },
    { id: 'structured-reporting', label: 'Structured Reporting' },
    { id: 'reinforcement-learning', label: 'Reinforcement Learning' },
    { id: 'icd10-coding', label: 'ICD-10 Coding' },
    { id: 'ai-incident', label: 'AI Incidents' },
    { id: 'presurgical-evaluation', label: 'Pre-Surgical Evaluation' },
    { id: 'medication-refills', label: 'Medication Refills' },
    { id: 'secure-messaging', label: 'Secure Messaging' },
    { id: 'consent-management', label: 'Consent Management' },
    { id: 'patient-documents', label: 'Patient Documents' },
    { id: 'bmad', label: 'BMAD Spec-Driven Agents' },
    { id: 'cross-patient-benchmark', label: 'Cross-Patient Benchmark' },
    { id: 'clinical-pharmacist', label: 'Clinical Pharmacist' },
    { id: 'decision-ai', label: 'Decision AI' },
    { id: 'patients-seen', label: 'Patients Seen' },
    { id: 'data-ops', label: 'Data Operations' },
    { id: 'population-health', label: 'Population Health' },
    { id: 'pharmacogenomics', label: 'Pharmacogenomics' },
    { id: 'surgical-outcomes', label: 'Surgical Outcomes' },
    { id: 'dataset-requirements', label: 'Dataset Requirements' }
  ]

  // API Calls
  const runClassification = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await axios.post(`${API_URL}/classify`, {
        disease: selectedDisease,
        modality: modality,
        n_channels: channelConfig,
        include_analysis: true
      })

      setClassificationResult(response.data)

      // Also fetch analyses
      if (analysisOptions.asis) fetchAsisAnalysis()
      if (analysisOptions.social) fetchSocialAnalysis()
      if (analysisOptions.tobeAuto) fetchTobeAnalysis()
      if (analysisOptions.statistical) fetchStatistics()
      if (analysisOptions.clinical) fetchClinicalAnalysis()

    } catch (err) {
      setError(err.message)
      // Use mock data for demo
      setClassificationResult(generateMockClassification())
    } finally {
      setIsLoading(false)
    }
  }, [selectedDisease, modality, channelConfig, analysisOptions])

  const fetchAsisAnalysis = async () => {
    try {
      const response = await axios.post(`${API_URL}/analysis`, {
        disease: selectedDisease,
        analysis_type: 'asis'
      })
      setAsisAnalysis(response.data.analysis)
    } catch {
      setAsisAnalysis(generateMockAsisAnalysis())
    }
  }

  const fetchSocialAnalysis = async () => {
    try {
      const response = await axios.post(`${API_URL}/analysis`, {
        disease: selectedDisease,
        analysis_type: 'social'
      })
      setSocialAnalysis(response.data.analysis)
    } catch {
      setSocialAnalysis(generateMockSocialAnalysis())
    }
  }

  const fetchTobeAnalysis = async (manual = false, manualData = null) => {
    try {
      const response = await axios.post(`${API_URL}/analysis`, {
        disease: selectedDisease,
        analysis_type: 'tobe',
        manual: manual,
        manual_data: manualData
      })
      setTobeAnalysis(response.data.analysis)
    } catch {
      setTobeAnalysis(generateMockTobeAnalysis())
    }
  }

  const fetchStatistics = async () => {
    try {
      const response = await axios.get(`${API_URL}/statistics/${selectedDisease}`)
      setStatisticalData(response.data)
    } catch {
      setStatisticalData(generateMockStatistics())
    }
  }

  const fetchClinicalAnalysis = async () => {
    try {
      const response = await axios.post(`${API_URL}/analysis`, {
        disease: selectedDisease,
        analysis_type: 'clinical'
      })
      setClinicalData(response.data.analysis)
    } catch {
      setClinicalData(generateMockClinicalAnalysis())
    }
  }

  // Mock data generators
  const generateMockClassification = () => ({
    success: true,
    disease: selectedDisease,
    modality: modality,
    classification: {
      predictions: {
        accuracy: 0.95 + Math.random() * 0.049,
        confidence: 0.85 + Math.random() * 0.14,
        predicted_class: selectedDisease,
        probabilities: {
          Healthy: Math.random() * 0.2,
          [selectedDisease]: 0.7 + Math.random() * 0.29
        }
      }
    }
  })

  const generateMockAsisAnalysis = () => ({
    report: {
      title: `AS-IS Analysis - ${selectedDisease}`,
      current_state: {
        detection_accuracy: 0.85 + Math.random() * 0.1,
        prevalence_rate: 0.01 + Math.random() * 0.09,
        avg_diagnosis_time_days: Math.floor(90 + Math.random() * 365),
        false_positive_rate: 0.05 + Math.random() * 0.1,
        false_negative_rate: 0.05 + Math.random() * 0.1
      },
      challenges: [
        'Late diagnosis',
        'Limited biomarkers',
        'High symptom variability',
        'Treatment resistance'
      ],
      severity_distribution: {
        mild: 0.3 + Math.random() * 0.1,
        moderate: 0.35 + Math.random() * 0.1,
        severe: 0.2 + Math.random() * 0.1
      }
    }
  })

  const generateMockSocialAnalysis = () => ({
    report: {
      title: `Social Analysis - ${selectedDisease}`,
      social_impact: {
        social_withdrawal_score: 3 + Math.random() * 6,
        communication_difficulty: 3 + Math.random() * 6,
        relationship_impact: 3 + Math.random() * 6,
        work_impact: 3 + Math.random() * 6,
        daily_activity_reduction: 20 + Math.random() * 40
      },
      phone_activity: {
        calls_per_day: 1 + Math.random() * 7,
        messages_per_day: 5 + Math.random() * 25,
        social_app_hours: 0.5 + Math.random() * 3.5
      },
      isolation_risk: 0.3 + Math.random() * 0.5
    }
  })

  const generateMockTobeAnalysis = () => ({
    report: {
      title: `To-Be Analysis - ${selectedDisease}`,
      target_state: {
        target_accuracy: 0.999,
        early_detection_improvement: 0.3 + Math.random() * 0.2,
        diagnosis_time_reduction: 0.4 + Math.random() * 0.2
      },
      recommendations: [
        'Multi-modal AI classification',
        'Real-time EEG monitoring',
        'Hybrid imaging analysis',
        'Continuous biomarker tracking'
      ],
      implementation_timeline: Math.floor(12 + Math.random() * 24),
      projected_benefits: {
        patients_helped: Math.floor(500 + Math.random() * 4500),
        cost_savings_percent: 15 + Math.random() * 20
      }
    }
  })

  const generateMockStatistics = () => ({
    statistics: {
      accuracy: 0.95 + Math.random() * 0.049,
      precision: 0.93 + Math.random() * 0.06,
      recall: 0.92 + Math.random() * 0.07,
      f1_score: 0.94 + Math.random() * 0.05,
      auc: 0.96 + Math.random() * 0.039,
      confusion_matrix: {
        tp: Math.floor(85 + Math.random() * 10),
        tn: Math.floor(80 + Math.random() * 10),
        fp: Math.floor(5 + Math.random() * 10),
        fn: Math.floor(5 + Math.random() * 10)
      }
    }
  })

  const generateMockClinicalAnalysis = () => ({
    report: {
      title: `Clinical Analysis - ${selectedDisease}`,
      diagnosis: {
        primary_condition: selectedDisease,
        confidence: 0.85 + Math.random() * 0.14,
        severity: ['Mild', 'Moderate', 'Severe'][Math.floor(Math.random() * 3)]
      },
      biomarkers: ['EEG patterns', 'Brain imaging', 'Clinical assessment'],
      recommendations: {
        diagnostic: ['Comprehensive neurological exam', 'Cognitive assessment'],
        monitoring: ['Regular follow-ups', 'EEG monitoring'],
        intervention: ['Medication as indicated', 'Therapy programs']
      }
    }
  })

  // Initialize with mock data
  useEffect(() => {
    setAsisAnalysis(generateMockAsisAnalysis())
    setSocialAnalysis(generateMockSocialAnalysis())
    setTobeAnalysis(generateMockTobeAnalysis())
    setStatisticalData(generateMockStatistics())
    setClinicalData(generateMockClinicalAnalysis())
  }, [selectedDisease])

  // Render sidebar
  const renderSidebar = () => (
    <aside className="sidebar">
      <div className="app-header">
        <span className="app-logo">🧠</span>
        <div>
          <div className="app-title">NeuroAI</div>
          <div className="app-subtitle">Disease Detector v2.0</div>
        </div>
      </div>


      <div className="sidebar-section">
        <div className="sidebar-section-title">Disease Selection</div>
        <div className="select-wrapper">
          <select
            className="select-input"
            value={selectedDisease}
            onChange={(e) => setSelectedDisease(e.target.value)}
          >
            {DISEASES.map(d => (
              <option key={d.id} value={d.id}>{d.name}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-section">
        <div className="sidebar-section-title">Main Menu · Departments</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: 360, overflowY: 'auto' }}>
          {[...DEPARTMENTS, ...extraDepartments].map(d => (
            <button
              key={d.id}
              onClick={() => setActiveDept(d.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 8, width: '100%', textAlign: 'left',
                border: 'none', cursor: 'pointer', borderRadius: 6, padding: '8px 10px', fontSize: 13,
                background: activeDept === d.id ? '#1e88e5' : 'transparent',
                color: activeDept === d.id ? '#fff' : '#475569', fontWeight: activeDept === d.id ? 600 : 400,
              }}
            ><span style={{ fontSize: 15 }}>{d.icon}</span><span>{d.name}</span></button>
          ))}
        </div>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-section">
        <div className="sidebar-section-title">Classification Mode</div>
        <div className="radio-group">
          {[
            { value: 'eeg', label: 'EEG Only' },
            { value: 'video_eeg', label: 'Video EEG' },
            { value: 'image', label: 'Image Only (MRI/CT)' },
            { value: 'hybrid', label: 'Hybrid (EEG + Image)' }
          ].map(opt => (
            <label
              key={opt.value}
              className={`radio-option ${modality === opt.value ? 'active' : ''}`}
            >
              <input
                type="radio"
                value={opt.value}
                checked={modality === opt.value}
                onChange={(e) => setModality(e.target.value)}
              />
              <span className="radio-dot" />
              <span className="radio-label">{opt.label}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-section">
        <div className="sidebar-section-title">Emotiv Device / Channels</div>
        <div className="select-wrapper">
          <select
            className="select-input"
            value={channelConfig}
            onChange={(e) => setChannelConfig(parseInt(e.target.value))}
          >
            {CHANNEL_CONFIGS.map(c => (
              <option key={c.channels} value={c.channels}>
                {c.name} ({c.channels} ch)
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-section">
        <div className="sidebar-section-title">Analysis Options</div>
        <div className="checkbox-group">
          {[
            { key: 'asis', label: 'AS-IS Analysis' },
            { key: 'social', label: 'Social Analysis' },
            { key: 'tobeManual', label: 'To-Be Manual' },
            { key: 'tobeAuto', label: 'To-Be Automatic' },
            { key: 'statistical', label: 'Statistical' },
            { key: 'clinical', label: 'Clinical' }
          ].map(opt => (
            <label
              key={opt.key}
              className={`checkbox-option ${analysisOptions[opt.key] ? 'checked' : ''}`}
            >
              <input
                type="checkbox"
                checked={analysisOptions[opt.key]}
                onChange={(e) => setAnalysisOptions(prev => ({
                  ...prev,
                  [opt.key]: e.target.checked
                }))}
              />
              <span className="checkbox-box" />
              <span className="checkbox-label">{opt.label}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-section">
        <div className="sidebar-section-title">Model Settings</div>

        <div className="slider-wrapper">
          <div className="slider-header">
            <span className="slider-label">Training Samples</span>
            <span className="slider-value">{trainingSamples}</span>
          </div>
          <input
            type="range"
            className="slider-input"
            min={100}
            max={1000}
            step={50}
            value={trainingSamples}
            onChange={(e) => setTrainingSamples(parseInt(e.target.value))}
          />
        </div>

        <div className="slider-wrapper">
          <div className="slider-header">
            <span className="slider-label">Epochs</span>
            <span className="slider-value">{epochs}</span>
          </div>
          <input
            type="range"
            className="slider-input"
            min={10}
            max={100}
            step={5}
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value))}
          />
        </div>

        <div className="slider-wrapper">
          <div className="slider-header">
            <span className="slider-label">Target Accuracy</span>
            <span className="slider-value">{targetAccuracy}%</span>
          </div>
          <input
            type="range"
            className="slider-input"
            min={90}
            max={99.9}
            step={0.1}
            value={targetAccuracy}
            onChange={(e) => setTargetAccuracy(parseFloat(e.target.value))}
          />
        </div>
      </div>

      <button
        className="btn btn-primary btn-full mt-4"
        onClick={runClassification}
        disabled={isLoading}
      >
        {isLoading ? 'Processing...' : 'Run Classification'}
      </button>
    </aside>
  )

  // Render classification tab
  const renderClassificationTab = () => {
    const accuracy = classificationResult?.classification?.predictions?.accuracy || 0.95
    const confidence = classificationResult?.classification?.predictions?.confidence || 0.9

    return (
      <div>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Accuracy</div>
            <div className="metric-value">{(accuracy * 100).toFixed(1)}%</div>
            <div className={`metric-change ${accuracy >= targetAccuracy/100 ? 'positive' : 'negative'}`}>
              {accuracy >= targetAccuracy/100 ? 'Target reached' : `${((targetAccuracy/100 - accuracy) * 100).toFixed(1)}% below target`}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Confidence</div>
            <div className="metric-value">{(confidence * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Modality</div>
            <div className="metric-value" style={{ fontSize: '20px' }}>
              {modality === 'eeg' ? 'EEG' : modality === 'image' ? 'MRI/CT' : 'Hybrid'}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Channels</div>
            <div className="metric-value">{channelConfig}</div>
          </div>
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-title">Accuracy Gauge</div>
            <GaugeChart value={accuracy * 100} label="Detection Accuracy" />
          </div>
          <div className="chart-card">
            <div className="chart-title">Classification Probabilities</div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={[
                { name: 'Healthy', value: 0.15 + Math.random() * 0.1 },
                { name: selectedDisease, value: 0.7 + Math.random() * 0.25 }
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                />
                <Bar dataKey="value" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {modality !== 'image' && (
          <EEGDisplay channels={channelConfig} />
        )}

        {classificationResult && (
          <div className="alert alert-success">
            <span className="alert-icon">✓</span>
            <div className="alert-content">
              <div className="alert-title">Classification Complete</div>
              <div className="alert-message">
                Predicted: {selectedDisease} with {(confidence * 100).toFixed(1)}% confidence
                using {channelConfig}-channel {modality.toUpperCase()} data
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Render AS-IS tab
  const renderAsisTab = () => {
    const data = asisAnalysis?.report || {}
    const current = data.current_state || {}
    const severity = data.severity_distribution || {}

    const severityData = Object.entries(severity).map(([key, value]) => ({
      name: key.charAt(0).toUpperCase() + key.slice(1),
      value: value * 100
    }))

    return (
      <div>
        <div className="analysis-section">
          <div className="analysis-header">
            <div>
              <div className="analysis-title">AS-IS Analysis - {selectedDisease}</div>
              <div className="analysis-description">Current state analysis of disease detection</div>
            </div>
            <span className="card-badge badge-warning">Current State</span>
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Detection Accuracy</div>
              <div className="metric-value">{((current.detection_accuracy || 0.85) * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Prevalence Rate</div>
              <div className="metric-value">{((current.prevalence_rate || 0.05) * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Avg Diagnosis Time</div>
              <div className="metric-value">{current.avg_diagnosis_time_days || 180} days</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">False Positive Rate</div>
              <div className="metric-value">{((current.false_positive_rate || 0.1) * 100).toFixed(1)}%</div>
            </div>
          </div>

          <div className="charts-grid">
            <div className="chart-card">
              <div className="chart-title">Severity Distribution</div>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={severityData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                  >
                    {severityData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card">
              <div className="chart-title">Error Rates</div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={[
                  { name: 'False Positive', value: (current.false_positive_rate || 0.1) * 100 },
                  { name: 'False Negative', value: (current.false_negative_rate || 0.1) * 100 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="value" fill="#f44336" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="analysis-grid">
            <div className="analysis-item">
              <div className="analysis-item-title">Current Challenges</div>
              <ul className="analysis-list">
                {(data.challenges || []).map((challenge, i) => (
                  <li key={i}>{challenge}</li>
                ))}
              </ul>
            </div>
            <div className="analysis-item">
              <div className="analysis-item-title">Risk Factors</div>
              <ul className="analysis-list">
                <li>Age-related factors</li>
                <li>Genetic predisposition</li>
                <li>Environmental factors</li>
                <li>Lifestyle factors</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Render Social Analysis tab
  const renderSocialTab = () => {
    const data = socialAnalysis?.report || {}
    const social = data.social_impact || {}
    const phone = data.phone_activity || {}

    const radarData = [
      { subject: 'Withdrawal', A: social.social_withdrawal_score || 5 },
      { subject: 'Communication', A: social.communication_difficulty || 5 },
      { subject: 'Relationships', A: social.relationship_impact || 5 },
      { subject: 'Work Impact', A: social.work_impact || 5 },
      { subject: 'Daily Activities', A: (social.daily_activity_reduction || 30) / 10 }
    ]

    return (
      <div>
        <div className="analysis-section">
          <div className="analysis-header">
            <div>
              <div className="analysis-title">Social Analysis - {selectedDisease}</div>
              <div className="analysis-description">Social interaction and activity patterns</div>
            </div>
            <span className="card-badge badge-info">Social Impact</span>
          </div>

          <div className="charts-grid">
            <div className="chart-card">
              <div className="chart-title">Social Impact Radar</div>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#334155" />
                  <PolarAngleAxis dataKey="subject" stroke="#94a3b8" />
                  <PolarRadiusAxis angle={30} domain={[0, 10]} stroke="#94a3b8" />
                  <Radar
                    name="Impact"
                    dataKey="A"
                    stroke="#1e88e5"
                    fill="#1e88e5"
                    fillOpacity={0.3}
                  />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card">
              <div className="chart-title">Phone Activity Patterns</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                  { name: 'Calls/Day', value: phone.calls_per_day || 3 },
                  { name: 'Messages/Day', value: phone.messages_per_day || 15 },
                  { name: 'Social Apps (hrs)', value: phone.social_app_hours || 2 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="value" fill="#7c4dff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Social Withdrawal</div>
              <div className="metric-value">{(social.social_withdrawal_score || 5).toFixed(1)}/10</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Daily Activity Reduction</div>
              <div className="metric-value">{(social.daily_activity_reduction || 30).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Isolation Risk</div>
              <div className="metric-value">{((data.isolation_risk || 0.5) * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Work Impact</div>
              <div className="metric-value">{(social.work_impact || 5).toFixed(1)}/10</div>
            </div>
          </div>

          <div className="progress-container">
            <div className="progress-header">
              <span className="progress-label">Social Isolation Risk</span>
              <span className="progress-value">{((data.isolation_risk || 0.5) * 100).toFixed(1)}%</span>
            </div>
            <div className="progress-bar">
              <div
                className={`progress-fill ${(data.isolation_risk || 0.5) > 0.7 ? 'danger' : (data.isolation_risk || 0.5) > 0.4 ? 'warning' : 'success'}`}
                style={{ width: `${(data.isolation_risk || 0.5) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Render To-Be Analysis tab
  const renderTobeTab = () => {
    const data = tobeAnalysis?.report || {}
    const target = data.target_state || {}
    const benefits = data.projected_benefits || {}

    return (
      <div>
        <div className="analysis-section">
          <div className="analysis-header">
            <div>
              <div className="analysis-title">To-Be Analysis - {selectedDisease}</div>
              <div className="analysis-description">Target state and improvement recommendations</div>
            </div>
            <span className="card-badge badge-success">Target State</span>
          </div>

          <div className="tabs-container">
            <div className="tabs-header">
              <button className="tab-btn active">Automatic (AI)</button>
              <button className="tab-btn">Manual Configuration</button>
            </div>
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Target Accuracy</div>
              <div className="metric-value">{((target.target_accuracy || 0.999) * 100).toFixed(1)}%</div>
              <div className="metric-change positive">+{((target.target_accuracy || 0.999) * 100 - 85).toFixed(1)}% improvement</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Early Detection Improvement</div>
              <div className="metric-value">{((target.early_detection_improvement || 0.4) * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Diagnosis Time Reduction</div>
              <div className="metric-value">{((target.diagnosis_time_reduction || 0.5) * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Implementation Timeline</div>
              <div className="metric-value">{data.implementation_timeline || 18} months</div>
            </div>
          </div>

          <div className="charts-grid">
            <div className="chart-card">
              <div className="chart-title">AS-IS vs To-Be Comparison</div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={[
                  { name: 'Accuracy', current: 85, target: 99.9 },
                  { name: 'Early Detection', current: 65, target: 95 },
                  { name: 'Treatment Effect', current: 60, target: 85 }
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Bar dataKey="current" name="Current" fill="#f44336" />
                  <Bar dataKey="target" name="Target" fill="#4caf50" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card">
              <div className="chart-title">Projected Benefits</div>
              <div className="analysis-item" style={{ background: 'transparent' }}>
                <ul className="analysis-list">
                  <li>
                    <span>Patients Helped</span>
                    <strong>{(benefits.patients_helped || 2500).toLocaleString()}</strong>
                  </li>
                  <li>
                    <span>Cost Savings</span>
                    <strong>{(benefits.cost_savings_percent || 25).toFixed(1)}%</strong>
                  </li>
                  <li>
                    <span>Accuracy Improvement</span>
                    <strong>+14.9%</strong>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <div className="analysis-grid">
            <div className="analysis-item">
              <div className="analysis-item-title">AI Recommendations</div>
              <ul className="analysis-list">
                {(data.recommendations || [
                  'Multi-modal AI classification',
                  'Real-time EEG monitoring',
                  'Hybrid imaging analysis',
                  'Continuous biomarker tracking'
                ]).map((rec, i) => (
                  <li key={i}>{rec}</li>
                ))}
              </ul>
            </div>
            <div className="analysis-item">
              <div className="analysis-item-title">Implementation Priorities</div>
              <ul className="analysis-list">
                <li>Phase 1: Multi-modal data integration</li>
                <li>Phase 2: Real-time processing pipeline</li>
                <li>Phase 3: Clinical workflow integration</li>
                <li>Phase 4: Outcome validation studies</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Render Statistical Analysis tab
  const renderStatisticalTab = () => {
    const stats = statisticalData?.statistics || {}
    const cm = stats.confusion_matrix || {}

    const rocData = Array.from({ length: 20 }, (_, i) => ({
      fpr: i / 20,
      tpr: Math.min(1, i / 20 + 0.3 + Math.random() * 0.1)
    }))

    const trainingHistory = Array.from({ length: 30 }, (_, i) => ({
      epoch: i + 1,
      trainLoss: 0.5 * Math.exp(-i / 10) + Math.random() * 0.05,
      valLoss: 0.6 * Math.exp(-i / 12) + Math.random() * 0.08,
      trainAcc: 1 - 0.5 * Math.exp(-i / 8),
      valAcc: 1 - 0.6 * Math.exp(-i / 10)
    }))

    return (
      <div>
        <div className="analysis-section">
          <div className="analysis-header">
            <div>
              <div className="analysis-title">Statistical Analysis - {selectedDisease}</div>
              <div className="analysis-description">Classification metrics and performance</div>
            </div>
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Accuracy</div>
              <div className="metric-value">{((stats.accuracy || 0.95) * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Precision</div>
              <div className="metric-value">{((stats.precision || 0.94) * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Recall</div>
              <div className="metric-value">{((stats.recall || 0.93) * 100).toFixed(2)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">F1 Score</div>
              <div className="metric-value">{((stats.f1_score || 0.94) * 100).toFixed(2)}%</div>
            </div>
          </div>

          <div className="charts-grid">
            <div className="chart-card">
              <div className="chart-title">ROC Curve (AUC = {(stats.auc || 0.97).toFixed(3)})</div>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={rocData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="fpr" stroke="#94a3b8" label={{ value: 'FPR', position: 'bottom' }} />
                  <YAxis stroke="#94a3b8" label={{ value: 'TPR', angle: -90, position: 'left' }} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Area type="monotone" dataKey="tpr" stroke="#1e88e5" fill="rgba(30, 136, 229, 0.3)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card">
              <div className="chart-title">Confusion Matrix</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', padding: '20px' }}>
                <div style={{ background: 'rgba(76, 175, 80, 0.2)', padding: '20px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#4caf50' }}>{cm.tp || 90}</div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>True Positive</div>
                </div>
                <div style={{ background: 'rgba(244, 67, 54, 0.2)', padding: '20px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f44336' }}>{cm.fp || 5}</div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>False Positive</div>
                </div>
                <div style={{ background: 'rgba(244, 67, 54, 0.2)', padding: '20px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f44336' }}>{cm.fn || 8}</div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>False Negative</div>
                </div>
                <div style={{ background: 'rgba(76, 175, 80, 0.2)', padding: '20px', borderRadius: '8px', textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#4caf50' }}>{cm.tn || 87}</div>
                  <div style={{ fontSize: '12px', color: '#94a3b8' }}>True Negative</div>
                </div>
              </div>
            </div>
          </div>

          <div className="charts-grid">
            <div className="chart-card">
              <div className="chart-title">Training History - Loss</div>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="epoch" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Line type="monotone" dataKey="trainLoss" name="Train Loss" stroke="#1e88e5" dot={false} />
                  <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke="#f44336" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="chart-card">
              <div className="chart-title">Training History - Accuracy</div>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="epoch" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                  <Line type="monotone" dataKey="trainAcc" name="Train Acc" stroke="#4caf50" dot={false} />
                  <Line type="monotone" dataKey="valAcc" name="Val Acc" stroke="#ff9800" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Render Clinical Analysis tab
  const renderClinicalTab = () => {
    const data = clinicalData?.report || {}
    const diagnosis = data.diagnosis || {}
    const recommendations = data.recommendations || {}

    return (
      <div>
        <div className="analysis-section">
          <div className="analysis-header">
            <div>
              <div className="analysis-title">Clinical Analysis - {selectedDisease}</div>
              <div className="analysis-description">Clinical interpretation and recommendations</div>
            </div>
            <span className={`card-badge ${diagnosis.severity === 'Severe' ? 'badge-danger' : diagnosis.severity === 'Moderate' ? 'badge-warning' : 'badge-success'}`}>
              {diagnosis.severity || 'Moderate'}
            </span>
          </div>

          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-label">Primary Condition</div>
              <div className="metric-value" style={{ fontSize: '18px' }}>{diagnosis.primary_condition || selectedDisease}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Confidence</div>
              <div className="metric-value">{((diagnosis.confidence || 0.9) * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Severity</div>
              <div className="metric-value" style={{ fontSize: '18px' }}>{diagnosis.severity || 'Moderate'}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Biomarkers</div>
              <div className="metric-value" style={{ fontSize: '18px' }}>{(data.biomarkers || []).length}</div>
            </div>
          </div>

          <div className="analysis-grid">
            <div className="analysis-item">
              <div className="analysis-item-title">Diagnostic Recommendations</div>
              <ul className="analysis-list">
                {(recommendations.diagnostic || [
                  'Comprehensive neurological exam',
                  'Cognitive assessment battery'
                ]).map((rec, i) => (
                  <li key={i}>
                    <input type="checkbox" style={{ marginRight: '8px' }} />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
            <div className="analysis-item">
              <div className="analysis-item-title">Monitoring</div>
              <ul className="analysis-list">
                {(recommendations.monitoring || [
                  'Regular follow-ups',
                  'EEG monitoring'
                ]).map((rec, i) => (
                  <li key={i}>
                    <input type="checkbox" style={{ marginRight: '8px' }} />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
            <div className="analysis-item">
              <div className="analysis-item-title">Intervention</div>
              <ul className="analysis-list">
                {(recommendations.intervention || [
                  'Medication as indicated',
                  'Therapy programs'
                ]).map((rec, i) => (
                  <li key={i}>
                    <input type="checkbox" style={{ marginRight: '8px' }} />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
            <div className="analysis-item">
              <div className="analysis-item-title">Biomarkers</div>
              <ul className="analysis-list">
                {(data.biomarkers || ['EEG patterns', 'Brain imaging', 'Clinical assessment']).map((bio, i) => (
                  <li key={i}>{bio}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="mt-4">
            <button className="btn btn-primary">
              Generate Clinical Report
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Departments-only app: ALL former top-tab views folded into the department (first) menu.
  // Defined here (after the render* fns) so the 6 inline views are in scope.
  const extraDepartments = [
    { id: 'tool_classification', name: 'Classification', icon: '🎯', element: renderClassificationTab() },
    { id: 'tool_analysis', name: 'AI Analysis', icon: '🔬', element: <AnalysisUI /> },
    { id: 'tool_metrics', name: 'Metrics Dashboard', icon: '📈', element: <MetricsDashboard /> },
    { id: 'tool_asis', name: 'AS-IS Analysis', icon: '📋', element: renderAsisTab() },
    { id: 'tool_social', name: 'Social Analysis', icon: '🫂', element: renderSocialTab() },
    { id: 'tool_tobe', name: 'To-Be Analysis', icon: '🚀', element: renderTobeTab() },
    { id: 'tool_statistical', name: 'Statistical', icon: '📐', element: renderStatisticalTab() },
    { id: 'tool_clinical', name: 'Clinical', icon: '🩺', element: renderClinicalTab() },
    { id: 'tool_monitoring', name: 'RAG Monitoring', icon: '📡', element: <MonitoringDashboard /> },
    { id: 'tool_pipelines', name: 'Pipelines', icon: '🛠️', element: <PipelineManager /> },
    { id: 'tool_jobs', name: 'Jobs', icon: '⏱️', element: <JobScheduler /> },
    { id: 'tool_inference', name: 'Inference Testing', icon: '🧪', element: <InferenceDashboard /> },
    { id: 'tool_integrations', name: 'Integrations', icon: '🔌', element: <IntegrationHub /> },
    { id: 'tool_infographics', name: 'Infographics', icon: '📊', element: <InfographicsDashboard /> },
    { id: 'tool_agent_eval', name: 'Agent Evaluation', icon: '🔍', element: <AgentEvaluationDashboard /> },
  ]

  // Render active tab content
  const renderTabContent = () => {
    switch (activeTab) {
      case 'departments':
        return <DepartmentsDashboard selectedDisease={selectedDisease} extraDepartments={extraDepartments}
                 activeDept={activeDept} setActiveDept={setActiveDept} />
      case 'classification':
        return renderClassificationTab()
      case 'analysis':
        return <AnalysisUI />
      case 'metrics':
        return <MetricsDashboard />
      case 'asis':
        return renderAsisTab()
      case 'social':
        return renderSocialTab()
      case 'tobe':
        return renderTobeTab()
      case 'statistical':
        return renderStatisticalTab()
      case 'clinical':
        return renderClinicalTab()
      case 'monitoring':
        return <MonitoringDashboard />
      case 'pipelines':
        return <PipelineManager />
      case 'jobs':
        return <JobScheduler />
      case 'inference':
        return <InferenceDashboard />
      case 'integrations':
        return <IntegrationHub />
      case 'infographics':
        return <InfographicsDashboard />
      case 'entropy':
        return <EntropyDashboard />
      case 'topomap':
        return <TopomapDashboard />
      case 'expert':
        return <ExpertDashboard />
      case 'datacleaning':
        return <DataCleaningDashboard />
      case 'icalabel':
        return <ICLabelDashboard />
      case 'seizuretimeline':
        return <SeizureTimelineDashboard />
      case 'synchrosqueezing':
        return <SynchrosqueezingDashboard />
      case 'xai':
        return <XAIDashboard />
      case 'great-expectations':
        return <GreatExpectationsDashboard />
      case 'datasharing':
        return <DataSharingDashboard />
      case 'datagovernance':
        return <DataGovernanceDashboard />
      case 'torchmetrics':
        return <TorchMetricsDashboard />
      case 'deepchecks':
        return <DeepchecksDashboard />
      case 'aif360':
        return <AIF360Dashboard />
      case 'torcheeg':
        return <TorchEEGDashboard />
      case 'ilae-classification':
        return <ILAEClassificationDashboard />
      case 'annotation':
        return <AnnotationDashboard />
      case 'ai-cost':
        return <AICostDashboard />
      case 'token-cost':
        return <TokenCostDashboard />
      case 'shadow-ai':
        return <ShadowAIDashboard />
      case 'noise-cleaning':
        return <NoiseCleaningDashboard />
      case 'moca-autoscoring':
        return <MoCAAutoscoringDashboard />
      case 'inference-gpu':
        return <InferenceGPUDashboard />
      case 'spike-overlay':
        return <SpikeOverlayDashboard />
      case 'epilepsy-nurse':
        return <EpilepsyNurseDashboard />
      case 'pharmacist':
        return <PharmacistDashboard />
      case 'embedding-drift':
        return <EmbeddingDriftDashboard />
      case 'slp':
        return <SLPDashboard />
      case 'psychologist':
        return <PsychologistDashboard />
      case 'dietitian':
        return <DietitianDashboard />
      case 'social-worker':
        return <SocialWorkerDashboard />
      case 'medication':
        return <MedicationDashboard />
      case 'executive-scorecard':
        return <ExecutiveScorecardDashboard />
      case 'ai-usage':
        return <AIUsageDashboard />
      case 'therapy':
        return <TherapyDashboard />
      case 'notifications':
        return <NotificationDashboard />
      case 'alerts':
        return <AlertsDashboard />
      case 'tool-execution':
        return <ToolExecutionDashboard />
      case 'reports':
        return <ReportsDashboard />
      case 'database-ops':
        return <DatabaseOpsDashboard />
      case 'campaigns':
        return <CampaignsDashboard />
      case 'ai-risk':
        return <AIRiskDashboard />
      case 'chunking':
        return <ChunkingDashboard />
      case 'hallucination':
        return <HallucinationDashboard />
      case 'devops':
        return <DevOpsDashboard />
      case 'content-freshness':
        return <ContentFreshnessDashboard />
      case 'ai-compliance':
        return <AIComplianceDashboard />
      case 'ai-red-team':
        return <AIRedTeamDashboard />
      case 'knowledge-mgmt':
        return <KnowledgeManagementDashboard />
      case 'fine-tuning':
        return <FineTuningDashboard />
      case 'response-quality':
        return <ResponseQualityDashboard />
      case 'retrieval':
        return <RetrievalDashboard />
      case 'retrieval-eval':
        return <RetrievalEvalDashboard />
      case 'agent-loop':
        return <AgentLoopDashboard />
      case 'executive-ai':
        return <ExecutiveAIDashboard />
      case 'event-queue':
        return <EventQueueDashboard />
      case 'routing':
        return <RoutingDashboard />
      case 'citation':
        return <CitationDashboard />
      case 'agent-memory':
        return <AgentMemoryDashboard />
      case 'mcp-overview':
        return <MCPOverviewDashboard />
      case 'mcp-federation':
        return <MCPFederationDashboard />
      case 'release':
        return <ReleaseDashboard />
      case 'agent-eval':
        return <AgentEvaluationDashboard />
      case 'integration-dash':
        return <IntegrationDashboard />
      case 'responsible-ai':
        return <ResponsibleAIDashboard />
      case 'appointments':
        return <AppointmentsDashboard />
      case 'billing-claims':
        return <BillingClaimsDashboard />
      case 'finops':
        return <FinOpsDashboard />
      case 'cssrs':
        return <CSSRSDashboard />
      case 'phq9':
        return <PHQ9Dashboard />
      case 'ica-noise-cleaning':
        return <ICANoiseCleaningDashboard />
      case 'fim':
        return <FIMDashboard />
      case 'copm':
        return <COPMDashboard />
      case 'wais':
        return <WAISDashboard />
      case 'digit-span':
        return <DigitSpanDashboard />
      case 'amps':
        return <AMPSDashboard />
      case 'video-eeg':
        return <VideoEEGDashboard />
      case 'ncv':
        return <NCVDashboard />
      case 'blink-reflex':
        return <BlinkReflexDashboard />
      case 'ssep':
        return <SSEPDashboard />
      case 'vep':
        return <VEPDashboard />
      case 'emg':
        return <EMGDashboard />
      case 'rns':
        return <RNSDashboard />
      case 'bera':
        return <BeraDashboard />
      case 'ssr':
        return <SSRDashboard />
      case 'abpm-holter':
        return <ABPMHolterDashboard />
      case 'feature-evaluation':
        return <FeatureEvaluationDashboard />
      case 'feature-selection':
        return <FeatureSelectionDashboard />
      case 'automatic-pipelines':
        return <AutomaticPipelinesDashboard />
      case 'transfer-learning':
        return <TransferLearningDashboard />
      case 'cross-patient-benchmark':
        return <CrossPatientDashboard />
      case 'data-augmentation':
        return <DataAugmentationDashboard />
      case 'seizure-prediction':
        return <SeizurePredictionDashboard />
      case 'data-steward':
        return <DataStewardDashboard />
      case 'hybrid-pipeline':
        return <HybridPipelineDashboard />
      case 'connectivity':
        return <ConnectivityDashboard />
      case 'change-management':
        return <ChangeManagementDashboard />
      case 'scalogram':
        return <ScalogramDashboard />
      case 'saliency-attention':
        return <SaliencyAttentionDashboard />
      case 'guardrails':
        return <GuardrailsDashboard />
      case 'spwvd':
        return <SpwvdDashboard />
      case 'patient-facing-report':
        return <PatientFacingReportDashboard />
      case 'rlhf-training':
        return <RLHFTrainingDashboard />
      case 'federated-learning':
        return <FederatedLearningDashboard />
      case 'gnn-electrode-connectivity':
        return <GNNElectrodeConnectivityDashboard />
      case 'patient-education':
        return <PatientEducationDashboard />
      case 'audio-converter':
        return <AudioConverterDashboard />
      case 'pac':
        return <PACDashboard />
      case 'body-movement':
        return <BodyMovementDashboard />
      case 'video-converter':
        return <VideoConverterDashboard />
      case 'survey-link':
        return <SurveyLinkDashboard />
      case 'autonomic':
        return <AutonomicDashboard />
      case 'hrv':
        return <HRVDashboard />
      case 'pnes-differential':
        return <PNESDifferentialDashboard />
      case 'eeg-mri-concordance':
        return <EEGMRIConcordanceDashboard />
      case 'data-versioning':
        return <DataVersioningDashboard />
      case 'model-ops':
        return <ModelOpsDashboard />
      case 'mlops':
        return <MLOpsDashboard />
      case 'llmops':
        return <LLMOpsDashboard />
      case 'trust-ai':
        return <TrustAIDashboard />
      case 'ethical-ai':
        return <EthicalAIDashboard />
      case 'data-drift':
        return <DataDriftDashboard />
      case 'feature-drift':
        return <FeatureDriftDashboard />
      case 'model-drift':
        return <ModelDriftDashboard />
      case 'output-drift':
        return <OutputDriftDashboard />
      case 'prompt-drift':
        return <PromptDriftDashboard />
      case 'exercise':
        return <ExerciseDashboard />
      case 'knowledge-graph':
        return <KnowledgeGraphDashboard />
      case 'anomaly-detection':
        return <AnomalyDetectionDashboard />
      case 'causal-ai':
        return <CausalAIDashboard />
      case 'deep-learning':
        return <DeepLearningDashboard />
      case 'bias-detection':
        return <BiasDetectionDashboard />
      case 'digital-twin':
        return <DigitalTwinDashboard />
      case 'ai-observability':
        return <AIObservabilityDashboard />
      case 'model-monitoring':
        return <ModelMonitoringDashboard />
      case 'continuous-monitoring':
        return <ContinuousMonitoringDashboard />
      case 'ai-control-tower':
        return <AIControlTowerDashboard />
      case 'generative-ai':
        return <GenerativeAIDashboard />
      case 'human-evaluation':
        return <HumanEvaluationDashboard />
      case 'model-governance':
        return <ModelGovernanceDashboard />
      case 'continuous-learning':
        return <ContinuousLearningDashboard />
      case 'multimodal-ai':
        return <MultimodalAIDashboard />
      case 'drift-detection':
        return <DriftDetectionDashboard />
      case 'explainable-ai':
        return <ExplainableAIDashboard />
      case 'xai-groundtruth':
        return <XAIGroundTruthDashboard />
      case 'communication-ai':
        return <CommunicationAIDashboard />
      case 'foundation-models':
        return <FoundationModelsDashboard />
      case 'neonatal-eeg':
        return <NeonatalEEGDashboard />
      case 'analytics-ai':
        return <AnalyticsAIDashboard />
      case 'interpretable-ai':
        return <InterpretableAIDashboard />
      case 'agentic-rag':
        return <AgenticRAGDashboard />
      case 'grounding':
        return <GroundingDashboard />
      case 'visits':
        return <VisitsDashboard />
      case 'prescriptions':
        return <PrescriptionsDashboard />
      case 'adl':
        return <ADLDashboard />
      case 'clinical-tasks':
        return <ClinicalTasksDashboard />
      case 'patients-seen':
        return <PatientSeenDashboard />
      case 'patient-dashboard':
        return <PatientDashboardPanel />
      case 'data-lineage':
        return <DataLineageDashboard />
      case 'ai-security':
        return <AISecurityDashboard />
      case 'data-acquisition':
        return <DataAcquisitionDashboard />
      case 'data-privacy':
        return <DataPrivacyDashboard />
      case 'data-quality':
        return <DataQualityDashboard />
      case 'embedding':
        return <EmbeddingDashboard />
      case 'ai-lifecycle':
        return <AILifecycleDashboard />
      case 'mcp-governance':
        return <MCPGovernanceDashboard />
      case 'agent-council':
        return <AgentCouncilDashboard />
      case 'vector-db':
        return <VectorDBDashboard />
      case 'image-segmentation':
        return <ImageSegmentationDashboard />
      case 'object-detection':
        return <ObjectDetectionDashboard />
      case 'yolo-detection':
        return <YOLODetectionDashboard />
      case 'speech-ai':
        return <SpeechAIDashboard />
      case 'voice-ai':
        return <VoiceAIDashboard />
      case 'text-to-audio':
        return <TextToAudioDashboard />
      case 'text-to-video':
        return <TextToVideoDashboard />
      case 'cognitive-profile':
        return <CognitiveProfileDashboard />
      case 'time-series-ai':
        return <TimeSeriesAIDashboard />
      case 'medication-interaction':
        return <MedicationInteractionDashboard />
      case 'conversational-ai':
        return <ConversationalAIDashboard />
      case 'patient-reporting':
        return <PatientReportingDashboard />
      case 'research-coordinator':
        return <ResearchCoordinatorDashboard />
      case 'neurologist':
        return <NeurologistDashboard />
      case 'neurosurgeon':
        return <NeurosurgeonDashboard />
      case 'cnn-resnet':
        return <CNNResNetDashboard />
      case 'neurophysiologist':
        return <NeurophysiologistDashboard />
      case 'rnn-lstm':
        return <RNNLSTMDashboard />
      case 'neuropsychologist':
        return <NeuropsychologistDashboard />
      case 'radiologist':
        return <RadiologistDashboard />
      case 'psychiatrist':
        return <PsychiatristDashboard />
      case 'occupational-therapist':
        return <OccupationalTherapistDashboard />
      case 'eeg-technician':
        return <EEGTechnicianDashboard />
      case 'irb-ethics':
        return <IRBEthicsDashboard />
      case 'clinical-data-manager':
        return <ClinicalDataManagerDashboard />
      case 'patient-caregiver':
        return <PatientCaregiverDashboard />
      case 'clinical-psychologist':
        return <ClinicalPsychologistDashboard />
      case 'iot-engineer':
        return <IoTEngineerDashboard />
      case 'ai-federation':
        return <AIFederationDashboard />
      case 'is-sop':
        return <ISSopDashboard />
      case 'trigger-tracking':
        return <TriggerTrackingDashboard />
      case 'emergency-caregiver':
        return <EmergencyCaregiverDashboard />
      case 'caregiver-readiness':
        return <CaregiverReadinessDashboard />
      case 'medication-management':
        return <MedicationManagementDashboard />
      case 'pro-outcomes':
        return <PROOutcomesDashboard />
      case 'demographics':
        return <DemographicsDashboard />
      case 'wearables-digital-twin':
        return <WearablesDigitalTwinDashboard />
      case 'self-service':
        return <SelfServiceDashboard />
      case 'qa-test-suite':
        return <QATestSuiteDashboard />
      case 'product-manager':
        return <ProductManagerDashboard />
      case 'admin-panel':
        return <AdminDashboard />
      case 'functional-ba':
        return <FunctionalBADashboard />
      case 'integration-role':
        return <IntegrationRoleDashboard />
      case 'dataset-coverage':
        return <DatasetCoverageDashboard />
      case 'dark-factory':
        return <AIDarkFactoryDashboard />
      case 'seizure-risk-forecast':
        return <SeizureRiskForecastingDashboard />
      case 'cloud-ops':
        return <CloudOpsDashboard />
      case 'observability':
        return <ObservabilityDashboard />
      case 'seizure-severity':
        return <SeizureSeverityDashboard />
      case 'seizure-diary':
        return <SeizureDiaryDashboard />
      case 'abpm-holter':
        return <AbpmDashboard />
      case 'edge-deploy':
        return <EdgeDeployDashboard />
      case 'closed-loop':
        return <ClosedLoopDashboard />
      case 'band-heatmap':
        return <BandHeatmapDashboard />
      case 'device-telemetry':
        return <DeviceTelemetryDashboard />
      case 'telehealth':
        return <TelehealthDashboard />
      case 'workflow':
        return <WorkflowDashboard />
      case 'clinical-flowcharts':
        return <ClinicalFlowchartsDashboard />
      case 'functional-recovery':
        return <FunctionalRecoveryDashboard />
      case 'incident-management':
        return <IncidentManagementDashboard />
      case 'segmentation':
        return <SegmentationDashboard />
      case 'assessment-analytics':
        return <AssessmentDashboard />
      case 'epilepsy-board':
        return <EpilepsyBoardDashboard />
      case 'consent-management':
        return <ConsentManagementDashboard />
      case 'referral-triage':
        return <ReferralTriageDashboard />
      case 'rag-metadata-filter':
        return <RAGMetadataFilterDashboard />
      case 'recovery-trajectory':
        return <RecoveryTrajectoryDashboard />
      case 'artifact-detection':
        return <ArtifactDetectionDashboard />
      case 'cognitive-decline':
        return <CognitiveDeclineDashboard />
      case 'mri-review':
        return <MRIReviewDashboard />
      case 'goal-attainment':
        return <GoalAttainmentDashboard />
      case 'autonomic-analysis':
        return <AutonomicAnalysisDashboard />
      case 'guided-assessment':
        return <GuidedAssessmentDashboard />
      case 'model-retirement':
        return <ModelRetirementDashboard />
      case 'ai-roi':
        return <AIROIDashboard />
      case 'daily-care-plan':
        return <DailyCarePlanDashboard />
      case 'patient-report':
        return <PatientReportDashboard />
      case 'user-management':
        return <UserManagementDashboard />
      case 'benchmark-validation':
        return <BenchmarkValidationDashboard />
      case 'groups-teams':
        return <GroupsTeamsDashboard />
      case 'rehab-plan':
        return <RehabPlanDashboard />
      case 'medication-adherence':
        return <MedicationAdherenceDashboard />
      case 'multimodal-fusion':
        return <MultimodalFusionDashboard />
      case 'pnes-screening':
        return <PnesScreeningDashboard />
      case 'snn-neuromorphic':
        return <SnnNeuromorphicDashboard />
      case 'patient-portal':
        return <PatientPortalDashboard />
      case 'mcp-server':
        return <MCPServerDashboard />
      case 'neurolab-readiness':
        return <NeuroLabReadinessDashboard />
      case 'comorbidity-analysis':
        return <ComorbidityAnalysisDashboard />
      case 'sleep-staging':
        return <SleepStagingDashboard />
      case 'semiology-classifier':
        return <SemiologyClassifierDashboard />
      case 'ai-governance':
        return <AIGovernanceDashboard />
      case 'epworth':
        return <EpworthDashboard />
      case 'realtime-eeg-qc':
        return <RealtimeEEGQCDashboard />
      case 'patient-video':
        return <PatientVideoDashboard />
      case 'rag-report-gen':
        return <RAGReportGenDashboard />
      case 'subtle-seizure':
        return <SubtleSeizureDashboard />
      case 'api-resilience':
        return <APIResilienceDashboard />
      case 'otel-llm':
        return <OTelLLMDashboard />
      case 'mobile-alerts':
        return <MobileAlertsDashboard />
      case 'resource-monitor':
        return <ResourceMonitorDashboard />
      case 'config-drift':
        return <ConfigDriftDashboard />
      case 'alert-fatigue':
        return <AlertFatigueDashboard />
      case 'data-completeness':
        return <DataCompletenessDashboard />
      case 'treatment-efficacy':
        return <TreatmentEfficacyDashboard />
      case 'structured-reporting':
        return <StructuredReportingDashboard />
      case 'reinforcement-learning':
        return <ReinforcementLearningDashboard />
      case 'icd10-coding':
        return <ICD10CodingDashboard />
      case 'ai-incident':
        return <AIIncidentDashboard />
      case 'presurgical-evaluation':
        return <PreSurgicalEvaluationDashboard />
      case 'medication-refills':
        return <MedicationRefillDashboard />
      case 'secure-messaging':
        return <SecureMessagingDashboard />
      case 'consent-management':
        return <ConsentDashboard />
      case 'patient-documents':
        return <PatientDocumentsDashboard />
      case 'bmad':
        return <BmadDashboard />
      case 'cross-patient-benchmark':
        return <CrossPatientBenchmarkDashboard />
      case 'clinical-pharmacist':
        return <ClinicalPharmacistDashboard />
      case 'decision-ai':
        return <DecisionAiDashboard />
      case 'patients-seen':
        return <PatientsSeenDashboard />
      case 'data-ops':
        return <DataOpsDashboard />
      case 'population-health':
        return <PopulationHealthDashboard />
      case 'pharmacogenomics':
        return <PharmacogenomicsDashboard />
      case 'surgical-outcomes':
        return <SurgicalOutcomeDashboard />
      case 'dataset-requirements':
        return <DatasetRequirementsDashboard />
      default:
        return renderClassificationTab()
    }
  }

  return (
    <div className="app-container">
      {/* GLOBAL REAL-ERROR banner — fixed, shows server 5xx / network / JS errors */}
      {globalErrors.length > 0 && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999, background: '#7f1d1d', color: '#fff', padding: '8px 14px', boxShadow: '0 2px 8px rgba(0,0,0,0.3)', maxHeight: 180, overflow: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
            <strong style={{ fontSize: 13 }}>⚠ {globalErrors.length} error{globalErrors.length > 1 ? 's' : ''}</strong>
            <button onClick={() => setGlobalErrors([])} style={{ marginLeft: 'auto', fontSize: 11, padding: '2px 10px', borderRadius: 4, border: '1px solid #fca5a5', background: 'transparent', color: '#fff', cursor: 'pointer' }}>dismiss all</button>
          </div>
          {globalErrors.map((e, i) => (
            <div key={i} style={{ fontSize: 12, fontFamily: 'monospace', padding: '2px 0', borderTop: i ? '1px solid #991b1b' : 'none' }}>
              <span style={{ opacity: 0.7 }}>{e.t}</span> <strong>[{e.source}]</strong> {e.message}
            </div>
          ))}
        </div>
      )}
      {renderSidebar()}

      <main className="main-content">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 16px', marginBottom: 8,
          background: 'linear-gradient(90deg,#dbeafe,#ecfdf5)', border: '1px solid #e5e7eb', borderRadius: 8 }}>
          <span style={{ fontSize: 24 }}>🧠</span>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#0f172a', textTransform: 'capitalize' }}>
              {(DISEASES.find(d => d.id === selectedDisease)?.name) || selectedDisease}
            </div>
            <div style={{ fontSize: 12, color: '#475569' }}>
              Selected disease · mode: {modality === 'video_eeg' ? 'Video EEG' : modality === 'eeg' ? 'EEG Only' : modality}
            </div>
          </div>
        </div>

        {isLoading ? (
          <div className="loading-container">
            <div className="loading-spinner" />
            <div className="loading-text">Processing {selectedDisease} classification...</div>
          </div>
        ) : (
          renderTabContent()
        )}

        {error && (
          <div className="alert alert-danger">
            <span className="alert-icon">!</span>
            <div className="alert-content">
              <div className="alert-title">Error</div>
              <div className="alert-message">{error}</div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

// Gauge Chart Component
function GaugeChart({ value, label }) {
  return (
    <div className="gauge-container">
      <div className="gauge-circle" style={{ '--value': value }}>
        <span className="gauge-value">{value.toFixed(1)}%</span>
      </div>
      <div className="gauge-label">{label}</div>
    </div>
  )
}

// EEG Display Component
function EEGDisplay({ channels }) {
  const channelNames = channels <= 8
    ? ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2'].slice(0, channels)
    : Array.from({ length: Math.min(8, channels) }, (_, i) => `Ch${i + 1}`)

  return (
    <div className="eeg-display">
      <div className="eeg-header">
        <span className="eeg-title">EEG Signal Preview ({channels} channels)</span>
        <div className="eeg-status">
          <span className="eeg-status-dot" />
          <span>Streaming</span>
        </div>
      </div>
      <div className="eeg-channels">
        {channelNames.map((name, i) => (
          <div key={i} className="eeg-channel">
            <span className="eeg-channel-name">{name}</span>
            <div className="eeg-channel-line">
              <div
                className="eeg-waveform"
                style={{ animationDelay: `${i * 0.1}s` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default App
