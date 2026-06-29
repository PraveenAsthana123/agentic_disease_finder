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
    { id: 'synchrosqueezing', label: 'Synchrosqueezing' }
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
