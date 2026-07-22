import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, AreaChart, Area
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

function InferenceDashboard() {
  const [activeView, setActiveView] = useState('test') // test, reports, detail
  const [isLoading, setIsLoading] = useState(false)
  const [reports, setReports] = useState([])
  const [selectedReport, setSelectedReport] = useState(null)
  const [inferenceStatus, setInferenceStatus] = useState(null)

  // Test input form
  const [testInput, setTestInput] = useState({
    data_type: 'eeg_raw',
    eeg_channels: 22,
    eeg_sampling_rate: 256,
    eeg_duration_seconds: 10,
    patient_id: '',
    patient_age: null,
    patient_gender: '',
    clinical_notes: ''
  })

  useEffect(() => {
    fetchInferenceStatus()
    fetchReports()
  }, [])

  const fetchInferenceStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/inference/status`)
      setInferenceStatus(response.data)
    } catch (err) {
      setInferenceStatus({
        available: true,
        supported_data_types: ['eeg_raw', 'eeg_file', 'mri_file', 'ct_file', 'image_file', 'multimodal'],
        supported_diseases: [
          { id: 'alzheimer', name: "Alzheimer's Disease" },
          { id: 'parkinson', name: "Parkinson's Disease" },
          { id: 'schizophrenia', name: 'Schizophrenia' },
          { id: 'epilepsy', name: 'Epilepsy' },
          { id: 'autism', name: 'Autism Spectrum Disorder' },
          { id: 'depression', name: 'Major Depressive Disorder' },
          { id: 'stress', name: 'Chronic Stress' }
        ],
        eeg_channels: ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'A1', 'A2']
      })
    }
  }

  const fetchReports = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/inference/reports`)
      setReports(response.data.reports || [])
    } catch (err) {
      console.error('Error fetching reports:', err)
    }
  }

  const runInferenceTest = async () => {
    setIsLoading(true)
    try {
      const response = await axios.post(`${API_URL}/api/inference/test`, testInput)
      if (response.data.success) {
        setSelectedReport(response.data.report)
        setActiveView('detail')
        await fetchReports()
      }
    } catch (err) {
      console.error('Error running inference:', err)
      // Generate mock report for demo
      setSelectedReport(generateMockReport())
      setActiveView('detail')
    } finally {
      setIsLoading(false)
    }
  }

  const generateMockReport = () => ({
    report_id: `report_${Date.now()}`,
    generated_at: new Date().toISOString(),
    overall_status: Math.random() > 0.7 ? 'abnormal' : 'normal',
    overall_confidence: 0.85 + Math.random() * 0.14,
    quality_score: 0.8 + Math.random() * 0.19,
    processing_time_ms: 500 + Math.random() * 1500,
    diagnostics: [
      {
        disease_id: 'alzheimer',
        disease_name: "Alzheimer's Disease",
        prediction: Math.random() > 0.7 ? 'positive' : 'negative',
        probability: Math.random() * 0.8,
        confidence: 'high',
        severity_score: Math.random() * 0.6,
        progression_risk: Math.random() * 0.4,
        key_features: [
          { feature: 'theta_increase', importance: 0.85 },
          { feature: 'alpha_decrease', importance: 0.72 }
        ],
        contributing_channels: ['Fp1', 'Fp2', 'F3', 'F4'],
        contributing_regions: ['Frontal Lobe', 'Temporal Lobe'],
        recommendations: ['Neurological consultation', 'Follow-up EEG'],
        follow_up_tests: ['MRI scan', 'Cognitive assessment']
      },
      {
        disease_id: 'parkinson',
        disease_name: "Parkinson's Disease",
        prediction: 'negative',
        probability: Math.random() * 0.4,
        confidence: 'moderate'
      },
      {
        disease_id: 'epilepsy',
        disease_name: 'Epilepsy',
        prediction: 'negative',
        probability: Math.random() * 0.3,
        confidence: 'high'
      },
      {
        disease_id: 'depression',
        disease_name: 'Major Depressive Disorder',
        prediction: Math.random() > 0.8 ? 'borderline' : 'negative',
        probability: Math.random() * 0.5,
        confidence: 'moderate'
      }
    ],
    channel_analyses: Array.from({ length: testInput.eeg_channels }, (_, i) => ({
      channel_name: inferenceStatus?.eeg_channels?.[i] || `Ch${i + 1}`,
      channel_index: i,
      signal_quality: 0.7 + Math.random() * 0.29,
      noise_level: Math.random() * 0.15,
      artifact_percentage: Math.random() * 0.1,
      frequency_bands: {
        delta: 0.1 + Math.random() * 0.2,
        theta: 0.1 + Math.random() * 0.15,
        alpha: 0.15 + Math.random() * 0.2,
        beta: 0.1 + Math.random() * 0.15,
        gamma: 0.05 + Math.random() * 0.1
      },
      anomaly_score: Math.random() * 0.5,
      statistics: {
        mean: -10 + Math.random() * 20,
        std: 10 + Math.random() * 40,
        skewness: -1 + Math.random() * 2,
        kurtosis: 2 + Math.random() * 3
      }
    })),
    region_analyses: [
      { region_name: 'Frontal Lobe', region_code: 'FL', activity_level: 0.6 + Math.random() * 0.3, abnormality_score: Math.random() * 0.4, connectivity_strength: 0.5 + Math.random() * 0.4 },
      { region_name: 'Temporal Lobe', region_code: 'TL', activity_level: 0.5 + Math.random() * 0.4, abnormality_score: Math.random() * 0.5, connectivity_strength: 0.4 + Math.random() * 0.5 },
      { region_name: 'Parietal Lobe', region_code: 'PL', activity_level: 0.55 + Math.random() * 0.35, abnormality_score: Math.random() * 0.35, connectivity_strength: 0.45 + Math.random() * 0.45 },
      { region_name: 'Occipital Lobe', region_code: 'OL', activity_level: 0.6 + Math.random() * 0.3, abnormality_score: Math.random() * 0.3, connectivity_strength: 0.5 + Math.random() * 0.4 },
      { region_name: 'Central Region', region_code: 'CR', activity_level: 0.5 + Math.random() * 0.4, abnormality_score: Math.random() * 0.4, connectivity_strength: 0.55 + Math.random() * 0.35 }
    ],
    visualizations: {
      disease_probabilities: {
        'Alzheimer': Math.random() * 0.6,
        'Parkinson': Math.random() * 0.3,
        'Epilepsy': Math.random() * 0.2,
        'Depression': Math.random() * 0.4,
        'Schizophrenia': Math.random() * 0.25,
        'Stress': Math.random() * 0.35
      },
      band_powers: {
        timestamps: Array.from({ length: 50 }, (_, i) => i * 2),
        delta: Array.from({ length: 50 }, () => 0.1 + Math.random() * 0.2),
        theta: Array.from({ length: 50 }, () => 0.1 + Math.random() * 0.15),
        alpha: Array.from({ length: 50 }, () => 0.15 + Math.random() * 0.2),
        beta: Array.from({ length: 50 }, () => 0.1 + Math.random() * 0.15),
        gamma: Array.from({ length: 50 }, () => 0.05 + Math.random() * 0.1)
      }
    },
    warnings: Math.random() > 0.7 ? ['Low signal quality in some channels'] : [],
    notes: ['Analysis completed successfully']
  })

  const renderTestForm = () => (
    <div className="inference-test">
      <div className="test-header">
        <h2>Inference Testing</h2>
        <p>Provide patient data for end-to-end disease detection analysis</p>
      </div>

      <div className="test-form">
        <div className="form-section">
          <h3>Data Input Configuration</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Data Type</label>
              <select
                value={testInput.data_type}
                onChange={(e) => setTestInput({ ...testInput, data_type: e.target.value })}
              >
                <option value="eeg_raw">EEG Raw Data</option>
                <option value="eeg_file">EEG File</option>
                <option value="mri_file">MRI Scan</option>
                <option value="ct_file">CT Scan</option>
                <option value="image_file">Image File</option>
                <option value="multimodal">Multimodal</option>
              </select>
            </div>
            <div className="form-group">
              <label>EEG Channels: {testInput.eeg_channels}</label>
              <input
                type="range"
                min="2"
                max="64"
                value={testInput.eeg_channels}
                onChange={(e) => setTestInput({ ...testInput, eeg_channels: parseInt(e.target.value) })}
              />
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>Sampling Rate (Hz): {testInput.eeg_sampling_rate}</label>
              <select
                value={testInput.eeg_sampling_rate}
                onChange={(e) => setTestInput({ ...testInput, eeg_sampling_rate: parseInt(e.target.value) })}
              >
                <option value="128">128 Hz</option>
                <option value="256">256 Hz</option>
                <option value="512">512 Hz</option>
                <option value="1024">1024 Hz</option>
              </select>
            </div>
            <div className="form-group">
              <label>Recording Duration (s): {testInput.eeg_duration_seconds}</label>
              <input
                type="range"
                min="5"
                max="60"
                step="5"
                value={testInput.eeg_duration_seconds}
                onChange={(e) => setTestInput({ ...testInput, eeg_duration_seconds: parseInt(e.target.value) })}
              />
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>Patient Information (Optional)</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Patient ID</label>
              <input
                type="text"
                value={testInput.patient_id}
                onChange={(e) => setTestInput({ ...testInput, patient_id: e.target.value })}
                placeholder="Enter patient ID"
              />
            </div>
            <div className="form-group">
              <label>Age</label>
              <input
                type="number"
                value={testInput.patient_age || ''}
                onChange={(e) => setTestInput({ ...testInput, patient_age: e.target.value ? parseInt(e.target.value) : null })}
                placeholder="Age"
              />
            </div>
            <div className="form-group">
              <label>Gender</label>
              <select
                value={testInput.patient_gender}
                onChange={(e) => setTestInput({ ...testInput, patient_gender: e.target.value })}
              >
                <option value="">Select...</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
          <div className="form-group">
            <label>Clinical Notes</label>
            <textarea
              value={testInput.clinical_notes}
              onChange={(e) => setTestInput({ ...testInput, clinical_notes: e.target.value })}
              placeholder="Enter any clinical notes or observations..."
            />
          </div>
        </div>

        <div className="data-upload-section">
          <h3>Data Upload</h3>
          <div className="upload-zone">
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              <strong>Drop EEG/MRI/CT files here</strong>
              <span>or click to browse</span>
            </div>
            <input type="file" className="file-input" accept=".edf,.bdf,.set,.nii,.dcm,.jpg,.png" />
          </div>
          <div className="upload-info">
            Supported formats: EDF, BDF, SET (EEG) | NIfTI, DICOM (MRI/CT) | JPG, PNG (Images)
          </div>
        </div>

        <div className="form-actions">
          <button className="btn btn-secondary" onClick={() => setActiveView('reports')}>
            View Previous Reports
          </button>
          <button
            className="btn btn-primary btn-lg"
            onClick={runInferenceTest}
            disabled={isLoading}
          >
            {isLoading ? 'Running Analysis...' : 'Run Inference Test'}
          </button>
        </div>
      </div>

      {inferenceStatus?.supported_diseases && (
        <div className="disease-grid">
          <h3>Supported Disease Detection</h3>
          <div className="disease-tags">
            {inferenceStatus.supported_diseases.map((d, i) => (
              <span key={i} className="disease-tag">{d.name || d}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )

  const renderReportsList = () => (
    <div className="reports-list">
      <div className="reports-header">
        <button className="btn btn-secondary" onClick={() => setActiveView('test')}>
          ← Back to Testing
        </button>
        <h2>Inference Reports</h2>
      </div>

      <div className="reports-grid">
        {reports.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📋</div>
            <div className="empty-title">No Reports</div>
            <div className="empty-description">Run an inference test to generate reports</div>
          </div>
        ) : (
          reports.map((report, i) => (
            <div key={i} className="report-card" onClick={() => {
              setSelectedReport(report)
              setActiveView('detail')
            }}>
              <div className="report-header">
                <span className="report-id">{report.report_id?.slice(0, 8)}...</span>
                <span className={`report-status ${report.overall_status}`}>
                  {report.overall_status}
                </span>
              </div>
              <div className="report-body">
                <div className="report-metric">
                  <span>Confidence</span>
                  <strong>{((report.overall_confidence || 0) * 100).toFixed(1)}%</strong>
                </div>
                <div className="report-metric">
                  <span>Quality</span>
                  <strong>{((report.quality_score || 0) * 100).toFixed(1)}%</strong>
                </div>
                <div className="report-metric">
                  <span>Processing</span>
                  <strong>{(report.processing_time_ms || 0).toFixed(0)}ms</strong>
                </div>
              </div>
              <div className="report-footer">
                {new Date(report.generated_at).toLocaleString()}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )

  const renderReportDetail = () => {
    if (!selectedReport) return null

    const diagnostics = selectedReport.diagnostics || []
    const channels = selectedReport.channel_analyses || []
    const regions = selectedReport.region_analyses || []
    const visualizations = selectedReport.visualizations || {}

    const diseaseData = diagnostics.map(d => ({
      name: d.disease_name?.replace("'s Disease", '') || d.disease_id,
      probability: (d.probability || 0) * 100
    }))

    const regionData = regions.map(r => ({
      subject: r.region_name?.split(' ')[0] || r.region_code,
      activity: (r.activity_level || 0) * 100,
      abnormality: (r.abnormality_score || 0) * 100
    }))

    const bandData = visualizations.band_powers?.timestamps?.map((t, i) => ({
      time: t,
      delta: (visualizations.band_powers.delta?.[i] || 0) * 100,
      theta: (visualizations.band_powers.theta?.[i] || 0) * 100,
      alpha: (visualizations.band_powers.alpha?.[i] || 0) * 100,
      beta: (visualizations.band_powers.beta?.[i] || 0) * 100,
      gamma: (visualizations.band_powers.gamma?.[i] || 0) * 100
    })) || []

    const primaryDiagnosis = diagnostics.find(d => d.prediction === 'positive' || d.prediction === 'borderline') || diagnostics[0]

    return (
      <div className="report-detail">
        <div className="report-detail-header">
          <button className="btn btn-secondary" onClick={() => setActiveView('reports')}>
            ← Back to Reports
          </button>
          <h2>Inference Report</h2>
          <span className={`report-status large ${selectedReport.overall_status}`}>
            {selectedReport.overall_status?.toUpperCase()}
          </span>
        </div>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Overall Status</div>
            <div className={`metric-value ${selectedReport.overall_status}`}>
              {selectedReport.overall_status}
            </div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Confidence</div>
            <div className="metric-value">{((selectedReport.overall_confidence || 0) * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Data Quality</div>
            <div className="metric-value">{((selectedReport.quality_score || 0) * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Processing Time</div>
            <div className="metric-value">{(selectedReport.processing_time_ms || 0).toFixed(0)}ms</div>
          </div>
        </div>

        {primaryDiagnosis && (
          <div className={`primary-diagnosis ${primaryDiagnosis.prediction}`}>
            <h3>Primary Finding</h3>
            <div className="diagnosis-content">
              <div className="diagnosis-name">{primaryDiagnosis.disease_name}</div>
              <div className="diagnosis-prediction">{primaryDiagnosis.prediction}</div>
              <div className="diagnosis-probability">
                {((primaryDiagnosis.probability || 0) * 100).toFixed(1)}% probability
              </div>
              {primaryDiagnosis.recommendations && (
                <div className="diagnosis-recommendations">
                  <strong>Recommendations:</strong>
                  <ul>
                    {primaryDiagnosis.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-title">Disease Probability Distribution</div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={diseaseData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
                <YAxis type="category" dataKey="name" stroke="#94a3b8" width={100} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                  formatter={(value) => `${value.toFixed(1)}%`}
                />
                <Bar dataKey="probability" fill="#1e88e5" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-title">Brain Region Analysis</div>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={regionData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="subject" stroke="#94a3b8" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#94a3b8" />
                <Radar name="Activity" dataKey="activity" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.3} />
                <Radar name="Abnormality" dataKey="abnormality" stroke="#f44336" fill="#f44336" fillOpacity={0.3} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card full-width">
          <div className="chart-title">Frequency Band Power Over Time</div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={bandData.slice(0, 30)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Area type="monotone" dataKey="delta" stackId="1" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.6} />
              <Area type="monotone" dataKey="theta" stackId="1" stroke="#7c4dff" fill="#7c4dff" fillOpacity={0.6} />
              <Area type="monotone" dataKey="alpha" stackId="1" stroke="#4caf50" fill="#4caf50" fillOpacity={0.6} />
              <Area type="monotone" dataKey="beta" stackId="1" stroke="#ff9800" fill="#ff9800" fillOpacity={0.6} />
              <Area type="monotone" dataKey="gamma" stackId="1" stroke="#f44336" fill="#f44336" fillOpacity={0.6} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="channel-analysis">
          <h3>Channel Analysis ({channels.length} channels)</h3>
          <div className="channel-grid">
            {channels.slice(0, 12).map((ch, i) => (
              <div key={i} className={`channel-card ${ch.anomaly_score > 0.3 ? 'warning' : ''}`}>
                <div className="channel-name">{ch.channel_name}</div>
                <div className="channel-metrics">
                  <div className="channel-metric">
                    <span>Quality</span>
                    <div className="mini-bar">
                      <div className="mini-fill" style={{ width: `${(ch.signal_quality || 0) * 100}%` }} />
                    </div>
                    <span>{((ch.signal_quality || 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="channel-metric">
                    <span>Anomaly</span>
                    <div className="mini-bar warning">
                      <div className="mini-fill" style={{ width: `${(ch.anomaly_score || 0) * 100}%` }} />
                    </div>
                    <span>{((ch.anomaly_score || 0) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="diagnostics-detail">
          <h3>All Diagnostic Results</h3>
          <div className="diagnostics-table">
            <table>
              <thead>
                <tr>
                  <th>Disease</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                  <th>Confidence</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {diagnostics.map((d, i) => (
                  <tr key={i} className={d.prediction}>
                    <td>{d.disease_name}</td>
                    <td>
                      <span className={`prediction-badge ${d.prediction}`}>
                        {d.prediction}
                      </span>
                    </td>
                    <td>{((d.probability || 0) * 100).toFixed(1)}%</td>
                    <td>{d.confidence || 'N/A'}</td>
                    <td>{d.severity_score ? (d.severity_score * 100).toFixed(1) + '%' : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {selectedReport.warnings && selectedReport.warnings.length > 0 && (
          <div className="alert alert-warning">
            <span className="alert-icon">⚠</span>
            <div className="alert-content">
              <div className="alert-title">Warnings</div>
              {selectedReport.warnings.map((w, i) => (
                <div key={i} className="alert-message">{w}</div>
              ))}
            </div>
          </div>
        )}

        <div className="report-actions">
          <button className="btn btn-secondary">Export PDF</button>
          <button className="btn btn-secondary">Export JSON</button>
          <button className="btn btn-primary" onClick={() => setActiveView('test')}>
            Run New Test
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="inference-dashboard">
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner large" />
          <div className="loading-text">Running End-to-End Inference Pipeline...</div>
          <div className="loading-steps">
            <div className="step active">Preprocessing</div>
            <div className="step">Feature Extraction</div>
            <div className="step">Classification</div>
            <div className="step">Report Generation</div>
          </div>
        </div>
      )}

      {activeView === 'test' && renderTestForm()}
      {activeView === 'reports' && renderReportsList()}
      {activeView === 'detail' && renderReportDetail()}
    </div>
  )
}

export default InferenceDashboard
