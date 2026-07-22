import React, { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, AreaChart, Area, Legend, ComposedChart
} from 'recharts'

// Colors
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#9c27b0']

// ============================================================================
// MAIN ANALYSIS UI COMPONENT
// ============================================================================
function AnalysisUI() {
  const [activeSection, setActiveSection] = useState('performance')

  const sections = [
    { id: 'performance', label: 'Performance Matrix', icon: '📊' },
    { id: 'trust', label: 'Trust AI', icon: '🛡️' },
    { id: 'ethical', label: 'Ethical AI', icon: '⚖️' },
    { id: 'explainable', label: 'Explainable AI', icon: '🔍' },
    { id: 'responsible', label: 'Responsible AI', icon: '✅' },
    { id: 'governance', label: 'Governance AI', icon: '🏛️' }
  ]

  const renderSection = () => {
    switch (activeSection) {
      case 'performance':
        return <PerformanceMatrix />
      case 'trust':
        return <TrustAI />
      case 'ethical':
        return <EthicalAI />
      case 'explainable':
        return <ExplainableAI />
      case 'responsible':
        return <ResponsibleAI />
      case 'governance':
        return <GovernanceAI />
      default:
        return <PerformanceMatrix />
    }
  }

  return (
    <div className="analysis-ui">
      {/* Section Navigation */}
      <div className="analysis-nav">
        {sections.map(section => (
          <button
            key={section.id}
            className={`analysis-nav-btn ${activeSection === section.id ? 'active' : ''}`}
            onClick={() => setActiveSection(section.id)}
          >
            <span className="nav-icon">{section.icon}</span>
            <span className="nav-label">{section.label}</span>
          </button>
        ))}
      </div>

      {/* Section Content */}
      <div className="analysis-content">
        {renderSection()}
      </div>
    </div>
  )
}

// ============================================================================
// 1. PERFORMANCE MATRIX
// ============================================================================
function PerformanceMatrix() {
  // Disease Performance Data
  const diseasePerformance = [
    { disease: 'Parkinson', accuracy: 100.0, sensitivity: 100.0, specificity: 100.0, f1: 1.0, auc: 1.0, subjects: 50 },
    { disease: 'Autism', accuracy: 97.67, sensitivity: 97.0, specificity: 98.3, f1: 0.976, auc: 0.985, subjects: 300 },
    { disease: 'Schizophrenia', accuracy: 97.17, sensitivity: 96.5, specificity: 97.8, f1: 0.971, auc: 0.992, subjects: 84 },
    { disease: 'Epilepsy', accuracy: 94.22, sensitivity: 93.5, specificity: 94.9, f1: 0.941, auc: 0.968, subjects: 102 },
    { disease: 'Stress', accuracy: 94.17, sensitivity: 93.0, specificity: 95.3, f1: 0.940, auc: 0.962, subjects: 120 },
    { disease: 'Depression', accuracy: 91.07, sensitivity: 89.5, specificity: 92.6, f1: 0.908, auc: 0.945, subjects: 112 }
  ]

  // Sensitivity Analysis Data
  const sensitivityData = [
    { parameter: 'OCR Threshold', baseline: 0.85, low: 0.75, high: 0.95, impact: 'High' },
    { parameter: 'Confidence Cutoff', baseline: 0.90, low: 0.80, high: 0.95, impact: 'Medium' },
    { parameter: 'Text Clarity Weight', baseline: 0.28, low: 0.20, high: 0.35, impact: 'High' },
    { parameter: 'Drug Match Score', baseline: 0.24, low: 0.18, high: 0.30, impact: 'Medium' },
    { parameter: 'Batch Size', baseline: 32, low: 16, high: 64, impact: 'Low' }
  ]

  // Model Comparison Data
  const modelComparison = [
    { model: 'VotingClassifier', accuracy: 97.2, f1: 0.971, time: 245 },
    { model: 'ExtraTrees', accuracy: 96.5, f1: 0.964, time: 180 },
    { model: 'RandomForest', accuracy: 95.8, f1: 0.957, time: 165 },
    { model: 'XGBoost', accuracy: 95.2, f1: 0.951, time: 120 },
    { model: 'DNN+XGB', accuracy: 94.5, f1: 0.943, time: 350 },
    { model: 'SVM', accuracy: 89.3, f1: 0.891, time: 95 }
  ]

  // Cross-validation fold data
  const cvFoldData = [
    { fold: 'Fold 1', Parkinson: 100, Autism: 96.7, Schizophrenia: 97.6, Epilepsy: 94.8, Stress: 96.7, Depression: 93.3 },
    { fold: 'Fold 2', Parkinson: 100, Autism: 100, Schizophrenia: 96.4, Epilepsy: 93.2, Stress: 98.3, Depression: 90.3 },
    { fold: 'Fold 3', Parkinson: 100, Autism: 96.7, Schizophrenia: 97.8, Epilepsy: 95.1, Stress: 93.3, Depression: 90.2 },
    { fold: 'Fold 4', Parkinson: 100, Autism: 96.7, Schizophrenia: 97.1, Epilepsy: 94.0, Stress: 90.0, Depression: 90.2 },
    { fold: 'Fold 5', Parkinson: 100, Autism: 98.3, Schizophrenia: 97.0, Epilepsy: 94.0, Stress: 92.5, Depression: 91.3 }
  ]

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Performance Matrix</h2>
        <p>Comprehensive model performance metrics across all diseases</p>
      </div>

      {/* Summary Cards */}
      <div className="metrics-grid-4">
        <div className="metric-card highlight-green">
          <div className="metric-icon">🎯</div>
          <div className="metric-label">Average Accuracy</div>
          <div className="metric-value">95.72%</div>
          <div className="metric-change positive">All above 90% target</div>
        </div>
        <div className="metric-card highlight-blue">
          <div className="metric-icon">📈</div>
          <div className="metric-label">Best Performer</div>
          <div className="metric-value">Parkinson</div>
          <div className="metric-change positive">100% Accuracy</div>
        </div>
        <div className="metric-card highlight-purple">
          <div className="metric-icon">🔬</div>
          <div className="metric-label">Total Subjects</div>
          <div className="metric-value">768</div>
          <div className="metric-change">Across 6 diseases</div>
        </div>
        <div className="metric-card highlight-orange">
          <div className="metric-icon">⚡</div>
          <div className="metric-label">Avg Processing</div>
          <div className="metric-value">198ms</div>
          <div className="metric-change positive">Below 250ms target</div>
        </div>
      </div>

      {/* Disease Performance Table */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Disease Classification Performance</h3>
          <span className="badge badge-success">5-Fold Cross-Validation</span>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Disease</th>
                <th>Accuracy</th>
                <th>Sensitivity</th>
                <th>Specificity</th>
                <th>F1 Score</th>
                <th>AUC-ROC</th>
                <th>Subjects</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {diseasePerformance.map((row, idx) => (
                <tr key={idx}>
                  <td className="disease-name">{row.disease}</td>
                  <td>
                    <span className={`value ${row.accuracy >= 95 ? 'excellent' : row.accuracy >= 90 ? 'good' : 'warning'}`}>
                      {row.accuracy.toFixed(2)}%
                    </span>
                  </td>
                  <td>{row.sensitivity.toFixed(1)}%</td>
                  <td>{row.specificity.toFixed(1)}%</td>
                  <td>{row.f1.toFixed(3)}</td>
                  <td>{row.auc.toFixed(3)}</td>
                  <td>{row.subjects}</td>
                  <td>
                    <span className={`status-badge ${row.accuracy >= 95 ? 'status-excellent' : row.accuracy >= 90 ? 'status-good' : 'status-warning'}`}>
                      {row.accuracy >= 95 ? 'Excellent' : row.accuracy >= 90 ? 'Good' : 'Needs Review'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts Row */}
      <div className="charts-grid mt-4">
        <div className="chart-card">
          <div className="chart-title">Accuracy by Disease</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={diseasePerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="disease" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[80, 100]} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="accuracy" fill="#4caf50" radius={[4, 4, 0, 0]}>
                {diseasePerformance.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.accuracy >= 95 ? '#4caf50' : entry.accuracy >= 90 ? '#ff9800' : '#f44336'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">5-Fold Cross-Validation Results</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={cvFoldData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="fold" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[85, 100]} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Legend />
              <Line type="monotone" dataKey="Parkinson" stroke="#4caf50" strokeWidth={2} />
              <Line type="monotone" dataKey="Autism" stroke="#1e88e5" strokeWidth={2} />
              <Line type="monotone" dataKey="Schizophrenia" stroke="#7c4dff" strokeWidth={2} />
              <Line type="monotone" dataKey="Epilepsy" stroke="#ff9800" strokeWidth={2} />
              <Line type="monotone" dataKey="Stress" stroke="#00bcd4" strokeWidth={2} />
              <Line type="monotone" dataKey="Depression" stroke="#e91e63" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sensitivity Analysis */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Sensitivity Analysis</h3>
          <span className="badge badge-info">Parameter Impact Assessment</span>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Baseline</th>
                <th>Low</th>
                <th>High</th>
                <th>Impact</th>
                <th>Recommendation</th>
              </tr>
            </thead>
            <tbody>
              {sensitivityData.map((row, idx) => (
                <tr key={idx}>
                  <td className="param-name">{row.parameter}</td>
                  <td><code>{row.baseline}</code></td>
                  <td>{row.low}</td>
                  <td>{row.high}</td>
                  <td>
                    <span className={`impact-badge impact-${row.impact.toLowerCase()}`}>
                      {row.impact}
                    </span>
                  </td>
                  <td>
                    {row.impact === 'High' ? 'Monitor closely' : row.impact === 'Medium' ? 'Regular check' : 'Standard'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Comparison */}
      <div className="charts-grid mt-4">
        <div className="chart-card">
          <div className="chart-title">Model Comparison</div>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={modelComparison}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="model" stroke="#94a3b8" />
              <YAxis yAxisId="left" stroke="#94a3b8" domain={[85, 100]} />
              <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Legend />
              <Bar yAxisId="left" dataKey="accuracy" name="Accuracy %" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              <Line yAxisId="right" type="monotone" dataKey="time" name="Time (ms)" stroke="#ff9800" strokeWidth={2} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Performance Radar</div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={[
              { metric: 'Accuracy', value: 95.7 },
              { metric: 'Sensitivity', value: 94.9 },
              { metric: 'Specificity', value: 96.5 },
              { metric: 'F1 Score', value: 95.6 },
              { metric: 'AUC-ROC', value: 97.5 },
              { metric: 'Speed', value: 92.0 }
            ]}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="metric" stroke="#94a3b8" />
              <PolarRadiusAxis angle={30} domain={[80, 100]} stroke="#94a3b8" />
              <Radar name="Performance" dataKey="value" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.3} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// 2. TRUST AI
// ============================================================================
function TrustAI() {
  const trustMetrics = [
    { dimension: 'Reliability', score: 92, description: 'Consistent performance across conditions', status: 'Strong' },
    { dimension: 'Robustness', score: 88, description: 'Handles edge cases and noise', status: 'Strong' },
    { dimension: 'Consistency', score: 90, description: 'Reproducible results', status: 'Strong' },
    { dimension: 'Calibration', score: 85, description: 'Confidence matches accuracy', status: 'Good' },
    { dimension: 'Uncertainty', score: 82, description: 'Proper uncertainty quantification', status: 'Good' },
    { dimension: 'Validation', score: 95, description: 'Cross-validation compliance', status: 'Excellent' }
  ]

  const trustRadarData = trustMetrics.map(m => ({ metric: m.dimension, value: m.score }))

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Trust AI</h2>
        <p>Measuring system trustworthiness and reliability</p>
      </div>

      {/* Trust Score Overview */}
      <div className="trust-score-card">
        <div className="trust-score-circle">
          <svg viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" strokeWidth="8" />
            <circle
              cx="50" cy="50" r="45" fill="none"
              stroke="#4caf50" strokeWidth="8"
              strokeDasharray={`${88 * 2.83} ${283 - 88 * 2.83}`}
              strokeLinecap="round"
              transform="rotate(-90 50 50)"
            />
          </svg>
          <div className="trust-score-value">88</div>
          <div className="trust-score-label">Trust Score</div>
        </div>
        <div className="trust-score-details">
          <h3>Overall Trust Assessment: STRONG</h3>
          <p>The AI system demonstrates high reliability and consistent performance across all evaluation dimensions.</p>
          <div className="trust-badges">
            <span className="trust-badge trust-verified">Validated</span>
            <span className="trust-badge trust-certified">Certified</span>
            <span className="trust-badge trust-audited">Audited</span>
          </div>
        </div>
      </div>

      {/* Trust Dimensions */}
      <div className="charts-grid mt-4">
        <div className="chart-card">
          <div className="chart-title">Trust Dimensions Radar</div>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={trustRadarData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="metric" stroke="#94a3b8" />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#94a3b8" />
              <Radar name="Trust Score" dataKey="value" stroke="#4caf50" fill="#4caf50" fillOpacity={0.3} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Trust Metrics Breakdown</div>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={trustMetrics} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
              <YAxis dataKey="dimension" type="category" stroke="#94a3b8" width={100} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="score" fill="#1e88e5" radius={[0, 4, 4, 0]}>
                {trustMetrics.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.score >= 90 ? '#4caf50' : entry.score >= 80 ? '#1e88e5' : '#ff9800'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Trust Dimensions Table */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Trust Dimension Details</h3>
        </div>
        <div className="trust-grid">
          {trustMetrics.map((metric, idx) => (
            <div key={idx} className="trust-item">
              <div className="trust-item-header">
                <span className="trust-item-name">{metric.dimension}</span>
                <span className={`trust-item-score ${metric.score >= 90 ? 'excellent' : metric.score >= 80 ? 'good' : 'warning'}`}>
                  {metric.score}%
                </span>
              </div>
              <div className="trust-item-bar">
                <div
                  className="trust-item-fill"
                  style={{ width: `${metric.score}%`, background: metric.score >= 90 ? '#4caf50' : metric.score >= 80 ? '#1e88e5' : '#ff9800' }}
                />
              </div>
              <div className="trust-item-desc">{metric.description}</div>
              <span className={`status-badge status-${metric.status.toLowerCase()}`}>{metric.status}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Confidence Calibration */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Confidence Calibration</h3>
          <span className="badge badge-success">ECE: 0.032</span>
        </div>
        <div className="calibration-info">
          <div className="calibration-metric">
            <span className="calibration-label">Expected Calibration Error</span>
            <span className="calibration-value">0.032</span>
            <span className="calibration-status good">Well Calibrated</span>
          </div>
          <div className="calibration-metric">
            <span className="calibration-label">Maximum Calibration Error</span>
            <span className="calibration-value">0.078</span>
            <span className="calibration-status good">Acceptable</span>
          </div>
          <div className="calibration-metric">
            <span className="calibration-label">Brier Score</span>
            <span className="calibration-value">0.045</span>
            <span className="calibration-status excellent">Excellent</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// 3. ETHICAL AI
// ============================================================================
function EthicalAI() {
  const ethicalPrinciples = [
    { principle: 'Beneficence', score: 95, description: 'AI should benefit patients and healthcare providers', icon: '❤️' },
    { principle: 'Non-maleficence', score: 92, description: 'AI should not cause harm; errors are minimized', icon: '🛡️' },
    { principle: 'Autonomy', score: 88, description: 'Users maintain control over AI recommendations', icon: '🎯' },
    { principle: 'Justice', score: 85, description: 'AI treats all users fairly without discrimination', icon: '⚖️' },
    { principle: 'Transparency', score: 90, description: 'AI decisions are explainable and auditable', icon: '🔍' }
  ]

  const fairnessMetrics = [
    { metric: 'Demographic Parity', value: 0.96, threshold: 0.80, status: 'Pass' },
    { metric: 'Equalized Odds', value: 0.94, threshold: 0.80, status: 'Pass' },
    { metric: 'Predictive Parity', value: 0.92, threshold: 0.80, status: 'Pass' },
    { metric: 'Treatment Equality', value: 0.89, threshold: 0.80, status: 'Pass' },
    { metric: 'Calibration Across Groups', value: 0.91, threshold: 0.80, status: 'Pass' }
  ]

  const biasAuditData = [
    { category: 'Age Groups', disparity: 2.3, status: 'Low' },
    { category: 'Gender', disparity: 1.8, status: 'Low' },
    { category: 'Ethnicity', disparity: 3.1, status: 'Low' },
    { category: 'Socioeconomic', disparity: 4.2, status: 'Acceptable' },
    { category: 'Geographic', disparity: 2.9, status: 'Low' }
  ]

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Ethical AI</h2>
        <p>Ensuring AI systems adhere to ethical principles and fairness standards</p>
      </div>

      {/* Ethics Score */}
      <div className="ethics-overview">
        <div className="ethics-score-card">
          <div className="ethics-score">90</div>
          <div className="ethics-label">Ethics Score</div>
          <div className="ethics-status">Compliant</div>
        </div>
        <div className="ethics-summary">
          <h3>Ethical Compliance Summary</h3>
          <ul className="ethics-list">
            <li><span className="check">✓</span> All 5 ethical principles met</li>
            <li><span className="check">✓</span> Fairness metrics within acceptable bounds</li>
            <li><span className="check">✓</span> Bias audit completed - Low risk</li>
            <li><span className="check">✓</span> Human oversight protocols in place</li>
            <li><span className="check">✓</span> Regular ethics review scheduled</li>
          </ul>
        </div>
      </div>

      {/* Ethical Principles */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>AI Ethics Principles</h3>
        </div>
        <div className="principles-grid">
          {ethicalPrinciples.map((p, idx) => (
            <div key={idx} className="principle-card">
              <div className="principle-icon">{p.icon}</div>
              <div className="principle-name">{p.principle}</div>
              <div className="principle-score">{p.score}%</div>
              <div className="principle-bar">
                <div className="principle-fill" style={{ width: `${p.score}%` }} />
              </div>
              <div className="principle-desc">{p.description}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Fairness Metrics */}
      <div className="charts-grid mt-4">
        <div className="chart-card">
          <div className="chart-title">Fairness Metrics</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={fairnessMetrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="metric" stroke="#94a3b8" angle={-45} textAnchor="end" height={100} />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="value" fill="#4caf50" radius={[4, 4, 0, 0]} />
              <Bar dataKey="threshold" fill="#334155" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Bias Audit Results</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={biasAuditData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 10]} stroke="#94a3b8" />
              <YAxis dataKey="category" type="category" stroke="#94a3b8" width={100} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="disparity" name="Disparity %" radius={[0, 4, 4, 0]}>
                {biasAuditData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.disparity <= 3 ? '#4caf50' : entry.disparity <= 5 ? '#ff9800' : '#f44336'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Fairness Table */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Fairness Assessment Details</h3>
          <span className="badge badge-success">All Metrics Passing</span>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Threshold</th>
                <th>Gap</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {fairnessMetrics.map((row, idx) => (
                <tr key={idx}>
                  <td>{row.metric}</td>
                  <td><strong>{row.value.toFixed(2)}</strong></td>
                  <td>{row.threshold.toFixed(2)}</td>
                  <td className="positive">+{(row.value - row.threshold).toFixed(2)}</td>
                  <td><span className="status-badge status-excellent">{row.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// 4. EXPLAINABLE AI
// ============================================================================
function ExplainableAI() {
  const featureImportance = [
    { feature: 'Beta Power', importance: 0.28, category: 'EEG' },
    { feature: 'Alpha Asymmetry', importance: 0.22, category: 'EEG' },
    { feature: 'Gamma Power', importance: 0.18, category: 'EEG' },
    { feature: 'Theta/Beta Ratio', importance: 0.12, category: 'EEG' },
    { feature: 'Delta Power', importance: 0.08, category: 'EEG' },
    { feature: 'Coherence', importance: 0.07, category: 'Connectivity' },
    { feature: 'Hjorth Mobility', importance: 0.05, category: 'Statistical' }
  ]

  const shapValues = [
    { feature: 'Beta Power (Frontal)', value: 0.45, direction: 'positive' },
    { feature: 'Alpha Suppression', value: 0.32, direction: 'positive' },
    { feature: 'Gamma Elevation', value: 0.28, direction: 'positive' },
    { feature: 'Delta Increase', value: -0.15, direction: 'negative' },
    { feature: 'Theta Power', value: -0.08, direction: 'negative' }
  ]

  const interpretabilityScores = [
    { method: 'SHAP Analysis', score: 92, coverage: '100%' },
    { method: 'LIME Explanations', score: 88, coverage: '95%' },
    { method: 'Feature Attribution', score: 95, coverage: '100%' },
    { method: 'Attention Maps', score: 85, coverage: '80%' },
    { method: 'Counterfactuals', score: 78, coverage: '70%' }
  ]

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Explainable AI</h2>
        <p>Understanding how the AI makes decisions and predictions</p>
      </div>

      {/* XAI Score */}
      <div className="metrics-grid-4">
        <div className="metric-card highlight-blue">
          <div className="metric-icon">🔍</div>
          <div className="metric-label">Explainability Score</div>
          <div className="metric-value">90</div>
          <div className="metric-change positive">Strong</div>
        </div>
        <div className="metric-card highlight-green">
          <div className="metric-icon">📊</div>
          <div className="metric-label">Feature Coverage</div>
          <div className="metric-value">100%</div>
          <div className="metric-change">All features explained</div>
        </div>
        <div className="metric-card highlight-purple">
          <div className="metric-icon">🎯</div>
          <div className="metric-label">Interpretation Methods</div>
          <div className="metric-value">5</div>
          <div className="metric-change">SHAP, LIME, etc.</div>
        </div>
        <div className="metric-card highlight-orange">
          <div className="metric-icon">📝</div>
          <div className="metric-label">Documentation</div>
          <div className="metric-value">95%</div>
          <div className="metric-change positive">Complete</div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="charts-grid mt-4">
        <div className="chart-card">
          <div className="chart-title">Feature Importance (Global)</div>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={featureImportance} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[0, 0.3]} stroke="#94a3b8" />
              <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={120} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="importance" fill="#1e88e5" radius={[0, 4, 4, 0]}>
                {featureImportance.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">SHAP Values (Sample Prediction)</div>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={shapValues} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" domain={[-0.3, 0.5]} stroke="#94a3b8" />
              <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={150} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {shapValues.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.value >= 0 ? '#4caf50' : '#f44336'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Interpretation Methods */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Interpretation Methods</h3>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Score</th>
                <th>Coverage</th>
                <th>Status</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {interpretabilityScores.map((row, idx) => (
                <tr key={idx}>
                  <td><strong>{row.method}</strong></td>
                  <td>
                    <div className="score-bar-container">
                      <div className="score-bar" style={{ width: `${row.score}%`, background: row.score >= 90 ? '#4caf50' : row.score >= 80 ? '#1e88e5' : '#ff9800' }} />
                      <span>{row.score}%</span>
                    </div>
                  </td>
                  <td>{row.coverage}</td>
                  <td><span className={`status-badge ${row.score >= 90 ? 'status-excellent' : 'status-good'}`}>Active</span></td>
                  <td className="text-muted">
                    {row.method === 'SHAP Analysis' && 'Game-theoretic feature attribution'}
                    {row.method === 'LIME Explanations' && 'Local interpretable explanations'}
                    {row.method === 'Feature Attribution' && 'Direct feature contribution analysis'}
                    {row.method === 'Attention Maps' && 'Neural attention visualization'}
                    {row.method === 'Counterfactuals' && 'What-if scenario analysis'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Decision Path Example */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Sample Decision Path</h3>
          <span className="badge badge-info">Depression Detection</span>
        </div>
        <div className="decision-path">
          <div className="decision-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <div className="step-title">EEG Signal Input</div>
              <div className="step-desc">22-channel EEG data collected at 256Hz</div>
            </div>
          </div>
          <div className="decision-arrow">→</div>
          <div className="decision-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <div className="step-title">Feature Extraction</div>
              <div className="step-desc">570 features extracted (power, connectivity, Hjorth)</div>
            </div>
          </div>
          <div className="decision-arrow">→</div>
          <div className="decision-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <div className="step-title">Key Features Identified</div>
              <div className="step-desc">Beta Power ↑, Alpha Asymmetry detected</div>
            </div>
          </div>
          <div className="decision-arrow">→</div>
          <div className="decision-step highlight">
            <div className="step-number">4</div>
            <div className="step-content">
              <div className="step-title">Prediction: Depression</div>
              <div className="step-desc">Confidence: 91.07% | F1: 0.908</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// 5. RESPONSIBLE AI
// ============================================================================
function ResponsibleAI() {
  const responsibleMetrics = [
    { category: 'Reliability', score: 92, status: 'Strong', icon: '🔒' },
    { category: 'Trustworthiness', score: 88, status: 'Strong', icon: '🛡️' },
    { category: 'Safety', score: 85, status: 'Good', icon: '⚠️' },
    { category: 'Fairness', score: 80, status: 'Good', icon: '⚖️' },
    { category: 'Explainability', score: 90, status: 'Strong', icon: '🔍' },
    { category: 'Interpretability', score: 85, status: 'Good', icon: '📊' },
    { category: 'Auditability', score: 95, status: 'Excellent', icon: '📋' },
    { category: 'Compliance', score: 75, status: 'Developing', icon: '📜' },
    { category: 'Human-in-Loop', score: 90, status: 'Strong', icon: '👤' },
    { category: 'Governance', score: 88, status: 'Strong', icon: '🏛️' }
  ]

  const slaTargets = [
    { metric: 'Accuracy SLO', target: '≥90%', current: '95.72%', status: 'Met', variance: '+5.72%' },
    { metric: 'Sensitivity SLO', target: '≥89%', current: '94.9%', status: 'Met', variance: '+5.9%' },
    { metric: 'Specificity SLO', target: '≥92%', current: '96.5%', status: 'Met', variance: '+4.5%' },
    { metric: 'Inference Latency', target: '<1s', current: '198ms', status: 'Met', variance: '-802ms' },
    { metric: 'Availability', target: '99.5%', current: '99.8%', status: 'Met', variance: '+0.3%' }
  ]

  const riskAssessment = [
    { risk: 'Model Drift', level: 'Low', mitigation: 'Continuous monitoring + retraining pipeline', status: 'Controlled' },
    { risk: 'Data Quality', level: 'Low', mitigation: 'Automated validation + outlier detection', status: 'Controlled' },
    { risk: 'Adversarial Attack', level: 'Medium', mitigation: 'Input validation + robustness testing', status: 'Monitored' },
    { risk: 'Privacy Breach', level: 'Low', mitigation: 'Data anonymization + access controls', status: 'Controlled' },
    { risk: 'Bias Amplification', level: 'Low', mitigation: 'Regular bias audits + fairness constraints', status: 'Controlled' }
  ]

  const overallScore = Math.round(responsibleMetrics.reduce((sum, m) => sum + m.score, 0) / responsibleMetrics.length)

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Responsible AI</h2>
        <p>Comprehensive responsible AI assessment and monitoring</p>
      </div>

      {/* Overall Score */}
      <div className="responsible-overview">
        <div className="responsible-score-card">
          <div className="responsible-gauge">
            <svg viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="54" fill="none" stroke="#334155" strokeWidth="10" />
              <circle
                cx="60" cy="60" r="54" fill="none"
                stroke={overallScore >= 85 ? '#4caf50' : overallScore >= 70 ? '#ff9800' : '#f44336'}
                strokeWidth="10"
                strokeDasharray={`${overallScore * 3.39} ${339 - overallScore * 3.39}`}
                strokeLinecap="round"
                transform="rotate(-90 60 60)"
              />
            </svg>
            <div className="responsible-score-text">
              <div className="score-value">{overallScore}</div>
              <div className="score-label">Overall</div>
            </div>
          </div>
          <div className="responsible-status">
            <span className={`status-indicator ${overallScore >= 85 ? 'good' : 'warning'}`}>
              {overallScore >= 85 ? 'GOOD' : 'DEVELOPING'}
            </span>
          </div>
        </div>

        <div className="responsible-summary">
          <h3>Responsible AI Assessment</h3>
          <div className="summary-grid">
            <div className="summary-item">
              <span className="summary-label">Dimensions Evaluated</span>
              <span className="summary-value">10</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Strong Performance</span>
              <span className="summary-value">5</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Good Performance</span>
              <span className="summary-value">4</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Needs Improvement</span>
              <span className="summary-value">1</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="responsible-metrics-grid mt-4">
        {responsibleMetrics.map((metric, idx) => (
          <div key={idx} className={`responsible-metric-card ${metric.status.toLowerCase()}`}>
            <div className="metric-header">
              <span className="metric-icon">{metric.icon}</span>
              <span className="metric-name">{metric.category}</span>
            </div>
            <div className="metric-score">{metric.score}</div>
            <div className="metric-bar">
              <div className="metric-fill" style={{ width: `${metric.score}%` }} />
            </div>
            <div className={`metric-status status-${metric.status.toLowerCase()}`}>{metric.status}</div>
          </div>
        ))}
      </div>

      {/* SLA Targets */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>SLA Targets</h3>
          <span className="badge badge-success">All Met</span>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Target</th>
                <th>Current</th>
                <th>Variance</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {slaTargets.map((row, idx) => (
                <tr key={idx}>
                  <td><strong>{row.metric}</strong></td>
                  <td><code>{row.target}</code></td>
                  <td className="highlight">{row.current}</td>
                  <td className="positive">{row.variance}</td>
                  <td><span className="status-badge status-excellent">{row.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Risk Assessment */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Risk Assessment</h3>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Risk</th>
                <th>Level</th>
                <th>Mitigation Strategy</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {riskAssessment.map((row, idx) => (
                <tr key={idx}>
                  <td><strong>{row.risk}</strong></td>
                  <td>
                    <span className={`risk-badge risk-${row.level.toLowerCase()}`}>{row.level}</span>
                  </td>
                  <td>{row.mitigation}</td>
                  <td><span className="status-badge status-good">{row.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// 6. GOVERNANCE AI
// ============================================================================
function GovernanceAI() {
  const governanceFramework = [
    { dimension: 'Model Lifecycle', maturity: 4, target: 5, status: 'Advanced' },
    { dimension: 'Data Governance', maturity: 4, target: 5, status: 'Advanced' },
    { dimension: 'Risk Management', maturity: 3, target: 5, status: 'Established' },
    { dimension: 'Compliance', maturity: 3, target: 5, status: 'Established' },
    { dimension: 'Monitoring', maturity: 5, target: 5, status: 'Optimized' },
    { dimension: 'Documentation', maturity: 4, target: 5, status: 'Advanced' }
  ]

  const modelRegistry = [
    { model: 'VotingClassifier-v2.4.1', status: 'Production', deployed: '2025-12-15', accuracy: '95.72%', owner: 'ML Team' },
    { model: 'DNN-XGB-Ensemble-v1.2', status: 'Production', deployed: '2025-12-10', accuracy: '91.07%', owner: 'DL Team' },
    { model: 'ExtraTrees-v3.0', status: 'Staging', deployed: '2025-12-20', accuracy: '96.5%', owner: 'ML Team' },
    { model: 'CNN-LSTM-v1.0', status: 'Development', deployed: '-', accuracy: '88.2%', owner: 'Research' }
  ]

  const auditLog = [
    { date: '2026-01-14', action: 'Model Performance Review', user: 'System', status: 'Completed' },
    { date: '2026-01-13', action: 'Bias Audit Executed', user: 'AI Ethics Team', status: 'Passed' },
    { date: '2026-01-12', action: 'Data Quality Check', user: 'Data Team', status: 'Passed' },
    { date: '2026-01-10', action: 'Security Scan', user: 'Security Team', status: 'Passed' },
    { date: '2026-01-08', action: 'Model Retraining', user: 'ML Ops', status: 'Completed' }
  ]

  const humanOversight = {
    reviewRequired: true,
    escalationRate: 3.2,
    overrideRate: 1.8,
    avgReviewTime: '45 seconds'
  }

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Governance AI</h2>
        <p>AI governance framework, model registry, and audit compliance</p>
      </div>

      {/* Governance Overview */}
      <div className="governance-overview">
        <div className="governance-score">
          <div className="gov-score-value">88</div>
          <div className="gov-score-label">Governance Score</div>
          <span className="gov-status strong">Strong</span>
        </div>
        <div className="governance-info">
          <div className="gov-info-item">
            <span className="gov-info-icon">📋</span>
            <div>
              <div className="gov-info-label">Model Version</div>
              <div className="gov-info-value">v2.4.1</div>
            </div>
          </div>
          <div className="gov-info-item">
            <span className="gov-info-icon">📅</span>
            <div>
              <div className="gov-info-label">Last Training</div>
              <div className="gov-info-value">2025-12-15</div>
            </div>
          </div>
          <div className="gov-info-item">
            <span className="gov-info-icon">📊</span>
            <div>
              <div className="gov-info-label">Dataset Size</div>
              <div className="gov-info-value">125,000</div>
            </div>
          </div>
          <div className="gov-info-item">
            <span className="gov-info-icon">✅</span>
            <div>
              <div className="gov-info-label">Ethics Review</div>
              <div className="gov-info-value">Approved</div>
            </div>
          </div>
        </div>
      </div>

      {/* Governance Maturity */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Governance Maturity Assessment</h3>
          <span className="badge badge-info">CMMI Level 4</span>
        </div>
        <div className="maturity-grid">
          {governanceFramework.map((item, idx) => (
            <div key={idx} className="maturity-item">
              <div className="maturity-header">
                <span className="maturity-name">{item.dimension}</span>
                <span className={`maturity-status status-${item.status.toLowerCase()}`}>{item.status}</span>
              </div>
              <div className="maturity-levels">
                {[1, 2, 3, 4, 5].map(level => (
                  <div
                    key={level}
                    className={`maturity-level ${level <= item.maturity ? 'filled' : ''} ${level === item.target ? 'target' : ''}`}
                  >
                    {level}
                  </div>
                ))}
              </div>
              <div className="maturity-progress">
                <span>Current: {item.maturity}/5</span>
                <span>Target: {item.target}/5</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Registry */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Model Registry</h3>
          <span className="badge badge-success">4 Models</span>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Status</th>
                <th>Deployed</th>
                <th>Accuracy</th>
                <th>Owner</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {modelRegistry.map((row, idx) => (
                <tr key={idx}>
                  <td><code>{row.model}</code></td>
                  <td>
                    <span className={`status-badge ${row.status === 'Production' ? 'status-excellent' : row.status === 'Staging' ? 'status-good' : 'status-warning'}`}>
                      {row.status}
                    </span>
                  </td>
                  <td>{row.deployed}</td>
                  <td><strong>{row.accuracy}</strong></td>
                  <td>{row.owner}</td>
                  <td>
                    <button className="btn-small">View</button>
                    <button className="btn-small">Audit</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Human Oversight */}
      <div className="charts-grid mt-4">
        <div className="card">
          <div className="card-header">
            <h3>Human Oversight Policy</h3>
          </div>
          <div className="oversight-content">
            <div className="oversight-item">
              <span className="oversight-icon">👤</span>
              <div className="oversight-info">
                <div className="oversight-label">Human Review Required</div>
                <div className="oversight-value">For predictions &lt;90% confidence</div>
              </div>
              <span className="oversight-status active">Active</span>
            </div>
            <div className="oversight-metrics">
              <div className="oversight-metric">
                <div className="metric-val">{humanOversight.escalationRate}%</div>
                <div className="metric-label">Escalation Rate</div>
              </div>
              <div className="oversight-metric">
                <div className="metric-val">{humanOversight.overrideRate}%</div>
                <div className="metric-label">Override Rate</div>
              </div>
              <div className="oversight-metric">
                <div className="metric-val">{humanOversight.avgReviewTime}</div>
                <div className="metric-label">Avg Review Time</div>
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3>Recent Audit Log</h3>
          </div>
          <div className="audit-log">
            {auditLog.map((log, idx) => (
              <div key={idx} className="audit-item">
                <div className="audit-date">{log.date}</div>
                <div className="audit-action">{log.action}</div>
                <div className="audit-user">{log.user}</div>
                <span className={`audit-status ${log.status.toLowerCase()}`}>{log.status}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Governance Chart */}
      <div className="chart-card mt-4">
        <div className="chart-title">Governance Maturity Radar</div>
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={governanceFramework.map(g => ({ dimension: g.dimension, current: g.maturity, target: g.target }))}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="dimension" stroke="#94a3b8" />
            <PolarRadiusAxis angle={30} domain={[0, 5]} stroke="#94a3b8" />
            <Radar name="Current" dataKey="current" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.3} />
            <Radar name="Target" dataKey="target" stroke="#4caf50" fill="#4caf50" fillOpacity={0.1} strokeDasharray="5 5" />
            <Legend />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default AnalysisUI
