import React, { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, AreaChart, Area, Legend, ScatterChart, Scatter
} from 'recharts'

const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#9c27b0']

// ============================================================================
// MAIN METRICS DASHBOARD
// ============================================================================
function MetricsDashboard() {
  const [activeSection, setActiveSection] = useState('model')

  const sections = [
    { id: 'model', label: 'Model Performance', icon: '📊' },
    { id: 'quality', label: 'AI Quality', icon: '✨' },
    { id: 'business', label: 'Business Metrics', icon: '💼' },
    { id: 'tech', label: 'Technology', icon: '⚡' },
    { id: 'security', label: 'Security & Compliance', icon: '🔒' }
  ]

  const renderSection = () => {
    switch (activeSection) {
      case 'model': return <ModelPerformance />
      case 'quality': return <AIQualityMetrics />
      case 'business': return <BusinessMetrics />
      case 'tech': return <TechnologyMetrics />
      case 'security': return <SecurityCompliance />
      default: return <ModelPerformance />
    }
  }

  return (
    <div className="metrics-dashboard">
      <div className="metrics-nav">
        {sections.map(section => (
          <button
            key={section.id}
            className={`metrics-nav-btn ${activeSection === section.id ? 'active' : ''}`}
            onClick={() => setActiveSection(section.id)}
          >
            <span className="nav-icon">{section.icon}</span>
            <span>{section.label}</span>
          </button>
        ))}
      </div>
      <div className="metrics-content">
        {renderSection()}
      </div>
    </div>
  )
}

// ============================================================================
// 1. MODEL PERFORMANCE SECTION
// ============================================================================
function ModelPerformance() {
  // Core metrics
  const coreMetrics = { accuracy: 92.4, precision: 91.8, recall: 93.1, f1: 92.4 }

  // ROC Curve data
  const rocData = [
    { fpr: 0, tpr: 0 },
    { fpr: 0.02, tpr: 0.45 },
    { fpr: 0.05, tpr: 0.72 },
    { fpr: 0.08, tpr: 0.85 },
    { fpr: 0.12, tpr: 0.91 },
    { fpr: 0.18, tpr: 0.95 },
    { fpr: 0.25, tpr: 0.97 },
    { fpr: 0.4, tpr: 0.98 },
    { fpr: 0.6, tpr: 0.99 },
    { fpr: 1, tpr: 1 }
  ]

  // Confusion Matrix
  const confusionMatrix = { tp: 186, fn: 14, fp: 16, tn: 178 }

  // Per-Class Performance
  const perClassData = [
    { class: 'Medication Name', precision: 95, recall: 93, f1: 94 },
    { class: 'Dosage', precision: 91, recall: 89, f1: 90 },
    { class: 'Frequency', precision: 88, recall: 92, f1: 90 },
    { class: 'Duration', precision: 85, recall: 87, f1: 86 }
  ]

  // t-SNE data
  const tsneData = [
    { x: 20, y: 30, name: 'Metformin', cluster: 'Diabetes' },
    { x: 25, y: 35, name: 'Sitagliptin', cluster: 'Diabetes' },
    { x: 22, y: 28, name: 'Glipizide', cluster: 'Diabetes' },
    { x: 28, y: 32, name: 'Empagliflozin', cluster: 'Diabetes' },
    { x: 60, y: 45, name: 'Lisinopril', cluster: 'Cardiac' },
    { x: 65, y: 50, name: 'Losartan', cluster: 'Cardiac' },
    { x: 58, y: 48, name: 'Atorvastatin', cluster: 'Cardiac' },
    { x: 62, y: 42, name: 'Aspirin', cluster: 'Cardiac' },
    { x: 45, y: 70, name: 'Insulin Glargine', cluster: 'Insulin' },
    { x: 48, y: 75, name: 'Semaglutide', cluster: 'Insulin' }
  ]

  // Feature importance
  const featureImportance = [
    { feature: 'Text Clarity', importance: 28, desc: 'Quality of OCR text extraction' },
    { feature: 'Drug Name Match', importance: 24, desc: 'Match with drug database' },
    { feature: 'Dosage Pattern', importance: 18, desc: 'Recognition of dosage format' },
    { feature: 'Frequency Keywords', importance: 15, desc: 'Detection of timing words' },
    { feature: 'Document Layout', importance: 10, desc: 'Prescription structure analysis' },
    { feature: 'Handwriting Score', importance: 5, desc: 'Legibility assessment' }
  ]

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Model Performance</h2>
        <p>Comprehensive model evaluation metrics and analysis</p>
      </div>

      {/* Core Metrics */}
      <div className="metrics-grid-4">
        <div className="metric-card highlight-blue">
          <div className="metric-value">{coreMetrics.accuracy}%</div>
          <div className="metric-label">Accuracy</div>
        </div>
        <div className="metric-card highlight-green">
          <div className="metric-value">{coreMetrics.precision}%</div>
          <div className="metric-label">Precision</div>
        </div>
        <div className="metric-card highlight-purple">
          <div className="metric-value">{coreMetrics.recall}%</div>
          <div className="metric-label">Recall</div>
        </div>
        <div className="metric-card highlight-orange">
          <div className="metric-value">{coreMetrics.f1}%</div>
          <div className="metric-label">F1 Score</div>
        </div>
      </div>

      {/* ROC Curve and Confusion Matrix */}
      <div className="charts-grid mt-4">
        <div className="chart-card">
          <div className="chart-title">ROC Curve (AUC: 0.956)</div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="fpr" stroke="#94a3b8" label={{ value: 'False Positive Rate', position: 'bottom', offset: -5 }} />
              <YAxis stroke="#94a3b8" label={{ value: 'True Positive Rate', angle: -90, position: 'left' }} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Area type="monotone" dataKey="tpr" stroke="#1e88e5" fill="rgba(30, 136, 229, 0.3)" strokeWidth={2} />
              <Line type="linear" dataKey="fpr" stroke="#64748b" strokeDasharray="5 5" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          <InfoBox
            title="What is ROC Curve?"
            content="ROC (Receiver Operating Characteristic) Curve shows the trade-off between sensitivity and specificity. AUC measures overall model discrimination ability."
            kpiTarget="AUC >= 0.90 for production deployment"
            roiImpact="Each 1% improvement reduces manual review costs by ~$15K annually"
            status="Good"
          />
        </div>

        <div className="chart-card">
          <div className="chart-title">Confusion Matrix</div>
          <div className="confusion-matrix">
            <div className="cm-header">
              <div></div>
              <div className="cm-label">Predicted +</div>
              <div className="cm-label">Predicted -</div>
            </div>
            <div className="cm-row">
              <div className="cm-label">Actual +</div>
              <div className="cm-cell tp">{confusionMatrix.tp}</div>
              <div className="cm-cell fn">{confusionMatrix.fn}</div>
            </div>
            <div className="cm-row">
              <div className="cm-label">Actual -</div>
              <div className="cm-cell fp">{confusionMatrix.fp}</div>
              <div className="cm-cell tn">{confusionMatrix.tn}</div>
            </div>
          </div>
          <InfoBox
            title="What is Confusion Matrix?"
            content="Confusion Matrix shows actual vs predicted classifications, revealing true positives, true negatives, false positives, and false negatives."
            kpiTarget="False Negative Rate < 2%, False Positive Rate < 5%"
            roiImpact="Reducing false negatives by 1% prevents ~50 potential adverse events annually"
            status="Good"
          />
        </div>
      </div>

      {/* Per-Class Performance */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Per-Class Performance</h3>
        </div>
        <div className="per-class-grid">
          {perClassData.map((item, idx) => (
            <div key={idx} className="per-class-item">
              <div className="per-class-name">{item.class}</div>
              <div className="per-class-metrics">
                <div className="pcm">
                  <span className="pcm-label">P</span>
                  <span className="pcm-value">{item.precision}%</span>
                </div>
                <div className="pcm">
                  <span className="pcm-label">R</span>
                  <span className="pcm-value">{item.recall}%</span>
                </div>
                <div className="pcm">
                  <span className="pcm-label">F1</span>
                  <span className="pcm-value">{item.f1}%</span>
                </div>
              </div>
              <div className="per-class-bar">
                <div className="per-class-fill" style={{ width: `${item.f1}%` }} />
              </div>
            </div>
          ))}
        </div>
        <InfoBox
          title="What is Per-Class Performance?"
          content="Per-class metrics show Precision (P), Recall (R), and F1 score for each medication category independently."
          kpiTarget="All classes: P >= 90%, R >= 90%, F1 >= 90%"
          roiImpact="Improving underperforming classes reduces support tickets by 30%"
          status="Good"
        />
      </div>

      {/* t-SNE and Feature Importance */}
      <div className="charts-grid mt-4">
        <div className="chart-card">
          <div className="chart-title">t-SNE: Medication Clustering</div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" dataKey="x" stroke="#94a3b8" />
              <YAxis type="number" dataKey="y" stroke="#94a3b8" />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Legend />
              <Scatter name="Diabetes" data={tsneData.filter(d => d.cluster === 'Diabetes')} fill="#4caf50" />
              <Scatter name="Cardiac" data={tsneData.filter(d => d.cluster === 'Cardiac')} fill="#1e88e5" />
              <Scatter name="Insulin" data={tsneData.filter(d => d.cluster === 'Insulin')} fill="#ff9800" />
            </ScatterChart>
          </ResponsiveContainer>
          <InfoBox
            title="What is t-SNE Clustering?"
            content="t-SNE (t-distributed Stochastic Neighbor Embedding) visualizes high-dimensional medication data in 2D, showing natural clustering patterns."
            kpiTarget="Clear cluster separation with inter-cluster distance > 20 units"
            roiImpact="Better clustering improves recommendation accuracy by 15-20%"
            status="Good"
          />
        </div>

        <div className="chart-card">
          <div className="chart-title">Feature Importance (SHAP Analysis)</div>
          <div className="feature-importance-list">
            {featureImportance.map((f, idx) => (
              <div key={idx} className="fi-item">
                <div className="fi-header">
                  <span className="fi-name">{f.feature}</span>
                  <span className="fi-value">{f.importance}%</span>
                </div>
                <div className="fi-desc">{f.desc}</div>
                <div className="fi-bar">
                  <div className="fi-fill" style={{ width: `${f.importance * 3}%`, background: COLORS[idx % COLORS.length] }} />
                </div>
              </div>
            ))}
          </div>
          <InfoBox
            title="What is Feature Importance?"
            content="SHAP (SHapley Additive exPlanations) analysis shows which input features most influence model predictions."
            kpiTarget="Top 3 features should contribute >= 60% of total importance"
            roiImpact="Optimizing high-importance features yields 2x efficiency improvement"
            status="Good"
          />
        </div>
      </div>

      {/* Decision Explanation */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Decision Explanation Example</h3>
        </div>
        <div className="decision-example">
          <div className="de-input">
            <div className="de-title">Input: Prescription Image</div>
            <div className="de-prescription">
              <div className="rx-header">Rx - Dr. Michael Chen</div>
              <div className="rx-patient">Patient: Susan Brown</div>
              <div className="rx-item">1. Tab Metformin 500mg BD</div>
              <div className="rx-item">2. Tab Atorvastatin 10mg OD</div>
            </div>
          </div>
          <div className="de-process">
            <div className="de-title">AI Decision Process</div>
            <div className="de-steps">
              <div className="de-step">
                <span className="step-num">1</span>
                <div>
                  <div className="step-title">Text Extraction</div>
                  <div className="step-detail">OCR confidence: 96.2%</div>
                </div>
              </div>
              <div className="de-step">
                <span className="step-num">2</span>
                <div>
                  <div className="step-title">Drug Recognition</div>
                  <div className="step-detail">"Metformin" matched with 99.1% confidence</div>
                </div>
              </div>
              <div className="de-step">
                <span className="step-num">3</span>
                <div>
                  <div className="step-title">Dosage Parsing</div>
                  <div className="step-detail">"500mg" extracted, "BD" → Twice daily</div>
                </div>
              </div>
              <div className="de-step">
                <span className="step-num">4</span>
                <div>
                  <div className="step-title">Validation</div>
                  <div className="step-detail">Cross-referenced with drug database</div>
                </div>
              </div>
            </div>
          </div>
          <div className="de-output">
            <div className="de-title">Output</div>
            <div className="de-result">
              <div className="de-confidence">94.2% Confidence</div>
              <div className="de-medication">Medication: Metformin 500mg</div>
              <div className="de-frequency">Frequency: Twice daily</div>
              <div className="de-explanation">
                Explanation: High confidence due to clear text, exact drug name match, and standard dosage format.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Tonality Analysis */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Tonality Analysis Model</h3>
          <span className="badge badge-info">Professional & Caring</span>
        </div>
        <div className="tonality-content">
          <div className="tonality-scores">
            {[
              { name: 'professional', score: 92 },
              { name: 'empathetic', score: 88 },
              { name: 'urgent', score: 45 },
              { name: 'informative', score: 95 },
              { name: 'reassuring', score: 85 }
            ].map((t, idx) => (
              <div key={idx} className="tonality-item">
                <div className="tonality-label">{t.name}</div>
                <div className="tonality-bar">
                  <div className="tonality-fill" style={{ width: `${t.score}%` }} />
                </div>
                <div className="tonality-value">{t.score}%</div>
              </div>
            ))}
          </div>
          <div className="message-analysis">
            <div className="ma-title">Recent Message Analysis</div>
            {[
              { msg: 'Time to take your medication', type: 'Reminder', score: 94 },
              { msg: 'Your glucose is above target range', type: 'Alert', score: 89 },
              { msg: 'Great job maintaining your schedule!', type: 'Encouragement', score: 96 },
              { msg: 'Please consult your doctor', type: 'Advisory', score: 91 }
            ].map((m, idx) => (
              <div key={idx} className="ma-item">
                <div className="ma-message">"{m.msg}"</div>
                <div className="ma-type">{m.type}</div>
                <div className="ma-score">{m.score}%</div>
              </div>
            ))}
          </div>
        </div>
        <InfoBox
          title="What is Tonality Analysis?"
          content="Tonality Analysis evaluates the emotional tone and communication style of AI-generated messages to patients."
          kpiTarget="Professional >= 90%, Empathetic >= 85%, Reassuring >= 80%"
          roiImpact="Improved tone increases user retention by 18% and satisfaction by 22%"
          status="Good"
        />
      </div>
    </div>
  )
}

// ============================================================================
// 2. AI QUALITY METRICS SECTION
// ============================================================================
function AIQualityMetrics() {
  const qualityDimensions = [
    {
      name: 'Fairness', score: 94, status: 'Excellent',
      description: 'Model treats all demographic groups equitably',
      subMetrics: [
        { name: 'demographicParity', score: 96 },
        { name: 'equalizedOdds', score: 93 },
        { name: 'individualFairness', score: 92 }
      ]
    },
    {
      name: 'Robustness', score: 91, status: 'Strong',
      description: 'Model performs well under various conditions',
      subMetrics: [
        { name: 'adversarialAccuracy', score: 88 },
        { name: 'noiseResilience', score: 93 },
        { name: 'outOfDistribution', score: 89 }
      ]
    },
    {
      name: 'Interpretability', score: 89, status: 'Good',
      description: 'Model decisions can be understood by humans',
      subMetrics: [
        { name: 'localExplanations', score: 92 },
        { name: 'globalExplanations', score: 87 },
        { name: 'featureAttribution', score: 90 }
      ]
    },
    {
      name: 'Portability', score: 87, status: 'Good',
      description: 'Model can be deployed across different environments',
      subMetrics: [
        { name: 'crossPlatform', score: 92 },
        { name: 'modelExport', score: 88 },
        { name: 'apiCompatibility', score: 85 }
      ]
    }
  ]

  const trustMetrics = {
    userAcceptance: 94.2,
    reliabilityIndex: 96,
    consistencyScore: 98
  }

  const responsibleAIChecklist = [
    { item: 'Human oversight for low-confidence predictions', checked: true },
    { item: 'Clear explanation of AI decisions provided', checked: true },
    { item: 'User can override AI recommendations', checked: true },
    { item: 'Regular bias audits conducted', checked: true },
    { item: 'Data minimization practices in place', checked: true },
    { item: 'Feedback loop for continuous improvement', checked: true },
    { item: 'Emergency escalation procedures defined', checked: true }
  ]

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>AI Quality Metrics</h2>
        <p>Fairness, Robustness, Interpretability, and Portability assessment</p>
      </div>

      {/* Quality Dimensions Grid */}
      <div className="quality-grid">
        {qualityDimensions.map((dim, idx) => (
          <div key={idx} className="quality-card">
            <div className="quality-header">
              <div className="quality-name">{dim.name}</div>
              <div className="quality-score">{dim.score}%</div>
            </div>
            <div className="quality-desc">{dim.description}</div>
            <div className="quality-bar">
              <div className="quality-fill" style={{ width: `${dim.score}%` }} />
            </div>
            <div className="quality-submetrics">
              {dim.subMetrics.map((sub, sidx) => (
                <div key={sidx} className="qsm">
                  <span className="qsm-name">{sub.name}</span>
                  <span className="qsm-score">{sub.score}%</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Trust Metrics */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Trust Metrics</h3>
        </div>
        <div className="trust-metrics-grid">
          <div className="trust-metric-item">
            <div className="tm-value">{trustMetrics.userAcceptance}%</div>
            <div className="tm-label">User Acceptance Rate</div>
          </div>
          <div className="trust-metric-item">
            <div className="tm-value">{trustMetrics.reliabilityIndex}%</div>
            <div className="tm-label">Reliability Index</div>
          </div>
          <div className="trust-metric-item">
            <div className="tm-value">{trustMetrics.consistencyScore}%</div>
            <div className="tm-label">Consistency Score</div>
          </div>
        </div>
        <InfoBox
          title="What is Trust Metrics?"
          content="Trust Metrics measure user confidence in AI recommendations through acceptance, reliability, and consistency scores."
          kpiTarget="User Acceptance >= 90%, Reliability >= 95%, Consistency >= 95%"
          roiImpact="Each 5% trust improvement correlates with 12% increase in daily active users"
          status="Good"
        />
      </div>

      {/* Responsible AI Checklist */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Responsible AI Checklist</h3>
          <span className="badge badge-success">All Complete</span>
        </div>
        <div className="checklist-grid">
          {responsibleAIChecklist.map((item, idx) => (
            <div key={idx} className={`checklist-item ${item.checked ? 'checked' : ''}`}>
              <span className="check-icon">{item.checked ? '✓' : '○'}</span>
              <span className="check-text">{item.item}</span>
            </div>
          ))}
        </div>
      </div>

      {/* User Feedback */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>User Feedback & Corrections</h3>
        </div>
        <div className="feedback-metrics">
          <div className="feedback-item">
            <div className="fb-value">4.6</div>
            <div className="fb-label">Avg Feedback Score (out of 5)</div>
            <div className="fb-stars">★★★★☆</div>
          </div>
          <div className="feedback-item">
            <div className="fb-value">5.8%</div>
            <div className="fb-label">Correction Rate</div>
            <div className="fb-trend positive">↓ Improving</div>
          </div>
          <div className="feedback-item">
            <div className="fb-value">4.8</div>
            <div className="fb-label">Transparency Rating (out of 5)</div>
            <div className="fb-stars">★★★★★</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// 3. BUSINESS METRICS SECTION
// ============================================================================
function BusinessMetrics() {
  const kpis = [
    { name: 'Patient Adherence', value: '87.5%', change: '+4.2%', trend: 'up' },
    { name: 'Cost Savings', value: '$125K', change: '+12.8%', trend: 'up' },
    { name: 'Time to Treatment', value: '2.3 days', change: '-18%', trend: 'down' },
    { name: 'Error Reduction', value: '94%', change: '+6%', trend: 'up' },
    { name: 'User Adoption', value: '78%', change: '+15%', trend: 'up' },
    { name: 'Support Tickets', value: '156', change: '-23%', trend: 'down' }
  ]

  const roiData = {
    roi: 250,
    payback: '8 months',
    npv: '$625K',
    irr: '156%'
  }

  const swot = {
    strengths: ['High OCR accuracy (92.4%)', 'HIPAA compliant', 'Real-time processing', 'Multi-language support'],
    weaknesses: ['Handwritten prescription challenges', 'Limited offline capability', 'High initial training data needs'],
    opportunities: ['Telemedicine integration', 'Insurance claim automation', 'Pharmacy network expansion', 'AI diagnostic assistance'],
    threats: ['Regulatory changes', 'Data privacy concerns', 'Competitor solutions', 'Technology obsolescence']
  }

  const valueProposition = [
    { icon: '⏱️', title: 'Time Savings', desc: 'Reduce manual data entry by 85%', score: 92 },
    { icon: '✓', title: 'Accuracy', desc: 'AI-powered verification reduces errors', score: 94 },
    { icon: '🛡️', title: 'Compliance', desc: 'Automated HIPAA & GDPR compliance', score: 98 },
    { icon: '💰', title: 'Cost Efficiency', desc: 'Lower operational costs by 40%', score: 88 },
    { icon: '❤️', title: 'Patient Safety', desc: 'Drug interaction alerts & dosage verification', score: 96 }
  ]

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Business Metrics</h2>
        <p>KPIs, ROI Analysis, and Strategic Assessment</p>
      </div>

      {/* KPIs Grid */}
      <div className="card">
        <div className="card-header">
          <h3>Key Performance Indicators (KPIs)</h3>
        </div>
        <div className="kpi-grid">
          {kpis.map((kpi, idx) => (
            <div key={idx} className="kpi-card">
              <div className="kpi-value">{kpi.value}</div>
              <div className="kpi-name">{kpi.name}</div>
              <div className={`kpi-change ${kpi.trend === 'up' ? 'positive' : 'negative'}`}>
                {kpi.change}
              </div>
            </div>
          ))}
        </div>
        <InfoBox
          title="What are KPIs?"
          content="Key Performance Indicators track critical business and clinical metrics for the AI system."
          kpiTarget="All KPIs trending positive with month-over-month improvement"
          roiImpact="KPI improvements directly correlate with $125K quarterly cost savings"
          status="Good"
        />
      </div>

      {/* ROI Analysis */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>ROI Analysis</h3>
        </div>
        <div className="roi-grid">
          <div className="roi-item highlight">
            <div className="roi-value">{roiData.roi}%</div>
            <div className="roi-label">Return on Investment</div>
          </div>
          <div className="roi-item">
            <div className="roi-value">{roiData.payback}</div>
            <div className="roi-label">Payback Period</div>
          </div>
          <div className="roi-item">
            <div className="roi-value">{roiData.npv}</div>
            <div className="roi-label">Net Present Value</div>
          </div>
          <div className="roi-item">
            <div className="roi-value">{roiData.irr}</div>
            <div className="roi-label">Internal Rate of Return</div>
          </div>
        </div>
        <InfoBox
          title="What is ROI Analysis?"
          content="Return on Investment analysis measures financial returns against initial investment in the AI system."
          kpiTarget="ROI >= 200%, Payback Period <= 12 months"
          roiImpact="Current 250% ROI exceeds industry benchmarks by 80%"
          status="Good"
        />
      </div>

      {/* SWOT Analysis */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>SWOT Analysis</h3>
        </div>
        <div className="swot-grid">
          <div className="swot-quadrant strengths">
            <div className="swot-title">Strengths</div>
            <ul>{swot.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul>
          </div>
          <div className="swot-quadrant weaknesses">
            <div className="swot-title">Weaknesses</div>
            <ul>{swot.weaknesses.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </div>
          <div className="swot-quadrant opportunities">
            <div className="swot-title">Opportunities</div>
            <ul>{swot.opportunities.map((o, i) => <li key={i}>{o}</li>)}</ul>
          </div>
          <div className="swot-quadrant threats">
            <div className="swot-title">Threats</div>
            <ul>{swot.threats.map((t, i) => <li key={i}>{t}</li>)}</ul>
          </div>
        </div>
        <InfoBox
          title="What is SWOT Analysis?"
          content="SWOT Analysis identifies Strengths, Weaknesses, Opportunities, and Threats for strategic planning."
          kpiTarget="Strengths > Weaknesses, Opportunities > Threats"
          roiImpact="Strategic SWOT execution can increase market share by 25%"
          status="Good"
        />
      </div>

      {/* Value Proposition */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Value Proposition Matrix</h3>
        </div>
        <div className="value-grid">
          {valueProposition.map((v, idx) => (
            <div key={idx} className="value-item">
              <div className="value-icon">{v.icon}</div>
              <div className="value-info">
                <div className="value-title">{v.title}</div>
                <div className="value-desc">{v.desc}</div>
              </div>
              <div className="value-score">{v.score}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// 4. TECHNOLOGY METRICS SECTION
// ============================================================================
function TechnologyMetrics() {
  const techPerformance = [
    { name: 'inferenceLatency', value: 145, unit: 'ms', target: 200 },
    { name: 'throughput', value: 1250, unit: 'req/s', target: 1000 },
    { name: 'gpuUtilization', value: 72, unit: '%', target: 80 },
    { name: 'memoryUsage', value: 68, unit: '%', target: 80 },
    { name: 'modelSize', value: 125, unit: 'MB', target: 200 },
    { name: 'batchProcessing', value: 32, unit: 'items', target: 32 }
  ]

  const sustainability = [
    { icon: '🌱', name: 'Carbon Footprint', value: '2.4 kg CO2/day', desc: 'Model inference emissions' },
    { icon: '⚡', name: 'Energy Efficiency', value: '94%', desc: 'Green computing score' },
    { icon: '📄', name: 'Paper Reduction', value: '12,500 sheets/month', desc: 'Digital prescriptions saved' },
    { icon: '🖥️', name: 'Server Utilization', value: '78%', desc: 'Optimal resource usage' }
  ]

  const qualityScore = {
    overall: 91.5,
    components: [
      { name: 'Data Quality', score: 94 },
      { name: 'Model Performance', score: 92 },
      { name: 'User Experience', score: 89 },
      { name: 'Security & Compliance', score: 98 },
      { name: 'Reliability', score: 88 },
      { name: 'Maintainability', score: 86 }
    ]
  }

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Technology Performance</h2>
        <p>System efficiency, sustainability, and quality metrics</p>
      </div>

      {/* Overall Quality Score */}
      <div className="quality-score-card">
        <div className="qs-main">
          <div className="qs-circle">
            <svg viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" strokeWidth="8" />
              <circle
                cx="50" cy="50" r="45" fill="none"
                stroke="#4caf50" strokeWidth="8"
                strokeDasharray={`${qualityScore.overall * 2.83} ${283 - qualityScore.overall * 2.83}`}
                strokeLinecap="round"
                transform="rotate(-90 50 50)"
              />
            </svg>
            <div className="qs-value">{qualityScore.overall}</div>
          </div>
          <div className="qs-label">Overall Quality Score</div>
        </div>
        <div className="qs-components">
          {qualityScore.components.map((c, idx) => (
            <div key={idx} className="qsc-item">
              <div className="qsc-name">{c.name}</div>
              <div className="qsc-bar">
                <div className="qsc-fill" style={{ width: `${c.score}%` }} />
              </div>
              <div className="qsc-score">{c.score}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Technology Performance */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Technology Performance</h3>
        </div>
        <div className="tech-perf-grid">
          {techPerformance.map((t, idx) => (
            <div key={idx} className="tech-perf-item">
              <div className="tp-name">{t.name}</div>
              <div className="tp-value">{t.value} <span className="tp-unit">{t.unit}</span></div>
              <div className="tp-bar">
                <div className="tp-fill" style={{ width: `${Math.min((t.value / t.target) * 100, 100)}%` }} />
              </div>
              <div className="tp-target">Target: {t.target} {t.unit}</div>
            </div>
          ))}
        </div>
        <InfoBox
          title="What is Technology Performance?"
          content="Technology Performance metrics measure system efficiency including latency, throughput, and resource utilization."
          kpiTarget="Latency < 200ms, Throughput > 1000 req/s, GPU < 80%"
          roiImpact="Performance optimization reduces infrastructure costs by 20%"
          status="Good"
        />
      </div>

      {/* Sustainability */}
      <div className="card mt-4">
        <div className="card-header">
          <h3>Sustainability Performance</h3>
          <span className="badge badge-success">Green Certified</span>
        </div>
        <div className="sustainability-grid">
          {sustainability.map((s, idx) => (
            <div key={idx} className="sustain-item">
              <div className="sustain-icon">{s.icon}</div>
              <div className="sustain-info">
                <div className="sustain-name">{s.name}</div>
                <div className="sustain-desc">{s.desc}</div>
              </div>
              <div className="sustain-value">{s.value}</div>
            </div>
          ))}
        </div>
        <InfoBox
          title="What is Sustainability Performance?"
          content="Sustainability Performance tracks environmental impact including carbon footprint, energy efficiency, and resource conservation."
          kpiTarget="Carbon footprint < 5 kg CO2/day, Energy efficiency >= 90%"
          roiImpact="Green initiatives attract ESG-focused investors and enterprise clients"
          status="Good"
        />
      </div>
    </div>
  )
}

// ============================================================================
// 5. SECURITY & COMPLIANCE SECTION
// ============================================================================
function SecurityCompliance() {
  const securityMetrics = [
    { name: 'Data Encryption', value: 'AES-256', status: 'active' },
    { name: 'Access Controls', value: 'Role-Based (RBAC)', status: 'active' },
    { name: 'Audit Logs', value: '1247 entries', status: 'active' },
    { name: 'Vulnerabilities', value: '0 found', status: 'secure' }
  ]

  const certifications = ['HIPAA', 'GDPR', 'SOC2']

  return (
    <div className="section-content">
      <div className="section-header">
        <h2>Security & Compliance</h2>
        <p>Data protection, access controls, and regulatory compliance</p>
      </div>

      {/* Security Metrics */}
      <div className="security-grid">
        {securityMetrics.map((s, idx) => (
          <div key={idx} className="security-item">
            <div className="sec-name">{s.name}</div>
            <div className="sec-value">{s.value}</div>
            <div className={`sec-status ${s.status}`}>{s.status}</div>
          </div>
        ))}
      </div>

      {/* Certifications */}
      <div className="certifications mt-4">
        <div className="cert-title">Certifications</div>
        <div className="cert-badges">
          {certifications.map((c, idx) => (
            <div key={idx} className="cert-badge">
              <span className="cert-icon">✓</span>
              <span className="cert-name">{c}</span>
            </div>
          ))}
        </div>
      </div>

      <InfoBox
        title="What is Security & Compliance?"
        content="Security & Compliance metrics track data protection, access controls, and regulatory compliance status."
        kpiTarget="Zero vulnerabilities, 100% compliance, Privacy Score >= 95%"
        roiImpact="Security compliance enables $500K+ in enterprise contracts"
        status="Good"
      />
    </div>
  )
}

// ============================================================================
// INFO BOX COMPONENT
// ============================================================================
function InfoBox({ title, content, kpiTarget, roiImpact, status }) {
  return (
    <div className="info-box">
      <div className="info-header">
        <span className="info-icon">ℹ️</span>
        <span className="info-title">{title}</span>
        <span className={`info-status status-${status.toLowerCase()}`}>{status}</span>
      </div>
      <div className="info-content">{content}</div>
      <div className="info-metrics">
        <div className="info-metric">
          <span className="im-label">KPI Target:</span>
          <span className="im-value">{kpiTarget}</span>
        </div>
        <div className="info-metric">
          <span className="im-label">ROI Impact:</span>
          <span className="im-value">{roiImpact}</span>
        </div>
      </div>
    </div>
  )
}

export default MetricsDashboard
