import React, { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, PieChart, Pie, Cell, Legend
} from 'recharts'

// Phase data for all 15 phases
const PHASES = [
  { id: 1, name: 'Knowledge & Data', modules: 16, category: 'data', threshold: 90 },
  { id: 2, name: 'Retrieval', modules: 17, category: 'data', threshold: 85 },
  { id: 3, name: 'Generation', modules: 17, category: 'data', threshold: 90 },
  { id: 4, name: 'Decision Policy', modules: 17, category: 'agent', threshold: 85 },
  { id: 5, name: 'Agent Behavior', modules: 17, category: 'agent', threshold: 85 },
  { id: 6, name: 'A2A Interaction', modules: 17, category: 'agent', threshold: 80 },
  { id: 7, name: 'MCP Compliance', modules: 17, category: 'agent', threshold: 95 },
  { id: 8, name: 'Explainability', modules: 17, category: 'trust', threshold: 85 },
  { id: 9, name: 'Robustness', modules: 17, category: 'trust', threshold: 80 },
  { id: 10, name: 'Statistical', modules: 17, category: 'trust', threshold: 85 },
  { id: 11, name: 'Benchmarking', modules: 17, category: 'ops', threshold: 80 },
  { id: 12, name: 'Scalability', modules: 17, category: 'ops', threshold: 85 },
  { id: 13, name: 'Governance', modules: 17, category: 'ops', threshold: 95 },
  { id: 14, name: 'Production', modules: 23, category: 'ops', threshold: 90 },
  { id: 15, name: 'Value & ROI', modules: 20, category: 'value', threshold: 80 }
]

const CATEGORY_COLORS = {
  data: '#ef4444',
  agent: '#3b82f6',
  trust: '#22c55e',
  ops: '#f97316',
  value: '#a855f7'
}

const CATEGORY_NAMES = {
  data: 'Data Quality',
  agent: 'Agent Behavior',
  trust: 'Trust & Safety',
  ops: 'Operations',
  value: 'Business Value'
}

// Generate mock results for a phase
const generatePhaseResults = (phase) => {
  const passed = Math.floor(phase.modules * (0.85 + Math.random() * 0.15))
  const score = (passed / phase.modules) * 100
  return {
    ...phase,
    passed,
    failed: phase.modules - passed,
    score: score.toFixed(1),
    status: score >= phase.threshold ? 'PASS' : 'FAIL'
  }
}

// Phase Details Component
function PhaseDetails({ phase, onClose }) {
  const moduleResults = Array.from({ length: phase.modules }, (_, i) => ({
    id: i + 1,
    name: `Module ${i + 1}`,
    status: Math.random() > 0.1 ? 'pass' : 'fail',
    score: (85 + Math.random() * 15).toFixed(1),
    latency: Math.floor(10 + Math.random() * 90)
  }))

  return (
    <div className="phase-details-modal">
      <div className="phase-details-content">
        <div className="phase-details-header">
          <h2>Phase {phase.id}: {phase.name}</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        <div className="phase-summary-stats">
          <div className="stat-box">
            <span className="stat-value">{phase.modules}</span>
            <span className="stat-label">Total Modules</span>
          </div>
          <div className="stat-box success">
            <span className="stat-value">{phase.passed}</span>
            <span className="stat-label">Passed</span>
          </div>
          <div className="stat-box danger">
            <span className="stat-value">{phase.failed}</span>
            <span className="stat-label">Failed</span>
          </div>
          <div className="stat-box">
            <span className="stat-value">{phase.score}%</span>
            <span className="stat-label">Score</span>
          </div>
        </div>

        <div className="modules-table-container">
          <table className="modules-table">
            <thead>
              <tr>
                <th>Module</th>
                <th>Status</th>
                <th>Score</th>
                <th>Latency (ms)</th>
              </tr>
            </thead>
            <tbody>
              {moduleResults.map(m => (
                <tr key={m.id}>
                  <td>{m.name}</td>
                  <td>
                    <span className={`status-badge ${m.status}`}>
                      {m.status.toUpperCase()}
                    </span>
                  </td>
                  <td>{m.score}%</td>
                  <td>{m.latency}ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// Main Monitoring Dashboard
function MonitoringDashboard() {
  const [phaseResults, setPhaseResults] = useState([])
  const [selectedPhase, setSelectedPhase] = useState(null)
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [isRunning, setIsRunning] = useState(false)

  useEffect(() => {
    // Initialize with results
    setPhaseResults(PHASES.map(generatePhaseResults))
  }, [])

  const runAnalysis = () => {
    setIsRunning(true)
    // Simulate running analysis
    setTimeout(() => {
      setPhaseResults(PHASES.map(generatePhaseResults))
      setIsRunning(false)
    }, 2000)
  }

  // Filter phases by category
  const filteredPhases = selectedCategory === 'all'
    ? phaseResults
    : phaseResults.filter(p => p.category === selectedCategory)

  // Calculate overall stats
  const totalModules = phaseResults.reduce((sum, p) => sum + p.modules, 0)
  const totalPassed = phaseResults.reduce((sum, p) => sum + p.passed, 0)
  const overallScore = totalModules > 0 ? ((totalPassed / totalModules) * 100).toFixed(1) : 0

  // Category summary data
  const categoryData = Object.keys(CATEGORY_NAMES).map(cat => {
    const catPhases = phaseResults.filter(p => p.category === cat)
    const catModules = catPhases.reduce((sum, p) => sum + p.modules, 0)
    const catPassed = catPhases.reduce((sum, p) => sum + p.passed, 0)
    return {
      name: CATEGORY_NAMES[cat],
      score: catModules > 0 ? ((catPassed / catModules) * 100).toFixed(1) : 0,
      modules: catModules,
      passed: catPassed
    }
  })

  // Radar chart data
  const radarData = categoryData.map(c => ({
    subject: c.name,
    score: parseFloat(c.score),
    fullMark: 100
  }))

  // Bar chart data for phases
  const barData = phaseResults.map(p => ({
    name: `P${p.id}`,
    score: parseFloat(p.score),
    threshold: p.threshold,
    fill: CATEGORY_COLORS[p.category]
  }))

  return (
    <div className="monitoring-dashboard">
      {/* Header */}
      <div className="monitoring-header">
        <div className="monitoring-title-section">
          <h1 className="monitoring-title">RAG/Agentic Monitoring Framework</h1>
          <p className="monitoring-subtitle">15-Phase Analysis with 260 Modules</p>
        </div>
        <div className="monitoring-actions">
          <button
            className={`run-analysis-btn ${isRunning ? 'running' : ''}`}
            onClick={runAnalysis}
            disabled={isRunning}
          >
            {isRunning ? 'Running Analysis...' : 'Run Full Analysis'}
          </button>
        </div>
      </div>

      {/* Overall Stats */}
      <div className="overall-stats">
        <div className="stat-card large">
          <div className="stat-icon">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
          </div>
          <div className="stat-content">
            <span className="stat-value">{overallScore}%</span>
            <span className="stat-label">Overall Score</span>
          </div>
          <span className={`status-indicator ${parseFloat(overallScore) >= 90 ? 'success' : parseFloat(overallScore) >= 80 ? 'warning' : 'danger'}`}>
            {parseFloat(overallScore) >= 90 ? 'APPROVED' : parseFloat(overallScore) >= 80 ? 'REVIEW' : 'BLOCKED'}
          </span>
        </div>

        <div className="stat-card">
          <span className="stat-value">{totalModules}</span>
          <span className="stat-label">Total Modules</span>
        </div>

        <div className="stat-card success">
          <span className="stat-value">{totalPassed}</span>
          <span className="stat-label">Passed</span>
        </div>

        <div className="stat-card danger">
          <span className="stat-value">{totalModules - totalPassed}</span>
          <span className="stat-label">Failed</span>
        </div>

        <div className="stat-card">
          <span className="stat-value">15</span>
          <span className="stat-label">Phases</span>
        </div>
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        {/* Radar Chart */}
        <div className="chart-container radar-chart">
          <h3 className="chart-title">Category Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="subject" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#94a3b8" />
              <Radar
                name="Score"
                dataKey="score"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.4}
              />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Bar Chart */}
        <div className="chart-container bar-chart">
          <h3 className="chart-title">Phase Scores</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 10 }} />
              <YAxis stroke="#94a3b8" domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                formatter={(value) => [`${value}%`, 'Score']}
              />
              <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                {barData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
              <Line type="monotone" dataKey="threshold" stroke="#ef4444" strokeDasharray="5 5" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie Chart */}
        <div className="chart-container pie-chart">
          <h3 className="chart-title">Module Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={categoryData}
                dataKey="modules"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
              >
                {categoryData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={Object.values(CATEGORY_COLORS)[index]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Category Filter */}
      <div className="category-filter">
        <button
          className={`filter-btn ${selectedCategory === 'all' ? 'active' : ''}`}
          onClick={() => setSelectedCategory('all')}
        >
          All Phases
        </button>
        {Object.entries(CATEGORY_NAMES).map(([key, name]) => (
          <button
            key={key}
            className={`filter-btn ${selectedCategory === key ? 'active' : ''}`}
            style={{ '--category-color': CATEGORY_COLORS[key] }}
            onClick={() => setSelectedCategory(key)}
          >
            {name}
          </button>
        ))}
      </div>

      {/* Phase Cards Grid */}
      <div className="phases-grid">
        {filteredPhases.map(phase => (
          <div
            key={phase.id}
            className={`phase-card ${phase.status.toLowerCase()}`}
            onClick={() => setSelectedPhase(phase)}
          >
            <div className="phase-card-header">
              <span className="phase-number">Phase {phase.id}</span>
              <span
                className="phase-category-badge"
                style={{ backgroundColor: CATEGORY_COLORS[phase.category] }}
              >
                {CATEGORY_NAMES[phase.category]}
              </span>
            </div>
            <h3 className="phase-name">{phase.name}</h3>

            <div className="phase-progress">
              <div className="progress-bar-container">
                <div
                  className="progress-bar-fill"
                  style={{
                    width: `${phase.score}%`,
                    backgroundColor: phase.status === 'PASS' ? '#22c55e' : '#ef4444'
                  }}
                />
              </div>
              <span className="progress-value">{phase.score}%</span>
            </div>

            <div className="phase-stats">
              <div className="phase-stat">
                <span className="phase-stat-value">{phase.modules}</span>
                <span className="phase-stat-label">Modules</span>
              </div>
              <div className="phase-stat success">
                <span className="phase-stat-value">{phase.passed}</span>
                <span className="phase-stat-label">Passed</span>
              </div>
              <div className="phase-stat danger">
                <span className="phase-stat-value">{phase.failed}</span>
                <span className="phase-stat-label">Failed</span>
              </div>
            </div>

            <div className="phase-footer">
              <span className={`status-badge ${phase.status.toLowerCase()}`}>
                {phase.status}
              </span>
              <span className="threshold-info">Threshold: {phase.threshold}%</span>
            </div>
          </div>
        ))}
      </div>

      {/* Category Summary Table */}
      <div className="summary-section">
        <h3 className="section-title">Category Summary</h3>
        <table className="summary-table">
          <thead>
            <tr>
              <th>Category</th>
              <th>Phases</th>
              <th>Modules</th>
              <th>Passed</th>
              <th>Score</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(CATEGORY_NAMES).map(([key, name]) => {
              const catPhases = phaseResults.filter(p => p.category === key)
              const catModules = catPhases.reduce((sum, p) => sum + p.modules, 0)
              const catPassed = catPhases.reduce((sum, p) => sum + p.passed, 0)
              const catScore = catModules > 0 ? ((catPassed / catModules) * 100).toFixed(1) : 0
              return (
                <tr key={key}>
                  <td>
                    <span className="category-indicator" style={{ backgroundColor: CATEGORY_COLORS[key] }} />
                    {name}
                  </td>
                  <td>{catPhases.length}</td>
                  <td>{catModules}</td>
                  <td>{catPassed}</td>
                  <td>{catScore}%</td>
                  <td>
                    <span className={`status-badge ${parseFloat(catScore) >= 85 ? 'pass' : 'fail'}`}>
                      {parseFloat(catScore) >= 85 ? 'PASS' : 'REVIEW'}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Sign-off Gate Section */}
      <div className="signoff-section">
        <h3 className="section-title">Sign-off Gate Results</h3>
        <div className="signoff-grid">
          {[
            { name: 'Quality Gate (P1-3)', required: 90, phases: [1,2,3] },
            { name: 'Behavior Gate (P4-7)', required: 85, phases: [4,5,6,7] },
            { name: 'Safety Gate (P8-10)', required: 85, phases: [8,9,10] },
            { name: 'Performance Gate (P11-12)', required: 80, phases: [11,12] },
            { name: 'Compliance Gate (P13)', required: 95, phases: [13] },
            { name: 'Production Gate (P14)', required: 90, phases: [14] },
            { name: 'Value Gate (P15)', required: 80, phases: [15] }
          ].map((gate, i) => {
            const gatePhases = phaseResults.filter(p => gate.phases.includes(p.id))
            const gateModules = gatePhases.reduce((sum, p) => sum + p.modules, 0)
            const gatePassed = gatePhases.reduce((sum, p) => sum + p.passed, 0)
            const gateScore = gateModules > 0 ? ((gatePassed / gateModules) * 100).toFixed(1) : 0
            const passed = parseFloat(gateScore) >= gate.required

            return (
              <div key={i} className={`signoff-card ${passed ? 'approved' : 'blocked'}`}>
                <div className="signoff-header">
                  <span className="signoff-name">{gate.name}</span>
                  <span className={`signoff-status ${passed ? 'approved' : 'blocked'}`}>
                    {passed ? 'APPROVE' : 'BLOCK'}
                  </span>
                </div>
                <div className="signoff-metrics">
                  <div className="signoff-metric">
                    <span className="metric-label">Required</span>
                    <span className="metric-value">{gate.required}%</span>
                  </div>
                  <div className="signoff-metric">
                    <span className="metric-label">Actual</span>
                    <span className="metric-value">{gateScore}%</span>
                  </div>
                </div>
                <div className="signoff-progress">
                  <div
                    className="signoff-progress-fill"
                    style={{
                      width: `${gateScore}%`,
                      backgroundColor: passed ? '#22c55e' : '#ef4444'
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Phase Details Modal */}
      {selectedPhase && (
        <PhaseDetails
          phase={selectedPhase}
          onClose={() => setSelectedPhase(null)}
        />
      )}
    </div>
  )
}

export default MonitoringDashboard
