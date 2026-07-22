import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const OK_COLOR = '#10b981'
const WARN_COLOR = '#f59e0b'
const ERR_COLOR = '#ef4444'

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

function pct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

function fmt(v, digits = 2) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(digits)) : String(v)
}

export default function MLOpsDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API}/api/mlops/overview`),
      axios.get(`${API}/api/mlops/breakdown`),
      axios.get(`${API}/api/mlops/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'experiments', label: 'Experiments & CV' },
    { id: 'features', label: 'Features & Models' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading MLOps...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const kpis = overview?.kpis || {}
  const trainingHistory = overview?.training_history || []
  const cvSummary = overview?.cv_summary || {}
  const evalStrategies = overview?.evaluation_strategies || []
  const multiDisease = breakdown?.multi_disease_accuracy || []
  const cvDetail = breakdown?.cross_validation_detail || {}
  const pipelineEvents = breakdown?.training_pipeline_events || []
  const features = breakdown?.feature_inventory || []
  const modelFiles = breakdown?.model_file_inventory || []
  const dailyActivity = breakdown?.daily_training_activity || []
  const defs = definitions || {}

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI Cards */}
      <Card>
        <KPI label="Training Runs" value={kpis.total_experiments || 0} sub="dated reports" color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Disease Models" value={kpis.total_models || 0} sub="trained classifiers" color="#10b981" />
      </Card>
      <Card>
        <KPI label="Mean Accuracy" value={pct(kpis.mean_accuracy)} sub={kpis.mean_accuracy_note || 'in-sample (optimistic)'} color={kpis.mean_accuracy >= 0.9 ? OK_COLOR : WARN_COLOR} />
      </Card>
      <Card>
        <KPI label="Patient-Specific Acc" value={pct(cvSummary.mean_accuracy)} sub="honest CV" color="#8b5cf6" />
      </Card>
      <Card>
        <KPI label="Mean Sensitivity" value={pct(cvSummary.mean_sensitivity)} sub="seizure detection" color="#06b6d4" />
      </Card>
      <Card>
        <KPI label="CV Subjects" value={kpis.total_cv_subjects || 0} sub="CHB-MIT patients" color="#64748b" />
      </Card>
      <Card>
        <KPI label="Best Model" value={kpis.best_model?.disease || kpis.best_model || '--'} sub={pct(kpis.best_model?.accuracy ?? kpis.best_accuracy)} color={OK_COLOR} />
      </Card>
      <Card>
        <KPI label="Pipeline Health" value={kpis.pipeline_health?.status || kpis.pipeline_health || '--'} color={(kpis.pipeline_health?.status || kpis.pipeline_health) === 'healthy' ? OK_COLOR : WARN_COLOR} />
      </Card>

      {/* Evaluation Strategies Comparison */}
      <Card title="Evaluation Strategies — Accuracy Comparison" span={2}>
        {evalStrategies.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={evalStrategies} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} tickFormatter={v => pct(v)} />
              <YAxis type="category" dataKey="strategy" width={180} tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => pct(v)} />
              <Bar dataKey="mean_accuracy" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8' }}>No evaluation data</div>}
      </Card>

      {/* Training Run History */}
      <Card title="Training Run History" span={2}>
        {trainingHistory.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Date</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Scripts</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Status</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px' }}>Duration</th>
                </tr>
              </thead>
              <tbody>
                {trainingHistory.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px' }}>{r.date}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{r.scripts_run}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>
                      <Badge text={r.all_ok ? 'PASS' : 'FAIL'} color={r.all_ok ? OK_COLOR : ERR_COLOR} />
                    </td>
                    <td style={{ textAlign: 'right', padding: '6px 8px' }}>{fmt(r.total_seconds, 0)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No training history</div>}
      </Card>

      {/* Per-Subject CV Results */}
      <Card title="Patient-Specific CV — Per Subject (CHB-MIT)" span={4}>
        {(cvSummary.per_subject || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={cvSummary.per_subject}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="subject" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tickFormatter={v => pct(v)} />
              <Tooltip formatter={v => pct(v)} />
              <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" radius={[4, 4, 0, 0]} />
              <Bar dataKey="sensitivity" fill="#8b5cf6" name="Sensitivity" radius={[4, 4, 0, 0]} />
              <Bar dataKey="f1" fill="#06b6d4" name="F1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8' }}>No CV data</div>}
      </Card>
    </div>
  )

  const renderExperiments = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Multi-Disease Accuracy */}
      <Card title="Multi-Disease Model Accuracy (in-sample, optimistic)" span={2}>
        {multiDisease.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Disease</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Samples</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Accuracy</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Precision</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Recall</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>F1</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Eval Type</th>
                </tr>
              </thead>
              <tbody>
                {multiDisease.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600, textTransform: 'capitalize' }}>{m.disease}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{m.n_samples}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{pct(m.accuracy)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{pct(m.precision)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{pct(m.recall)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{pct(m.f1)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>
                      <Badge text="in-sample" color={WARN_COLOR} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No multi-disease data</div>}
      </Card>

      {/* Cross-Validation Detail by Strategy */}
      {Object.entries(cvDetail).map(([strategy, data]) => (
        <Card key={strategy} title={`CV: ${strategy.replace(/_/g, ' ')}`}>
          <div style={{ marginBottom: 8, fontSize: 12, color: '#64748b' }}>
            Mean accuracy: <strong>{pct(data.mean_accuracy)}</strong>
            {data.method && <span> — {data.method}</span>}
          </div>
          {(data.folds || data.per_subject || []).length > 0 && (
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={data.folds || data.per_subject}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={data.folds ? 'held_out' : 'subject'} tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => pct(v)} />
                <Tooltip formatter={v => pct(v)} />
                <Bar dataKey="accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      ))}

      {/* Daily Training Activity */}
      <Card title="Daily Training Pipeline Activity" span={2}>
        {dailyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={dailyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Events" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8' }}>No daily activity data</div>}
      </Card>

      {/* Pipeline Events */}
      <Card title="Recent Pipeline Events" span={2}>
        {pipelineEvents.length > 0 ? (
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Timestamp</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Component</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Action</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Details</th>
                </tr>
              </thead>
              <tbody>
                {pipelineEvents.map((e, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', whiteSpace: 'nowrap' }}>{e.timestamp}</td>
                    <td style={{ padding: '6px 8px' }}>
                      <Badge text={e.component} color="#3b82f6" />
                    </td>
                    <td style={{ padding: '6px 8px' }}>{e.action}</td>
                    <td style={{ padding: '6px 8px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.details}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No pipeline events</div>}
      </Card>
    </div>
  )

  const renderFeatures = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Feature Inventory */}
      <Card title="EEG Feature Inventory (15 features)">
        {features.length > 0 ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {features.map((f, i) => (
              <span key={i} style={{
                display: 'inline-block', padding: '4px 10px', borderRadius: 8,
                fontSize: 11, fontWeight: 500, background: COLORS[i % COLORS.length] + '18',
                color: COLORS[i % COLORS.length], border: `1px solid ${COLORS[i % COLORS.length]}30`
              }}>{f}</span>
            ))}
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No feature data</div>}
      </Card>

      {/* Model File Inventory */}
      <Card title="Model File Inventory">
        {modelFiles.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>File</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px' }}>Size (MB)</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Modified</th>
                </tr>
              </thead>
              <tbody>
                {modelFiles.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>{m.file}</td>
                    <td style={{ textAlign: 'right', padding: '6px 8px' }}>{fmt(m.size_mb)}</td>
                    <td style={{ padding: '6px 8px' }}>{m.modified}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No model files</div>}
      </Card>

      {/* Model Size Comparison Chart */}
      <Card title="Model Size Comparison" span={2}>
        {modelFiles.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={modelFiles}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
              <YAxis label={{ value: 'MB', angle: -90, position: 'insideLeft', fontSize: 11 }} />
              <Tooltip formatter={v => `${fmt(v)} MB`} />
              <Bar dataKey="size_mb" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Size (MB)">
                {modelFiles.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8' }}>No model files</div>}
      </Card>
    </div>
  )

  const renderDefinitions = () => {
    const sections = defs.sections || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {sections.map((sec, i) => (
          <Card key={i} title={sec.name}>
            {sec.description && (
              <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: '0 0 10px' }}>{sec.description}</p>
            )}
            {sec.fields && (
              <ul style={{ fontSize: 13, color: '#475569', lineHeight: 1.8, margin: 0, paddingLeft: 20 }}>
                {sec.fields.map((f, j) => (
                  <li key={j}><strong>{f.name}:</strong> {f.description}</li>
                ))}
              </ul>
            )}
            {sec.items && (
              <ul style={{ fontSize: 13, color: '#475569', lineHeight: 1.8, margin: 0, paddingLeft: 20 }}>
                {sec.items.map((item, j) => (
                  <li key={j}>{typeof item === 'string' ? item : <><strong>{item.name || item.term}:</strong> {item.description || item.definition}</>}</li>
                ))}
              </ul>
            )}
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>MLOps Dashboard</h2>
        <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>
          Training pipeline monitoring, experiment tracking, feature store, cross-validation metrics
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'experiments' && renderExperiments()}
      {tab === 'features' && renderFeatures()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
