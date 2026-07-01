import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, ReferenceLine, Legend
} from 'recharts'

const API = '/api'

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

export default function ModelDriftDashboard() {
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
      axios.get(`${API}/model-drift/overview`),
      axios.get(`${API}/model-drift/breakdown`),
      axios.get(`${API}/model-drift/definitions`),
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
    { id: 'performance', label: 'Performance & CV' },
    { id: 'history', label: 'Training History' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Model Drift dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No model drift data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Model Drift Monitor</h2>
        <Badge text={overview.performance_verdict} color={
          overview.performance_verdict === 'STABLE' ? '#10b981' :
          overview.performance_verdict === 'IMPROVED' ? '#3b82f6' : '#ef4444'
        } />
        <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>Last training: {overview.run_at}</span>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'performance' && renderPerformance()}
      {tab === 'history' && renderHistory()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const kpis = overview.kpis || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((k, i) => (
          <Card key={i}><KPI label={k.label} value={k.value} sub={k.sub} color={k.color} /></Card>
        ))}

        <Card title="Per-Subject Accuracy" span={2}>
          {breakdown?.per_subject && (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={breakdown.per_subject} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subject" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} />
                <Tooltip formatter={(v) => (v * 100).toFixed(1) + '%'} />
                <Legend />
                <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="sensitivity" name="Sensitivity" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Evaluation Strategies Comparison" span={2}>
          {breakdown?.evaluation_strategies && (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={breakdown.evaluation_strategies} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} />
                <YAxis type="category" dataKey="method" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => (v * 100).toFixed(1) + '%'} />
                <ReferenceLine x={0.95} stroke="#10b981" strokeDasharray="5 5" label={{ value: '95%', fill: '#10b981', fontSize: 10 }} />
                <ReferenceLine x={0.80} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: '80%', fill: '#f59e0b', fontSize: 10 }} />
                <Bar dataKey="accuracy" name="Accuracy" radius={[0, 4, 4, 0]}>
                  {(breakdown.evaluation_strategies || []).map((s, i) => (
                    <React.Fragment key={i}>
                      {/* color by accuracy level */}
                    </React.Fragment>
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Bootstrap Confidence Intervals" span={2}>
          {overview.bootstrap_ci && (
            <div style={{ padding: 12 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Metric</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Mean</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>95% CI Low</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>95% CI High</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.bootstrap_ci || []).map((ci, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{ci.metric}</td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{(ci.mean * 100).toFixed(2)}%</td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{(ci.ci95_low * 100).toFixed(2)}%</td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{(ci.ci95_high * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title="External Validation (Bonn)" span={2}>
          <div style={{ textAlign: 'center', padding: 20 }}>
            <div style={{ fontSize: 36, fontWeight: 700, color: '#10b981' }}>
              {overview.bonn_external_accuracy != null ? (overview.bonn_external_accuracy * 100).toFixed(0) + '%' : 'N/A'}
            </div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 6 }}>Bonn University EEG dataset — independent validation</div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>200 samples, stratified 5-fold CV, RF + ensemble</div>
          </div>
        </Card>
      </div>
    )
  }

  function renderPerformance() {
    if (!breakdown?.available) return null
    const subjects = breakdown.per_subject || []
    const strategies = breakdown.evaluation_strategies || []
    const folds = breakdown.cross_validation_folds || {}
    const comparison = breakdown.bootstrap_comparison || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Per-Subject Performance (Patient-Specific Temporal Split)">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px' }}>Subject</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>Total Windows</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>Seizure Windows</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>Accuracy</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>Sensitivity</th>
                <th style={{ textAlign: 'right', padding: '8px 6px' }}>F1</th>
              </tr>
            </thead>
            <tbody>
              {subjects.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontWeight: 600 }}>{s.subject}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{s.n_total}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{s.n_seizure}</td>
                  <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{(s.accuracy * 100).toFixed(1)}%</td>
                  <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{(s.sensitivity * 100).toFixed(1)}%</td>
                  <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{(s.f1 * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        {Object.keys(folds).map(strategy => (
          <Card key={strategy} title={`Cross-Validation Folds — ${strategy}`}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={folds[strategy]} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="held_out" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} />
                <Tooltip formatter={(v) => (v * 100).toFixed(1) + '%'} />
                <Legend />
                <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        ))}

        {comparison.length > 0 && (
          <Card title="Literature Comparison">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Method</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Setting</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>Reported</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Source</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{c.method}</td>
                    <td style={{ padding: '6px' }}>{c.setting}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{c.reported}</td>
                    <td style={{ padding: '6px', fontSize: 12, color: '#64748b' }}>{c.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>
    )
  }

  function renderHistory() {
    const timeline = breakdown?.training_timeline || []
    const events = breakdown?.training_events || []
    const models = breakdown?.model_inventory || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Training Run Timeline">
          {timeline.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={timeline} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 2]} allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="success_count" name="Successful Runs" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="total_count" name="Total Runs" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No training timeline data.</p>
          )}
        </Card>

        <Card title={`Training Events (${events.length} total)`}>
          {events.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No training events recorded.</p>
          ) : (
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Timestamp</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Detail</th>
                    <th style={{ textAlign: 'center', padding: '8px 6px' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev, i) => {
                    const ok = ev.detail?.includes('succeeded')
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 12 }}>{ev.ts}</td>
                        <td style={{ padding: '6px' }}>{ev.detail}</td>
                        <td style={{ padding: '6px', textAlign: 'center' }}>
                          <Badge text={ok ? 'OK' : 'FAIL'} color={ok ? '#10b981' : '#ef4444'} />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title={`Model Inventory (${models.length} models)`}>
          {models.length > 0 && (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Model</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>Size (MB)</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Last Modified</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{m.size_mb?.toFixed(2)}</td>
                    <td style={{ padding: '6px', fontSize: 12, color: '#64748b' }}>{m.modified}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    if (!definitions?.sections) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {definitions.sections.map((sec, i) => (
          <Card key={i} title={sec.title}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <tbody>
                {sec.items.map((item, j) => (
                  <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600, width: '25%', verticalAlign: 'top', color: '#334155' }}>{item.term}</td>
                    <td style={{ padding: '8px 6px', color: '#475569', lineHeight: 1.5 }}>{item.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        ))}
      </div>
    )
  }
}
