import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, LineChart, Line, PieChart, Pie, Cell
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

export default function OutputDriftDashboard() {
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
      axios.get(`${API}/output-drift/overview`),
      axios.get(`${API}/output-drift/breakdown`),
      axios.get(`${API}/output-drift/definitions`),
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
    { id: 'outputs', label: 'Output Analysis' },
    { id: 'rag', label: 'RAG Pipeline' },
    { id: 'correlation', label: 'Correlation' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Output/RAG Drift dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No output drift data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Output / RAG Drift Monitor</h2>
        <Badge text={overview.verdict} color={overview.verdict === 'STABLE' ? '#10b981' : overview.verdict?.includes('MODERATE') ? '#f59e0b' : '#ef4444'} />
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
      {tab === 'outputs' && renderOutputs()}
      {tab === 'rag' && renderRAG()}
      {tab === 'correlation' && renderCorrelation()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const kpis = overview.kpis || []
    const cd = overview.confidence_drift || {}
    const ld = overview.label_drift || {}
    const rq = overview.rag_quality || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((k, i) => (
          <Card key={i}><KPI label={k.label} value={k.value} color={k.color} /></Card>
        ))}

        <Card title="Confidence Drift" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, padding: 8 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Reference Mean</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{cd.ref_mean?.toFixed(4)}</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>{cd.ref_n} samples</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Live Mean</div>
              <div style={{ fontSize: 22, fontWeight: 700 }}>{cd.live_mean?.toFixed(4)}</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>{cd.live_n} samples</div>
            </div>
          </div>
          <div style={{ textAlign: 'center', marginTop: 8 }}>
            <Badge text={`Shift: ${cd.shift?.toFixed(4)} (${cd.severity})`} color={cd.severity === 'high' ? '#ef4444' : cd.severity === 'moderate' ? '#f59e0b' : '#10b981'} />
          </div>
        </Card>

        <Card title="Label Distribution Drift" span={1}>
          <div style={{ textAlign: 'center', padding: 12 }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: ld.severity === 'high' ? '#ef4444' : ld.severity === 'moderate' ? '#f59e0b' : '#10b981' }}>
              {ld.jsd?.toFixed(4)}
            </div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Jensen-Shannon Divergence</div>
            <Badge text={ld.severity} color={ld.severity === 'high' ? '#ef4444' : ld.severity === 'moderate' ? '#f59e0b' : '#10b981'} />
          </div>
        </Card>

        <Card title="RAG Pipeline Health" span={1}>
          <div style={{ textAlign: 'center', padding: 12 }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: rq.severity === 'high' ? '#ef4444' : rq.severity === 'moderate' ? '#f59e0b' : '#10b981' }}>
              {(rq.success_rate * 100)?.toFixed(1)}%
            </div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Success Rate</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              {rq.n_queries} queries | {rq.n_blocked} blocked
            </div>
          </div>
        </Card>

        {overview.input_drift_correlation && (
          <Card title="Input-Output Drift Correlation" span={4}>
            <p style={{ margin: 0, fontSize: 14, color: '#475569', lineHeight: 1.6 }}>
              {overview.input_drift_correlation.note}
              {overview.input_drift_correlation.input_frac_drifted != null && (
                <span style={{ marginLeft: 8, fontSize: 12, color: '#94a3b8' }}>
                  (Input drift fraction: {(overview.input_drift_correlation.input_frac_drifted * 100).toFixed(1)}%)
                </span>
              )}
            </p>
          </Card>
        )}
      </div>
    )
  }

  function renderOutputs() {
    if (!breakdown?.available) return null
    const timeline = breakdown.conf_timeline || []
    const histogram = breakdown.conf_histogram || []
    const comparison = breakdown.period_comparison || []
    const patients = breakdown.patient_summary || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Confidence Timeline (${breakdown.n_analyses} analyses)`}>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={timeline} margin={{ left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="ts" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
              <YAxis domain={[0.5, 0.7]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => v.toFixed(4)} labelFormatter={l => `Time: ${l}`} />
              <ReferenceLine y={breakdown.ref_stats?.mean} stroke="#3b82f6" strokeDasharray="5 5" label={{ value: `Ref mean: ${breakdown.ref_stats?.mean?.toFixed(3)}`, fill: '#3b82f6', fontSize: 10 }} />
              <Line type="monotone" dataKey="confidence" stroke="#8b5cf6" dot={{ r: 3 }} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Confidence Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={histogram}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Reference vs Live Period">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={comparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                <YAxis domain={[0.5, 0.7]} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => v.toFixed(4)} />
                <Bar dataKey="mean_confidence" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                  {comparison.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <Card title="Per-Patient Confidence Summary">
          <div style={{ maxHeight: 350, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Patient</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>Mean Conf.</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>Min</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>Max</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>Analyses</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontFamily: 'monospace' }}>{p.patient}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{p.mean.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{p.min.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{p.max.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'right' }}>{p.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderRAG() {
    if (!breakdown?.available) return null
    const ragTimeline = breakdown.rag_timeline || []
    const ragComponents = breakdown.rag_component_chart || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="RAG Pipeline — Component Breakdown">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={ragComponents} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="component" width={90} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="total" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title={`RAG Event Log (${ragTimeline.length} events)`}>
          {ragTimeline.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No RAG events recorded.</p>
          ) : (
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Timestamp</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Component</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Action</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Detail</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Actor</th>
                  </tr>
                </thead>
                <tbody>
                  {ragTimeline.map((ev, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 11 }}>{ev.ts}</td>
                      <td style={{ padding: '6px' }}>
                        <Badge text={ev.component} color={ev.component === 'council' ? '#8b5cf6' : ev.component === 'patient_chat' ? '#3b82f6' : '#06b6d4'} />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <Badge text={ev.action} color={ev.action === 'blocked' ? '#ef4444' : '#10b981'} />
                      </td>
                      <td style={{ padding: '6px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ev.detail}</td>
                      <td style={{ padding: '6px', fontSize: 12, color: '#64748b' }}>{ev.actor}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    )
  }

  function renderCorrelation() {
    const corr = breakdown?.input_output_correlation
    const confDrift = overview.confidence_drift || {}
    const ragQuality = overview.rag_quality || {}

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Input-Output Drift Correlation" span={1}>
          {corr ? (
            <div style={{ padding: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, marginBottom: 20 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Input Drift Fraction</div>
                  <div style={{ fontSize: 28, fontWeight: 700, color: corr.input_drift_frac > 0.5 ? '#ef4444' : '#f59e0b' }}>
                    {(corr.input_drift_frac * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Output Confidence Shift</div>
                  <div style={{ fontSize: 28, fontWeight: 700, color: confDrift.severity === 'high' ? '#ef4444' : confDrift.severity === 'moderate' ? '#f59e0b' : '#10b981' }}>
                    {corr.output_conf_shift?.toFixed(4)}
                  </div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>RAG Success Rate</div>
                  <div style={{ fontSize: 28, fontWeight: 700, color: ragQuality.severity === 'high' ? '#ef4444' : '#10b981' }}>
                    {(ragQuality.success_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <p style={{ margin: 0, fontSize: 14, color: '#475569', lineHeight: 1.6, background: '#f8fafc', padding: 16, borderRadius: 8 }}>
                {corr.correlation_note}
              </p>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No input drift data available for correlation analysis. Run the Data Drift monitor first.</p>
          )}
        </Card>

        <Card title="Drift Severity Summary">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, padding: 16 }}>
            <div style={{ textAlign: 'center', padding: 16, borderRadius: 8, background: '#f8fafc' }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Confidence Drift</div>
              <Badge text={confDrift.severity || 'unknown'} color={confDrift.severity === 'high' ? '#ef4444' : confDrift.severity === 'moderate' ? '#f59e0b' : '#10b981'} />
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>Shift: {confDrift.shift?.toFixed(4)}</div>
            </div>
            <div style={{ textAlign: 'center', padding: 16, borderRadius: 8, background: '#f8fafc' }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Label Drift</div>
              <Badge text={overview.label_drift?.severity || 'unknown'} color={overview.label_drift?.severity === 'high' ? '#ef4444' : overview.label_drift?.severity === 'moderate' ? '#f59e0b' : '#10b981'} />
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>JSD: {overview.label_drift?.jsd?.toFixed(4)}</div>
            </div>
            <div style={{ textAlign: 'center', padding: 16, borderRadius: 8, background: '#f8fafc' }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>RAG Quality</div>
              <Badge text={ragQuality.severity || 'unknown'} color={ragQuality.severity === 'high' ? '#ef4444' : ragQuality.severity === 'moderate' ? '#f59e0b' : '#10b981'} />
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>Success: {(ragQuality.success_rate * 100).toFixed(1)}%</div>
            </div>
          </div>
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
