import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, PieChart, Pie, Cell
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

export default function DataDriftDashboard() {
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
      axios.get(`${API}/data-drift/overview`),
      axios.get(`${API}/data-drift/breakdown`),
      axios.get(`${API}/data-drift/definitions`),
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
    { id: 'features', label: 'Feature Drift' },
    { id: 'history', label: 'History & Alerts' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Data Drift dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No drift data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Data Drift Monitor</h2>
        <Badge text={overview.verdict} color={overview.verdict?.includes('SEVERE') ? '#ef4444' : overview.verdict?.includes('MODERATE') ? '#f59e0b' : '#10b981'} />
        <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>Last run: {overview.run_at}</span>
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
      {tab === 'features' && renderFeatures()}
      {tab === 'history' && renderHistory()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const kpis = overview.kpis || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((k, i) => (
          <Card key={i}><KPI label={k.label} value={k.value} color={k.color} /></Card>
        ))}

        <Card title="Drift Interpretation" span={4}>
          <p style={{ margin: 0, fontSize: 14, color: '#475569', lineHeight: 1.6 }}>{overview.interpretation}</p>
          <div style={{ marginTop: 12, fontSize: 12, color: '#94a3b8' }}>
            Method: {overview.method} | Disease: {overview.disease} | Reference: {overview.n_reference} samples | Live: {overview.n_live} samples
          </div>
        </Card>

        <Card title="Severity Distribution" span={2}>
          {breakdown?.severity_chart && (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={breakdown.severity_chart.filter(s => s.value > 0)} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {breakdown.severity_chart.filter(s => s.value > 0).map((s, i) => (
                    <Cell key={i} fill={s.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Worst Drifted Feature" span={2}>
          <div style={{ textAlign: 'center', padding: 20 }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#ef4444' }}>{overview.worst_feature}</div>
            <div style={{ fontSize: 14, color: '#64748b', marginTop: 8 }}>PSI: {overview.worst_psi?.toFixed(4)}</div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
              Max PSI: {overview.max_psi?.toFixed(4)} | Max KS: {overview.max_ks?.toFixed(4)}
            </div>
          </div>
        </Card>
      </div>
    )
  }

  function renderFeatures() {
    if (!breakdown?.available) return null
    const top10 = breakdown.top10_psi || []
    const allFeatures = breakdown.feature_psi || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Top 10 Features by PSI (${breakdown.n_features_total} total, ${breakdown.n_ks_significant} KS-significant)`} span={1}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={top10} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="feature" width={90} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => v.toFixed(4)} />
              <ReferenceLine x={0.25} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'High', fill: '#ef4444', fontSize: 10 }} />
              <ReferenceLine x={0.1} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: 'Moderate', fill: '#f59e0b', fontSize: 10 }} />
              <Bar dataKey="psi" radius={[0, 4, 4, 0]}>
                {top10.map((f, i) => <Cell key={i} fill={f.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="KS Statistic — Top 10 Features" span={1}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={breakdown.ks_chart || []} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis type="category" dataKey="feature" width={90} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => v.toFixed(4)} />
              <Bar dataKey="ks_stat" radius={[0, 4, 4, 0]}>
                {(breakdown.ks_chart || []).map((f, i) => <Cell key={i} fill={f.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="All Features — Drift Detail" span={1}>
          <div style={{ maxHeight: 400, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Feature</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>PSI</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>KS Stat</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>KS p-value</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px' }}>Severity</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px' }}>KS Sig?</th>
                </tr>
              </thead>
              <tbody>
                {allFeatures.map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px' }}>{f.feature}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{f.psi.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{f.ks_stat.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{f.ks_p < 0.0001 ? '< 0.0001' : f.ks_p.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'center' }}>
                      <Badge text={f.severity} color={f.color} />
                    </td>
                    <td style={{ padding: '6px', textAlign: 'center' }}>
                      {f.ks_significant ? <span style={{ color: '#ef4444', fontWeight: 600 }}>Yes</span> : <span style={{ color: '#10b981' }}>No</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderHistory() {
    const events = breakdown?.event_timeline || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Drift Monitor Events (${events.length} total)`} span={1}>
          {events.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No drift monitoring events recorded yet.</p>
          ) : (
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Timestamp</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Detail</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Drift Fraction</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>High Features</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 12 }}>{ev.ts}</td>
                      <td style={{ padding: '6px' }}>{ev.detail}</td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {ev.frac_drifted != null ? (ev.frac_drifted * 100).toFixed(1) + '%' : '--'}
                      </td>
                      <td style={{ padding: '6px', textAlign: 'right' }}>
                        {ev.n_high != null ? ev.n_high : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title="Threshold Reference" span={1}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, padding: 16 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ width: 16, height: 16, borderRadius: '50%', background: '#ef4444', margin: '0 auto 8px' }} />
              <div style={{ fontSize: 14, fontWeight: 600 }}>High</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>PSI &ge; 0.25</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>Severe distributional shift</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ width: 16, height: 16, borderRadius: '50%', background: '#f59e0b', margin: '0 auto 8px' }} />
              <div style={{ fontSize: 14, fontWeight: 600 }}>Moderate</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>0.1 &le; PSI &lt; 0.25</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>Noticeable shift, monitor</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ width: 16, height: 16, borderRadius: '50%', background: '#10b981', margin: '0 auto 8px' }} />
              <div style={{ fontSize: 14, fontWeight: 600 }}>Low</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>PSI &lt; 0.1</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>Stable, no action</div>
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
