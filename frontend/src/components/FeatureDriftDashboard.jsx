import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

export default function FeatureDriftDashboard() {
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
      axios.get(`${API}/api/feature-drift/overview`),
      axios.get(`${API}/api/feature-drift/breakdown`),
      axios.get(`${API}/api/feature-drift/definitions`),
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
    { id: 'categories', label: 'Category Analysis' },
    { id: 'models', label: 'Model Importance' },
    { id: 'history', label: 'Drift History' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Feature Drift dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No feature drift data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Feature Drift Monitor</h2>
        <Badge
          text={`${overview.n_high_drift} high-drift features`}
          color={overview.n_high_drift > 5 ? '#ef4444' : overview.n_high_drift > 0 ? '#f59e0b' : '#10b981'}
        />
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
      {tab === 'categories' && renderCategories()}
      {tab === 'models' && renderModels()}
      {tab === 'history' && renderHistory()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const kpis = overview.kpis || []
    const catSummary = overview.category_summary || []
    const topDrifted = overview.top_important_drifted || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((k, i) => (
          <Card key={i}><KPI label={k.label} value={k.value} color={k.color} /></Card>
        ))}

        <Card title="Feature Drift Interpretation" span={4}>
          <p style={{ margin: 0, fontSize: 14, color: '#475569', lineHeight: 1.6 }}>{overview.interpretation}</p>
          <div style={{ marginTop: 12, fontSize: 12, color: '#94a3b8' }}>
            Method: {overview.method} | Disease: {overview.disease}
          </div>
        </Card>

        <Card title="Category Drift Summary" span={2}>
          {catSummary.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={catSummary} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <ReferenceLine y={0.25} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'High', fill: '#ef4444', fontSize: 10 }} />
                <ReferenceLine y={0.1} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: 'Mod', fill: '#f59e0b', fontSize: 10 }} />
                <Bar dataKey="avg_psi" name="Avg PSI" radius={[4, 4, 0, 0]}>
                  {catSummary.map((c, i) => <Cell key={i} fill={c.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Top Drifted Features (by PSI)" span={2}>
          {topDrifted.length > 0 ? (
            <div>
              {topDrifted.map((f, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '10px 0', borderBottom: i < topDrifted.length - 1 ? '1px solid #f1f5f9' : 'none'
                }}>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{f.feature}</div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{f.category}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, fontFamily: 'monospace', color: '#ef4444' }}>{f.psi.toFixed(4)}</div>
                    <Badge text={f.severity} color={f.severity === 'high' ? '#ef4444' : f.severity === 'moderate' ? '#f59e0b' : '#10b981'} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No drifted features detected.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderCategories() {
    if (!breakdown?.available) return null
    const catChart = breakdown.category_chart || []
    const featuresByCategory = breakdown.features_by_category || {}
    const featureDetail = breakdown.feature_detail || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Category-Level Drift (${breakdown.n_categories} categories, ${breakdown.n_features} features)`}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={catChart} margin={{ left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
              <ReferenceLine y={0.25} stroke="#ef4444" strokeDasharray="5 5" />
              <ReferenceLine y={0.1} stroke="#f59e0b" strokeDasharray="5 5" />
              <Bar dataKey="avg_psi" name="Avg PSI" radius={[4, 4, 0, 0]}>
                {catChart.map((c, i) => <Cell key={i} fill={c.color} />)}
              </Bar>
              <Bar dataKey="max_psi" name="Max PSI" radius={[4, 4, 0, 0]} fillOpacity={0.4}>
                {catChart.map((c, i) => <Cell key={i} fill={c.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Category Radar — Average KS Statistic">
          {catChart.length >= 3 && (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={catChart}>
                <PolarGrid />
                <PolarAngleAxis dataKey="category" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Radar dataKey="avg_ks" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} name="Avg KS" />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
              </RadarChart>
            </ResponsiveContainer>
          )}
          {catChart.length < 3 && (
            <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>
              Need at least 3 categories for radar chart. Showing {catChart.length}.
            </div>
          )}
        </Card>

        <Card title="All Features — Category-Grouped Detail">
          <div style={{ maxHeight: 500, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Feature</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Category</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>PSI</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>KS Stat</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px' }}>KS p-value</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px' }}>Severity</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px' }}>KS Sig?</th>
                </tr>
              </thead>
              <tbody>
                {featureDetail.map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px' }}>{f.feature}</td>
                    <td style={{ padding: '6px' }}>
                      <Badge text={f.category} color={f.category_color} />
                    </td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{f.psi.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{f.ks_stat.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{f.ks_p < 0.0001 ? '< 0.0001' : f.ks_p.toFixed(4)}</td>
                    <td style={{ padding: '6px', textAlign: 'center' }}>
                      <Badge text={f.severity} color={f.severity_color} />
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

  function renderModels() {
    if (!breakdown?.available) return null
    const models = breakdown.model_comparison || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Cross-Model Feature Importance (${models.length} models)`}>
          {models.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No trained models found with feature importances.</p>
          ) : (
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Model Type</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Features</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Max Importance</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Gini Conc.</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Top-10 Feature Indices</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{m.model_type}</td>
                      <td style={{ padding: '6px', textAlign: 'right' }}>{m.n_features}</td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{m.max_importance.toFixed(6)}</td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>{m.gini_concentration.toFixed(4)}</td>
                      <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 11 }}>{m.top10_indices.join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {models.length > 0 && (
          <Card title="Top-10 Feature Importances — First Model">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart
                data={models[0].top10_indices.map((idx, i) => ({
                  feature: `Feature ${idx}`,
                  importance: models[0].top10_importances[i],
                }))}
                layout="vertical"
                margin={{ left: 80 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" width={70} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(6) : v} />
                <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}

        <Card title="Importance Concentration">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, padding: 16 }}>
            {models.map((m, i) => (
              <div key={i} style={{ textAlign: 'center', padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{m.model_type}</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: '#3b82f6', marginTop: 8 }}>{m.gini_concentration.toFixed(2)}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>Gini Concentration</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{m.n_features} features</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  function renderHistory() {
    const driftEvents = breakdown?.event_timeline || []
    const trainEvents = breakdown?.training_timeline || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Drift Monitor Events (${driftEvents.length} total)`}>
          {driftEvents.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No drift monitoring events recorded yet.</p>
          ) : (
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Timestamp</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Detail</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Drift Fraction</th>
                  </tr>
                </thead>
                <tbody>
                  {driftEvents.map((ev, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 12 }}>{ev.ts}</td>
                      <td style={{ padding: '6px' }}>{ev.detail}</td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {ev.frac_drifted != null ? (ev.frac_drifted * 100).toFixed(1) + '%' : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title={`Training Events (${trainEvents.length} recent)`}>
          {trainEvents.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No training events recorded.</p>
          ) : (
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Timestamp</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Action</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {trainEvents.map((ev, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 12 }}>{ev.ts}</td>
                      <td style={{ padding: '6px' }}><Badge text={ev.action} color="#3b82f6" /></td>
                      <td style={{ padding: '6px', fontSize: 12 }}>{ev.detail}</td>
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
