import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const PIE_COLORS = ['#16a34a', '#3b82f6', '#eab308', '#ef4444', '#8b5cf6', '#ec4899', '#f59e0b', '#06b6d4']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(4)) : String(v)
}

function fmtScore(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toFixed(2) : String(v)
}

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

export default function FeatureEvaluationDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/feature-evaluation/overview`),
          axios.get(`${API_URL}/api/feature-evaluation/breakdown`),
          axios.get(`${API_URL}/api/feature-evaluation/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Feature Evaluation data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'anova', label: 'ANOVA Rankings' },
    { id: 'features', label: 'Feature Table' },
    { id: 'correlations', label: 'Correlations' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const classDist = Object.entries(overview?.class_distribution || {}).map(([name, count]) => ({ name, count }))
  const categoryAvgF = (overview?.category_avg_f_score || []).map(d => ({
    ...d,
    category: d.category || d.name,
    f_score: d.f_score || d.value || 0
  }))
  const anovaChart = (overview?.anova_chart || []).slice(0, 15)

  const featureTable = breakdown?.feature_table || []
  const correlationPairs = breakdown?.correlation_pairs || []

  const categories = defs?.categories || []
  const methods = defs?.methods || []
  const clinicalRelevance = defs?.clinical_relevance || []

  const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
  const thStyle = { textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }
  const tdStyle = { padding: '7px 10px', borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        Feature Evaluation Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        ANOVA-based feature discrimination analysis: F-scores, significance testing, and correlation structure across EEG features
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* --- Overview Tab --- */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Analyses" value={fmt(kpis.total_analyses)} />
              <KPI label="Total Features" value={fmt(kpis.total_features)} />
              <KPI label="Diseases Covered" value={fmt(kpis.diseases_covered)} />
              <KPI label="Top Feature" value={kpis.top_discriminative_feature || '--'} sub="Most discriminative" />
              <KPI label="Mean F-Score" value={fmtScore(kpis.mean_f_score)} color="#2563eb" />
              <KPI label="Above Significance" value={fmt(kpis.features_above_significance)} color="#16a34a" />
            </div>
          </Card>

          <Card title="Class Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={classDist} dataKey="count" nameKey="name" cx="50%" cy="50%"
                     outerRadius={80} label={({ name, count }) => `${name}: ${count}`}>
                  {classDist.map((d, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Category Average F-Score" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={categoryAvgF}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="f_score" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* --- ANOVA Rankings Tab --- */}
      {tab === 'anova' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Top 15 Features by F-Score (ANOVA)">
            <ResponsiveContainer width="100%" height={500}>
              <BarChart data={anovaChart} layout="vertical" margin={{ left: 150 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => [fmtScore(v), 'F-Score']} />
                <Bar dataKey="f_score" radius={[0, 4, 4, 0]}>
                  {anovaChart.map((d, i) => (
                    <Cell key={i} fill={d.significant ? '#16a34a' : '#94a3b8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ marginTop: 12, display: 'flex', gap: 16, fontSize: 12, color: '#64748b' }}>
              <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#16a34a', borderRadius: 2, marginRight: 4, verticalAlign: 'middle' }}></span>Significant (p &lt; 0.05)</span>
              <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#94a3b8', borderRadius: 2, marginRight: 4, verticalAlign: 'middle' }}></span>Not Significant</span>
            </div>
          </Card>
        </div>
      )}

      {/* --- Feature Table Tab --- */}
      {tab === 'features' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="All Features — ANOVA Results">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Rank</th>
                    <th style={thStyle}>Feature</th>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>F-Score</th>
                    <th style={thStyle}>p-value</th>
                    <th style={thStyle}>Significant</th>
                    <th style={thStyle}>Mean Control</th>
                    <th style={thStyle}>Mean Epilepsy</th>
                    <th style={thStyle}>Effect Size</th>
                  </tr>
                </thead>
                <tbody>
                  {[...featureTable].sort((a, b) => (a.rank || 0) - (b.rank || 0)).map((r, i) => (
                    <tr key={i} style={{
                      background: r.significant ? '#f0fdf4' : (i % 2 === 0 ? '#fff' : '#f8fafc')
                    }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{r.rank}</td>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{r.feature}</td>
                      <td style={tdStyle}>{r.category}</td>
                      <td style={{ ...tdStyle, fontWeight: 600, color: '#2563eb' }}>{fmtScore(r.f_score)}</td>
                      <td style={{ ...tdStyle, color: r.p_value < 0.05 ? '#16a34a' : '#64748b' }}>{fmt(r.p_value)}</td>
                      <td style={tdStyle}>
                        {r.significant
                          ? <span style={{ color: '#16a34a', fontWeight: 700, fontSize: 11 }}>YES</span>
                          : <span style={{ color: '#94a3b8', fontSize: 11 }}>No</span>}
                      </td>
                      <td style={tdStyle}>{fmtScore(r.mean_control)}</td>
                      <td style={tdStyle}>{fmtScore(r.mean_epilepsy)}</td>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{fmtScore(r.effect_size)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* --- Correlations Tab --- */}
      {tab === 'correlations' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Top Correlated Feature Pairs">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Feature A</th>
                  <th style={thStyle}>Feature B</th>
                  <th style={thStyle}>Correlation</th>
                </tr>
              </thead>
              <tbody>
                {correlationPairs.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 500 }}>{r.feature_a}</td>
                    <td style={{ ...tdStyle, fontWeight: 500 }}>{r.feature_b}</td>
                    <td style={{
                      ...tdStyle, fontWeight: 700,
                      color: Math.abs(r.correlation) > 0.8 ? '#ef4444' : Math.abs(r.correlation) > 0.5 ? '#f59e0b' : '#16a34a'
                    }}>{fmtScore(r.correlation)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* --- Definitions Tab --- */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {categories.length > 0 && (
            <Card title="Feature Categories">
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {categories.map((c, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{c.name || c.category || (typeof c === 'string' ? c : '--')}</td>
                      <td style={tdStyle}>{c.description || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {methods.length > 0 && (
            <Card title="Statistical Methods">
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Method</th>
                    <th style={thStyle}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {methods.map((m, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{m.name || m.method || (typeof m === 'string' ? m : '--')}</td>
                      <td style={tdStyle}>{m.description || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {clinicalRelevance.length > 0 && (
            <Card title="Clinical Relevance">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {clinicalRelevance.map((item, i) => (
                  <li key={i} style={{ fontSize: 13, marginBottom: 6, color: '#334155' }}>
                    {typeof item === 'string' ? item : item.description || item.text || JSON.stringify(item)}
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* Fallback: render raw definitions if none of the above matched */}
          {categories.length === 0 && methods.length === 0 && clinicalRelevance.length === 0 && defs && (
            <Card title="Definitions">
              {Object.entries(defs).map(([key, val]) => (
                <div key={key} style={{ marginBottom: 16 }}>
                  <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px', textTransform: 'capitalize' }}>
                    {key.replace(/_/g, ' ')}
                  </h4>
                  {Array.isArray(val) ? (
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      {val.map((item, i) => (
                        <li key={i} style={{ fontSize: 13, marginBottom: 4, color: '#334155' }}>
                          {typeof item === 'string' ? item : JSON.stringify(item)}
                        </li>
                      ))}
                    </ul>
                  ) : typeof val === 'object' ? (
                    <pre style={{ fontSize: 12, color: '#475569', background: '#f1f5f9', padding: 12, borderRadius: 8, overflow: 'auto' }}>
                      {JSON.stringify(val, null, 2)}
                    </pre>
                  ) : (
                    <p style={{ fontSize: 13, color: '#334155' }}>{String(val)}</p>
                  )}
                </div>
              ))}
            </Card>
          )}
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 32, textAlign: 'center', color: '#94a3b8', fontSize: 11 }}>
        Feature Evaluation Dashboard — ANOVA Feature Discrimination Analysis
      </div>
    </div>
  )
}
