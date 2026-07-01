import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
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

const TIER_COLORS = {
  production: '#10b981', advanced: '#3b82f6', component: '#8b5cf6',
  baseline: '#f59e0b', experimental: '#94a3b8'
}

const CAT_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981', '#06b6d4', '#ec4899', '#64748b']

export default function FoundationModelsDashboard() {
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
      axios.get(`${API}/foundation-models/overview`),
      axios.get(`${API}/foundation-models/breakdown`),
      axios.get(`${API}/foundation-models/definitions`),
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
    { id: 'catalog', label: 'Model Catalog' },
    { id: 'disease', label: 'Disease Coverage' },
    { id: 'performance', label: 'Performance' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Foundation Models…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const k = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Foundation Models Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Clinical AI model catalog — {k.total_models} models across {k.diseases_covered} diseases
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(overview, k)}
      {tab === 'catalog' && renderCatalog(breakdown)}
      {tab === 'disease' && renderDisease(overview, breakdown)}
      {tab === 'performance' && renderPerformance(breakdown)}
      {tab === 'definitions' && renderDefinitions(definitions)}
    </div>
  )
}

function renderOverview(ov, k) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI Row */}
      <Card><KPI label="Total Models" value={k.total_models} color="#3b82f6" /></Card>
      <Card><KPI label="Production" value={k.production_models} color="#10b981" /></Card>
      <Card><KPI label="Experimental" value={k.experimental_models} color="#8b5cf6" /></Card>
      <Card><KPI label="Architectures" value={k.neural_architectures} color="#f59e0b" /></Card>
      <Card><KPI label="Diseases Covered" value={k.diseases_covered} color="#06b6d4" /></Card>
      <Card><KPI label="Total Size" value={k.total_model_size} color="#64748b" /></Card>
      <Card><KPI label="Predictions" value={k.total_predictions} color="#3b82f6" /></Card>
      <Card><KPI label="Avg Confidence" value={`${(k.avg_prediction_confidence * 100).toFixed(1)}%`} color="#10b981" /></Card>

      {/* Tier Distribution Pie */}
      <Card title="Model Tiers" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={ov.tier_distribution || []} dataKey="count" nameKey="tier" cx="50%" cy="50%"
                 outerRadius={90} label={({ tier, count }) => `${tier}: ${count}`}>
              {(ov.tier_distribution || []).map((d, i) => (
                <Cell key={i} fill={TIER_COLORS[d.tier] || CAT_COLORS[i % CAT_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Algorithm Distribution Bar */}
      <Card title="Top Algorithms" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={(ov.algorithm_distribution || []).slice(0, 10)} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="algorithm" width={130} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Production Models Table */}
      <Card title="Production Models" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Disease</th>
                <th style={{ textAlign: 'left', padding: 8 }}>File</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Size</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Framework</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Modified</th>
              </tr>
            </thead>
            <tbody>
              {(ov.production_models || []).map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8 }}><Badge text={m.disease} color="#10b981" /></td>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{m.filename}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{m.size_human}</td>
                  <td style={{ padding: 8 }}>{m.framework}</td>
                  <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{m.modified ? m.modified.split('T')[0] : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Neural Architectures */}
      {ov.architectures && ov.architectures.length > 0 && (
        <Card title="Neural Architectures" span={4}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {ov.architectures.map((a, i) => (
              <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{a.class_name}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{a.framework}</div>
                <div style={{ fontSize: 12, color: '#475569', marginTop: 6 }}>{a.description}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

function renderCatalog(bd) {
  const models = bd?.all_saved_models || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Model Catalog (${models.length} models)`}>
        <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Disease</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Algorithm</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Tier</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Size</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Acc %</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Framework</th>
                <th style={{ textAlign: 'left', padding: 8 }}>File</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8 }}>{m.disease}</td>
                  <td style={{ padding: 8 }}>{m.algorithm}</td>
                  <td style={{ padding: 8 }}>
                    <Badge text={m.tier} color={TIER_COLORS[m.tier] || '#94a3b8'} />
                  </td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{m.size_human}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>
                    {m.accuracy_pct != null ? m.accuracy_pct : '—'}
                  </td>
                  <td style={{ padding: 8 }}>{m.framework}</td>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 10, color: '#64748b' }}>{m.filename}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderDisease(ov, bd) {
  const summaries = bd?.disease_model_summaries || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Patient Disease Distribution */}
      <Card title="Patient Disease Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={ov.disease_patient_distribution || []} dataKey="count" nameKey="disease"
                 cx="50%" cy="50%" outerRadius={90} label={({ disease, count }) => `${disease}: ${count}`}>
              {(ov.disease_patient_distribution || []).map((d, i) => (
                <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Framework Distribution */}
      <Card title="Framework Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.framework_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="framework" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Per-Disease Model Summaries */}
      <Card title="Disease Model Coverage" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Disease</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Variants</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Production</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Best Acc %</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Total Size</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Algorithms</th>
              </tr>
            </thead>
            <tbody>
              {summaries.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{s.disease}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{s.total_variants}</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>
                    {s.production_model ? <Badge text="Yes" color="#10b981" /> : <Badge text="No" color="#94a3b8" />}
                  </td>
                  <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>
                    {s.best_accuracy_pct != null ? s.best_accuracy_pct : '—'}
                  </td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{s.total_size}</td>
                  <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{(s.algorithms || []).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderPerformance(bd) {
  const topModels = bd?.top_models_by_accuracy || []
  const perfByDisease = bd?.prediction_performance_by_disease || []
  const largest = bd?.largest_models || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Prediction Performance by Disease */}
      <Card title="Prediction Performance by Disease" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={perfByDisease}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="total_predictions" fill="#3b82f6" name="Predictions" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Top Models by Accuracy */}
      <Card title="Top Models by Accuracy" span={1}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Disease</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Algorithm</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Acc %</th>
              </tr>
            </thead>
            <tbody>
              {topModels.slice(0, 15).map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{m.disease}</td>
                  <td style={{ padding: 6 }}>{m.algorithm}</td>
                  <td style={{ padding: 6, textAlign: 'right', fontWeight: 700 }}>{m.accuracy_pct}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Largest Models */}
      <Card title="Largest Models" span={1}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Disease</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Algorithm</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Size</th>
              </tr>
            </thead>
            <tbody>
              {largest.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{m.disease}</td>
                  <td style={{ padding: 6 }}>{m.algorithm}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{m.size_human}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderDefinitions(defs) {
  if (!defs || !defs.sections) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {defs.sections.map((sec, si) => (
        <Card key={si} title={sec.title}>
          <div style={{ display: 'grid', gap: 12 }}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#475569', marginTop: 4, lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
