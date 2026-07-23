import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function QualityBadge({ quality }) {
  const q = String(quality || '')
  const color = q === 'Excellent' ? '#10b981' : q === 'Good' ? '#06b6d4' : q === 'Fair' ? '#f59e0b' : '#ef4444'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{q || '--'}</span>
  )
}

function ConfidenceBadge({ confidence }) {
  const v = Number(confidence) || 0
  const color = v >= 0.8 ? '#10b981' : v >= 0.6 ? '#f59e0b' : '#ef4444'
  return (
    <span style={{ fontWeight: 600, color }}>{(v * 100).toFixed(0)}%</span>
  )
}

function DiseaseBadge({ disease }) {
  const d = String(disease || '')
  const colorMap = { epilepsy: '#ef4444', depression: '#8b5cf6', parkinsons: '#3b82f6', alzheimers: '#f59e0b', sleep_disorder: '#06b6d4' }
  const color = colorMap[d] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{d.replace(/_/g, ' ') || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'analyses', label: 'All Analyses' },
  { id: 'patients', label: 'By Patient' },
  { id: 'diseases', label: 'By Disease' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EEGAnalysisResultsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/eeg-analysis-results/overview`),
      axios.get(`${API_URL}/api/eeg-analysis-results/breakdown`),
      axios.get(`${API_URL}/api/eeg-analysis-results/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EEG analysis results…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>EEG Analysis Results Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        AI-powered EEG classification results — {fmt(k.total_analyses)} analyses, {fmt(k.total_patients)} patients, {fmt(k.diseases_covered)} diseases
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'analyses' && <AnalysesTab analyses={breakdown?.analyses || []} />}
      {tab === 'patients' && <PatientsTab patients={breakdown?.by_patient || []} />}
      {tab === 'diseases' && <DiseasesTab diseases={breakdown?.by_disease || []} confidence={overview?.disease_confidence || []} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const k = data.kpis || {}
  const diseaseData = (data.disease_dist || []).map(d => ({ name: d.disease.replace(/_/g, ' '), value: d.count }))
  const labelData = (data.label_dist || []).map(d => ({ name: d.label, value: d.count }))
  const qualityData = (data.quality_dist || []).map(d => ({ name: d.quality, value: d.count }))
  const tierData = (data.confidence_tiers || []).map(d => ({ name: d.tier, value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 12 }}>
          <KPI label="Total Analyses" value={k.total_analyses} color="#3b82f6" />
          <KPI label="Patients" value={k.total_patients} color="#8b5cf6" />
          <KPI label="Diseases Covered" value={k.diseases_covered} color="#06b6d4" />
          <KPI label="Avg Confidence" value={`${((k.avg_confidence || 0) * 100).toFixed(0)}%`} color="#10b981" />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          <KPI label="High Confidence (≥80%)" value={k.high_confidence_count} color="#10b981" />
          <KPI label="Low Confidence (<50%)" value={k.low_confidence_count} color="#ef4444" />
          <KPI label="Excellent Quality" value={`${k.signal_quality_excellent_pct}%`} color="#10b981" />
          <KPI label="Poor Quality" value={`${k.signal_quality_poor_pct}%`} color="#ef4444" />
        </div>
      </Card>

      <Card title="Disease Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={diseaseData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
              {diseaseData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Prediction Labels">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={labelData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={60} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Signal Quality">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={qualityData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
              {qualityData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Confidence Tiers">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={tierData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Confidence by Disease" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={(data.disease_confidence || []).map(d => ({ ...d, disease: d.disease.replace(/_/g, ' ') }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
            <Legend />
            <Bar dataKey="avg_confidence" name="Avg" fill="#3b82f6" radius={[6, 6, 0, 0]} />
            <Bar dataKey="max_confidence" name="Max" fill="#10b981" radius={[6, 6, 0, 0]} />
            <Bar dataKey="min_confidence" name="Min" fill="#ef4444" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Analysis Trend" span={3}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.monthly_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="left" allowDecimals={false} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="analyses" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Analyses" />
            <Line yAxisId="right" type="monotone" dataKey="avg_confidence" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Avg Confidence" />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function AnalysesTab({ analyses }) {
  const [sort, setSort] = useState({ key: 'created_at', dir: 'desc' })
  const [filter, setFilter] = useState('')
  const filtered = analyses.filter(a =>
    !filter || a.disease === filter || a.signal_quality === filter
  )
  const sorted = [...filtered].sort((a, b) => {
    const av = a[sort.key] ?? '', bv = b[sort.key] ?? ''
    const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
    return sort.dir === 'asc' ? cmp : -cmp
  })
  const toggle = k => setSort(prev => ({ key: k, dir: prev.key === k && prev.dir === 'asc' ? 'desc' : 'asc' }))
  const th = (k, label) => (
    <th onClick={() => toggle(k)} style={{ padding: '8px 10px', cursor: 'pointer', fontSize: 12, color: '#475569', textAlign: 'left', whiteSpace: 'nowrap', borderBottom: '1px solid #e2e8f0' }}>
      {label} {sort.key === k ? (sort.dir === 'asc' ? '▲' : '▼') : ''}
    </th>
  )

  const diseases = [...new Set(analyses.map(a => a.disease))].sort()
  const qualities = [...new Set(analyses.map(a => a.signal_quality))].sort()

  return (
    <Card title={`All Analyses (${filtered.length})`}>
      <div style={{ marginBottom: 12, display: 'flex', gap: 8 }}>
        <select value={filter} onChange={e => setFilter(e.target.value)} style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 13 }}>
          <option value="">All</option>
          <optgroup label="Disease">
            {diseases.map(d => <option key={d} value={d}>{d.replace(/_/g, ' ')}</option>)}
          </optgroup>
          <optgroup label="Signal Quality">
            {qualities.map(q => <option key={q} value={q}>{q}</option>)}
          </optgroup>
        </select>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr>{th('id', 'ID')}{th('patient_id', 'Patient')}{th('disease', 'Disease')}{th('predicted_label', 'Prediction')}{th('confidence', 'Confidence')}{th('signal_quality', 'Signal Quality')}{th('created_at', 'Date')}</tr>
          </thead>
          <tbody>
            {sorted.map((a, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.id}</td>
                <td style={{ padding: '6px 10px' }}>{a.patient_id}</td>
                <td style={{ padding: '6px 10px' }}><DiseaseBadge disease={a.disease} /></td>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.predicted_label}</td>
                <td style={{ padding: '6px 10px' }}><ConfidenceBadge confidence={a.confidence} /></td>
                <td style={{ padding: '6px 10px' }}><QualityBadge quality={a.signal_quality} /></td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{(a.created_at || '').slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function PatientsTab({ patients }) {
  const [sort, setSort] = useState({ key: 'analyses', dir: 'desc' })
  const sorted = [...patients].sort((a, b) => {
    const av = a[sort.key] ?? '', bv = b[sort.key] ?? ''
    const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
    return sort.dir === 'asc' ? cmp : -cmp
  })
  const toggle = k => setSort(prev => ({ key: k, dir: prev.key === k && prev.dir === 'asc' ? 'desc' : 'asc' }))
  const th = (k, label) => (
    <th onClick={() => toggle(k)} style={{ padding: '8px 10px', cursor: 'pointer', fontSize: 12, color: '#475569', textAlign: 'left', whiteSpace: 'nowrap', borderBottom: '1px solid #e2e8f0' }}>
      {label} {sort.key === k ? (sort.dir === 'asc' ? '▲' : '▼') : ''}
    </th>
  )
  return (
    <Card title={`Patient Analysis Summary (${patients.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr>{th('patient_id', 'Patient')}{th('analyses', 'Analyses')}{th('diseases', 'Diseases')}{th('avg_confidence', 'Avg Confidence')}{th('last_analysis', 'Last Analysis')}</tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                <td style={{ padding: '6px 10px' }}>{p.analyses}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{p.diseases}</td>
                <td style={{ padding: '6px 10px' }}><ConfidenceBadge confidence={p.avg_confidence} /></td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{(p.last_analysis || '').slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function DiseasesTab({ diseases, confidence }) {
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title="Disease Breakdown">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr>
                <th style={{ padding: '8px 10px', fontSize: 12, color: '#475569', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Disease</th>
                <th style={{ padding: '8px 10px', fontSize: 12, color: '#475569', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Total</th>
                <th style={{ padding: '8px 10px', fontSize: 12, color: '#475569', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Labels</th>
                <th style={{ padding: '8px 10px', fontSize: 12, color: '#475569', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Avg Confidence</th>
                <th style={{ padding: '8px 10px', fontSize: 12, color: '#475569', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Latest</th>
              </tr>
            </thead>
            <tbody>
              {diseases.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px' }}><DiseaseBadge disease={d.disease} /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.total}</td>
                  <td style={{ padding: '6px 10px', fontSize: 12 }}>{d.labels}</td>
                  <td style={{ padding: '6px 10px' }}><ConfidenceBadge confidence={d.avg_confidence} /></td>
                  <td style={{ padding: '6px 10px', fontSize: 12 }}>{(d.latest || '').slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Confidence Comparison by Disease">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={(confidence || []).map(d => ({ ...d, disease: d.disease.replace(/_/g, ' ') }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
            <Legend />
            <Bar dataKey="avg_confidence" name="Avg Confidence" fill="#3b82f6" radius={[6, 6, 0, 0]} />
            <Bar dataKey="max_confidence" name="Max" fill="#10b981" radius={[6, 6, 0, 0]} />
            <Bar dataKey="min_confidence" name="Min" fill="#ef4444" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  if (!data) return null
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title="Field Definitions">
        <div style={{ display: 'grid', gap: 8 }}>
          {Object.entries(data.fields || {}).map(([k, v]) => (
            <div key={k}><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 4, fontSize: 12 }}>{k}</code> <span style={{ color: '#475569', marginLeft: 4 }}>{v}</span></div>
          ))}
        </div>
      </Card>

      <Card title="Diseases">
        <div style={{ display: 'grid', gap: 8 }}>
          {(data.diseases || []).map((d, i) => (
            <div key={i}><strong style={{ color: '#1e293b', textTransform: 'capitalize' }}>{(d.name || '').replace(/_/g, ' ')}:</strong> <span style={{ color: '#475569' }}>{d.description}</span></div>
          ))}
        </div>
      </Card>

      <Card title="Signal Quality Levels">
        <div style={{ display: 'grid', gap: 8 }}>
          {Object.entries(data.signal_quality_levels || {}).map(([k, v]) => (
            <div key={k}><QualityBadge quality={k} /> <span style={{ color: '#475569', marginLeft: 8 }}>{v}</span></div>
          ))}
        </div>
      </Card>

      <Card title="Confidence Score Interpretation">
        <div style={{ display: 'grid', gap: 8 }}>
          {Object.entries(data.confidence_interpretation || {}).map(([k, v]) => (
            <div key={k}><strong style={{ color: '#1e293b' }}>{k}:</strong> <span style={{ color: '#475569' }}>{v}</span></div>
          ))}
        </div>
      </Card>

      <Card title="Data Source">
        <div><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 4, fontSize: 12 }}>analyses</code> <span style={{ color: '#475569', marginLeft: 4 }}>EEG analysis results from AI classification pipeline — predictions, confidence scores, signal quality assessments</span></div>
      </Card>
    </div>
  )
}
