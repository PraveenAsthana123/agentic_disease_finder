import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#a855f7']

function QualityBadge({ quality }) {
  const styles = {
    Excellent: { bg: '#dcfce7', color: '#16a34a' },
    Good: { bg: '#dbeafe', color: '#1d4ed8' },
    Fair: { bg: '#fef3c7', color: '#d97706' },
    Poor: { bg: '#fee2e2', color: '#dc2626' },
  }
  const s = styles[quality] || { bg: '#f1f5f9', color: '#475569' }
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 600,
      background: s.bg, color: s.color
    }}>{quality}</span>
  )
}

function DiseaseBadge({ disease }) {
  const styles = {
    epilepsy: { bg: '#fee2e2', color: '#dc2626' },
    depression: { bg: '#e0e7ff', color: '#4338ca' },
    alzheimers: { bg: '#fef3c7', color: '#d97706' },
    parkinsons: { bg: '#dcfce7', color: '#16a34a' },
    sleep_disorder: { bg: '#f3e8ff', color: '#7c3aed' },
  }
  const s = styles[disease] || { bg: '#f1f5f9', color: '#475569' }
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 600,
      background: s.bg, color: s.color
    }}>{disease}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'all', label: 'All Analyses' },
  { id: 'patients', label: 'By Patient' },
  { id: 'diseases', label: 'By Disease' },
  { id: 'definitions', label: 'Glossary' },
]

export default function AnalysesDashboard() {
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
      axios.get(`${API_URL}/api/analyses/overview`),
      axios.get(`${API_URL}/api/analyses/breakdown`),
      axios.get(`${API_URL}/api/analyses/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EEG Analyses data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>EEG Analyses Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          AI classification results — 5 diseases, 49 patients, confidence scoring, signal quality assessment
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'all' && breakdown && <AllTab data={breakdown} />}
      {tab === 'patients' && breakdown && <PatientsTab data={breakdown} />}
      {tab === 'diseases' && breakdown && <DiseasesTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const { total_analyses, unique_patients, avg_confidence, disease_count, date_range,
          disease_distribution, label_distribution, quality_distribution,
          confidence_by_disease, daily_trend, high_confidence_rate } = data

  const diseaseChart = (disease_distribution || []).map(d => ({ name: d.disease, value: d.count }))
  const labelChart = (label_distribution || []).map(l => ({ name: l.predicted_label, value: l.count }))
  const qualityChart = (quality_distribution || []).map(q => ({ name: q.signal_quality, value: q.count }))
  const dailyChart = (daily_trend || []).map(d => ({ date: d.day?.slice(5), total: d.total }))
  const confChart = (confidence_by_disease || []).map(c => ({
    name: c.disease,
    avg: parseFloat(c.avg_confidence) || 0,
    min: parseFloat(c.min_confidence) || 0,
    max: parseFloat(c.max_confidence) || 0,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card>
        <KPI label="Total Analyses" value={total_analyses?.toLocaleString()} color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Unique Patients" value={unique_patients} sub={`${disease_count} diseases`} color="#10b981" />
      </Card>
      <Card>
        <KPI label="Avg Confidence" value={`${((avg_confidence || 0) * 100).toFixed(1)}%`} sub={`${date_range?.active_days} active days`} color="#8b5cf6" />
      </Card>
      <Card>
        <KPI label="High Confidence" value={`${high_confidence_rate}%`} sub="analyses >= 70%" color="#f59e0b" />
      </Card>

      <Card title="Daily Analysis Volume" span={4}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={dailyChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" height={50} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="total" fill="#3b82f6" name="Analyses" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Disease Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={diseaseChart} dataKey="value" nameKey="name" cx="50%" cy="50%"
                 outerRadius={85} label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                 labelLine={{ stroke: '#94a3b8' }}>
              {diseaseChart.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Signal Quality" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={qualityChart} dataKey="value" nameKey="name" cx="50%" cy="50%"
                 outerRadius={85} label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                 labelLine={{ stroke: '#94a3b8' }}>
              {qualityChart.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Prediction Labels" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={labelChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={140} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" name="Count" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Confidence by Disease" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={confChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
            <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="avg" fill="#3b82f6" name="Average" radius={[4, 4, 0, 0]} />
            <Bar dataKey="max" fill="#10b981" name="Max" radius={[4, 4, 0, 0]} />
            <Bar dataKey="min" fill="#ef4444" name="Min" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function AllTab({ data }) {
  const [filter, setFilter] = useState('')
  const [diseaseFilter, setDiseaseFilter] = useState('all')
  const execs = (data.all_analyses || [])
    .filter(e => diseaseFilter === 'all' || e.disease === diseaseFilter)
    .filter(e => !filter || (e.patient_id || '').toLowerCase().includes(filter.toLowerCase())
      || (e.predicted_label || '').toLowerCase().includes(filter.toLowerCase()))

  return (
    <Card>
      <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
        <input
          placeholder="Search patient or label..."
          value={filter} onChange={e => setFilter(e.target.value)}
          style={{ flex: 1, padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
        />
        <select value={diseaseFilter} onChange={e => setDiseaseFilter(e.target.value)}
          style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}>
          <option value="all">All Diseases</option>
          <option value="epilepsy">Epilepsy</option>
          <option value="depression">Depression</option>
          <option value="alzheimers">Alzheimers</option>
          <option value="parkinsons">Parkinsons</option>
          <option value="sleep_disorder">Sleep Disorder</option>
        </select>
      </div>
      <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>Showing {execs.length} analyses</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              {['Patient', 'Disease', 'Prediction', 'Confidence', 'Signal Quality', 'Date'].map(h =>
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {execs.slice(0, 100).map((e, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 500 }}>{e.patient_id}</td>
                <td style={{ padding: '6px' }}><DiseaseBadge disease={e.disease} /></td>
                <td style={{ padding: '6px' }}>{e.predicted_label}</td>
                <td style={{ padding: '6px' }}>
                  <span style={{
                    fontWeight: 600,
                    color: e.confidence >= 0.8 ? '#16a34a' : e.confidence >= 0.6 ? '#d97706' : '#dc2626'
                  }}>{(e.confidence * 100).toFixed(1)}%</span>
                </td>
                <td style={{ padding: '6px' }}><QualityBadge quality={e.signal_quality} /></td>
                <td style={{ padding: '6px', whiteSpace: 'nowrap' }}>{e.created_at?.slice(0, 16).replace('T', ' ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function PatientsTab({ data }) {
  const patients = data.by_patient || []
  const confDist = data.confidence_distribution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Confidence Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={confDist}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" name="Analyses" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title={`Per-Patient Summary (${patients.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Patient', 'Analyses', 'Diseases', 'Avg Confidence', 'Latest'].map(h =>
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontWeight: 500 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px' }}>{p.total_analyses}</td>
                  <td style={{ padding: '6px' }}>{p.diseases}</td>
                  <td style={{ padding: '6px' }}>
                    <span style={{
                      fontWeight: 600,
                      color: p.avg_confidence >= 0.8 ? '#16a34a' : p.avg_confidence >= 0.6 ? '#d97706' : '#dc2626'
                    }}>{(p.avg_confidence * 100).toFixed(1)}%</span>
                  </td>
                  <td style={{ padding: '6px', whiteSpace: 'nowrap' }}>{p.latest_date?.slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DiseasesTab({ data }) {
  const diseases = data.by_disease || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {diseases.map((d, i) => (
        <Card key={i} title={d.disease}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#3b82f6' }}>{d.total}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Total Analyses</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#10b981' }}>{((d.avg_confidence || 0) * 100).toFixed(1)}%</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Avg Confidence</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 14, fontWeight: 500, color: '#475569' }}>{d.labels}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>Prediction Labels</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              {d.quality_breakdown && Object.entries(d.quality_breakdown).map(([q, c]) => (
                <span key={q} style={{ marginRight: 6 }}><QualityBadge quality={q} /> {c}</span>
              ))}
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Field Glossary">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600, width: 180 }}>Field</th>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {(data.field_glossary || []).map((f, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 500, fontFamily: 'monospace', fontSize: 11 }}>{f.field}</td>
                <td style={{ padding: '6px', color: '#475569' }}>{f.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {data.diseases && (
        <Card title="Disease Types">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <tbody>
              {Object.entries(data.diseases).map(([k, v], i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', width: 160 }}><DiseaseBadge disease={k} /></td>
                  <td style={{ padding: '6px', color: '#475569' }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {data.signal_quality_levels && (
        <Card title="Signal Quality Levels">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <tbody>
              {Object.entries(data.signal_quality_levels).map(([k, v], i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', width: 120 }}><QualityBadge quality={k} /></td>
                  <td style={{ padding: '6px', color: '#475569' }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {data.prediction_labels && (
        <Card title="Prediction Labels">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {data.prediction_labels.map((l, i) => (
              <span key={i} style={{
                padding: '4px 12px', borderRadius: 8, fontSize: 12, fontWeight: 500,
                background: '#f1f5f9', color: '#334155'
              }}>{l}</span>
            ))}
          </div>
        </Card>
      )}

      <Card title="Data Source">
        <p style={{ margin: 0, fontSize: 13, color: '#475569' }}>{data.data_source}</p>
      </Card>
    </div>
  )
}
