import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const SEV_COLORS = { Normal: '#10b981', Mild: '#f59e0b', Moderate: '#f97316', Severe: '#ef4444' }
const FLAG_COLORS = { normal: '#10b981', high: '#ef4444', low: '#3b82f6' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'abpm', label: 'ABPM Parameters' },
  { id: 'holter', label: 'Holter ECG' },
  { id: 'patients', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AbpmDashboard() {
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
      axios.get(`${API_URL}/api/abpm-holter/overview`),
      axios.get(`${API_URL}/api/abpm-holter/breakdown`),
      axios.get(`${API_URL}/api/abpm-holter/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ABPM/Holter data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const renderTabs = () => (
    <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
      {TABS.map(t => (
        <button key={t.id} onClick={() => setTab(t.id)} style={{
          padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
          background: tab === t.id ? '#3b82f6' : '#f1f5f9',
          color: tab === t.id ? '#fff' : '#475569',
          fontWeight: tab === t.id ? 600 : 400, fontSize: 13
        }}>{t.label}</button>
      ))}
    </div>
  )

  const renderOverview = () => {
    if (!overview) return null
    const { kpis, severity_distribution, pattern_distribution, dipping_distribution, patient_summary } = overview

    const sevData = Object.entries(severity_distribution).map(([k, v]) => ({ name: k, value: v }))
    const patternData = Object.entries(pattern_distribution).filter(([, v]) => v > 0).map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v }))
    const dipData = Object.entries(dipping_distribution).filter(([, v]) => v > 0).map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v }))

    return (
      <>
        <Card title="Key Performance Indicators" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
            <KPI label="Total Studies" value={kpis.total_studies} />
            <KPI label="Abnormal" value={kpis.abnormal_count} color="#ef4444" />
            <KPI label="Abnormal Rate" value={`${kpis.abnormal_rate_pct}%`} color="#f97316" />
            <KPI label="Mean Systolic 24h" value={`${kpis.mean_systolic_24h}`} sub="mmHg" />
            <KPI label="Mean Diastolic 24h" value={`${kpis.mean_diastolic_24h}`} sub="mmHg" />
            <KPI label="Mean QTc" value={`${kpis.mean_qtc_ms}`} sub="ms" />
          </div>
        </Card>

        <Card title="Severity Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                {sevData.map((entry, i) => <Cell key={i} fill={SEV_COLORS[entry.name] || COLORS[i]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Diagnostic Patterns">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={patternData} layout="vertical" margin={{ left: 100 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Dipping Status Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={dipData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                {dipData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Patient Summary" span={2}>
          <div style={{ overflowX: 'auto', maxHeight: 400 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px' }}>Patient</th>
                  <th style={{ padding: '8px 6px' }}>Age</th>
                  <th style={{ padding: '8px 6px' }}>Disease</th>
                  <th style={{ padding: '8px 6px' }}>Severity</th>
                  <th style={{ padding: '8px 6px' }}>Pattern</th>
                  <th style={{ padding: '8px 6px' }}>Score</th>
                  <th style={{ padding: '8px 6px' }}>SBP 24h</th>
                  <th style={{ padding: '8px 6px' }}>Dipping</th>
                  <th style={{ padding: '8px 6px' }}>QTc</th>
                </tr>
              </thead>
              <tbody>
                {patient_summary.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px' }}>{p.name}</td>
                    <td style={{ padding: '6px' }}>{p.age}</td>
                    <td style={{ padding: '6px', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.disease}</td>
                    <td style={{ padding: '6px' }}><Badge text={p.severity} color={SEV_COLORS[p.severity] || '#64748b'} /></td>
                    <td style={{ padding: '6px' }}>{p.pattern_label}</td>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{p.cardiac_score}</td>
                    <td style={{ padding: '6px' }}>{p.systolic_24h}</td>
                    <td style={{ padding: '6px' }}>{p.dipping_pct}%</td>
                    <td style={{ padding: '6px' }}>{p.qtc_ms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  const renderAbpm = () => {
    if (!breakdown) return null
    const { abpm_summary, systolic_histogram, dipping_histogram } = breakdown

    return (
      <>
        <Card title="ABPM Parameter Summary" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px' }}>Parameter</th>
                  <th style={{ padding: '8px 6px' }}>Mean</th>
                  <th style={{ padding: '8px 6px' }}>Min</th>
                  <th style={{ padding: '8px 6px' }}>Max</th>
                  <th style={{ padding: '8px 6px' }}>Unit</th>
                  <th style={{ padding: '8px 6px' }}>Ref Range</th>
                  <th style={{ padding: '8px 6px' }}>Abnormal</th>
                </tr>
              </thead>
              <tbody>
                {abpm_summary.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 500 }}>{p.parameter}</td>
                    <td style={{ padding: '6px' }}>{p.mean}</td>
                    <td style={{ padding: '6px' }}>{p.min}</td>
                    <td style={{ padding: '6px' }}>{p.max}</td>
                    <td style={{ padding: '6px' }}>{p.unit}</td>
                    <td style={{ padding: '6px' }}>{p.ref_low}–{p.ref_high}</td>
                    <td style={{ padding: '6px' }}><Badge text={`${p.abnormal_n} (${p.abnormal_pct}%)`} color={p.abnormal_pct > 30 ? '#ef4444' : p.abnormal_pct > 10 ? '#f59e0b' : '#10b981'} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="24h Systolic BP Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={systolic_histogram}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={(l) => `${l} mmHg`} />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Nocturnal Dipping Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={dipping_histogram}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={(l) => `${l}%`} />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </>
    )
  }

  const renderHolter = () => {
    if (!breakdown) return null
    const { holter_summary, qtc_histogram, pvc_histogram, cardiac_score_histogram } = breakdown

    return (
      <>
        <Card title="Holter ECG Parameter Summary" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px' }}>Parameter</th>
                  <th style={{ padding: '8px 6px' }}>Mean</th>
                  <th style={{ padding: '8px 6px' }}>Min</th>
                  <th style={{ padding: '8px 6px' }}>Max</th>
                  <th style={{ padding: '8px 6px' }}>Unit</th>
                  <th style={{ padding: '8px 6px' }}>Ref Range</th>
                  <th style={{ padding: '8px 6px' }}>Abnormal</th>
                </tr>
              </thead>
              <tbody>
                {holter_summary.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 500 }}>{p.parameter}</td>
                    <td style={{ padding: '6px' }}>{p.mean}</td>
                    <td style={{ padding: '6px' }}>{p.min}</td>
                    <td style={{ padding: '6px' }}>{p.max}</td>
                    <td style={{ padding: '6px' }}>{p.unit}</td>
                    <td style={{ padding: '6px' }}>{p.ref_low}–{p.ref_high}</td>
                    <td style={{ padding: '6px' }}><Badge text={`${p.abnormal_n} (${p.abnormal_pct}%)`} color={p.abnormal_pct > 30 ? '#ef4444' : p.abnormal_pct > 10 ? '#f59e0b' : '#10b981'} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="QTc Interval Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={qtc_histogram}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={(l) => `${l} ms`} />
              <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="PVC Count Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={pvc_histogram}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={(l) => `${l} PVCs`} />
              <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Cardiac-Autonomic Risk Score Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={cardiac_score_histogram}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={(l) => `Score ${l}`} />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </>
    )
  }

  const renderPatients = () => {
    if (!breakdown) return null
    const { patient_detail_cards } = breakdown

    return (
      <>
        {patient_detail_cards.map((p, i) => (
          <Card key={i} title={`${p.name} (Age ${p.age}) — ${p.disease || 'N/A'}`} span={2}>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
              <Badge text={p.severity} color={SEV_COLORS[p.severity] || '#64748b'} />
              <Badge text={p.pattern_label} color="#3b82f6" />
              <Badge text={`Dipping: ${p.dipping_category.replace(/_/g, ' ')}`} color="#8b5cf6" />
              <span style={{ fontSize: 12, color: '#475569' }}>Score: <strong>{p.cardiac_score}</strong>/100</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <h4 style={{ fontSize: 13, margin: '0 0 8px', color: '#334155' }}>ABPM</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
                  {Object.entries(p.abpm).map(([key, v]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 4px', background: v.flag !== 'normal' ? (v.flag === 'high' ? '#fef2f2' : '#eff6ff') : '#f8fafc', borderRadius: 4 }}>
                      <span style={{ color: '#475569' }}>{key.replace(/_/g, ' ').replace(/mmhg|pct/g, '').trim()}</span>
                      <span style={{ fontWeight: 600, color: FLAG_COLORS[v.flag] || '#1e293b' }}>{v.value} {v.unit}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h4 style={{ fontSize: 13, margin: '0 0 8px', color: '#334155' }}>Holter ECG</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
                  {Object.entries(p.holter).map(([key, v]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 4px', background: v.flag !== 'normal' ? (v.flag === 'high' ? '#fef2f2' : '#eff6ff') : '#f8fafc', borderRadius: 4 }}>
                      <span style={{ color: '#475569' }}>{key.replace(/_/g, ' ').replace(/bpm|ms|count/g, '').trim()}</span>
                      <span style={{ fontWeight: 600, color: FLAG_COLORS[v.flag] || '#1e293b' }}>{v.value} {v.unit}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        ))}
      </>
    )
  }

  const renderDefinitions = () => {
    if (!definitions) return null

    return (
      <>
        <Card title="Protocol" span={2}>
          <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{definitions.protocol?.description}</p>
          <div style={{ marginTop: 12 }}>
            <strong style={{ fontSize: 12 }}>ABPM Recording:</strong>
            <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0' }}>{definitions.protocol?.recording_methods?.abpm}</p>
            <strong style={{ fontSize: 12 }}>Holter Recording:</strong>
            <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0' }}>{definitions.protocol?.recording_methods?.holter}</p>
          </div>
          {definitions.protocol?.indications && (
            <div style={{ marginTop: 12 }}>
              <strong style={{ fontSize: 12 }}>Indications:</strong>
              <ul style={{ margin: '4px 0', paddingLeft: 20, fontSize: 12, color: '#64748b' }}>
                {definitions.protocol.indications.map((ind, i) => <li key={i}>{ind}</li>)}
              </ul>
            </div>
          )}
        </Card>

        <Card title="Dipping Categories">
          {definitions.dipping_categories?.map((d, i) => (
            <div key={i} style={{ padding: '8px 0', borderBottom: i < definitions.dipping_categories.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{d.label} <span style={{ fontWeight: 400, color: '#64748b' }}>({d.range})</span></div>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{d.risk}</div>
            </div>
          ))}
        </Card>

        <Card title="Diagnostic Patterns">
          {definitions.diagnostic_patterns?.map((d, i) => (
            <div key={i} style={{ padding: '8px 0', borderBottom: i < definitions.diagnostic_patterns.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{d.label}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{d.description}</div>
            </div>
          ))}
        </Card>

        <Card title="Severity Levels">
          {definitions.severity_levels?.map((s, i) => (
            <div key={i} style={{ padding: '6px 0', display: 'flex', gap: 12, alignItems: 'center' }}>
              <Badge text={s.level} color={SEV_COLORS[s.level] || '#64748b'} />
              <span style={{ fontSize: 12, color: '#64748b' }}>Score {s.score_range}: {s.description}</span>
            </div>
          ))}
        </Card>
      </>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>ABPM / Holter Dashboard</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>Ambulatory Blood Pressure Monitoring + Holter ECG — Dipping status, arrhythmia burden, cardiac-autonomic correlation</p>
      {renderTabs()}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 20 }}>
        {tab === 'overview' && renderOverview()}
        {tab === 'abpm' && renderAbpm()}
        {tab === 'holter' && renderHolter()}
        {tab === 'patients' && renderPatients()}
        {tab === 'definitions' && renderDefinitions()}
      </div>
    </div>
  )
}
