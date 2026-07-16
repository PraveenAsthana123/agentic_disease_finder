import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line
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

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const QUALITY_COLORS = { Diagnostic: '#10b981', Adequate: '#3b82f6', Suboptimal: '#f59e0b', 'Non-diagnostic': '#ef4444' }
const CLASS_COLORS = { Lesional: '#ef4444', 'Non-Lesional': '#3b82f6', Equivocal: '#f59e0b', Normal: '#10b981' }
const CONFIDENCE_COLORS = { High: '#10b981', Low: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Findings & Patients' },
  { id: 'definitions', label: 'Definitions' },
]

export default function MRIFindingsDashboard() {
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
      axios.get(`${API_URL}/api/mri-findings/overview`),
      axios.get(`${API_URL}/api/mri-findings/breakdown`),
      axios.get(`${API_URL}/api/mri-findings/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading MRI Findings data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>MRI Findings Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Structural MRI analytics — lesion classification, location mapping, laterality analysis, hippocampal sclerosis tracking
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const lesionData = Object.entries(data.lesion_type_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const locationData = Object.entries(data.location_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const lateralityData = Object.entries(data.laterality_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const classData = Object.entries(data.classification_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const qualityData = Object.entries(data.quality_distribution || {}).map(([k, v]) => ({ name: k, value: v }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total MRI Scans">
        <KPI value={data.total_scans} label="MRI studies" sub={`${data.lesional_count} lesional`} color="#3b82f6" />
      </Card>
      <Card title="Lesional Rate">
        <KPI value={`${data.lesional_rate_pct}%`} label="lesional findings"
          sub={`${data.lesional_count} of ${data.total_scans}`} color="#ef4444" />
      </Card>
      <Card title="Hippocampal Sclerosis">
        <KPI value={`${data.hs_rate_pct}%`} label="HS positive"
          sub={`${data.hs_count} patients`} color="#8b5cf6" />
      </Card>
      <Card title="High Confidence">
        <KPI value={`${data.high_confidence_rate_pct}%`} label="radiologist confidence"
          sub={`Avg asymmetry: ${data.avg_volume_asymmetry}`} color="#10b981" />
      </Card>

      <Card title="Lesion Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={lesionData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
              {lesionData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Lesion Location" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={locationData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Scans" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Laterality" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={lateralityData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {lateralityData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Classification" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={classData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {classData.map((entry, i) => (
                <Cell key={i} fill={CLASS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Scan Quality" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={qualityData} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={90} />
            <Tooltip />
            <Bar dataKey="value" name="Scans">
              {qualityData.map((entry, i) => (
                <Cell key={i} fill={QUALITY_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Trend" span={2}>
        {(data.monthly_timeline || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data.monthly_timeline}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="scans" stroke="#3b82f6" strokeWidth={2} name="MRI Scans" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  // Flatten cross-tab dicts into arrays for table rendering
  const locationCrosstab = []
  Object.entries(data.lesion_location_crosstab || {}).forEach(([lesion, locs]) => {
    Object.entries(locs).forEach(([loc, cnt]) => {
      locationCrosstab.push({ lesion_label: lesion, lesion_location: loc, cnt })
    })
  })
  const lateralityCrosstab = []
  Object.entries(data.lesion_laterality_crosstab || {}).forEach(([lesion, lats]) => {
    Object.entries(lats).forEach(([lat, cnt]) => {
      lateralityCrosstab.push({ lesion_label: lesion, laterality: lat, cnt })
    })
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {(data.low_confidence_scans || []).length > 0 && (
        <Card title={`Low Confidence Scans (${data.low_confidence_scans.length})`} span={1}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Lesion</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Quality</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Classification</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                </tr>
              </thead>
              <tbody>
                {data.low_confidence_scans.map((s, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#fff5f5' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{s.patient_id}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{s.lesion_label}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                      <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                        background: QUALITY_COLORS[s.quality] ? QUALITY_COLORS[s.quality] + '20' : '#f1f5f9',
                        color: QUALITY_COLORS[s.quality] || '#64748b'
                      }}>{s.quality}</span>
                    </td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                      <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                        background: CLASS_COLORS[s.classification] ? CLASS_COLORS[s.classification] + '20' : '#f1f5f9',
                        color: CLASS_COLORS[s.classification] || '#64748b'
                      }}>{s.classification}</span>
                    </td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{(s.created_at || '').slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {(data.hs_patients || []).length > 0 && (
        <Card title={`Hippocampal Sclerosis Patients (${data.hs_patients.length})`} span={1}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f5f3ff' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Location</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Laterality</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Vol Asymmetry</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>T2/FLAIR</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {data.hs_patients.map((p, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#faf5ff' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{p.location || 'N/A'}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{p.laterality || 'N/A'}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', fontWeight: 600,
                      color: p.volume_asymmetry > 0.15 ? '#ef4444' : '#10b981' }}>{p.volume_asymmetry}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{p.t2_flair_signal}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                      <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                        background: CONFIDENCE_COLORS[p.confidence] ? CONFIDENCE_COLORS[p.confidence] + '20' : '#f1f5f9',
                        color: CONFIDENCE_COLORS[p.confidence] || '#64748b'
                      }}>{p.confidence}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Lesion Type x Location Cross-Tab" span={1}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Lesion Type</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Location</th>
                <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {locationCrosstab.map((r, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{r.lesion_label}</td>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{r.lesion_location}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{r.cnt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Lesion Type x Laterality Cross-Tab" span={1}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Lesion Type</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Laterality</th>
                <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {lateralityCrosstab.map((r, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{r.lesion_label}</td>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{r.laterality}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{r.cnt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title={`All MRI Findings (${(data.per_patient || []).length} scans)`} span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Lesion</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Location</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Laterality</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Classification</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Quality</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>HS</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Asymmetry</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Confidence</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {(data.per_patient || []).map((f, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{f.patient_id}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{f.lesion_label}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{f.location || 'N/A'}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{f.laterality || 'N/A'}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                      background: CLASS_COLORS[f.classification] ? CLASS_COLORS[f.classification] + '20' : '#f1f5f9',
                      color: CLASS_COLORS[f.classification] || '#64748b'
                    }}>{f.classification}</span>
                  </td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                      background: QUALITY_COLORS[f.quality] ? QUALITY_COLORS[f.quality] + '20' : '#f1f5f9',
                      color: QUALITY_COLORS[f.quality] || '#64748b'
                    }}>{f.quality}</span>
                  </td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', color: f.hippocampal_sclerosis === 'Yes' ? '#8b5cf6' : undefined, fontWeight: f.hippocampal_sclerosis === 'Yes' ? 600 : 400 }}>{f.hippocampal_sclerosis}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9',
                    color: f.volume_asymmetry > 0.15 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{f.volume_asymmetry}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                      background: CONFIDENCE_COLORS[f.confidence] ? CONFIDENCE_COLORS[f.confidence] + '20' : '#f1f5f9',
                      color: CONFIDENCE_COLORS[f.confidence] || '#64748b'
                    }}>{f.confidence}</span>
                  </td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{(f.created_at || '').slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  const lesionEntries = Object.entries(data.lesion_types || {})
  const qualityEntries = Object.entries(data.quality_tiers || {})
  const classEntries = Object.entries(data.classification_definitions || {})
  const lateralityEntries = Object.entries(data.laterality_definitions || {})

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title={`Clinical Glossary (${(data.glossary || []).length} terms)`} span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {(data.glossary || []).map((g, i) => (
            <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
              <strong style={{ color: '#1e293b' }}>{g.term}</strong>
              <div style={{ color: '#64748b', marginTop: 2 }}>{g.definition}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Lesion Types">
        {lesionEntries.map(([label, desc], i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (lesionEntries.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <strong style={{ color: '#ef4444' }}>{label}</strong>
            <div style={{ color: '#64748b', marginTop: 2 }}>{desc}</div>
          </div>
        ))}
      </Card>

      <Card title="Quality Tiers">
        {qualityEntries.map(([tier, desc], i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (qualityEntries.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: QUALITY_COLORS[tier] || '#94a3b8', marginRight: 6 }} />
            <strong>{tier}</strong>
            <div style={{ color: '#64748b', marginTop: 2, marginLeft: 16 }}>{desc}</div>
          </div>
        ))}
      </Card>

      <Card title="Classification Categories">
        {classEntries.map(([label, desc], i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (classEntries.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: CLASS_COLORS[label] || '#94a3b8', marginRight: 6 }} />
            <strong>{label}</strong>
            <div style={{ color: '#64748b', marginTop: 2, marginLeft: 16 }}>{desc}</div>
          </div>
        ))}
      </Card>

      <Card title="Laterality Definitions">
        {lateralityEntries.map(([side, desc], i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (lateralityEntries.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <strong style={{ color: '#3b82f6' }}>{side}</strong>
            <div style={{ color: '#64748b', marginTop: 2 }}>{desc}</div>
          </div>
        ))}
      </Card>

      {(data.clinical_notes || []).length > 0 && (
        <Card title="Clinical Notes" span={2}>
          {data.clinical_notes.map((n, i) => (
            <div key={i} style={{ padding: '8px 12px', background: i % 2 ? '#f0fdf4' : '#f8fafc', borderRadius: 8, marginBottom: 6, fontSize: 12 }}>
              <div style={{ color: '#1e293b' }}>{n}</div>
            </div>
          ))}
        </Card>
      )}
    </div>
  )
}
