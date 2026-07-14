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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const ENGEL_COLORS = { 'I': '#10b981', 'II': '#f59e0b', 'III': '#f97316', 'IV': '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'details', label: 'Surgery Details' },
  { id: 'pathology', label: 'Pathology' },
  { id: 'complications', label: 'Complications' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SurgicalOutcomeDashboard() {
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
      axios.get(`${API_URL}/api/surgical-outcomes/overview`),
      axios.get(`${API_URL}/api/surgical-outcomes/breakdown`),
      axios.get(`${API_URL}/api/surgical-outcomes/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Surgical Outcomes data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Surgical Outcome Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Epilepsy surgery outcomes — Engel classification, ILAE scale, seizure freedom, complication tracking
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'details' && breakdown && <DetailsTab surgeries={breakdown.surgeries} />}
      {tab === 'pathology' && breakdown && <PathologyTab data={breakdown} />}
      {tab === 'complications' && breakdown && <ComplicationsTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total Surgeries"><KPI label="Procedures performed" value={data.total_surgeries} /></Card>
      <Card title="Seizure Free"><KPI label={`${data.seizure_free_rate}% of surgeries`} value={`${data.seizure_free_rate}%`} color="#10b981" /></Card>
      <Card title="Mean Follow-up"><KPI label="Months post-surgery" value={data.mean_followup} color="#3b82f6" /></Card>
      <Card title="Complication Rate"><KPI label={`${data.complication_count} cases`} value={`${data.complication_rate}%`} color="#ef4444" /></Card>

      <Card title="Engel Classification Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.engel_major}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="class" fontSize={12} label={{ value: 'Engel Class', position: 'insideBottom', offset: -2, fontSize: 11 }} />
            <YAxis fontSize={12} />
            <Tooltip formatter={(v) => [v, 'Patients']} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {data.engel_major.map((entry, i) => (
                <Cell key={i} fill={ENGEL_COLORS[entry.class] || COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Surgery Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={data.surgery_type_distribution} dataKey="count" nameKey="type" cx="50%" cy="50%"
              outerRadius={90} label={({ type, count }) => `${type} (${count})`} labelLine={false}
              fontSize={10}>
              {data.surgery_type_distribution.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="ILAE Outcome Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.ilae_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="outcome" fontSize={12} label={{ value: 'ILAE Outcome', position: 'insideBottom', offset: -2, fontSize: 11 }} />
            <YAxis fontSize={12} />
            <Tooltip formatter={(v) => [v, 'Patients']} />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Outcome by Surgery Type" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.outcome_by_type} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={12} />
            <YAxis dataKey="surgery_type" type="category" fontSize={10} width={120} />
            <Tooltip />
            <Legend />
            <Bar dataKey="seizure_free" fill="#10b981" name="Seizure Free" stackId="a" />
            <Bar dataKey="complications" fill="#ef4444" name="Complications" stackId="b" />
            <Bar dataKey="total" fill="#3b82f6" name="Total" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DetailsTab({ surgeries }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`All Surgeries (${surgeries.length})`}>
        <div style={{ maxHeight: 600, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Surgery Type</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Date</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Engel</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>ILAE</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Seizure Free</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Follow-up (mo)</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Complications</th>
              </tr>
            </thead>
            <tbody>
              {surgeries.map((s, i) => {
                const engelMajor = s.engel_class?.replace(/[A-D]$/, '')
                const engelColor = ENGEL_COLORS[engelMajor] || '#334155'
                return (
                  <tr key={i} style={{ background: s.seizure_free ? '#f0fdf4' : undefined }}>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{s.patient_id}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{s.surgery_type}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{s.surgery_date}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center',
                      color: engelColor, fontWeight: 600 }}>{s.engel_class}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{s.ilae_outcome}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center',
                      color: s.seizure_free ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                      {s.seizure_free ? 'Yes' : 'No'}
                    </td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{s.follow_up_months}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12,
                      color: s.complications ? '#991b1b' : '#64748b' }}>
                      {s.complications || 'None'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function PathologyTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Outcomes by Pathology" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.outcome_by_pathology} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={12} />
            <YAxis dataKey="pathology" type="category" fontSize={10} width={180} />
            <Tooltip />
            <Legend />
            <Bar dataKey="total" fill="#3b82f6" name="Total" />
            <Bar dataKey="seizure_free" fill="#10b981" name="Seizure Free" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Pre vs Post Seizure Frequency" span={2}>
        <div style={{ maxHeight: 400, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Surgery</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Pre (sz/mo)</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Post (sz/mo)</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Reduction %</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Engel</th>
              </tr>
            </thead>
            <tbody>
              {data.freq_comparison.map((r, i) => {
                const reduction = r.pre_surgery_frequency > 0
                  ? Math.round((1 - r.post_surgery_frequency / r.pre_surgery_frequency) * 100)
                  : 0
                return (
                  <tr key={i}>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{r.patient_id}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12 }}>{r.surgery_type}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{r.pre_surgery_frequency}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center',
                      color: r.post_surgery_frequency === 0 ? '#10b981' : '#334155', fontWeight: r.post_surgery_frequency === 0 ? 600 : 400 }}>
                      {r.post_surgery_frequency}
                    </td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center',
                      color: reduction >= 90 ? '#10b981' : reduction >= 50 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
                      {reduction}%
                    </td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{r.engel_class}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Hemisphere Distribution" span={1}>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={data.hemisphere_distribution} dataKey="cnt" nameKey="hemisphere" cx="50%" cy="50%"
              outerRadius={70} label={({ hemisphere, cnt }) => `${hemisphere} (${cnt})`}>
              {data.hemisphere_distribution.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Outcome by Pathology (Avg Frequency)" span={1}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data.outcome_by_pathology.slice(0, 6)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="pathology" fontSize={9} angle={-20} textAnchor="end" height={60} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Legend />
            <Bar dataKey="avg_pre_freq" fill="#ef4444" name="Pre-surgery" />
            <Bar dataKey="avg_post_freq" fill="#10b981" name="Post-surgery" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ComplicationsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Complication Type Distribution" span={1}>
        {data.complication_breakdown.length === 0 ? (
          <p style={{ color: '#10b981', fontWeight: 600 }}>No complications recorded</p>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data.complication_breakdown}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="complications" fontSize={11} angle={-15} textAnchor="end" height={50} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Bar dataKey="cnt" fill="#ef4444" radius={[4, 4, 0, 0]} name="Cases" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Card>

      <Card title="Severity Breakdown" span={1}>
        {data.complication_breakdown.length === 0 ? (
          <p style={{ color: '#10b981', fontWeight: 600 }}>No complications</p>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={(() => {
                const sev = {}
                data.complication_breakdown.forEach(c => {
                  const s = c.complication_severity || 'unknown'
                  sev[s] = (sev[s] || 0) + c.cnt
                })
                return Object.entries(sev).map(([name, value]) => ({ name, value }))
              })()} dataKey="value" nameKey="name" cx="50%" cy="50%"
                outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
                <Cell fill="#f59e0b" />
                <Cell fill="#ef4444" />
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        )}
      </Card>

      <Card title={`Cases with Complications (${data.complication_cases.length})`} span={2}>
        {data.complication_cases.length === 0 ? (
          <p style={{ color: '#10b981', fontWeight: 600 }}>No complications recorded</p>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Surgery</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Date</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Complication</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Severity</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Engel</th>
              </tr>
            </thead>
            <tbody>
              {data.complication_cases.map((r, i) => (
                <tr key={i} style={{ background: r.complication_severity === 'major' ? '#fef2f2' : '#fffbeb' }}>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{r.surgery_type}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{r.surgery_date}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', color: '#991b1b', fontWeight: 600 }}>{r.complications}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center',
                    color: r.complication_severity === 'major' ? '#991b1b' : '#92400e', fontWeight: 600 }}>
                    {r.complication_severity}
                  </td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{r.engel_class}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Engel Classification (Surgical Outcome Scale)">
        {data.engel_classification.map((ec, i) => (
          <div key={i} style={{ marginBottom: 16, paddingBottom: 16, borderBottom: i < data.engel_classification.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span style={{
                display: 'inline-block', padding: '2px 10px', borderRadius: 6, fontWeight: 700, fontSize: 14,
                background: ENGEL_COLORS[ec.class] ? `${ENGEL_COLORS[ec.class]}20` : '#f1f5f9',
                color: ENGEL_COLORS[ec.class] || '#334155'
              }}>Class {ec.class}</span>
              <strong style={{ color: '#1e293b', fontSize: 14 }}>{ec.label}</strong>
            </div>
            <div style={{ paddingLeft: 20 }}>
              {ec.subclasses.map((sc, j) => (
                <div key={j} style={{ marginBottom: 4, fontSize: 13, color: '#475569' }}>
                  <strong style={{ fontFamily: 'monospace', color: '#334155' }}>{sc.sub}</strong>: {sc.definition}
                </div>
              ))}
            </div>
          </div>
        ))}
      </Card>

      <Card title="ILAE Outcome Scale (Wieser et al., 2001)">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', width: 80 }}>Outcome</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {data.ilae_scale.map((il, i) => (
              <tr key={i}>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center', fontWeight: 700,
                  color: il.outcome <= 2 ? '#10b981' : il.outcome <= 4 ? '#f59e0b' : '#ef4444' }}>{il.outcome}</td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>{il.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Surgery Type Descriptions">
        {data.surgery_types.map((st, i) => (
          <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < data.surgery_types.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <strong style={{ color: '#1e293b', fontSize: 14 }}>{st.type}</strong>
            <p style={{ margin: '4px 0 0', fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{st.description}</p>
          </div>
        ))}
      </Card>

      <Card title="Data Sources">
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {data.data_sources.map((s, i) => (
            <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 6 }}>{s}</li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
