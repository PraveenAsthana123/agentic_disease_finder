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

function Badge({ engel }) {
  const color = engelColor(engel)
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
      background: color + '20', color: color
    }}>{engel}</span>
  )
}

function engelColor(cls) {
  if (!cls) return '#64748b'
  const s = String(cls).toUpperCase()
  if (s.startsWith('IV')) return '#ef4444'
  if (s.startsWith('III')) return '#f59e0b'
  if (s.startsWith('II')) return '#3b82f6'
  if (s.startsWith('I')) return '#10b981'
  return '#64748b'
}

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const ENGEL_MAJOR_COLORS = { I: '#10b981', II: '#3b82f6', III: '#f59e0b', IV: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SurgicalOutcomesDashboard() {
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
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Surgical Outcomes Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Post-surgical outcome analytics — Engel classification, ILAE scale, complication tracking, seizure freedom rates
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

/* ─── Overview Tab ─── */
function OverviewTab({ data }) {
  const engelMajorData = (data.engel_major || []).map(d => ({ name: d.class, value: d.count }))
  const surgeryTypeData = (data.surgery_type_distribution || []).map(d => ({ name: d.type, value: d.count }))
  const ilaeData = (data.ilae_distribution || []).map(d => ({ name: `ILAE ${d.outcome}`, value: d.count }))
  const engelDetailData = (data.engel_detail || []).map(d => ({ name: d.class, value: d.count }))
  const followupData = (data.followup_trend || []).map(d => ({ name: d.period, total: d.total, seizure_free: d.seizure_free }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI row 1 — 4 cards */}
      <Card title="Total Surgeries">
        <KPI value={data.total_surgeries} label="surgical procedures" sub={`${data.total_patients} patients`} color="#3b82f6" />
      </Card>
      <Card title="Seizure Free Rate">
        <KPI value={`${data.seizure_free_rate}%`} label="seizure freedom"
          sub={`${data.seizure_free_count} of ${data.total_surgeries}`} color="#10b981" />
      </Card>
      <Card title="Mean Follow-up">
        <KPI value={data.mean_followup} label="months" sub="average follow-up duration" color="#8b5cf6" />
      </Card>
      <Card title="Complication Rate">
        <KPI value={`${data.complication_rate}%`} label="complications"
          sub={`${data.complication_count} cases`} color="#ef4444" />
      </Card>

      {/* KPI row 2 — 3 cards (span across 4 cols: 1+1+2 or use auto) */}
      <Card title="Total Patients">
        <KPI value={data.total_patients} label="unique patients" color="#06b6d4" />
      </Card>
      <Card title="AED Reduction Rate">
        <KPI value={`${data.aed_reduction_rate}%`} label="AED reduction"
          sub={`${data.aed_reduction_count} patients`} color="#f59e0b" />
      </Card>
      <Card title="Seizure Free Count">
        <KPI value={data.seizure_free_count} label="seizure-free patients" color="#10b981" />
      </Card>

      {/* Engel Major Class PieChart */}
      <Card title="Engel Major Class Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={engelMajorData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name}: ${value}`}>
              {engelMajorData.map((entry, i) => (
                <Cell key={i} fill={ENGEL_MAJOR_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Surgery Type horizontal bar */}
      <Card title="Surgery Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={surgeryTypeData} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={110} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Surgeries" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* ILAE Outcome bar */}
      <Card title="ILAE Outcome Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={ilaeData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" name="Patients" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Engel Detail subclasses bar */}
      <Card title="Engel Subclass Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={engelDetailData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" name="Patients" radius={[4, 4, 0, 0]}>
              {engelDetailData.map((entry, i) => (
                <Cell key={i} fill={engelColor(entry.name)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Follow-up Trend */}
      <Card title="Follow-up Trend (Seizure Free vs Total)" span={2}>
        {followupData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={followupData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="total" fill="#94a3b8" name="Total" radius={[4, 4, 0, 0]} />
              <Bar dataKey="seizure_free" fill="#10b981" name="Seizure Free" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>

      {/* Outcome by Surgery Type table */}
      <Card title="Outcome by Surgery Type" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Surgery Type</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Total</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Seizure Free</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Avg Follow-up (mo)</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Complications</th>
              </tr>
            </thead>
            <tbody>
              {(data.outcome_by_type || []).map((r, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 500 }}>{r.surgery_type}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{r.total}</td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: '#10b981', fontWeight: 600 }}>{r.seizure_free}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{r.avg_followup}</td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: r.complications > 0 ? '#ef4444' : undefined, fontWeight: r.complications > 0 ? 600 : 400 }}>{r.complications}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ─── Breakdown Tab ─── */
function BreakdownTab({ data }) {
  const hemisphereData = (data.hemisphere_distribution || []).map(d => ({ name: d.hemisphere, value: d.cnt, seizure_free: d.seizure_free }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Outcome by Pathology */}
      <Card title="Outcome by Pathology">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Pathology</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Total</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Seizure Free</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Avg Pre Freq</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Avg Post Freq</th>
              </tr>
            </thead>
            <tbody>
              {(data.outcome_by_pathology || []).map((r, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 500 }}>{r.pathology}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{r.total}</td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: '#10b981', fontWeight: 600 }}>{r.seizure_free}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{r.avg_pre_freq}</td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: r.avg_post_freq < r.avg_pre_freq ? '#10b981' : '#ef4444', fontWeight: 600 }}>{r.avg_post_freq}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Hemisphere PieChart */}
      <Card title="Hemisphere Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={hemisphereData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
              label={({ name, value }) => `${name}: ${value}`}>
              {hemisphereData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip formatter={(value, name, props) => [`${value} (SF: ${props.payload.seizure_free})`, name]} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Complication cases alert table */}
      {(data.complication_cases || []).length > 0 && (
        <Card title={`Complication Cases (${data.complication_cases.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={{ ...thStyle, color: '#dc2626' }}>Patient</th>
                  <th style={{ ...thStyle, color: '#dc2626' }}>Surgery Type</th>
                  <th style={{ ...thStyle, color: '#dc2626' }}>Date</th>
                  <th style={{ ...thStyle, color: '#dc2626' }}>Complications</th>
                  <th style={{ ...thStyle, color: '#dc2626' }}>Severity</th>
                  <th style={{ ...thStyle, color: '#dc2626' }}>Engel Class</th>
                </tr>
              </thead>
              <tbody>
                {data.complication_cases.map((c, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#fff5f5' : '#fff' }}>
                    <td style={{ ...tdStyle, fontWeight: 500 }}>{c.patient_id}</td>
                    <td style={tdStyle}>{c.surgery_type}</td>
                    <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>{(c.surgery_date || '').slice(0, 10)}</td>
                    <td style={tdStyle}>{c.complications}</td>
                    <td style={tdStyle}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                        background: c.complication_severity === 'major' ? '#fef2f2' : '#fefce8',
                        color: c.complication_severity === 'major' ? '#ef4444' : '#f59e0b'
                      }}>{c.complication_severity}</span>
                    </td>
                    <td style={tdStyle}><Badge engel={c.engel_class} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Pre vs Post seizure frequency comparison */}
      <Card title="Pre vs Post Seizure Frequency Comparison">
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={thStyle}>Patient</th>
                <th style={thStyle}>Surgery Type</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Pre Frequency</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Post Frequency</th>
                <th style={thStyle}>Change</th>
                <th style={thStyle}>Engel Class</th>
              </tr>
            </thead>
            <tbody>
              {(data.freq_comparison || []).map((r, i) => {
                const improved = r.post_surgery_frequency < r.pre_surgery_frequency
                return (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ ...tdStyle, fontWeight: 500 }}>{r.patient_id}</td>
                    <td style={tdStyle}>{r.surgery_type}</td>
                    <td style={{ ...tdStyle, textAlign: 'right' }}>{r.pre_surgery_frequency}</td>
                    <td style={{ ...tdStyle, textAlign: 'right', fontWeight: 600, color: improved ? '#10b981' : '#ef4444' }}>{r.post_surgery_frequency}</td>
                    <td style={tdStyle}>
                      {improved && (
                        <span style={{
                          padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                          background: '#dcfce7', color: '#10b981'
                        }}>Improved</span>
                      )}
                      {!improved && r.post_surgery_frequency === r.pre_surgery_frequency && (
                        <span style={{
                          padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                          background: '#f1f5f9', color: '#64748b'
                        }}>No change</span>
                      )}
                      {!improved && r.post_surgery_frequency > r.pre_surgery_frequency && (
                        <span style={{
                          padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                          background: '#fef2f2', color: '#ef4444'
                        }}>Worsened</span>
                      )}
                    </td>
                    <td style={tdStyle}><Badge engel={r.engel_class} /></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* All surgeries detail table */}
      <Card title={`All Surgeries (${(data.surgeries || []).length} records)`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={thStyle}>Patient</th>
                <th style={thStyle}>Surgery Type</th>
                <th style={thStyle}>Date</th>
                <th style={thStyle}>Hemisphere</th>
                <th style={thStyle}>Pathology</th>
                <th style={thStyle}>Engel</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>ILAE</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Follow-up (mo)</th>
                <th style={thStyle}>Seizure Free</th>
                <th style={thStyle}>AED Reduction</th>
                <th style={thStyle}>Complications</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Pre Freq</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Post Freq</th>
                <th style={thStyle}>Notes</th>
              </tr>
            </thead>
            <tbody>
              {(data.surgeries || []).map((s, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 500 }}>{s.patient_id}</td>
                  <td style={tdStyle}>{s.surgery_type}</td>
                  <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>{(s.surgery_date || '').slice(0, 10)}</td>
                  <td style={tdStyle}>{s.hemisphere}</td>
                  <td style={tdStyle}>{s.pathology}</td>
                  <td style={tdStyle}><Badge engel={s.engel_class} /></td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{s.ilae_outcome}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{s.follow_up_months}</td>
                  <td style={tdStyle}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                      background: s.seizure_free ? '#dcfce7' : '#fef2f2',
                      color: s.seizure_free ? '#10b981' : '#ef4444'
                    }}>{s.seizure_free ? 'Yes' : 'No'}</span>
                  </td>
                  <td style={tdStyle}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                      background: s.aed_reduction ? '#dbeafe' : '#f1f5f9',
                      color: s.aed_reduction ? '#3b82f6' : '#64748b'
                    }}>{s.aed_reduction ? 'Yes' : 'No'}</span>
                  </td>
                  <td style={tdStyle}>
                    {s.complications ? (
                      <span style={{
                        padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                        background: s.complication_severity === 'major' ? '#fef2f2' : '#fefce8',
                        color: s.complication_severity === 'major' ? '#ef4444' : '#f59e0b'
                      }}>{s.complications}</span>
                    ) : <span style={{ color: '#94a3b8' }}>None</span>}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{s.pre_surgery_frequency}</td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: s.post_surgery_frequency < s.pre_surgery_frequency ? '#10b981' : undefined, fontWeight: s.post_surgery_frequency < s.pre_surgery_frequency ? 600 : 400 }}>{s.post_surgery_frequency}</td>
                  <td style={{ ...tdStyle, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={s.notes}>{s.notes || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ─── Definitions Tab ─── */
function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Engel Classification with subclasses */}
      <Card title="Engel Classification" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Class</th>
                <th style={thStyle}>Label</th>
                <th style={thStyle}>Subclass</th>
                <th style={thStyle}>Definition</th>
              </tr>
            </thead>
            <tbody>
              {(data.engel_classification || []).map((cls, ci) => {
                const subs = cls.subclasses || []
                if (subs.length === 0) {
                  return (
                    <tr key={ci} style={{ background: ci % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={tdStyle}><Badge engel={cls.class} /></td>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{cls.label}</td>
                      <td style={tdStyle}>--</td>
                      <td style={tdStyle}>--</td>
                    </tr>
                  )
                }
                return subs.map((sub, si) => (
                  <tr key={`${ci}-${si}`} style={{ background: (ci + si) % 2 ? '#f8fafc' : '#fff' }}>
                    {si === 0 && (
                      <>
                        <td style={{ ...tdStyle, verticalAlign: 'top' }} rowSpan={subs.length}><Badge engel={cls.class} /></td>
                        <td style={{ ...tdStyle, fontWeight: 500, verticalAlign: 'top' }} rowSpan={subs.length}>{cls.label}</td>
                      </>
                    )}
                    <td style={tdStyle}><Badge engel={sub.sub} /></td>
                    <td style={{ ...tdStyle, color: '#64748b' }}>{sub.definition}</td>
                  </tr>
                ))
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* ILAE Scale */}
      <Card title="ILAE Outcome Scale">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ ...thStyle, textAlign: 'center' }}>Outcome</th>
                <th style={thStyle}>Definition</th>
              </tr>
            </thead>
            <tbody>
              {(data.ilae_scale || []).map((row, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600 }}>{row.outcome}</td>
                  <td style={{ ...tdStyle, color: '#64748b' }}>{row.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Surgery Types */}
      <Card title="Surgery Types">
        {(data.surgery_types || []).map((st, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.surgery_types.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <strong style={{ color: '#3b82f6' }}>{st.type}</strong>
            <div style={{ color: '#64748b', marginTop: 2 }}>{st.description}</div>
          </div>
        ))}
      </Card>

      {/* Data Sources */}
      {(data.data_sources || []).length > 0 && (
        <Card title="Data Sources" span={2}>
          <ul style={{ margin: 0, padding: '0 0 0 20px', fontSize: 12, color: '#64748b' }}>
            {data.data_sources.map((src, i) => (
              <li key={i} style={{ marginBottom: 4 }}>{src}</li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}

/* ─── Shared styles ─── */
const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }
const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }
