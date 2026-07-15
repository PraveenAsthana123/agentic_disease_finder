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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function MedicationRefillsDashboard() {
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
      axios.get(`${API_URL}/api/medication-refills/overview`),
      axios.get(`${API_URL}/api/medication-refills/breakdown`),
      axios.get(`${API_URL}/api/medication-refills/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Medication Refills data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Medication Refills Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Refill analytics — drug distribution, pharmacy tracking, auto-refill rates, gap analysis, per-patient detail
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
  const kpis = data.kpis || {}
  const drugDist = data.drug_distribution || []
  const pharmacyDist = data.pharmacy_distribution || []
  const monthlyTrend = data.monthly_trend || []
  const autoRefill = data.auto_refill_breakdown || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 16 }}>
      <Card title="Total Refills">
        <KPI value={kpis.total_refills} label="refills" color="#3b82f6" />
      </Card>
      <Card title="Total Patients">
        <KPI value={kpis.total_patients} label="unique patients" color="#8b5cf6" />
      </Card>
      <Card title="Total Drugs">
        <KPI value={kpis.total_drugs} label="unique drugs" color="#10b981" />
      </Card>
      <Card title="Avg Quantity">
        <KPI value={kpis.avg_quantity} label="per refill" color="#f59e0b" />
      </Card>
      <Card title="Avg Days Supply">
        <KPI value={kpis.avg_days_supply} label="days" color="#ef4444" />
      </Card>
      <Card title="Auto-Refill %">
        <KPI value={`${kpis.auto_refill_pct}%`} label="auto-refill rate" color="#06b6d4" />
      </Card>
      <Card title="Total Pharmacies">
        <KPI value={kpis.total_pharmacies} label="pharmacies" color="#ec4899" />
      </Card>

      {/* Drug Distribution Pie */}
      <Card title="Drug Distribution" span={4}>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={drugDist} cx="50%" cy="50%" outerRadius={100} dataKey="count" nameKey="drug_name"
              label={({ drug_name, pct }) => `${drug_name}: ${pct}%`}>
              {drugDist.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Auto-Refill Breakdown Pie */}
      <Card title="Auto-Refill Breakdown" span={3}>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={autoRefill} cx="50%" cy="50%" outerRadius={100} dataKey="count" nameKey="type"
              label={({ type, pct }) => `${type}: ${pct}%`}>
              {autoRefill.map((entry, i) => (
                <Cell key={i} fill={entry.type === 'Auto' ? '#10b981' : '#f59e0b'} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Pharmacy Distribution Horizontal Bar */}
      <Card title="Pharmacy Distribution" span={4}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={pharmacyDist} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="pharmacy" tick={{ fontSize: 11 }} width={120} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Monthly Trend Line */}
      <Card title="Monthly Refill Trend" span={3}>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={monthlyTrend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="refills" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Refills" />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  const perPatient = data.per_patient || []
  const recentRefills = data.recent_refills || []
  const drugDetails = data.drug_details || []
  const gapAnalysis = data.gap_analysis || []

  const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569', whiteSpace: 'nowrap' }
  const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontSize: 12, color: '#334155' }
  const maxRefills = Math.max(...perPatient.map(p => p.refills || 0), 1)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Per-Patient Summary */}
      <Card title={`Per-Patient Summary (${perPatient.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={thStyle}>Patient</th>
              <th style={thStyle}>Refills</th>
              <th style={thStyle}>Drugs</th>
              <th style={thStyle}>Avg Quantity</th>
              <th style={thStyle}>Avg Days Supply</th>
            </tr></thead>
            <tbody>
              {perPatient.map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={tdStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{
                        width: 60, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${(p.refills / maxRefills) * 100}%`, height: '100%', borderRadius: 3,
                          background: '#3b82f6'
                        }} />
                      </div>
                      <span>{p.refills}</span>
                    </div>
                  </td>
                  <td style={tdStyle}>{p.drugs}</td>
                  <td style={tdStyle}>{p.avg_quantity}</td>
                  <td style={tdStyle}>{p.avg_days_supply}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Drug Details */}
      <Card title={`Drug Details (${drugDetails.length} drugs)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={thStyle}>Drug</th>
              <th style={thStyle}>Total Refills</th>
              <th style={thStyle}>Unique Patients</th>
              <th style={thStyle}>Avg Quantity</th>
              <th style={thStyle}>Avg Days Supply</th>
              <th style={thStyle}>Auto-Refill %</th>
            </tr></thead>
            <tbody>
              {drugDetails.map((d, i) => {
                const pct = d.auto_refill_pct || 0
                const barColor = pct > 60 ? '#10b981' : pct >= 40 ? '#f59e0b' : '#ef4444'
                return (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{d.drug_name}</td>
                    <td style={tdStyle}>{d.total_refills}</td>
                    <td style={tdStyle}>{d.unique_patients}</td>
                    <td style={tdStyle}>{d.avg_quantity}</td>
                    <td style={tdStyle}>{d.avg_days_supply}</td>
                    <td style={tdStyle}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{
                          width: 60, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${pct}%`, height: '100%', borderRadius: 3,
                            background: barColor
                          }} />
                        </div>
                        <span style={{ color: barColor, fontWeight: 600 }}>{pct}%</span>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Gap Analysis */}
      {gapAnalysis.length > 0 && (
        <Card title={`Refill Gap Analysis (${gapAnalysis.length} gaps detected)`}>
          <div style={{ background: '#fefce8', borderRadius: 8, padding: 12 }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Drug</th>
                  <th style={thStyle}>Last Refill</th>
                  <th style={thStyle}>Previous Refill</th>
                  <th style={thStyle}>Gap (days)</th>
                  <th style={thStyle}>Days Supply</th>
                  <th style={thStyle}>Status</th>
                </tr></thead>
                <tbody>
                  {gapAnalysis.map((g, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.5)' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{g.patient_id}</td>
                      <td style={tdStyle}>{g.drug_name}</td>
                      <td style={tdStyle}>{g.last_refill}</td>
                      <td style={tdStyle}>{g.previous_refill}</td>
                      <td style={{ ...tdStyle, color: '#b45309', fontWeight: 600 }}>{g.gap_days}</td>
                      <td style={tdStyle}>{g.days_supply}</td>
                      <td style={tdStyle}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 600,
                          background: g.status === 'gap' ? '#fef3c7' : '#dcfce7',
                          color: g.status === 'gap' ? '#92400e' : '#166534'
                        }}>{g.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}

      {/* Recent Refills */}
      <Card title="Recent Refills (last 20)">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={thStyle}>Patient</th>
              <th style={thStyle}>Drug</th>
              <th style={thStyle}>Refill Date</th>
              <th style={thStyle}>Quantity</th>
              <th style={thStyle}>Pharmacy</th>
              <th style={thStyle}>Auto-Refill</th>
              <th style={thStyle}>Days Supply</th>
            </tr></thead>
            <tbody>
              {recentRefills.map((r, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={tdStyle}>{r.drug_name}</td>
                  <td style={tdStyle}>{r.refill_date}</td>
                  <td style={tdStyle}>{r.quantity}</td>
                  <td style={tdStyle}>{r.pharmacy}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>
                    {r.auto_refill ? <span style={{ color: '#10b981', fontSize: 16 }}>&#10003;</span> : <span style={{ color: '#ef4444', fontSize: 14 }}>&#10007;</span>}
                  </td>
                  <td style={tdStyle}>{r.days_supply}</td>
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
  const notes = [
    { term: 'Auto-Refill', desc: data.auto_refill_note },
    { term: 'Days Supply', desc: data.days_supply_note },
    { term: 'Gap Analysis', desc: data.gap_analysis_note },
    { term: 'Pharmacy', desc: data.pharmacy_note },
  ].filter(n => n.desc)

  const drugClasses = data.drug_classes || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Definition Notes */}
      <Card title="Key Definitions">
        <dl style={{ margin: 0 }}>
          {notes.map((n, i) => (
            <div key={i} style={{ marginBottom: 14 }}>
              <dt style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{n.term}</dt>
              <dd style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{n.desc}</dd>
            </div>
          ))}
        </dl>
      </Card>

      {/* Drug Classes Grid */}
      <Card title={`Drug Classes (${Object.keys(drugClasses).length})`}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
          {Object.entries(drugClasses).map(([name, desc], i) => (
            <div key={i} style={{
              background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0'
            }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{name}</div>
              <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{desc}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
