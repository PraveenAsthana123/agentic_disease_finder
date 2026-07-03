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

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'adherence_analysis', label: 'Adherence Analysis' },
  { id: 'side_effects', label: 'Side Effects' },
  { id: 'patient_detail', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function MedicationManagementDashboard() {
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
      axios.get(`${API_URL}/api/medication-management/overview`),
      axios.get(`${API_URL}/api/medication-management/breakdown`),
      axios.get(`${API_URL}/api/medication-management/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Medication Management Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Medication Self-Management Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Medication adherence monitoring — dose tracking, side effects, refill management, and rescue medication usage
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* KPI Row 1 */}
          <Card title="Total Patients"><KPI value={ov.total_patients} label="Patients tracked" color="#3b82f6" /></Card>
          <Card title="Adherence Logs"><KPI value={ov.total_adherence_logs} label="Total log entries" color="#8b5cf6" /></Card>
          <Card title="Overall Adherence"><KPI value={ov.overall_adherence_rate != null ? `${ov.overall_adherence_rate.toFixed(1)}%` : '--'} label="Adherence rate" color="#10b981" /></Card>
          <Card title="Missed Dose Rate"><KPI value={ov.missed_dose_rate != null ? `${ov.missed_dose_rate.toFixed(1)}%` : '--'} label="Missed doses" color="#e74c3c" /></Card>

          {/* KPI Row 2 */}
          <Card title="Avg Side Effect Severity"><KPI value={ov.avg_side_effect_severity != null ? ov.avg_side_effect_severity.toFixed(2) : '--'} label="Severity (1-10)" color="#f59e0b" /></Card>
          <Card title="Most Common Side Effect"><KPI value={ov.most_common_side_effect || '--'} label="Top side effect" color="#ec4899" /></Card>
          <Card title="Total Refills"><KPI value={ov.total_refills} label="Refills recorded" color="#06b6d4" /></Card>
          <Card title="Rescue Med Usage"><KPI value={ov.rescue_med_usage} label="Rescue doses" color="#f97316" /></Card>

          {/* Adherence by Drug Bar Chart */}
          <Card title="Adherence by Drug" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.adherence_by_drug || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="drug" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="adherence_pct" name="Adherence %" radius={[4, 4, 0, 0]}>
                  {(ov.adherence_by_drug || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Side Effect Distribution Bar Chart */}
          <Card title="Side Effect Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.side_effect_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="side_effect" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]}>
                  {(ov.side_effect_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Drug Distribution Pie Chart */}
          <Card title="Drug Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.drug_distribution || []} dataKey="count" nameKey="drug" cx="50%" cy="50%" outerRadius={80} label={e => `${e.drug}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.drug_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Adherence Trend 30d Line Chart */}
          <Card title="Adherence Trend (30 Days)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={ov.adherence_trend_30d || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={10} angle={-30} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Line type="monotone" dataKey="adherence_pct" name="Adherence %" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Adherence by Time of Day Bar Chart */}
          <Card title="Adherence by Time of Day" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.adherence_by_time_of_day || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time_of_day" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="adherence_pct" name="Adherence %" radius={[4, 4, 0, 0]}>
                  {(ov.adherence_by_time_of_day || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Spacer for grid alignment */}
          <Card title="" span={2}>
            <div style={{ height: 220 }} />
          </Card>
        </div>
      )}

      {/* ADHERENCE ANALYSIS TAB */}
      {tab === 'adherence_analysis' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Adherence by Drug Table */}
          <Card title="Adherence by Drug">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Drug', 'Adherence %', 'Total Doses', 'Missed'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.adherence_by_drug || ov.adherence_by_drug || []).map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(d.drug)}</td>
                      <td style={{ padding: '6px', fontWeight: 600, color: d.adherence_pct >= 80 ? '#10b981' : '#e74c3c' }}>
                        {d.adherence_pct != null ? `${d.adherence_pct.toFixed(1)}%` : '--'}
                      </td>
                      <td style={{ padding: '6px' }}>{fmt(d.total_doses)}</td>
                      <td style={{ padding: '6px', color: '#e74c3c', fontWeight: 600 }}>{fmt(d.missed)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Adherence by Time of Day Breakdown */}
          <Card title="Adherence by Time of Day">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.adherence_by_time_of_day || ov.adherence_by_time_of_day || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time_of_day" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="adherence_pct" name="Adherence %" radius={[4, 4, 0, 0]}>
                  {(bd.adherence_by_time_of_day || ov.adherence_by_time_of_day || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* SIDE EFFECTS TAB */}
      {tab === 'side_effects' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Avg Side Effect Severity KPI */}
          <Card title="Average Side Effect Severity">
            <KPI
              value={ov.avg_side_effect_severity != null ? ov.avg_side_effect_severity.toFixed(2) : '--'}
              label="Severity score (1-10)"
              color="#f59e0b"
            />
          </Card>

          {/* Side Effect Distribution Bar Chart */}
          <Card title="Side Effect Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.side_effect_distribution || ov.side_effect_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="side_effect" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count" radius={[4, 4, 0, 0]}>
                  {(bd.side_effect_distribution || ov.side_effect_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Side Effect Table */}
          <Card title="Side Effect Breakdown">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Side Effect', 'Count'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.side_effect_distribution || ov.side_effect_distribution || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(s.side_effect)}</td>
                      <td style={{ padding: '6px' }}>{fmt(s.count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patient_detail' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {(bd.patients || []).slice(0, 5).map((p, pi) => (
            <Card key={pi} title={`Patient ${p.patient_id}`}>
              {/* Stats Summary */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12, marginBottom: 16 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#3b82f6' }}>
                    {Array.isArray(p.drugs) ? p.drugs.length : fmt(p.drugs)}
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Drugs</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: p.adherence_rate >= 80 ? '#10b981' : '#e74c3c' }}>
                    {p.adherence_rate != null ? `${p.adherence_rate.toFixed(1)}%` : '--'}
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Adherence Rate</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#e74c3c' }}>{fmt(p.missed_doses_30d)}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Missed (30d)</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#06b6d4' }}>{fmt(p.last_refill_date)}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Last Refill</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 18, fontWeight: 700 }}>
                    <Badge
                      text={p.refill_due || '--'}
                      color={p.refill_due === 'overdue' ? '#e74c3c' : p.refill_due === 'due_soon' ? '#f59e0b' : '#10b981'}
                    />
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Refill Status</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#8b5cf6' }}>
                    {Array.isArray(p.drugs) ? p.drugs.join(', ') : fmt(p.drugs)}
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>Drug List</div>
                </div>
              </div>

              {/* Side Effects Badges */}
              <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Side Effects</h4>
              {Array.isArray(p.side_effects) && p.side_effects.length > 0 ? (
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {p.side_effects.map((se, si) => (
                    <Badge key={si} text={se} color={COLORS[si % COLORS.length]} />
                  ))}
                </div>
              ) : (
                <p style={{ color: '#94a3b8', fontSize: 12 }}>No side effects reported.</p>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Medication Management Concepts">
            {(defs.concepts || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No definitions available.</p>
            ) : (
              (defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.term || item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.definition || item.description}</div>
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
