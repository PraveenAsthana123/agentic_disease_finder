import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function SeizureBadge({ occurred }) {
  const yes = occurred === true || occurred === 1 || occurred === 'yes'
  const color = yes ? '#ef4444' : '#10b981'
  const label = yes ? 'SEIZURE' : 'NONE'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{label}</span>
  )
}

function AdherenceBadge({ adherence }) {
  const yes = adherence === true || adherence === 1 || adherence === 'yes'
  const color = yes ? '#10b981' : '#ef4444'
  const label = yes ? 'ADHERENT' : 'MISSED'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{label}</span>
  )
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

export default function TriggerLogDashboard() {
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
          axios.get(`${API_URL}/trigger-logs/overview`),
          axios.get(`${API_URL}/trigger-logs/breakdown`),
          axios.get(`${API_URL}/trigger-logs/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load trigger log data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading trigger log data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Trigger log data not available</div>

  const tabs = ['overview', 'triggers', 'patients', 'definitions']
  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Seizure Trigger Log Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Seizure trigger tracking and lifestyle analytics — {fmt(kpis.total_logs)} logs across {fmt(kpis.total_patients)} patients
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Total Logs" value={kpis.total_logs} />
              <KPI label="Total Patients" value={kpis.total_patients} />
              <KPI label="Seizure Rate" value={kpis.seizure_rate_pct} sub="%" color="#ef4444" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Total Seizure Events" value={kpis.total_seizure_events} color="#ef4444" />
              <KPI label="Avg Sleep (hrs)" value={kpis.avg_sleep_hours} color="#3b82f6" />
              <KPI label="Med Adherence" value={kpis.medication_adherence_pct} sub="%" color="#10b981" />
            </div>
          </Card>

          <Card title="Primary Trigger Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.primary_trigger_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="trigger" type="category" tick={{ fontSize: 11 }} width={140} />
                <Tooltip formatter={(v, name, props) => [`${v} (${props.payload.pct}%)`, 'Count']} />
                <Bar dataKey="count" name="Logs" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                  {(overview.primary_trigger_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Monthly Seizure Trend" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={overview.seizure_by_month || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Line type="monotone" dataKey="seizures" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Seizures" />
                <Line type="monotone" dataKey="total_logs" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Total Logs" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Sleep Quality Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.sleep_quality_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="quality" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Logs" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {(overview.sleep_quality_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Lifestyle Averages: Seizure vs No Seizure" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Factor</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>With Seizure</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Without Seizure</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.lifestyle_averages || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{(row.factor || '').replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#ef4444' }}>{fmt(row.with_seizure)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#10b981' }}>{fmt(row.without_seizure)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Triggers tab ── */}
      {tab === 'triggers' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Stress Level vs Seizure Rate">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Stress Level</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Seizure Rate (%)</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Log Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.stress_vs_seizure || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{fmt(row.stress_level)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#ef4444' }}>{fmt(row.seizure_pct)}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="High Risk Days">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#fef2f2' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Primary Trigger</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Sleep (hrs)</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Stress</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Fatigue</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Missed Doses</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.high_risk_days || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.log_date || '').slice(0, 10)}</td>
                      <td style={{ padding: '8px 12px' }}>{(row.primary_trigger || '').replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.sleep_hours)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.stress_level)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.fatigue_level)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: row.missed_doses > 0 ? '#ef4444' : '#64748b' }}>{fmt(row.missed_doses)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Adherence Issues">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Primary Trigger</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Seizure</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Adherence</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.adherence_issues || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.log_date || '').slice(0, 10)}</td>
                      <td style={{ padding: '8px 12px' }}>{(row.primary_trigger || '').replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><SeizureBadge occurred={row.seizure_occurred} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><AdherenceBadge adherence={row.medication_adherence} /></td>
                      <td style={{ padding: '8px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, color: '#64748b' }}>{row.notes || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Patients tab ── */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Logs</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Seizures</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Seizure Rate</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Sleep</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Stress</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Mood</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Adherence %</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.total_logs)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: row.seizures > 0 ? '#ef4444' : '#64748b' }}>{fmt(row.seizures)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ flex: 1, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.min(row.seizure_rate || 0, 100)}%`, height: '100%',
                              background: (row.seizure_rate || 0) > 50 ? '#ef4444' : (row.seizure_rate || 0) > 25 ? '#f59e0b' : '#10b981',
                              borderRadius: 4
                            }} />
                          </div>
                          <span style={{ fontSize: 12, color: '#64748b', minWidth: 36 }}>{fmt(row.seizure_rate)}%</span>
                        </div>
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.avg_sleep)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.avg_stress)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.avg_mood)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: (row.adherence_pct || 0) < 80 ? '#ef4444' : '#10b981' }}>{fmt(row.adherence_pct)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Recent Logs">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Date</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Sleep (hrs)</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Sleep Quality</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Stress</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Mood</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Seizure</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Trigger</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Adherence</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_logs || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.log_date || '').slice(0, 10)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.sleep_hours)}</td>
                      <td style={{ padding: '8px 12px' }}>{row.sleep_quality || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.stress_level)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.mood_score)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><SeizureBadge occurred={row.seizure_occurred} /></td>
                      <td style={{ padding: '8px 12px' }}>{(row.primary_trigger || '--').replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><AdherenceBadge adherence={row.medication_adherence} /></td>
                      <td style={{ padding: '8px 12px', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, color: '#64748b' }}>{row.notes || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {defs.trigger_descriptions && (
            <Card title="Trigger Descriptions">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.trigger_descriptions).map(([key, desc]) => (
                  <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {defs.field_descriptions && (
            <Card title="Field Descriptions">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.field_descriptions).map(([key, desc]) => (
                  <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {defs.clinical_notes && (
            <Card title="Clinical Notes">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {(Array.isArray(defs.clinical_notes) ? defs.clinical_notes : Object.values(defs.clinical_notes)).map((note, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 8, lineHeight: 1.5 }}>{note}</li>
                ))}
              </ul>
            </Card>
          )}

          {defs.glossary && (
            <Card title="Glossary">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.glossary).map(([term, definition]) => (
                  <div key={term} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>{term}</div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
