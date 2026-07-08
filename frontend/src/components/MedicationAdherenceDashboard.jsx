import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend
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

const fmt = v => (v != null ? v : '--')
const pct = v => (v != null ? `${v}%` : '--')

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const ADHERENCE_COLORS = { taken: '#10b981', late: '#f59e0b', missed: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'by_drug', label: 'By Drug' },
  { id: 'by_patient', label: 'By Patient' },
  { id: 'missed_late', label: 'Missed & Late' },
  { id: 'definitions', label: 'Definitions' },
]

export default function MedicationAdherenceDashboard() {
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
      axios.get(`${API_URL}/api/medication-adherence/overview`),
      axios.get(`${API_URL}/api/medication-adherence/breakdown`),
      axios.get(`${API_URL}/api/medication-adherence/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Medication Adherence Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Medication Adherence Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Antiseizure medication adherence tracking — taken/late/missed rates, drug-level analysis, side effects, refill monitoring
      </p>

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
          <Card title="Total Dose Logs"><KPI value={fmt(ov.total_logs)} label="Dose records" color="#3b82f6" /></Card>
          <Card title="Patients"><KPI value={fmt(ov.total_patients)} label="Unique patients" color="#8b5cf6" /></Card>
          <Card title="Adherence Rate"><KPI value={pct(ov.adherence_rate)} label="Doses taken on time" color="#10b981" /></Card>
          <Card title="Late Rate"><KPI value={pct(ov.late_rate)} label="Doses taken late" color="#f59e0b" /></Card>

          <Card title="Missed Rate"><KPI value={pct(ov.missed_rate)} label="Doses missed" color="#ef4444" /></Card>
          <Card title="Avg Minutes Late"><KPI value={ov.avg_minutes_late != null ? `${ov.avg_minutes_late} min` : '--'} label="When late" color="#f97316" /></Card>
          <Card title="Side Effect Severity"><KPI value={fmt(ov.avg_side_effect_severity)} label="Average (1-10)" color="#ec4899" /></Card>
          <Card title="Unique Drugs"><KPI value={fmt(ov.unique_drugs)} label="ASM medications" color="#06b6d4" /></Card>

          {/* Adherence by Drug */}
          <Card title="Adherence by Drug" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov.adherence_by_drug || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} fontSize={11} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="drug" fontSize={11} width={110} />
                <Tooltip formatter={v => `${v}%`} />
                <Legend />
                <Bar dataKey="taken_pct" name="Taken" fill={ADHERENCE_COLORS.taken} stackId="a" />
                <Bar dataKey="late_pct" name="Late" fill={ADHERENCE_COLORS.late} stackId="a" />
                <Bar dataKey="missed_pct" name="Missed" fill={ADHERENCE_COLORS.missed} stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Weekly Adherence Trend */}
          <Card title="Weekly Adherence Trend" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={ov.adherence_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="week" fontSize={10} angle={-20} textAnchor="end" height={50} />
                <YAxis domain={[0, 100]} fontSize={11} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={v => `${v}%`} />
                <Line type="monotone" dataKey="adherence_rate" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Adherence %" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Frequency Distribution */}
          <Card title="Dose Frequency Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.frequency_distribution || []} dataKey="count" nameKey="frequency" cx="50%" cy="50%" outerRadius={80} label={e => `${e.frequency}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.frequency_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Top Side Effects */}
          <Card title="Top Side Effects" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.side_effect_frequency || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="side_effect" fontSize={10} angle={-25} textAnchor="end" height={60} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Occurrences" fill="#ec4899" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Time of Day */}
          <Card title="Adherence by Time of Day" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.time_of_day || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scheduled_time" fontSize={11} />
                <YAxis domain={[0, 100]} fontSize={11} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={v => `${v}%`} />
                <Legend />
                <Bar dataKey="taken_pct" name="Taken" fill={ADHERENCE_COLORS.taken} />
                <Bar dataKey="late_pct" name="Late" fill={ADHERENCE_COLORS.late} />
                <Bar dataKey="missed_pct" name="Missed" fill={ADHERENCE_COLORS.missed} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Mood Distribution */}
          <Card title="Post-Dose Mood Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.mood_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mood_score" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Dose logs" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* BY DRUG TAB */}
      {tab === 'by_drug' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Drug Adherence Analysis">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Drug</th>
                    <th style={{ padding: '8px 12px' }}>Total Doses</th>
                    <th style={{ padding: '8px 12px' }}>Adherence</th>
                    <th style={{ padding: '8px 12px' }}>Late</th>
                    <th style={{ padding: '8px 12px' }}>Missed</th>
                    <th style={{ padding: '8px 12px' }}>Avg Severity</th>
                    <th style={{ padding: '8px 12px' }}>Common Side Effects</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.per_drug || []).map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{d.drug_name}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(d.total_doses)}</td>
                      <td style={{ padding: '8px 12px', color: d.adherence_rate >= 80 ? '#10b981' : d.adherence_rate >= 50 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>{pct(d.adherence_rate)}</td>
                      <td style={{ padding: '8px 12px', color: '#f59e0b' }}>{pct(d.late_rate)}</td>
                      <td style={{ padding: '8px 12px', color: '#ef4444' }}>{pct(d.missed_rate)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(d.avg_severity)}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{(d.common_side_effects || []).join(', ') || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* BY PATIENT TAB */}
      {tab === 'by_patient' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Adherence Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Patient</th>
                    <th style={{ padding: '8px 12px' }}>Total Doses</th>
                    <th style={{ padding: '8px 12px' }}>Adherence</th>
                    <th style={{ padding: '8px 12px' }}>Late</th>
                    <th style={{ padding: '8px 12px' }}>Missed</th>
                    <th style={{ padding: '8px 12px' }}>Avg Min Late</th>
                    <th style={{ padding: '8px 12px' }}>Drugs</th>
                    <th style={{ padding: '8px 12px' }}>Top Side Effects</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.per_patient || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(p.total_doses)}</td>
                      <td style={{ padding: '8px 12px', color: p.adherence_rate >= 80 ? '#10b981' : p.adherence_rate >= 50 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>{pct(p.adherence_rate)}</td>
                      <td style={{ padding: '8px 12px', color: '#f59e0b' }}>{pct(p.late_rate)}</td>
                      <td style={{ padding: '8px 12px', color: '#ef4444' }}>{pct(p.missed_rate)}</td>
                      <td style={{ padding: '8px 12px' }}>{p.avg_minutes_late != null ? `${p.avg_minutes_late} min` : '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{(p.drugs || []).join(', ')}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{(p.top_side_effects || []).join(', ') || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Refills */}
          <Card title="Medication Refills">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Patient</th>
                    <th style={{ padding: '8px 12px' }}>Drug</th>
                    <th style={{ padding: '8px 12px' }}>Refill Date</th>
                    <th style={{ padding: '8px 12px' }}>Qty</th>
                    <th style={{ padding: '8px 12px' }}>Pharmacy</th>
                    <th style={{ padding: '8px 12px' }}>Auto-Refill</th>
                    <th style={{ padding: '8px 12px' }}>Days Supply</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.refills || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}>{r.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{r.drug_name}</td>
                      <td style={{ padding: '8px 12px' }}>{r.refill_date}</td>
                      <td style={{ padding: '8px 12px' }}>{r.quantity}</td>
                      <td style={{ padding: '8px 12px' }}>{r.pharmacy}</td>
                      <td style={{ padding: '8px 12px' }}>{r.auto_refill ? 'Yes' : 'No'}</td>
                      <td style={{ padding: '8px 12px' }}>{r.days_supply}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* MISSED & LATE TAB */}
      {tab === 'missed_late' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Recent Missed Doses">
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px' }}>Patient</th>
                    <th style={{ padding: '6px 8px' }}>Drug</th>
                    <th style={{ padding: '6px 8px' }}>Date</th>
                    <th style={{ padding: '6px 8px' }}>Time</th>
                    <th style={{ padding: '6px 8px' }}>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.recent_missed || []).map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px' }}>{m.patient_id}</td>
                      <td style={{ padding: '6px 8px' }}>{m.drug_name}</td>
                      <td style={{ padding: '6px 8px' }}>{m.log_date}</td>
                      <td style={{ padding: '6px 8px' }}>{m.scheduled_time}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{m.notes || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Recent Late Doses">
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px' }}>Patient</th>
                    <th style={{ padding: '6px 8px' }}>Drug</th>
                    <th style={{ padding: '6px 8px' }}>Date</th>
                    <th style={{ padding: '6px 8px' }}>Time</th>
                    <th style={{ padding: '6px 8px' }}>Min Late</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.recent_late || []).map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px' }}>{l.patient_id}</td>
                      <td style={{ padding: '6px 8px' }}>{l.drug_name}</td>
                      <td style={{ padding: '6px 8px' }}>{l.log_date}</td>
                      <td style={{ padding: '6px 8px' }}>{l.scheduled_time}</td>
                      <td style={{ padding: '6px 8px', color: l.minutes_late > 60 ? '#ef4444' : '#f59e0b', fontWeight: 600 }}>{l.minutes_late}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {(defs.terms || []).map((d, i) => (
            <Card key={i} title={d.term}>
              <p style={{ fontSize: 13, color: '#475569', margin: 0, lineHeight: 1.6 }}>{d.definition}</p>
              {d.threshold && <p style={{ fontSize: 12, color: '#64748b', marginTop: 8, fontStyle: 'italic' }}>Threshold: {d.threshold}</p>}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
