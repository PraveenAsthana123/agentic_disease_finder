import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const map = {
    'caregiver-responded': { bg: '#dcfce7', fg: '#166534' },
    'resolved-home': { bg: '#dbeafe', fg: '#1e40af' },
    'ems-dispatched': { bg: '#fef3c7', fg: '#92400e' },
    'er-visit': { bg: '#fee2e2', fg: '#991b1b' },
    'false-alarm': { bg: '#f1f5f9', fg: '#475569' },
  }
  const s = map[status] || { bg: '#f1f5f9', fg: '#475569' }
  return (
    <span style={{
      background: s.bg, color: s.fg, padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600
    }}>{status}</span>
  )
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(4)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

function fmtSec(v) {
  if (v == null) return '--'
  if (v < 60) return `${Math.round(v)}s`
  return `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`
}

const TABS = ['Overview', 'Patient Audit', 'Event Timeline', 'Alert Analysis', 'Methodology']

export default function ClosedLoopDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/closed-loop/overview`),
      axios.get(`${API}/closed-loop/breakdown`),
      axios.get(`${API}/closed-loop/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading closed-loop data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const kpis = ov?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: '0 0 4px' }}>
        Closed-Loop Neurostimulation Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 20px' }}>
        Pre-ictal detection → alert → response → outcome audit loop
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#1e293b' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: 600, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} ov={ov} />}
      {tab === 1 && <PatientAuditTab bd={bd} />}
      {tab === 2 && <TimelineTab bd={bd} />}
      {tab === 3 && <AlertTab bd={bd} ov={ov} />}
      {tab === 4 && <MethodologyTab df={df} />}
    </div>
  )
}

function OverviewTab({ kpis, ov }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Loop KPIs" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
          <KPI label="SOS Events" value={kpis.total_sos_events} color="#3b82f6" />
          <KPI label="Seizures Logged" value={kpis.total_seizures_logged} color="#8b5cf6" />
          <KPI label="Avg Response" value={fmtSec(kpis.avg_response_seconds)} color="#f59e0b" />
          <KPI label="Notification Rate" value={fmtPct(kpis.notification_rate)} color="#10b981" />
          <KPI label="False Alarm Rate" value={fmtPct(kpis.false_alarm_rate)} color="#ef4444" />
          <KPI label="Patient Coverage" value={fmtPct(kpis.loop_patient_coverage)} color="#06b6d4" />
        </div>
      </Card>

      <Card title="Outcome Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={ov?.outcome_distribution || []} dataKey="count" nameKey="outcome"
              cx="50%" cy="50%" outerRadius={90} label={({ outcome, pct }) => `${outcome} ${fmtPct(pct)}`}>
              {(ov?.outcome_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Trigger Method" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={ov?.trigger_breakdown || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="method" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Response Time Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={ov?.response_time_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Event Types" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={ov?.event_type_breakdown || []} dataKey="count" nameKey="type"
              cx="50%" cy="50%" outerRadius={90} label={({ type, pct }) => `${type} ${fmtPct(pct)}`}>
              {(ov?.event_type_breakdown || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Response Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, padding: '20px 0' }}>
          <KPI label="Median Response" value={fmtSec(kpis.median_response_seconds)} color="#3b82f6" />
          <KPI label="P95 Response" value={fmtSec(kpis.p95_response_seconds)} color="#f59e0b" />
          <KPI label="Location Shared" value={fmtPct(kpis.location_sharing_rate)} color="#10b981" />
        </div>
      </Card>

      <Card title="Risk Score Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={ov?.risk_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="tier" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function PatientAuditTab({ bd }) {
  const patients = bd?.per_patient || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Per-Patient Closed-Loop Audit (${patients.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Patient', 'SOS Events', 'Seizures', 'Avg Response', 'Min/Max', 'False Alarms', 'FAR', 'Top Outcome'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => {
                const topOutcome = Object.entries(p.outcomes || {}).sort((a, b) => b[1] - a[1])[0]
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 10px' }}>{p.sos_events}</td>
                    <td style={{ padding: '8px 10px' }}>{p.seizures_logged}</td>
                    <td style={{ padding: '8px 10px' }}>{fmtSec(p.avg_response_s)}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b' }}>
                      {p.min_response_s != null ? `${fmtSec(p.min_response_s)} / ${fmtSec(p.max_response_s)}` : '--'}
                    </td>
                    <td style={{ padding: '8px 10px' }}>{p.false_alarms}</td>
                    <td style={{ padding: '8px 10px' }}>{fmtPct(p.false_alarm_rate)}</td>
                    <td style={{ padding: '8px 10px' }}>{topOutcome ? <StatusBadge status={topOutcome[0]} /> : '--'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Response Time by Event Type">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={bd?.severity_analysis || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="event_type" tick={{ fontSize: 11 }} />
            <YAxis label={{ value: 'Avg Response (s)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
            <Tooltip />
            <Bar dataKey="avg_response_s" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Avg Response (s)" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function TimelineTab({ bd }) {
  const timeline = bd?.event_timeline || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Event Timeline (${timeline.length} events)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Date', 'Patient', 'Event Type', 'Trigger', 'Response Time', 'Notified', 'Location', 'Outcome', 'Notes'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {timeline.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 12 }}>{e.date?.split('T')[0] || '--'}</td>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{e.patient_id}</td>
                  <td style={{ padding: '8px 10px' }}>{e.event_type}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12 }}>{e.trigger_method}</td>
                  <td style={{ padding: '8px 10px' }}>{fmtSec(e.response_time_s)}</td>
                  <td style={{ padding: '8px 10px' }}>{e.responder_notified ? '✓' : '✗'}</td>
                  <td style={{ padding: '8px 10px' }}>{e.location_shared ? '✓' : '✗'}</td>
                  <td style={{ padding: '8px 10px' }}><StatusBadge status={e.outcome} /></td>
                  <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>{e.notes || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function AlertTab({ bd, ov }) {
  const audit = bd?.alert_audit || {}
  const trig = bd?.trigger_effectiveness || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="IoT Alert Audit" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
          <KPI label="Total IoT Alerts" value={audit.total_iot_alerts} color="#3b82f6" />
          <KPI label="Seizure-Specific" value={audit.seizure_specific} color="#ef4444" />
          <KPI label="Critical" value={audit.critical_alerts} color="#f59e0b" />
          <KPI label="Resolved" value={audit.resolved_count} color="#10b981" />
          <KPI label="Acknowledged" value={audit.acknowledged_count} color="#8b5cf6" />
        </div>
      </Card>

      <Card title="Alerts by Type">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={Object.entries(audit.by_type || {}).map(([k, v]) => ({ type: k, count: v }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="type" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Alerts by Severity">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={Object.entries(audit.by_severity || {}).map(([k, v]) => ({ name: k, value: v }))}
              dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
              label={({ name, value }) => `${name}: ${value}`}>
              {Object.keys(audit.by_severity || {}).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Trigger Method → Outcome Effectiveness" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b' }}>Trigger Method</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b' }}>Outcomes</th>
              </tr>
            </thead>
            <tbody>
              {trig.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{t.method}</td>
                  <td style={{ padding: '8px 10px' }}>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {Object.entries(t.outcomes || {}).map(([k, v]) => (
                        <span key={k}><StatusBadge status={k} /> <span style={{ fontSize: 12, color: '#475569' }}>×{v}</span></span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function MethodologyTab({ df }) {
  if (!df) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Closed-Loop Stages">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {(df.loop_stages || []).map((s, i) => (
            <div key={i} style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
              <div style={{
                background: COLORS[i % COLORS.length], color: '#fff', borderRadius: '50%',
                width: 36, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontWeight: 700, fontSize: 16, flexShrink: 0
              }}>{s.stage}</div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{s.name}</div>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>{s.description}</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>Data: {s.data_source}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Glossary">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {(df.glossary || []).map((g, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b' }}>{g.term}</div>
              <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>{g.definition}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Clinical References">
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {(df.clinical_references || []).map((r, i) => (
            <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{r}</li>
          ))}
        </ul>
      </Card>

      <Card title="Data Sources">
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {(df.data_tables_used || []).map((t, i) => (
            <li key={i} style={{ fontSize: 13, color: '#334155', marginBottom: 4 }}>{t}</li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
