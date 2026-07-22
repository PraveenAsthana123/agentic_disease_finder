import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(4)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

const TABS = ['Overview', 'Input Rails', 'Output Rails', 'Dialog Flows', 'Methodology']

export default function GuardrailsDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/guardrails/overview`),
      axios.get(`${API}/api/guardrails/breakdown`),
      axios.get(`${API}/api/guardrails/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Guardrails analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>NeMo Guardrails Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        AI safety rail monitoring — {fmt(kpis.total_requests)} requests processed, {fmt(kpis.rails_triggered)} rails triggered
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#1e293b' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontSize: 13, fontWeight: 500
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} charts={ov.charts || {}} />}
      {tab === 1 && <InputRailsTab data={bd?.input_rails || []} severity={bd?.severity_breakdown || []} />}
      {tab === 2 && <OutputRailsTab data={bd?.output_rails || []} />}
      {tab === 3 && <DialogFlowsTab data={bd?.dialog_flows || []} />}
      {tab === 4 && <MethodologyTab data={df} />}
    </div>
  )
}

function OverviewTab({ kpis, charts }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Requests" value={fmt(kpis.total_requests)} /></Card>
      <Card><KPI label="Rails Triggered" value={fmt(kpis.rails_triggered)} color="#f59e0b" /></Card>
      <Card><KPI label="Block Rate" value={fmtPct(kpis.block_rate)} color="#ef4444" /></Card>
      <Card><KPI label="Avg Latency" value={`${kpis.avg_latency_ms}ms`} color="#3b82f6" /></Card>

      <Card title="Rail Triggers Over Time" span={4}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={charts.rail_triggers_over_time || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="input_rails" stroke={COLORS[0]} strokeWidth={2} name="Input Rails" />
            <Line type="monotone" dataKey="output_rails" stroke={COLORS[1]} strokeWidth={2} name="Output Rails" />
            <Line type="monotone" dataKey="dialog_rails" stroke={COLORS[2]} strokeWidth={2} name="Dialog Rails" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Rail Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={charts.rail_type_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="type" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill={COLORS[1]} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Safety Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, padding: '12px 0' }}>
          <KPI label="Topic Violations" value={fmt(kpis.topic_violations)} color="#f59e0b" />
          <KPI label="Output Filtered" value={fmt(kpis.output_filtered)} color="#8b5cf6" />
          <KPI label="Jailbreak Blocked" value={fmt(kpis.jailbreak_blocked)} color="#ef4444" />
          <KPI label="Hallucination Caught" value={fmt(kpis.hallucination_caught)} color="#06b6d4" />
        </div>
      </Card>
    </div>
  )
}

function InputRailsTab({ data, severity }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Input Rail Performance" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="rail_name" type="category" tick={{ fontSize: 11 }} width={130} />
            <Tooltip />
            <Bar dataKey="triggers" fill={COLORS[0]} radius={[0, 4, 4, 0]} name="Triggers" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Input Rails Detail">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Rail</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Triggers</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Block Rate</th>
              </tr>
            </thead>
            <tbody>
              {data.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{r.rail_name}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{r.triggers}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmtPct(r.block_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Severity Breakdown">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={severity}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="severity" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {severity.map((_, i) => (
                <rect key={i} fill={[COLORS[3], COLORS[2], COLORS[1], COLORS[0]][i] || COLORS[7]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function OutputRailsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Output Rail Triggers" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="rail_name" type="category" tick={{ fontSize: 11 }} width={140} />
            <Tooltip />
            <Bar dataKey="triggers" fill={COLORS[4]} radius={[0, 4, 4, 0]} name="Triggers" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Output Rails Detail" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Rail</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Triggers</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Block Rate</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {data.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6, fontWeight: 500 }}>{r.rail_name}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{r.triggers}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmtPct(r.block_rate)}</td>
                  <td style={{ padding: 6, color: '#64748b' }}>{r.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DialogFlowsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Dialog Flow Activations" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="flow_name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="activations" fill={COLORS[1]} radius={[4, 4, 0, 0]} name="Activations" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Flow Success Rates" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="flow_name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} tickFormatter={v => fmtPct(v)} />
            <Tooltip formatter={v => fmtPct(v)} />
            <Bar dataKey="success_rate" fill={COLORS[0]} radius={[4, 4, 0, 0]} name="Success Rate" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Dialog Flow Details" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Flow</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Activations</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Success Rate</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Avg Turns</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {data.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6, fontWeight: 500 }}>{r.flow_name}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{r.activations}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmtPct(r.success_rate)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{r.avg_turns}</td>
                  <td style={{ padding: 6, color: '#64748b' }}>{r.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function MethodologyTab({ data }) {
  if (!data) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Methodology" span={2}>
        <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{data.methodology}</p>
      </Card>

      <Card title="Rail Types">
        {(data.rail_types || []).map((r, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{r.name}</div>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{r.description}</div>
          </div>
        ))}
      </Card>

      <Card title="Clinical Relevance">
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
          {(data.clinical_relevance || []).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </Card>

      <Card title="Strengths">
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#10b981', lineHeight: 1.8 }}>
          {(data.strengths || []).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </Card>

      <Card title="Limitations">
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#f59e0b', lineHeight: 1.8 }}>
          {(data.limitations || []).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </Card>

      <Card title="Interpretation Notes" span={2}>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
          {(data.interpretation_notes || []).map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </Card>
    </div>
  )
}
