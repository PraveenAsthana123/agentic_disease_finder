import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { built: '#4caf50', partial: '#ff9800', missing: '#f44336' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function Badge({ status }) {
  const bg = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {status}
    </span>
  )
}

export default function ProductManagerDashboard() {
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
          axios.get(`${API_URL}/product-manager/overview`),
          axios.get(`${API_URL}/product-manager/breakdown`),
          axios.get(`${API_URL}/product-manager/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Product Manager data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading product manager data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Product manager data not available.'}
    </div>
  )

  const summary = overview.summary || {}
  const stakeholders = overview.stakeholder_readiness || []
  const processStatus = overview.process_status || []
  const funcStatus = overview.functionality_status || []
  const bkStakeholders = breakdown?.stakeholders || []
  const costLevers = breakdown?.cost_levers || []
  const revenueLevers = breakdown?.revenue_levers || []
  const productivityLevers = breakdown?.productivity_levers || []
  const phases = breakdown?.phases || []
  const patientSections = breakdown?.patient_sections || []
  const opsDashboards = breakdown?.ops_dashboards || []
  const definitions = defs?.definitions || []

  const stakeholderChart = stakeholders.map(s => ({ name: s.role, readiness: s.readiness_pct, built: s.built, missing: s.missing }))
  const processPie = [
    { name: 'Built', value: summary.processes_built || 0 },
    { name: 'Partial', value: summary.processes_partial || 0 },
    { name: 'Missing', value: summary.processes_missing || 0 },
  ].filter(d => d.value > 0)
  const funcPie = [
    { name: 'Built', value: summary.functionality_built || 0 },
    { name: 'Partial', value: summary.functionality_partial || 0 },
    { name: 'Missing', value: summary.functionality_missing || 0 },
  ].filter(d => d.value > 0)
  const phasePie = [
    { name: 'Built', value: summary.phases_built || 0 },
    { name: 'Missing', value: summary.phases_missing || 0 },
  ].filter(d => d.value > 0)

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 130
  })
  const tabs = ['overview', 'stakeholders', 'business case', 'roadmap', 'definitions']
  const tabStyle = (t) => ({
    padding: '8px 16px', cursor: 'pointer', borderRadius: '6px 6px 0 0', fontSize: 13, fontWeight: 600,
    background: tab === t ? '#1e88e5' : '#f1f5f9', color: tab === t ? '#fff' : '#475569',
    border: tab === t ? '1px solid #1e88e5' : '1px solid #e2e8f0', borderBottom: 'none'
  })

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 22, color: '#1e293b' }}>Product Manager Dashboard</h2>

      <div style={{ display: 'flex', gap: 4, marginBottom: 18, flexWrap: 'wrap' }}>
        {tabs.map(t => <div key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</div>)}
      </div>

      {tab === 'overview' && (
        <>
          {/* KPI cards */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {[
              { label: 'Overall Readiness', value: `${summary.overall_readiness_pct}%`, color: COLORS[3] },
              { label: 'Stakeholder Roles', value: summary.total_stakeholder_roles, color: COLORS[4] },
              { label: 'Capabilities Built', value: summary.total_built_capabilities, color: COLORS[0] },
              { label: 'Capabilities Missing', value: summary.total_missing_capabilities, color: COLORS[2] },
              { label: 'Process Maturity', value: `${summary.process_maturity_pct}%`, color: COLORS[5] },
              { label: 'Patient Module', value: `${summary.patient_module_built}/${summary.patient_module_total}`, color: COLORS[0] },
              { label: 'Ops Dashboards', value: `${summary.ops_dashboards_built}/${summary.ops_dashboards_total}`, color: COLORS[0] },
              { label: 'Business Levers', value: summary.business_case_levers, color: COLORS[6] },
            ].map((k, i) => (
              <div key={i} style={kpiStyle(k.color)}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>
                  {typeof k.value === 'number' ? fmt(k.value) : (k.value || '--')}
                </div>
              </div>
            ))}
          </div>

          {/* Stakeholder readiness bar + Process pie */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Stakeholder Readiness</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={stakeholderChart} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
                  <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="readiness" fill={COLORS[3]} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Process Maturity</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={processPie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {processPie.map((entry, i) => <Cell key={i} fill={STATUS_COLORS[entry.name.toLowerCase()] || COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Functionality coverage + Phase status */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Functionality Coverage</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={funcPie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {funcPie.map((entry, i) => <Cell key={i} fill={STATUS_COLORS[entry.name.toLowerCase()] || COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Implementation Phases</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={phasePie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {phasePie.map((entry, i) => <Cell key={i} fill={STATUS_COLORS[entry.name.toLowerCase()] || COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Process status table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Process Status</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Process</th>
                    <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {processStatus.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}>{p.name}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><Badge status={p.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Functionality status table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Functionality Coverage Detail</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Capability</th>
                    <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {funcStatus.map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}>{f.capability}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><Badge status={f.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {tab === 'stakeholders' && (
        <>
          {bkStakeholders.map((sh, idx) => (
            <div key={idx} style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>{sh.icon} {sh.role}</h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
                <div>
                  <h5 style={{ margin: '0 0 8px', color: '#4caf50', fontSize: 13 }}>Built ({sh.built.length})</h5>
                  <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: 13, color: '#334155' }}>
                    {sh.built.map((b, i) => <li key={i} style={{ marginBottom: 4 }}>{b}</li>)}
                  </ul>
                </div>
                <div>
                  <h5 style={{ margin: '0 0 8px', color: '#f44336', fontSize: 13 }}>Missing ({sh.missing.length})</h5>
                  <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: 13, color: '#64748b' }}>
                    {sh.missing.map((m, i) => <li key={i} style={{ marginBottom: 4 }}>{m}</li>)}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </>
      )}

      {tab === 'business case' && (
        <>
          {/* Cost decrease levers */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Cost Decrease Levers</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Lever</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {costLevers.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{c.lever}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.impact}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Revenue increase levers */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Revenue Increase Levers</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Lever</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {revenueLevers.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.lever}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{r.impact}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Productivity levers */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Productivity Increase Levers</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Role / Area</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {productivityLevers.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.lever}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{p.impact}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Strategy */}
          {breakdown?.strategy && (
            <div style={{ ...cardStyle, background: '#eff6ff', border: '1px solid #bfdbfe' }}>
              <h4 style={{ margin: '0 0 8px', color: '#1e40af', fontSize: 14 }}>Strategy</h4>
              <p style={{ margin: 0, fontSize: 13, color: '#1e3a5f', lineHeight: 1.6 }}>{breakdown.strategy}</p>
            </div>
          )}
        </>
      )}

      {tab === 'roadmap' && (
        <>
          {/* Implementation phases */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Implementation Phases</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Phase</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Scope</th>
                    <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {phases.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.phase}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{p.scope}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><Badge status={p.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Patient module sections */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Patient Module Sections</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Section</th>
                    <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Fields</th>
                    <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {patientSections.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}>{s.section}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{s.fields}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><Badge status={s.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Ops dashboards */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Ops Dashboards</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Dashboard</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Purpose</th>
                    <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {opsDashboards.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}>{d.label}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{d.purpose}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><Badge status={d.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {tab === 'definitions' && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Definitions</h4>
          {definitions.map((d, i) => (
            <div key={i} style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, padding: '12px 16px', marginBottom: 10 }}>
              <strong style={{ color: '#1e40af' }}>{d.term}</strong>
              <p style={{ margin: '4px 0 0', fontSize: 13, color: '#334155', lineHeight: 1.5 }}>{d.definition}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
