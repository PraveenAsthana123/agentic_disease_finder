import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { pass: '#4caf50', partial: '#ff9800', planned: '#f44336', built: '#4caf50', missing: '#f44336', Accepted: '#4caf50', 'In Progress': '#ff9800', Pending: '#f44336', Built: '#4caf50', Partial: '#ff9800', Missing: '#f44336' }

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

function Card({ title, children, style }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, border: '1px solid #e2e8f0',
      padding: 16, ...style
    }}>
      {title && <div style={{ fontWeight: 600, fontSize: 13, color: '#475569', marginBottom: 8 }}>{title}</div>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <Card>
      <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{fmt(value)}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </Card>
  )
}

export default function FunctionalBADashboard() {
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
          axios.get(`${API_URL}/functional-ba/overview`),
          axios.get(`${API_URL}/functional-ba/breakdown`),
          axios.get(`${API_URL}/functional-ba/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Functional/BA data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading Functional/BA data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 24, background: '#fef2f2', borderRadius: 8, color: '#dc2626', margin: 16 }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 24, background: '#fffbeb', borderRadius: 8, color: '#b45309', margin: 16 }}>
      Functional/BA data not available. {overview?.note || ''}
    </div>
  )

  const tabs = ['overview', 'traceability', 'gaps', 'acceptance', 'definitions']
  const s = overview.summary || {}

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 20, color: '#1e293b' }}>Functional / BA Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Requirements traceability, acceptance criteria, UAT readiness, process maturity
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '1px solid #e2e8f0', paddingBottom: 8 }}>
        {tabs.map(t => (
          <div key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t ? '#1e293b' : 'transparent',
            color: tab === t ? '#fff' : '#64748b',
            transition: 'all 0.15s'
          }}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </div>
        ))}
      </div>

      {/* ===== OVERVIEW TAB ===== */}
      {tab === 'overview' && (
        <div>
          {/* KPI row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 16 }}>
            <KPI label="Requirements" value={s.total_requirements} sub="user stories" />
            <KPI label="Acceptance Criteria" value={s.total_acceptance_criteria} sub={`${s.accepted} accepted`} />
            <KPI label="Acceptance %" value={`${s.overall_acceptance_pct}%`} sub="weighted coverage" />
            <KPI label="Process Maturity" value={`${s.process_maturity_pct}%`} sub={`${s.processes_built}/${s.total_processes} built`} />
            <KPI label="Functionality" value={`${s.functionality_built}/${s.functionality_total}`} sub="capabilities built" />
            <KPI label="Test Dimensions" value={`${s.dimensions_built}/${s.testing_dimensions}`} sub="dimensions built" />
            <KPI label="Patient Module" value={`${s.patient_sections_built}/${s.patient_sections_total}`} sub="sections built" />
            <KPI label="Admin Roles" value={`${s.admin_roles_built}/${s.admin_roles_total}`} sub="roles built" />
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            {/* Acceptance distribution pie */}
            <Card title="Acceptance Criteria Status">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.acceptance_distribution || []} dataKey="count" nameKey="status"
                    cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                    {(overview.acceptance_distribution || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* UAT readiness by role */}
            <Card title="UAT Readiness by Role">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.role_uat_readiness || []} layout="vertical" margin={{ left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="role" width={110} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="uat_readiness_pct" fill="#1e88e5" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Process maturity pie */}
            <Card title="Process Status">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.process_distribution || []} dataKey="count" nameKey="status"
                    cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                    {(overview.process_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={STATUS_COLORS[entry.status] || COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Functionality coverage pie */}
            <Card title="Functionality Coverage">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.functionality_distribution || []} dataKey="count" nameKey="status"
                    cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                    {(overview.functionality_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={STATUS_COLORS[entry.status] || COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </div>
      )}

      {/* ===== TRACEABILITY TAB ===== */}
      {tab === 'traceability' && breakdown && (
        <div>
          <Card title="Requirements Traceability Matrix" style={{ marginBottom: 16 }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Persona', 'User Story', 'Endpoint', 'Tests', 'Pass', 'Partial', 'Planned'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.traceability || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 600 }}>{row.persona}</td>
                      <td style={{ padding: '8px 6px', maxWidth: 320, fontSize: 11, color: '#475569' }}>{row.story}</td>
                      <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 10, color: '#1e88e5' }}>{row.endpoint}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center' }}>{row.linked_tests}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center', color: '#4caf50', fontWeight: 600 }}>{row.pass}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center', color: '#ff9800', fontWeight: 600 }}>{row.partial}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center', color: '#f44336', fontWeight: 600 }}>{row.planned}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Demo Stories" style={{ marginBottom: 16 }}>
            {(breakdown.demo_stories || []).map((ds, i) => (
              <div key={i} style={{ padding: '10px 0', borderBottom: i < (breakdown.demo_stories || []).length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{ds.title}</div>
                <div style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>{ds.script}</div>
                <div style={{ fontSize: 11, color: '#1e88e5', marginTop: 2 }}>Shows: {ds.shows}</div>
              </div>
            ))}
          </Card>

          <Card title="Testing Dimensions">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Dimension', 'Tests', 'Method', 'Status'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.testing_dimensions || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 600 }}>{row.dimension}</td>
                      <td style={{ padding: '8px 6px', fontSize: 11, color: '#475569' }}>{row.tests}</td>
                      <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 10, color: '#64748b' }}>{row.method}</td>
                      <td style={{ padding: '8px 6px' }}><Badge status={row.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ===== GAPS TAB ===== */}
      {tab === 'gaps' && breakdown && (
        <div>
          {/* Stakeholder gap analysis */}
          <Card title="Stakeholder Gap Analysis" style={{ marginBottom: 16 }}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={breakdown.stakeholder_gaps || []} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="role" width={130} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="built_count" name="Built" fill="#4caf50" stackId="a" />
                <Bar dataKey="missing_count" name="Missing" fill="#f44336" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Stakeholder detail */}
          {(breakdown.stakeholder_gaps || []).map((sg, i) => (
            <Card key={i} title={`${sg.role} (${sg.coverage_pct}% coverage)`} style={{ marginBottom: 12 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#4caf50', marginBottom: 4, textTransform: 'uppercase' }}>Built ({sg.built_count})</div>
                  {(sg.built_items || []).map((item, j) => (
                    <div key={j} style={{ fontSize: 12, color: '#475569', padding: '2px 0' }}>{item}</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#f44336', marginBottom: 4, textTransform: 'uppercase' }}>Missing ({sg.missing_count})</div>
                  {(sg.missing_items || []).map((item, j) => (
                    <div key={j} style={{ fontSize: 12, color: '#475569', padding: '2px 0' }}>{item}</div>
                  ))}
                </div>
              </div>
            </Card>
          ))}

          {/* Process detail */}
          <Card title="Clinical Process Status" style={{ marginTop: 16, marginBottom: 16 }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Process</th>
                    <th style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.processes || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px' }}>{p.name}</td>
                      <td style={{ padding: '8px 6px' }}><Badge status={p.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Functionality gaps */}
          <Card title="Functionality Coverage">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Capability</th>
                    <th style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.functionality || []).map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px' }}>{f.capability}</td>
                      <td style={{ padding: '8px 6px' }}><Badge status={f.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ===== ACCEPTANCE TAB ===== */}
      {tab === 'acceptance' && breakdown && (
        <div>
          {(breakdown.role_criteria || []).map((role, i) => (
            <Card key={i} title={role.role} style={{ marginBottom: 12 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Dimension</th>
                    <th style={{ padding: '6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Acceptance Criterion</th>
                    <th style={{ padding: '6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(role.criteria || []).map((c, j) => (
                    <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 500 }}>{c.dimension}</td>
                      <td style={{ padding: '6px', color: '#475569' }}>{c.criterion}</td>
                      <td style={{ padding: '6px' }}><Badge status={c.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </div>
      )}

      {/* ===== DEFINITIONS TAB ===== */}
      {tab === 'definitions' && defs?.available && (
        <div>
          {(defs.definitions || []).map((d, i) => (
            <Card key={i} style={{ marginBottom: 8 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{d.term}</div>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{d.definition}</div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
