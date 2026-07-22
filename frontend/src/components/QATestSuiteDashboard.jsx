import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { pass: '#4caf50', partial: '#ff9800', planned: '#f44336' }

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

export default function QATestSuiteDashboard() {
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
          axios.get(`${API_URL}/api/qa-test-suite/overview`),
          axios.get(`${API_URL}/api/qa-test-suite/breakdown`),
          axios.get(`${API_URL}/api/qa-test-suite/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load QA data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading QA test suite data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'QA test suite data not available.'}
    </div>
  )

  const summary = overview.summary || {}
  const roleSummaries = overview.role_summaries || []
  const dimBreakdown = overview.dimension_breakdown || []
  const dimMatrix = overview.dimension_matrix || []
  const statusDist = overview.status_distribution || {}
  const roles = breakdown?.roles || []
  const userStories = breakdown?.user_stories || []
  const demoStories = breakdown?.demo_stories || []

  const statusPie = Object.entries(statusDist).map(([name, value]) => ({ name, value }))
  const roleChart = roleSummaries.map(r => ({ name: r.role, pass_rate: r.pass_rate, pass: r.pass, partial: r.partial, planned: r.planned }))
  const dimChart = dimBreakdown.map(d => ({ name: d.dimension, pass: d.pass, partial: d.partial, planned: d.planned }))

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 130
  })
  const tabs = ['overview', 'role matrix', 'stories & demos', 'test detail', 'definitions']
  const tabStyle = (t) => ({
    padding: '8px 16px', cursor: 'pointer', borderRadius: '6px 6px 0 0', fontSize: 13, fontWeight: 600,
    background: tab === t ? '#1e88e5' : '#f1f5f9', color: tab === t ? '#fff' : '#475569',
    border: tab === t ? '1px solid #1e88e5' : '1px solid #e2e8f0', borderBottom: 'none'
  })

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 22, color: '#1e293b' }}>QA Test Suite Dashboard</h2>

      <div style={{ display: 'flex', gap: 4, marginBottom: 18, flexWrap: 'wrap' }}>
        {tabs.map(t => <div key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</div>)}
      </div>

      {tab === 'overview' && (
        <>
          {/* KPI cards */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {[
              { label: 'Total Tests', value: summary.total_tests, color: COLORS[3] },
              { label: 'Passing', value: summary.pass, color: COLORS[0] },
              { label: 'Partial', value: summary.partial, color: COLORS[1] },
              { label: 'Planned', value: summary.planned, color: COLORS[2] },
              { label: 'Coverage', value: `${summary.coverage_pct}%`, color: COLORS[4] },
              { label: 'Roles Tested', value: summary.total_roles, color: COLORS[5] },
              { label: 'User Stories', value: summary.user_stories, color: COLORS[6] },
              { label: 'Dimensions', value: `${summary.dimensions_built}/${summary.testing_dimensions} built`, color: COLORS[7] },
            ].map((k, i) => (
              <div key={i} style={kpiStyle(k.color)}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>
                  {typeof k.value === 'number' ? fmt(k.value) : (k.value || '--')}
                </div>
              </div>
            ))}
          </div>

          {/* Status distribution + dimension breakdown side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Test Status Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={statusPie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {statusPie.map((entry, i) => <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Tests by Dimension</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={dimChart}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="pass" stackId="a" fill={STATUS_COLORS.pass} />
                  <Bar dataKey="partial" stackId="a" fill={STATUS_COLORS.partial} />
                  <Bar dataKey="planned" stackId="a" fill={STATUS_COLORS.planned} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Role pass rates */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Pass Rate by Role</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={roleChart} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
                <YAxis dataKey="name" type="category" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="pass_rate" fill={COLORS[0]} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 9-Dimension Matrix */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>9-Dimension Testing Matrix</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Dimension</th>
                    <th style={{ padding: '8px 10px' }}>Tests</th>
                    <th style={{ padding: '8px 10px' }}>How</th>
                    <th style={{ padding: '8px 10px' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {dimMatrix.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{d.dimension}</td>
                      <td style={{ padding: '8px 10px', color: '#475569' }}>{d.tests}</td>
                      <td style={{ padding: '8px 10px', color: '#64748b', fontSize: 12, fontFamily: 'monospace' }}>{d.how}</td>
                      <td style={{ padding: '8px 10px' }}><Badge status={d.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {tab === 'role matrix' && (
        <>
          {/* Role summary table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Role Testing Summary</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Role</th>
                    <th style={{ padding: '8px 10px' }}>Total</th>
                    <th style={{ padding: '8px 10px' }}>Pass</th>
                    <th style={{ padding: '8px 10px' }}>Partial</th>
                    <th style={{ padding: '8px 10px' }}>Planned</th>
                    <th style={{ padding: '8px 10px' }}>Pass Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {roleSummaries.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.role}</td>
                      <td style={{ padding: '8px 10px' }}>{r.total}</td>
                      <td style={{ padding: '8px 10px', color: STATUS_COLORS.pass }}>{r.pass}</td>
                      <td style={{ padding: '8px 10px', color: STATUS_COLORS.partial }}>{r.partial}</td>
                      <td style={{ padding: '8px 10px', color: STATUS_COLORS.planned }}>{r.planned}</td>
                      <td style={{ padding: '8px 10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <div style={{ flex: 1, height: 8, background: '#f1f5f9', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{ width: `${r.pass_rate}%`, height: '100%', background: STATUS_COLORS.pass, borderRadius: 4 }} />
                          </div>
                          <span style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>{r.pass_rate}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Stacked bar: role × status */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Tests by Role (stacked)</h4>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={roleChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="pass" stackId="a" fill={STATUS_COLORS.pass} name="Pass" />
                <Bar dataKey="partial" stackId="a" fill={STATUS_COLORS.partial} name="Partial" />
                <Bar dataKey="planned" stackId="a" fill={STATUS_COLORS.planned} name="Planned" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {tab === 'stories & demos' && (
        <>
          {/* User stories */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>User Stories ({userStories.length})</h4>
            {userStories.map((s, i) => (
              <div key={i} style={{ padding: '12px 14px', borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ background: '#1e88e522', color: '#1e88e5', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>
                    {s.persona}
                  </span>
                  <span style={{ fontFamily: 'monospace', fontSize: 11, color: '#7c4dff' }}>{s.endpoint}</span>
                </div>
                <div style={{ fontSize: 13, color: '#334155', lineHeight: 1.5 }}>{s.story}</div>
              </div>
            ))}
          </div>

          {/* Demo stories */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Demo Stories ({demoStories.length})</h4>
            {demoStories.map((d, i) => (
              <div key={i} style={{ padding: '12px 14px', borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{d.title}</div>
                <div style={{ fontSize: 13, color: '#475569', marginBottom: 4 }}>{d.script}</div>
                <div style={{ fontSize: 12, color: '#7c4dff', fontStyle: 'italic' }}>Shows: {d.shows}</div>
              </div>
            ))}
          </div>
        </>
      )}

      {tab === 'test detail' && (
        <>
          {roles.map((role, ri) => (
            <div key={ri} style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>{role.role}</h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '6px 10px', width: 80 }}>Dimension</th>
                      <th style={{ padding: '6px 10px' }}>Test Case</th>
                      <th style={{ padding: '6px 10px', width: 80 }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(role.tests || []).map((t, ti) => (
                      <tr key={ti} style={{ borderBottom: '1px solid #f1f5f9', background: ti % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600, color: '#475569' }}>{t.dimension}</td>
                        <td style={{ padding: '6px 10px', color: '#334155' }}>{t.case}</td>
                        <td style={{ padding: '6px 10px' }}><Badge status={t.status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </>
      )}

      {tab === 'definitions' && defs?.definitions && (
        <div style={{ ...cardStyle, background: '#f0f9ff', border: '1px solid #bae6fd' }}>
          <h4 style={{ margin: '0 0 12px', color: '#0369a1' }}>QA Testing Definitions</h4>
          {defs.definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 10 }}>
              <strong style={{ color: '#0c4a6e' }}>{d.term}:</strong>{' '}
              <span style={{ color: '#334155', fontSize: 13 }}>{d.definition}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
