import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const LEVEL_COLORS = { basic: '#4caf50', intermediate: '#ff9800', high: '#f44336' }

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function LevelBadge({ level }) {
  const bg = LEVEL_COLORS[level] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {level}
    </span>
  )
}

export default function EpilepsyChallengesDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/epilepsy-challenges/overview`),
      axios.get(`${API_URL}/epilepsy-challenges/breakdown`),
      axios.get(`${API_URL}/epilepsy-challenges/definitions`),
    ])
      .then(([o, b, d]) => { setOverview(o.data); setBreakdown(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Epilepsy Challenges...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>Epilepsy Challenges data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'all', label: 'All Challenges' },
    { id: 'star', label: 'STAR Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]
  const s = overview.summary || {}

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 16 }}>
        Epilepsy Challenges Dashboard
      </h2>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#2563eb' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ────── OVERVIEW TAB ────── */}
      {tab === 'overview' && (
        <>
          <Card title="Key Metrics">
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Challenges" value={fmt(s.total_challenges)} />
              <KPI label="Basic" value={fmt(s.basic)} sub="acquisition / data" />
              <KPI label="Intermediate" value={fmt(s.intermediate)} sub="analysis / modeling" />
              <KPI label="High" value={fmt(s.high)} sub="prediction / governance" />
              <KPI label="AI Help Coverage" value={`${fmt(s.ai_help_coverage_pct)}%`} />
              <KPI label="STAR Documented" value={fmt(s.star_documented)} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Challenge Distribution by Level">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={overview.level_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.level_distribution || []).map((_, i) => (
                      <Cell key={i} fill={Object.values(LEVEL_COLORS)[i] || COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Challenges per Level">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview.level_distribution || []} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={75} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {(overview.level_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={LEVEL_COLORS[entry.name] || COLORS[i]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Challenge Summary Table">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['#', 'Level', 'Challenge', 'AI Solution'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.challenge_table || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.n}</td>
                      <td style={{ padding: '6px 10px' }}><LevelBadge level={c.level} /></td>
                      <td style={{ padding: '6px 10px' }}>{c.challenge}</td>
                      <td style={{ padding: '6px 10px', color: '#475569' }}>{c.ai_help}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ────── ALL CHALLENGES TAB ────── */}
      {tab === 'all' && breakdown && (
        <>
          {['basic', 'intermediate', 'high'].map(lvl => {
            const items = (breakdown.by_level || {})[lvl] || []
            if (!items.length) return null
            return (
              <Card key={lvl} title={<><LevelBadge level={lvl} /> <span style={{ marginLeft: 8 }}>{lvl.charAt(0).toUpperCase() + lvl.slice(1)} ({items.length})</span></>}>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: '#f8fafc' }}>
                        {['#', 'Challenge', 'AI Help'].map(h => (
                          <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((c, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                          <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.n}</td>
                          <td style={{ padding: '6px 10px' }}>{c.challenge}</td>
                          <td style={{ padding: '6px 10px', color: '#475569' }}>{c.ai_help}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )
          })}
        </>
      )}

      {/* ────── STAR DETAIL TAB ────── */}
      {tab === 'star' && breakdown && (
        <Card title="STAR Justification — All 30 Challenges">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['#', 'Level', 'Challenge', 'Situation', 'Task', 'Action', 'Result'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.all_challenges || []).map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.n}</td>
                    <td style={{ padding: '6px 10px' }}><LevelBadge level={c.level} /></td>
                    <td style={{ padding: '6px 10px', fontWeight: 500 }}>{c.challenge}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', minWidth: 140 }}>{c.situation}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', minWidth: 120 }}>{c.task}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', minWidth: 140 }}>{c.action}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', minWidth: 140 }}>{c.result}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ────── DEFINITIONS TAB ────── */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Difficulty Levels">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Level', 'Description'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(defs.levels || []).map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px' }}><LevelBadge level={l.name} /></td>
                      <td style={{ padding: '6px 10px', color: '#475569' }}>{l.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="STAR Method">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Letter', 'Meaning'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(defs.star_method || {}).map(([k, v], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600, textTransform: 'uppercase' }}>{k}</td>
                      <td style={{ padding: '6px 10px', color: '#475569' }}>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Glossary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Term', 'Definition'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{g.term}</td>
                      <td style={{ padding: '6px 10px', color: '#475569' }}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
              {(defs.clinical_notes || []).map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
              {(defs.references || []).map((r, i) => <li key={i} style={{ marginBottom: 4 }}>{r}</li>)}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
