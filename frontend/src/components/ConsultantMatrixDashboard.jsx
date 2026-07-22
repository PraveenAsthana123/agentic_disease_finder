import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b', '#795548', '#9e9e9e']
const TIER_COLORS = { 1: '#1e88e5', 2: '#ff9800' }
const DATA_COLORS = { yes: '#4caf50', optional: '#ff9800', no: '#e0e0e0', metadata: '#7c4dff', aggregated: '#00bcd4' }

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

function TierBadge({ tier }) {
  const bg = TIER_COLORS[tier] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      Tier {tier}
    </span>
  )
}

function MandatoryBadge({ mandatory }) {
  const bg = mandatory ? '#4caf50' : '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, marginLeft: 4
    }}>
      {mandatory ? 'Mandatory' : 'Optional'}
    </span>
  )
}

function DataBadge({ value }) {
  const bg = DATA_COLORS[value] || '#e0e0e0'
  const color = value === 'no' ? '#999' : bg
  return (
    <span style={{
      background: `${color}22`, color: color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '1px 6px', fontSize: 10, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {value}
    </span>
  )
}

function Pill({ text }) {
  return (
    <span style={{
      background: '#f1f5f9', color: '#475569', borderRadius: 12, padding: '2px 8px',
      fontSize: 11, display: 'inline-block', margin: '2px 3px'
    }}>
      {text}
    </span>
  )
}

export default function ConsultantMatrixDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/consultant-matrix/overview`),
      axios.get(`${API_URL}/api/consultant-matrix/breakdown`),
      axios.get(`${API_URL}/api/consultant-matrix/definitions`),
    ])
      .then(([o, b, d]) => { setOverview(o.data); setBreakdown(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Consultant Matrix...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>Consultant Matrix data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'roles', label: 'By Role' },
    { id: 'data-matrix', label: 'Data Matrix' },
    { id: 'ai-solutions', label: 'AI Solutions' },
    { id: 'definitions', label: 'Definitions' },
  ]
  const s = overview.summary || {}

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
        Consultant Engagement Matrix
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        10 clinical consultant roles — tasks, challenges, AI solutions, data requirements &amp; compliance docs
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#1e293b' : '#f1f5f9', color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <>
          <Card title="Key Metrics">
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 8 }}>
              <KPI label="Total Consultants" value={s.total_consultants} />
              <KPI label="Tier 1 (Core)" value={s.tier_1} />
              <KPI label="Tier 2 (Recommended)" value={s.tier_2} />
              <KPI label="Mandatory" value={s.mandatory} />
              <KPI label="Total Tasks" value={s.total_tasks} />
              <KPI label="Total Challenges" value={s.total_challenges} />
              <KPI label="AI Solutions" value={s.total_ai_solutions} sub={`${s.ai_coverage_pct}% coverage`} />
              <KPI label="Documents" value={s.total_documents} />
              <KPI label="Compliance Docs" value={s.total_compliance_docs} />
              <KPI label="Assessments" value={s.total_assessments} />
              <KPI label="Tools" value={s.total_tools} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Tier Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.tier_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {overview.tier_distribution.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Tasks & Challenges per Role">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.role_summary} margin={{ left: 0, right: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="id" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={50} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="tasks" fill="#1e88e5" name="Tasks" />
                  <Bar dataKey="challenges" fill="#ff9800" name="Challenges" />
                  <Bar dataKey="ai_solutions" fill="#4caf50" name="AI Solutions" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Role Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Consultant', 'Tier', 'Status', 'Role', 'Tasks', 'Challenges', 'AI Solutions', 'Tools', 'Assessments'].map(h =>
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {(overview.role_summary || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{r.name}</td>
                      <td style={{ padding: '6px' }}><TierBadge tier={r.tier} /></td>
                      <td style={{ padding: '6px' }}><MandatoryBadge mandatory={r.mandatory} /></td>
                      <td style={{ padding: '6px', color: '#64748b' }}>{r.role}</td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>{r.tasks}</td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>{r.challenges}</td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>{r.ai_solutions}</td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>{r.tools}</td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>{r.assessments}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* By Role Tab */}
      {tab === 'roles' && breakdown?.roles?.map((role, ri) => (
        <Card key={ri} title={
          <span>
            {role.name}
            <TierBadge tier={role.tier} />
            <MandatoryBadge mandatory={role.mandatory} />
            <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>{role.role} — {role.objective}</span>
          </span>
        }>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Tasks</div>
              {role.tasks.map((t, i) => <div key={i} style={{ fontSize: 12, color: '#334155', padding: '2px 0' }}>• {t}</div>)}
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Challenges</div>
              {role.challenges.map((c, i) => <div key={i} style={{ fontSize: 12, color: '#ef4444', padding: '2px 0' }}>• {c}</div>)}
            </div>
          </div>

          {role.ai_solutions.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>AI Solutions</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#f0fdf4' }}>
                    <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600 }}>Challenge</th>
                    <th style={{ padding: '4px 6px', textAlign: 'left', fontWeight: 600 }}>AI Mitigation</th>
                  </tr>
                </thead>
                <tbody>
                  {role.ai_solutions.map((ai, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '4px 6px', color: '#ef4444' }}>{ai.challenge}</td>
                      <td style={{ padding: '4px 6px', color: '#16a34a' }}>{ai.ai}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Documents</div>
              {role.documents.map((d, i) => <Pill key={i} text={d} />)}
              {role.documents.length === 0 && <span style={{ fontSize: 11, color: '#94a3b8' }}>None</span>}
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Compliance Docs</div>
              {role.compliance_docs.map((d, i) => <Pill key={i} text={d} />)}
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Tools</div>
              {role.tools.map((t, i) => <Pill key={i} text={t} />)}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Assessment Domains</div>
              {role.assessment.map((a, i) => <Pill key={i} text={a} />)}
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Patient Questionnaire</div>
              {role.patient_questionnaire.map((q, i) => <Pill key={i} text={q} />)}
              {role.patient_questionnaire.length === 0 && <span style={{ fontSize: 11, color: '#94a3b8' }}>N/A</span>}
            </div>
          </div>
        </Card>
      ))}

      {/* Data Matrix Tab */}
      {tab === 'data-matrix' && (
        <>
          <Card title="Data Requirements Matrix — What Each Consultant Needs">
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
              Cross-reference: 10 consultant roles × 7 data types. Shows which data each role requires (yes), can optionally use (optional), or does not need (no/metadata/aggregated).
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Consultant</th>
                    {(breakdown?.data_fields || []).map(f =>
                      <th key={f} style={{ padding: '6px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569', textTransform: 'uppercase', fontSize: 10 }}>
                        {f.replace(/_/g, ' ')}
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.data_matrix || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600, whiteSpace: 'nowrap' }}>{row.consultant}</td>
                      {(breakdown?.data_fields || []).map(f =>
                        <td key={f} style={{ padding: '4px', textAlign: 'center' }}>
                          <DataBadge value={row[f]} />
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Data Coverage Summary">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={(breakdown?.data_fields || []).map(f => ({
                name: f.replace(/_/g, ' '),
                yes: overview.data_coverage[f]?.yes || 0,
                optional: overview.data_coverage[f]?.optional || 0,
                no: overview.data_coverage[f]?.no || 0,
              }))} margin={{ bottom: 30 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="yes" stackId="a" fill="#4caf50" name="Required" />
                <Bar dataKey="optional" stackId="a" fill="#ff9800" name="Optional" />
                <Bar dataKey="no" stackId="a" fill="#e0e0e0" name="Not Needed" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {/* AI Solutions Tab */}
      {tab === 'ai-solutions' && (
        <Card title={`All AI Solutions (${breakdown?.ai_solutions?.length || 0})`}>
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            Every challenge → AI mitigation pair across all 10 consultant roles.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Consultant</th>
                  <th style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Challenge</th>
                  <th style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>AI Mitigation</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.ai_solutions || []).map((ai, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600, whiteSpace: 'nowrap' }}>{ai.consultant}</td>
                    <td style={{ padding: '6px', color: '#ef4444' }}>{ai.challenge}</td>
                    <td style={{ padding: '6px', color: '#16a34a' }}>{ai.ai}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Tier Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <tbody>
                {(defs.tiers || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}><TierBadge tier={t.tier} /> {t.label}</td>
                    <td style={{ padding: '6px', color: '#64748b' }}>{t.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Data Requirement Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <tbody>
                {(defs.data_requirement_legend || []).map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px' }}><DataBadge value={d.value} /></td>
                    <td style={{ padding: '6px', color: '#64748b' }}>{d.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: '4px 0', fontSize: 12 }}>
                  <strong style={{ color: '#1e293b' }}>{g.term}</strong>
                  <span style={{ color: '#64748b' }}> — {g.definition}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes">
            {(defs.clinical_notes || []).map((n, i) =>
              <div key={i} style={{ fontSize: 12, color: '#475569', padding: '3px 0' }}>• {n}</div>
            )}
          </Card>

          <Card title="References">
            {(defs.references || []).map((r, i) =>
              <div key={i} style={{ fontSize: 12, color: '#475569', padding: '3px 0' }}>{i + 1}. {r}</div>
            )}
          </Card>
        </>
      )}
    </div>
  )
}
