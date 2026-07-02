import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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

function LevelBadge({ level }) {
  const colors = {
    normal: { bg: '#dcfce7', fg: '#166534', border: '#bbf7d0' },
    moderate: { bg: '#fef3c7', fg: '#92400e', border: '#fde68a' },
    severe: { bg: '#fecaca', fg: '#991b1b', border: '#fca5a5' }
  }
  const c = colors[level] || { bg: '#f1f5f9', fg: '#475569', border: '#e2e8f0' }
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, color: c.fg, background: c.bg,
      padding: '2px 8px', borderRadius: 10, border: `1px solid ${c.border}`
    }}>{(level || 'unknown').toUpperCase()}</span>
  )
}

export default function ADLDashboard() {
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
          axios.get(`${API_URL}/adl/overview`),
          axios.get(`${API_URL}/adl/breakdown`),
          axios.get(`${API_URL}/adl/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load ADL data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#129489;&#8205;&#9878;&#65039;</div>
      Loading ADL assessment data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No ADL data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'instruments', label: 'Instrument Analytics' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'recent', label: 'Recent Assessments' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>ADL Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(overview.total_assessments)} functional assessments across {fmt(overview.instruments_tracked)} instruments | {fmt(overview.patients_assessed)} patients assessed | Coverage: {fmt(overview.coverage_pct)}%
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'instruments' && <InstrumentAnalyticsTab overview={overview} breakdown={breakdown} />}
      {tab === 'patients' && <PatientProfilesTab breakdown={breakdown} />}
      {tab === 'recent' && <RecentAssessmentsTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const indepColor = overview.independence_rate >= 70 ? '#10b981' : overview.independence_rate >= 50 ? '#f59e0b' : '#ef4444'
  const coverageColor = overview.coverage_pct >= 80 ? '#10b981' : overview.coverage_pct >= 50 ? '#f59e0b' : '#ef4444'
  const severeColor = overview.severe_patients_count > 0 ? '#ef4444' : '#10b981'

  const severityData = Object.entries(overview.severity_distribution || {}).map(([k, v]) => ({ level: k, count: v }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Assessments" value={overview.total_assessments} color="#3b82f6" /></Card>
      <Card><KPI label="Patients Assessed" value={overview.patients_assessed} sub={`of ${overview.total_patients} total`} color="#10b981" /></Card>
      <Card><KPI label="ADL Coverage" value={`${fmt(overview.coverage_pct)}%`} color={coverageColor} sub="patients with functional assessment" /></Card>
      <Card><KPI label="Instruments Tracked" value={overview.instruments_tracked} color="#8b5cf6" /></Card>

      <Card><KPI label="Independence Rate" value={`${fmt(overview.independence_rate)}%`} color={indepColor} sub="Barthel >= 80" /></Card>
      <Card><KPI label="Avg QOLIE-31" value={overview.avg_qolie} color="#06b6d4" sub="quality of life (0-100)" /></Card>
      <Card><KPI label="Avg Epworth" value={overview.avg_epworth} color="#f59e0b" sub="sleepiness (0-24)" /></Card>
      <Card><KPI label="Severe Patients" value={overview.severe_patients_count} color={severeColor} sub="need intervention" /></Card>

      {/* Severity distribution pie */}
      <Card title="Severity Distribution (All Instruments)" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={severityData} dataKey="count" nameKey="level"
              cx="50%" cy="50%" outerRadius={85} label={({ level, count }) => `${level} (${count})`}>
              {severityData.map((e, i) => (
                <Cell key={i} fill={
                  e.level === 'normal' ? '#10b981' : e.level === 'moderate' ? '#f59e0b' : e.level === 'severe' ? '#ef4444' : COLORS[i % COLORS.length]
                } />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Per-instrument bar */}
      <Card title="Assessments per Instrument" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={overview.instrument_stats || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="instrument" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Assessments" />
            <Bar dataKey="unique_patients" fill="#10b981" radius={[4, 4, 0, 0]} name="Patients" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Instrument summary cards */}
      {(overview.instrument_stats || []).map((inst, i) => (
        <Card key={i} title={inst.label} span={inst.instrument === 'BARTHEL' ? 2 : 1}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 12 }}>
            <KPI label="Count" value={inst.count} color="#3b82f6" />
            <KPI label="Avg Score" value={inst.avg_score} color="#8b5cf6" />
            <KPI label="Patients" value={inst.unique_patients} color="#10b981" />
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            Score range: {fmt(inst.min_score)} &ndash; {fmt(inst.max_score)} | {' '}
            {Object.entries(inst.level_distribution || {}).map(([l, c]) => `${l}: ${c}`).join(', ')}
          </div>
        </Card>
      ))}
    </div>
  )
}

function InstrumentAnalyticsTab({ overview, breakdown }) {
  const distributions = breakdown?.score_distributions || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Barthel score distribution */}
      <Card title="Barthel Index Score Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={distributions.BARTHEL || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Patients" />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
          0-20: Totally dependent | 21-40: Severely dependent | 41-60: Moderately dependent | 61-80: Mildly dependent | 81-100: Independent
        </div>
      </Card>

      {/* QOLIE-31 score distribution */}
      <Card title="QOLIE-31 Score Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={distributions.QOLIE31 || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} name="Patients" />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
          Higher = better quality of life | Target: &gt; 70
        </div>
      </Card>

      {/* Epworth score distribution */}
      <Card title="Epworth Sleepiness Scale Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={distributions.EPWORTH || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={50} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Patients" />
          </BarChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
          0-10: Normal | 11-12: Mild excessive sleepiness | 13-15: Moderate | 16-24: Severe
        </div>
      </Card>

      {/* Per-instrument level breakdown table */}
      <Card title="Level Breakdown by Instrument">
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead><tr style={{ borderBottom: '2px solid #e2e8f0' }}>
            <th style={{ textAlign: 'left', padding: 8 }}>Instrument</th>
            <th style={{ textAlign: 'right', padding: 8 }}>Total</th>
            <th style={{ textAlign: 'right', padding: 8 }}>Normal</th>
            <th style={{ textAlign: 'right', padding: 8 }}>Moderate</th>
            <th style={{ textAlign: 'right', padding: 8 }}>Severe</th>
            <th style={{ textAlign: 'right', padding: 8 }}>Avg Score</th>
          </tr></thead>
          <tbody>{(overview.instrument_stats || []).map((inst, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={{ padding: 8, fontWeight: 600 }}>{inst.label}</td>
              <td style={{ padding: 8, textAlign: 'right' }}>{inst.count}</td>
              <td style={{ padding: 8, textAlign: 'right', color: '#10b981', fontWeight: 600 }}>{inst.level_distribution?.normal || 0}</td>
              <td style={{ padding: 8, textAlign: 'right', color: '#f59e0b', fontWeight: 600 }}>{inst.level_distribution?.moderate || 0}</td>
              <td style={{ padding: 8, textAlign: 'right', color: '#ef4444', fontWeight: 600 }}>{inst.level_distribution?.severe || 0}</td>
              <td style={{ padding: 8, textAlign: 'right' }}>{fmt(inst.avg_score)}</td>
            </tr>
          ))}</tbody>
        </table>
      </Card>
    </div>
  )
}

function PatientProfilesTab({ breakdown }) {
  const patients = breakdown?.patient_profiles || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Patient Functional Profiles (${patients.length} patients)`}>
        <div style={{ display: 'grid', gap: 16 }}>
          {patients.map((p, i) => (
            <div key={i} style={{
              padding: 16, background: '#f8fafc', borderRadius: 8,
              border: p.overall_level === 'severe' ? '2px solid #ef4444' : p.overall_level === 'moderate' ? '2px solid #f59e0b' : '1px solid #e2e8f0'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <span style={{ fontWeight: 700, fontSize: 15, color: '#1e293b' }}>{p.patient_id}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>{p.instruments_completed} instrument{p.instruments_completed !== 1 ? 's' : ''}</span>
                <LevelBadge level={p.overall_level} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8 }}>
                {Object.entries(p.scores || {}).map(([inst, data], j) => (
                  <div key={j} style={{
                    padding: 10, background: '#fff', borderRadius: 6,
                    border: '1px solid #e2e8f0', fontSize: 12
                  }}>
                    <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>
                      {inst === 'BARTHEL' ? 'Barthel (ADL)' : inst === 'QOLIE31' ? 'QOLIE-31 (QoL)' : 'Epworth (Sleep)'}
                    </div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: data.level === 'severe' ? '#ef4444' : data.level === 'moderate' ? '#f59e0b' : '#10b981' }}>
                      {fmt(data.score)}<span style={{ fontSize: 12, fontWeight: 400, color: '#94a3b8' }}>/{fmt(data.max_score)}</span>
                    </div>
                    <div style={{ color: '#64748b', marginTop: 2 }}>
                      {data.category || data.interpretation}
                    </div>
                    <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>
                      {data.assessed_at ? data.assessed_at.substring(0, 10) : ''}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function RecentAssessmentsTab({ breakdown }) {
  const recent = breakdown?.recent_assessments || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Recent Functional Assessments (last ${recent.length})`}>
        <div style={{ maxHeight: 600, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Instrument</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Score</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Max</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Interpretation</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Level</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Date</th>
            </tr></thead>
            <tbody>{recent.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{r.patient_id}</td>
                <td style={{ padding: 8 }}>{r.instrument}</td>
                <td style={{ padding: 8, textAlign: 'right', fontWeight: 600,
                  color: r.level === 'severe' ? '#ef4444' : r.level === 'moderate' ? '#f59e0b' : '#10b981'
                }}>{fmt(r.score)}</td>
                <td style={{ padding: 8, textAlign: 'right', color: '#94a3b8' }}>{fmt(r.max_score)}</td>
                <td style={{ padding: 8 }}>{r.interpretation}</td>
                <td style={{ padding: 8 }}><LevelBadge level={r.level} /></td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>
                  {r.assessed_at ? r.assessed_at.substring(0, 10) : '--'}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div style={{ color: '#94a3b8' }}>No definitions available.</div>
  const sections = defs.sections || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {sections.map((section, si) => (
        <Card key={si} title={section.title}>
          <div style={{ display: 'grid', gap: 12 }}>
            {(section.items || []).map((item, ii) => (
              <div key={ii}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
