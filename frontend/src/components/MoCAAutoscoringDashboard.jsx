import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ScatterChart, Scatter, ZAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}
function fmtPct(v) { return v == null ? '--' : (v * 100).toFixed(1) + '%' }

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
  const colorMap = {
    normal: '#22c55e', mci: '#eab308', moderate: '#f97316', severe: '#ef4444',
    minimal: '#22c55e', mild: '#3b82f6',
    'moderately-severe': '#f97316',
    ok: '#22c55e', healthy: '#22c55e', complete: '#22c55e',
    warning: '#eab308', partial: '#eab308',
    critical: '#ef4444', failed: '#ef4444',
    pending: '#94a3b8', low: '#94a3b8',
    high: '#22c55e', medium: '#eab308',
  }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function MoCAAutoscoringDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/api/moca-autoscoring/overview`),
          axios.get(`${API_URL}/api/moca-autoscoring/breakdown`),
          axios.get(`${API_URL}/api/moca-autoscoring/definitions`)
        ])
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading MoCA Auto-Scoring...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No MoCA data available</div>

  const tabs = ['overview', 'patients', 'domains', 'comorbidity', 'definitions']

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>MoCA Auto-Scoring Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Montreal Cognitive Assessment — auto-scoring with normative comparison and domain analysis
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#e2e8f0', color: tab === t ? '#fff' : '#475569'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'domains' && <DomainsTab overview={overview} breakdown={breakdown} />}
      {tab === 'comorbidity' && <ComorbidityTab overview={overview} breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const k = overview.kpis
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card>
        <KPI label="Total Assessments" value={fmt(k.total_assessments)} sub={`${k.unique_patients} patients`} color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Mean MoCA Score" value={fmt(k.mean_score)} sub={`Median: ${k.median_score}`} color="#8b5cf6" />
      </Card>
      <Card>
        <KPI label="Below Cutoff (<26)" value={fmt(k.below_cutoff)} sub={fmtPct(k.below_cutoff_pct)} color="#f97316" />
      </Card>
      <Card>
        <KPI label="Normal (>=26)" value={fmt(k.normal_count)} sub={`MCI: ${k.mci_count} | Mod: ${k.moderate_count}`} color="#22c55e" />
      </Card>

      <Card title="Classification Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={overview.classification_distribution} dataKey="count" nameKey="category"
              cx="50%" cy="50%" outerRadius={90} label={({ category, count }) => `${category}: ${count}`}>
              {overview.classification_distribution.map((e, i) => (
                <Cell key={i} fill={e.color} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Score Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={overview.score_histogram}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" fontSize={12} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="MoCA vs MMSE Correlation" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="moca" name="MoCA" fontSize={12} label={{ value: 'MoCA', position: 'bottom', fontSize: 11 }} />
            <YAxis dataKey="mmse" name="MMSE" fontSize={12} label={{ value: 'MMSE', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={overview.moca_vs_mmse} fill="#8b5cf6" />
          </ScatterChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Assessors" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={overview.assessor_stats} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={12} />
            <YAxis dataKey="assessor" type="category" fontSize={11} width={120} />
            <Tooltip />
            <Bar dataKey="count" fill="#06b6d4" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Classification Groups">
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {breakdown.classification_groups.map((g, i) => (
            <div key={i} style={{ padding: 12, background: '#f1f5f9', borderRadius: 8, minWidth: 140 }}>
              <StatusBadge status={g.classification} />
              <div style={{ fontSize: 24, fontWeight: 700, marginTop: 8 }}>{g.count}</div>
              <div style={{ fontSize: 11, color: '#64748b' }}>patients</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Per-Patient MoCA Scores" span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                {['Patient', 'MoCA', 'MMSE', 'Classification', 'Impaired Domains', 'PHQ-9', 'GAD-7', 'Battery', 'Assessor', 'Date'].map(h => (
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.per_patient.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px' }}>{p.patient_id}</td>
                  <td style={{ padding: '6px', fontWeight: 700 }}>{p.moca_total}</td>
                  <td style={{ padding: '6px' }}>{fmt(p.mmse)}</td>
                  <td style={{ padding: '6px' }}><StatusBadge status={p.classification} /></td>
                  <td style={{ padding: '6px', fontSize: 11 }}>{p.impaired_domains.length > 0 ? p.impaired_domains.join(', ') : '--'}</td>
                  <td style={{ padding: '6px' }}>{fmt(p.phq9)}</td>
                  <td style={{ padding: '6px' }}>{fmt(p.gad7)}</td>
                  <td style={{ padding: '6px' }}>{p.battery_type}</td>
                  <td style={{ padding: '6px', fontSize: 11 }}>{p.assessor}</td>
                  <td style={{ padding: '6px', fontSize: 11 }}>{p.assessed_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DomainsTab({ overview, breakdown }) {
  const radarData = overview.domain_averages.map(d => ({
    domain: d.domain.replace('Visuospatial / Executive', 'Visuospatial'),
    score: d.pct,
    fullMark: 100
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Domain Averages (% of Max)" span={1}>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="domain" fontSize={11} />
            <PolarRadiusAxis domain={[0, 100]} fontSize={10} />
            <Radar dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
            <Tooltip formatter={v => `${v}%`} />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Domain Averages (Points)" span={1}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={overview.domain_averages}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="domain" fontSize={10} angle={-30} textAnchor="end" height={60}
              tickFormatter={v => v.replace('Visuospatial / Executive', 'Visuosp.')} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Bar dataKey="average" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Average" />
            <Bar dataKey="max" fill="#e2e8f0" radius={[4, 4, 0, 0]} name="Max Possible" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Most Vulnerable Domains" span={2}>
        {breakdown.domain_vulnerability.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={breakdown.domain_vulnerability} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" fontSize={12} />
              <YAxis dataKey="domain" type="category" fontSize={11} width={140} />
              <Tooltip formatter={(v, name) => name === 'pct' ? fmtPct(v) : v} />
              <Bar dataKey="impaired_count" fill="#ef4444" radius={[0, 4, 4, 0]} name="Impaired Count" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ color: '#94a3b8', fontSize: 13 }}>No domains below 50% threshold</p>
        )}
      </Card>
    </div>
  )
}

function ComorbidityTab({ overview, breakdown }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Depression (PHQ-9) vs MoCA" span={1}>
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="phq9" name="PHQ-9" fontSize={12} label={{ value: 'PHQ-9', position: 'bottom', fontSize: 11 }} />
            <YAxis dataKey="moca" name="MoCA" fontSize={12} label={{ value: 'MoCA', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={breakdown.depression_cognition} fill="#ef4444" />
          </ScatterChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Referral Reason Breakdown" span={1}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={overview.referral_breakdown}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="reason" fontSize={10} angle={-20} textAnchor="end" height={50}
              tickFormatter={v => v.replace('_', ' ')} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Legend />
            <Bar dataKey="normal" stackId="a" fill="#22c55e" />
            <Bar dataKey="mci" stackId="a" fill="#eab308" />
            <Bar dataKey="moderate" stackId="a" fill="#f97316" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Comorbidity Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Patient', 'MoCA', 'Classification', 'PHQ-9', 'Depression', 'GAD-7', 'Anxiety', 'Trail A (s)', 'Trail B (s)'].map(h => (
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.per_patient.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px' }}>{p.patient_id}</td>
                  <td style={{ padding: '6px', fontWeight: 700 }}>{p.moca_total}</td>
                  <td style={{ padding: '6px' }}><StatusBadge status={p.classification} /></td>
                  <td style={{ padding: '6px' }}>{fmt(p.phq9)}</td>
                  <td style={{ padding: '6px' }}>{p.depression_severity ? <StatusBadge status={p.depression_severity} /> : '--'}</td>
                  <td style={{ padding: '6px' }}>{fmt(p.gad7)}</td>
                  <td style={{ padding: '6px' }}>{p.anxiety_severity ? <StatusBadge status={p.anxiety_severity} /> : '--'}</td>
                  <td style={{ padding: '6px' }}>{fmt(p.trail_a)}</td>
                  <td style={{ padding: '6px' }}>{fmt(p.trail_b)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Metric Definitions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Metric</th>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {definitions.metrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 600 }}>{m.name}</td>
                <td style={{ padding: '6px' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="MoCA Domains">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Domain</th>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Max Score</th>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {definitions.domains.map((d, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 600 }}>{d.domain}</td>
                <td style={{ padding: '6px' }}>{d.max_score}</td>
                <td style={{ padding: '6px' }}>{d.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Scoring Guide">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {Object.entries(definitions.scoring_guide).map(([k, v]) => (
            <div key={k} style={{ padding: 8, background: '#f8fafc', borderRadius: 6 }}>
              <div style={{ fontSize: 11, color: '#64748b', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{v}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Data Sources">
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#475569' }}>
          {definitions.data_sources.map((s, i) => <li key={i} style={{ marginBottom: 4 }}>{s}</li>)}
        </ul>
      </Card>

      <Card title="Clinical Caveat">
        <p style={{ margin: 0, fontSize: 13, color: '#64748b', fontStyle: 'italic', lineHeight: 1.5 }}>
          {definitions.clinical_caveat}
        </p>
      </Card>
    </div>
  )
}
