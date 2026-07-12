import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'

const CANDIDACY_COLORS = {
  strong_candidate: '#22c55e',
  possible_candidate: '#eab308',
  not_candidate: '#ef4444'
}
const CANDIDACY_LABELS = {
  strong_candidate: 'Strong Candidate',
  possible_candidate: 'Possible Candidate',
  not_candidate: 'Not Candidate'
}
const PIE_COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#8b5cf6', '#f97316', '#06b6d4', '#ec4899']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function CandidacyBadge({ classification }) {
  const color = CANDIDACY_COLORS[classification] || '#94a3b8'
  const label = CANDIDACY_LABELS[classification] || classification || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

export default function PreSurgicalEvaluationDashboard() {
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
          axios.get(`${API_URL}/presurgical-evaluation/overview`),
          axios.get(`${API_URL}/presurgical-evaluation/breakdown`),
          axios.get(`${API_URL}/presurgical-evaluation/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading pre-surgical evaluation data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No data available</div>

  const tabs = ['overview', 'patients', 'workup', 'definitions']

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: '#1e293b' }}>Pre-Surgical Evaluation Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Epilepsy surgery candidacy assessment — ILAE criteria scoring with MRI, EEG, seizure burden, and drug resistance
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t ? '#1e293b' : '#f1f5f9', color: tab === t ? '#fff' : '#475569',
            fontSize: 13, fontWeight: 500
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'workup' && <WorkupTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const { kpis, candidacy_distribution, lesion_type_distribution, laterality_distribution, score_histogram } = overview
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Total Evaluated" value={fmt(kpis.total_evaluated)} />
          <KPI label="Strong Candidates" value={fmt(kpis.strong_candidates)} color="#22c55e" />
          <KPI label="Possible Candidates" value={fmt(kpis.possible_candidates)} color="#eab308" />
          <KPI label="Not Candidates" value={fmt(kpis.not_candidates)} color="#ef4444" />
          <KPI label="Avg Candidacy Score" value={fmt(kpis.avg_candidacy_score)} sub="/ 100" />
          <KPI label="Lesional Cases" value={fmt(kpis.lesional_cases)} color="#8b5cf6" />
          <KPI label="Hippocampal Sclerosis" value={fmt(kpis.hippocampal_sclerosis)} color="#3b82f6" />
        </div>
      </Card>

      <Card title="Candidacy Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={candidacy_distribution} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {candidacy_distribution.map((_, i) => (
                <Cell key={i} fill={['#22c55e', '#eab308', '#ef4444'][i]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Lesion Type Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={lesion_type_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="type" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip formatter={(v, name, props) => {
              const labels = overview.lesion_type_labels || {}
              return [v, labels[props.payload.type] || props.payload.type]
            }} />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Laterality">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={laterality_distribution} dataKey="count" nameKey="side" cx="50%" cy="50%" outerRadius={80} label>
              {laterality_distribution.map((_, i) => (
                <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Candidacy Score Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={score_histogram}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  if (!breakdown) return null
  const { patients } = breakdown
  return (
    <Card title={`Patient Candidacy Assessment (${patients.length} evaluated)`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              {['Patient', 'Age', 'Score', 'Status', 'Lesion', 'AEDs', 'Seizures', 'Workup %'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {patients.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}>{p.patient_id}</td>
                <td style={{ padding: '6px' }}>{p.age || '--'}</td>
                <td style={{ padding: '6px', fontWeight: 600 }}>{p.score}</td>
                <td style={{ padding: '6px' }}><CandidacyBadge classification={p.classification} /></td>
                <td style={{ padding: '6px' }}>{p.lesion_info}</td>
                <td style={{ padding: '6px' }}>{p.aed_count}</td>
                <td style={{ padding: '6px' }}>{p.seizure_count}</td>
                <td style={{ padding: '6px' }}>{p.workup_completeness_pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function WorkupTab({ breakdown }) {
  if (!breakdown) return null
  const { gap_analysis, patients } = breakdown
  const gapData = [
    { item: 'MRI', pct: gap_analysis.mri_coverage },
    { item: 'EEG', pct: gap_analysis.eeg_coverage },
    { item: 'Seizure Diary', pct: gap_analysis.seizure_diary_coverage },
    { item: 'Medication Hx', pct: gap_analysis.medication_coverage },
  ]
  const incomplete = patients.filter(p => p.workup_completeness_pct < 100)
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
      <Card title="Pre-Surgical Workup Coverage">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={gapData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} fontSize={11} />
            <YAxis dataKey="item" type="category" fontSize={11} width={100} />
            <Tooltip formatter={v => `${v}%`} />
            <Bar dataKey="pct" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title={`Incomplete Workups (${incomplete.length} patients)`}>
        <div style={{ maxHeight: 250, overflow: 'auto' }}>
          {incomplete.length === 0 ? (
            <p style={{ color: '#22c55e', fontSize: 13 }}>All workups complete</p>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Patient', 'MRI', 'EEG', 'Diary', 'Meds', 'Complete'].map(h => (
                    <th key={h} style={{ padding: '6px', textAlign: 'left', color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {incomplete.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '4px 6px' }}>{p.patient_id}</td>
                    <td style={{ padding: '4px 6px' }}>{p.has_mri ? '✓' : '✗'}</td>
                    <td style={{ padding: '4px 6px' }}>{p.has_eeg ? '✓' : '✗'}</td>
                    <td style={{ padding: '4px 6px' }}>{p.has_seizure_diary ? '✓' : '✗'}</td>
                    <td style={{ padding: '4px 6px' }}>{p.has_medication_history ? '✓' : '✗'}</td>
                    <td style={{ padding: '4px 6px' }}>{p.workup_completeness_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
      <Card title="Scoring Criteria" span={2}>
        <p style={{ fontSize: 13, color: '#475569', marginBottom: 12 }}>{defs.description}</p>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              {['Metric', 'Weight', 'Description'].map(h => (
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {defs.metrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 600 }}>{m.name}</td>
                <td style={{ padding: '6px' }}>{m.weight || m.range}</td>
                <td style={{ padding: '6px', color: '#64748b' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Lesion Types">
        {Object.entries(defs.lesion_types).map(([code, desc]) => (
          <div key={code} style={{ marginBottom: 8 }}>
            <span style={{ fontWeight: 700, color: '#1e293b', fontSize: 12 }}>{code}</span>
            <span style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>{desc}</span>
          </div>
        ))}
      </Card>

      <Card title="Surgical Procedures">
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {defs.surgical_procedures.map((p, i) => (
            <li key={i} style={{ padding: '4px 0', fontSize: 12, color: '#334155', borderBottom: '1px solid #f8fafc' }}>{p}</li>
          ))}
        </ul>
      </Card>

      <Card title="Engel Classification (Outcomes)">
        {Object.entries(defs.engel_outcomes).map(([cls, desc]) => (
          <div key={cls} style={{ marginBottom: 6 }}>
            <span style={{ fontWeight: 700, fontSize: 13, color: '#1e293b' }}>Class {cls}:</span>
            <span style={{ fontSize: 12, color: '#475569', marginLeft: 8 }}>{desc}</span>
          </div>
        ))}
      </Card>
    </div>
  )
}
