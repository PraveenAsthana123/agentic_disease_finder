import React, { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const LABEL_COLORS = {
  seizure: '#ef4444',
  spike: '#f59e0b',
  artifact: '#94a3b8',
  normal: '#22c55e',
  slowing: '#6366f1',
  burst_suppression: '#ec4899',
}

const ANNOTATOR_COLORS = ['#1e88e5', '#22c55e', '#f59e0b', '#7c4dff']

const cardStyle = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
}

export default function AnnotationDashboard() {
  const [overview, setOverview] = useState(null)
  const [agreement, setAgreement] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ovRes, agRes, defRes] = await Promise.all([
          axios.get(`${API_URL}/api/annotation/overview`),
          axios.get(`${API_URL}/api/annotation/agreement`),
          axios.get(`${API_URL}/api/annotation/definitions`),
        ])
        setOverview(ovRes.data)
        setAgreement(agRes.data)
        setDefinitions(defRes.data)
      } catch (e) {
        setError(e.message || 'Failed to load annotation data')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const kpis = useMemo(() => {
    if (!overview?.available) return null
    return {
      totalAnnotations: overview.total_annotations,
      totalSubjects: overview.total_subjects,
      goldSeizures: overview.total_seizures_gold,
      seizureSeconds: overview.total_seizure_seconds,
      kappaMean: overview.agreement?.cohens_kappa_mean,
      alpha: overview.agreement?.krippendorff_alpha,
      nAnnotators: overview.agreement?.n_annotators,
    }
  }, [overview])

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>
        Loading annotation quality data...
      </div>
    )
  }
  if (error) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>
        Error: {error}
      </div>
    )
  }
  if (!overview?.available) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>
        No annotation data available. Place CHB-MIT data in data/real_eeg/epilepsy_physionet/
      </div>
    )
  }

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'agreement', label: 'Agreement' },
    { id: 'definitions', label: 'Definitions' },
  ]

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          Label Studio / CVAT — Annotation Quality Dashboard
        </h2>
        <p style={{ color: '#64748b', margin: '4px 0 0', fontSize: 14 }}>
          Inter-annotator agreement on CHB-MIT EEG seizure annotations
          ({kpis?.nAnnotators} annotators · {kpis?.totalSubjects} subjects · {kpis?.goldSeizures} gold seizures)
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            style={{
              padding: '8px 18px',
              borderRadius: 8,
              border: 'none',
              background: activeTab === t.id ? '#1e88e5' : '#f1f5f9',
              color: activeTab === t.id ? '#fff' : '#475569',
              fontWeight: 600,
              fontSize: 13,
              cursor: 'pointer',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && renderOverview()}
      {activeTab === 'agreement' && renderAgreement()}
      {activeTab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    return (
      <div>
        {/* KPI Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 24 }}>
          {[
            { label: 'Total Annotations', value: kpis.totalAnnotations, color: '#1e88e5', icon: '🏷️' },
            { label: 'Gold Seizures', value: kpis.goldSeizures, color: '#ef4444', icon: '⚡' },
            { label: "Cohen's κ (mean)", value: kpis.kappaMean?.toFixed(3), color: '#22c55e', icon: '🤝' },
            { label: "Krippendorff's α", value: kpis.alpha?.toFixed(3), color: '#7c4dff', icon: '📐' },
            { label: 'Seizure Time', value: `${Math.round(kpis.seizureSeconds / 60)}m`, color: '#f59e0b', icon: '⏱️' },
            { label: 'Annotators', value: kpis.nAnnotators, color: '#ec4899', icon: '👥' },
          ].map((kpi, i) => (
            <div key={i} style={{ ...cardStyle, borderLeft: `4px solid ${kpi.color}`, display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 28 }}>{kpi.icon}</span>
              <div>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#0f172a' }}>{kpi.value}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>{kpi.label}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Label Distribution */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
          <div style={cardStyle}>
            <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>
              Label Distribution
            </h3>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={overview.label_distribution}
                  dataKey="count"
                  nameKey="label"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  label={({ label, percent }) => `${label} ${percent}%`}
                >
                  {overview.label_distribution.map((d, i) => (
                    <Cell key={i} fill={LABEL_COLORS[d.label] || '#94a3b8'} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div style={cardStyle}>
            <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>
              Per-Subject Annotations
            </h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.subject_stats}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subject" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="n_seizures" fill="#ef4444" name="Seizures (gold)" />
                <Bar dataKey="n_annotations" fill="#1e88e5" name="All annotations" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Annotator Stats Table */}
        <div style={cardStyle}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>
            Annotator Performance
          </h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Annotator</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Annotations</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Labels Used</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Subjects</th>
              </tr>
            </thead>
            <tbody>
              {overview.annotator_stats.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 500 }}>
                    <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: ANNOTATOR_COLORS[i % ANNOTATOR_COLORS.length], marginRight: 8 }} />
                    {a.annotator}
                  </td>
                  <td style={{ textAlign: 'right', padding: '8px 12px' }}>{a.n_annotations}</td>
                  <td style={{ textAlign: 'right', padding: '8px 12px' }}>{a.labels_used}</td>
                  <td style={{ textAlign: 'right', padding: '8px 12px' }}>{a.subjects_covered}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Tools Info */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 20 }}>
          {overview.tools && Object.values(overview.tools).map((tool, i) => (
            <div key={i} style={{ ...cardStyle, borderTop: `3px solid ${i === 0 ? '#1e88e5' : '#22c55e'}` }}>
              <h3 style={{ fontSize: 15, fontWeight: 600, margin: '0 0 6px' }}>{tool.name} {tool.version}</h3>
              <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 10px' }}>{tool.role}</p>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569' }}>
                {tool.features.map((f, j) => <li key={j} style={{ marginBottom: 3 }}>{f}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </div>
    )
  }

  function renderAgreement() {
    if (!agreement?.available) {
      return <div style={{ color: '#64748b', padding: 20 }}>No agreement data available.</div>
    }

    const scaleData = agreement.scale?.map(s => ({
      ...s,
      fill: s.label === 'Almost perfect' ? '#22c55e' :
            s.label === 'Substantial' ? '#4ade80' :
            s.label === 'Moderate' ? '#f59e0b' :
            s.label === 'Fair' ? '#fb923c' : '#ef4444',
    })) || []

    return (
      <div>
        {/* Agreement KPIs */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
          <div style={{ ...cardStyle, textAlign: 'center', borderTop: '4px solid #22c55e' }}>
            <div style={{ fontSize: 14, color: '#64748b', marginBottom: 8 }}>Cohen's Kappa (mean)</div>
            <div style={{ fontSize: 36, fontWeight: 700, color: '#0f172a' }}>{agreement.cohens_kappa_mean}</div>
            <div style={{
              display: 'inline-block', marginTop: 8, padding: '4px 12px', borderRadius: 20,
              fontSize: 12, fontWeight: 600,
              background: agreement.cohens_kappa_mean >= 0.61 ? '#dcfce7' : agreement.cohens_kappa_mean >= 0.41 ? '#fef3c7' : '#fee2e2',
              color: agreement.cohens_kappa_mean >= 0.61 ? '#166534' : agreement.cohens_kappa_mean >= 0.41 ? '#92400e' : '#991b1b',
            }}>
              {agreement.kappa_interpretation}
            </div>
          </div>

          <div style={{ ...cardStyle, textAlign: 'center', borderTop: '4px solid #7c4dff' }}>
            <div style={{ fontSize: 14, color: '#64748b', marginBottom: 8 }}>Krippendorff's Alpha</div>
            <div style={{ fontSize: 36, fontWeight: 700, color: '#0f172a' }}>{agreement.krippendorff_alpha}</div>
            <div style={{
              display: 'inline-block', marginTop: 8, padding: '4px 12px', borderRadius: 20,
              fontSize: 12, fontWeight: 600,
              background: agreement.krippendorff_alpha >= 0.80 ? '#dcfce7' : agreement.krippendorff_alpha >= 0.67 ? '#fef3c7' : '#fee2e2',
              color: agreement.krippendorff_alpha >= 0.80 ? '#166534' : agreement.krippendorff_alpha >= 0.67 ? '#92400e' : '#991b1b',
            }}>
              {agreement.alpha_interpretation}
            </div>
          </div>
        </div>

        {/* Kappa Scale */}
        <div style={cardStyle}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>
            Cohen's Kappa Interpretation Scale
          </h3>
          <div style={{ display: 'flex', gap: 8 }}>
            {scaleData.map((s, i) => (
              <div key={i} style={{
                flex: 1, padding: '10px 8px', borderRadius: 8, textAlign: 'center',
                background: s.fill + '22', border: `2px solid ${s.fill}`,
              }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: s.fill }}>{s.label}</div>
                <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{s.range}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Pairwise Kappa Table */}
        <div style={{ ...cardStyle, marginTop: 20 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>
            Pairwise Cohen's Kappa
          </h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Annotator A</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Annotator B</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Kappa (κ)</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Level</th>
              </tr>
            </thead>
            <tbody>
              {agreement.pairwise_kappas?.map((p, i) => {
                const level = p.kappa >= 0.81 ? 'Almost perfect' :
                              p.kappa >= 0.61 ? 'Substantial' :
                              p.kappa >= 0.41 ? 'Moderate' :
                              p.kappa >= 0.21 ? 'Fair' : 'Slight'
                const color = p.kappa >= 0.61 ? '#22c55e' : p.kappa >= 0.41 ? '#f59e0b' : '#ef4444'
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px' }}>{p.annotator_a}</td>
                    <td style={{ padding: '8px 12px' }}>{p.annotator_b}</td>
                    <td style={{ textAlign: 'right', padding: '8px 12px', fontWeight: 600, color }}>{p.kappa}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12, color }}>{level}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Bar chart of pairwise kappas */}
        <div style={{ ...cardStyle, marginTop: 20 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>
            Pairwise Agreement Chart
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={agreement.pairwise_kappas?.map(p => ({
              pair: `${p.annotator_a.replace('annotator_', 'A')} vs ${p.annotator_b.replace('annotator_', 'A')}`,
              kappa: p.kappa,
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="pair" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="kappa" name="Cohen's κ">
                {agreement.pairwise_kappas?.map((p, i) => (
                  <Cell key={i} fill={p.kappa >= 0.61 ? '#22c55e' : p.kappa >= 0.41 ? '#f59e0b' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    )
  }

  function renderDefinitions() {
    if (!definitions) return null
    return (
      <div>
        <div style={cardStyle}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 8 }}>
            {definitions.title}
          </h3>
          <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
            {definitions.description}
          </p>
          <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>
            Data source: {definitions.data_source}
          </p>
        </div>

        {/* Label Taxonomy */}
        <div style={{ ...cardStyle, marginTop: 20 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>
            Annotation Label Taxonomy
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            {definitions.annotation_labels?.map((lbl, i) => (
              <div key={i} style={{
                padding: '10px 14px', borderRadius: 8, borderLeft: `4px solid ${lbl.color}`,
                background: lbl.color + '10',
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#0f172a' }}>{lbl.name}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>
                  ID: {lbl.id} · Hotkey: {lbl.hotkey}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Metrics */}
        <div style={{ ...cardStyle, marginTop: 20 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>
            Agreement Metrics Reference
          </h3>
          {definitions.metrics?.map((m, i) => (
            <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < definitions.metrics.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{m.name}</div>
              <div style={{ fontSize: 13, color: '#475569', marginTop: 2 }}>{m.description}</div>
              {m.reference && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Ref: {m.reference}</div>}
            </div>
          ))}
        </div>

        {/* Tools */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginTop: 20 }}>
          {definitions.tools && Object.values(definitions.tools).map((tool, i) => (
            <div key={i} style={cardStyle}>
              <h4 style={{ fontSize: 14, fontWeight: 600, color: '#0f172a', margin: '0 0 6px' }}>{tool.name}</h4>
              <p style={{ fontSize: 12, color: '#475569', margin: '0 0 6px' }}>{tool.description}</p>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>License: {tool.license}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }
}
