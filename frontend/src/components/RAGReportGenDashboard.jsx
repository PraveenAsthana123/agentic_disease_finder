import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8010'
const COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'
]

export default function RAGReportGenDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    const endpoints = {
      overview: '/api/rag-report-gen/overview',
      breakdown: '/api/rag-report-gen/breakdown',
      definitions: '/api/rag-report-gen/definitions',
    }
    const url = endpoints[tab]
    if (!url) return
    axios.get(`${API}${url}`)
      .then(r => {
        if (tab === 'overview') setOverview(r.data)
        else if (tab === 'breakdown') setBreakdown(r.data)
        else setDefinitions(r.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [tab])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Patient Reports' },
    { id: 'sources', label: 'Data Sources' },
    { id: 'quality', label: 'Quality Analysis' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const sty = {
    card: { background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 },
    kpi: { background: '#334155', borderRadius: 8, padding: 16, textAlign: 'center', flex: 1 },
    kpiVal: { fontSize: 28, fontWeight: 700, color: '#818cf8' },
    kpiLabel: { fontSize: 12, color: '#94a3b8', marginTop: 4 },
    grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 },
    tab: (active) => ({
      padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600,
      background: active ? '#6366f1' : '#334155', color: active ? '#fff' : '#94a3b8', fontSize: 13
    }),
    badge: (level) => ({
      display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
      background: level === 'full' ? '#10b98133' : level === 'partial' ? '#f59e0b33' : '#ef444433',
      color: level === 'full' ? '#10b981' : level === 'partial' ? '#f59e0b' : '#ef4444'
    }),
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
    th: { textAlign: 'left', padding: '8px 10px', borderBottom: '1px solid #334155', color: '#94a3b8', fontWeight: 600 },
    td: { padding: '8px 10px', borderBottom: '1px solid #1e293b' },
  }

  // ── Tab: Overview ────────────────────────────────────────────────
  const renderOverview = () => {
    if (!overview) return null
    const d = overview
    const cov = d.report_coverage || {}
    const qual = d.report_quality || {}
    return (<>
      <div style={sty.grid}>
        {[
          [cov.total_patients, 'Total Patients'],
          [cov.reportable_with_analysis, 'Reportable'],
          [`${cov.coverage_pct}%`, 'Coverage'],
          [`${qual.avg_sections_populated}/5`, 'Avg Quality'],
          [`${qual.quality_pct}%`, 'Quality Score'],
        ].map(([val, label], i) => (
          <div key={i} style={sty.kpi}>
            <div style={sty.kpiVal}>{val ?? '—'}</div>
            <div style={sty.kpiLabel}>{label}</div>
          </div>
        ))}
      </div>

      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Report Sections</h3>
        {(d.report_sections || []).map((s, i) => (
          <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #334155', color: '#cbd5e1', fontSize: 13 }}>
            {i + 1}. {s}
          </div>
        ))}
      </div>

      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Report Coverage Distribution</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={[
                { name: 'Reportable', value: cov.reportable_with_analysis || 0 },
                { name: 'No Analysis', value: (cov.total_patients || 0) - (cov.reportable_with_analysis || 0) }
              ]}
              cx="50%" cy="50%" outerRadius={80} dataKey="value" label
            >
              {[COLORS[2], COLORS[3]].map((c, i) => <Cell key={i} fill={c} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Data Coverage Radar</h3>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={[
            { axis: 'XAI', val: cov.xai_covered || 0 },
            { axis: 'RAG', val: cov.rag_covered || 0 },
            { axis: 'Analysis', val: cov.reportable_with_analysis || 0 },
            { axis: 'Total', val: cov.total_patients || 0 },
          ]}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="axis" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 10 }} />
            <Radar dataKey="val" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </>)
  }

  // ── Tab: Patient Reports ──────────────────────────────────────
  const renderBreakdown = () => {
    if (!breakdown) return null
    const reports = breakdown.reports || []
    return (<>
      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>
          Patient Report Summary ({reports.length} patients)
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={sty.table}>
            <thead>
              <tr>
                <th style={sty.th}>Patient</th>
                <th style={sty.th}>Quality</th>
                <th style={sty.th}>Classification</th>
                <th style={sty.th}>Confidence</th>
                <th style={sty.th}>Biomarkers</th>
                <th style={sty.th}>XAI Features</th>
                <th style={sty.th}>Evidence</th>
                <th style={sty.th}>Seizures</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r, i) => (
                <tr key={i}>
                  <td style={sty.td}>P{r.patient_id}</td>
                  <td style={sty.td}>
                    <span style={sty.badge(
                      r.report_quality?.startsWith('4') || r.report_quality?.startsWith('5') ? 'full' :
                      r.report_quality?.startsWith('3') ? 'partial' : 'none'
                    )}>{r.report_quality}</span>
                  </td>
                  <td style={sty.td}>{r.classification || '—'}</td>
                  <td style={sty.td}>{r.confidence != null ? `${(r.confidence * 100).toFixed(0)}%` : '—'}</td>
                  <td style={sty.td}>{r.biomarker_count}</td>
                  <td style={sty.td}>{r.xai_features}</td>
                  <td style={sty.td}>{r.evidence_items}</td>
                  <td style={sty.td}>{r.seizure_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Report Quality Distribution</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={reports.map(r => ({
            patient: `P${r.patient_id}`,
            quality: parseInt(r.report_quality) || 0,
            biomarkers: r.biomarker_count || 0
          }))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="patient" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
            <Bar dataKey="quality" fill="#6366f1" name="Quality Score" />
            <Bar dataKey="biomarkers" fill="#10b981" name="Biomarkers" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>)
  }

  // ── Tab: Data Sources ──────────────────────────────────────────
  const renderSources = () => {
    if (!overview) return null
    const ds = overview.data_sources || {}
    const sourceData = Object.entries(ds).map(([k, v]) => ({
      name: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      count: v
    }))
    return (<>
      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Data Source Inventory</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={sourceData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={130} />
            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
            <Bar dataKey="count" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={sty.grid}>
        {sourceData.map((s, i) => (
          <div key={i} style={sty.kpi}>
            <div style={{ ...sty.kpiVal, color: COLORS[i % COLORS.length] }}>{s.count}</div>
            <div style={sty.kpiLabel}>{s.name}</div>
          </div>
        ))}
      </div>
    </>)
  }

  // ── Tab: Quality Analysis ──────────────────────────────────────
  const renderQuality = () => {
    if (!overview) return null
    const qual = overview.report_quality || {}
    return (<>
      <div style={sty.grid}>
        {[
          [qual.sample_size, 'Sample Size'],
          [`${qual.avg_sections_populated}/5`, 'Avg Sections'],
          [`${qual.quality_pct}%`, 'Quality %'],
        ].map(([val, label], i) => (
          <div key={i} style={sty.kpi}>
            <div style={sty.kpiVal}>{val ?? '—'}</div>
            <div style={sty.kpiLabel}>{label}</div>
          </div>
        ))}
      </div>

      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Quality Criteria</h3>
        {[
          'Prediction: at least one EEG analysis result',
          'Biomarkers: at least one spectral/MRI/cognitive marker',
          'XAI: SHAP feature importance or AI explanation present',
          'Evidence: at least one RAG conversation on file',
          'Metadata: demographics + medications + seizure history (2+ present)'
        ].map((c, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #334155', color: '#cbd5e1', fontSize: 13 }}>
            <span style={{ color: '#10b981', marginRight: 8 }}>Section {i + 1}:</span> {c}
          </div>
        ))}
      </div>

      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Coverage by Feature</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={[
                { name: 'XAI Covered', value: overview.report_coverage?.xai_covered || 0 },
                { name: 'RAG Covered', value: overview.report_coverage?.rag_covered || 0 },
                { name: 'With Analysis', value: overview.report_coverage?.reportable_with_analysis || 0 },
              ]}
              cx="50%" cy="50%" outerRadius={80} dataKey="value" label
            >
              {COLORS.slice(0, 3).map((c, i) => <Cell key={i} fill={c} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </>)
  }

  // ── Tab: Definitions ────────────────────────────────────────────
  const renderDefinitions = () => {
    if (!definitions) return null
    const sections = definitions.report_sections || {}
    return (<>
      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0' }}>{definitions.title}</h3>
        <p style={{ color: '#94a3b8', fontSize: 13 }}>Pipeline Step {definitions.pipeline_step} — {definitions.purpose}</p>
      </div>

      {Object.entries(sections).map(([key, sec]) => (
        <div key={key} style={sty.card}>
          <h4 style={{ color: '#818cf8', marginBottom: 6 }}>{key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</h4>
          <p style={{ color: '#cbd5e1', fontSize: 13, marginBottom: 6 }}>{sec.description}</p>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>
            Sources: {(sec.sources || []).join(', ')}
          </div>
        </div>
      ))}

      <div style={sty.card}>
        <h4 style={{ color: '#818cf8', marginBottom: 6 }}>Quality Scoring</h4>
        {(definitions.quality_scoring?.criteria || []).map((c, i) => (
          <div key={i} style={{ padding: '4px 0', color: '#cbd5e1', fontSize: 13 }}>{c}</div>
        ))}
      </div>

      <div style={sty.card}>
        <h4 style={{ color: '#818cf8', marginBottom: 6 }}>References</h4>
        {(definitions.references || []).map((r, i) => (
          <div key={i} style={{ padding: '4px 0', color: '#94a3b8', fontSize: 12 }}>{r}</div>
        ))}
      </div>
    </>)
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ color: '#e2e8f0', marginBottom: 4 }}>RAG Report Generation</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
        Pipeline Step 20 — Prediction + Biomarkers + XAI + Evidence + Metadata
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} style={sty.tab(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {loading && <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading...</div>}
      {error && <div style={{ color: '#ef4444', padding: 20 }}>Error: {error}</div>}
      {!loading && !error && tab === 'overview' && renderOverview()}
      {!loading && !error && tab === 'breakdown' && renderBreakdown()}
      {!loading && !error && tab === 'sources' && renderSources()}
      {!loading && !error && tab === 'quality' && renderQuality()}
      {!loading && !error && tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
