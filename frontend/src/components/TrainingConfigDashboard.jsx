import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
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

function ParamBadge({ label, value }) {
  return (
    <span style={{
      background: '#f1f5f9', color: '#334155', border: '1px solid #e2e8f0',
      borderRadius: 4, padding: '2px 8px', fontSize: 11, marginRight: 6, marginBottom: 4, display: 'inline-block'
    }}>
      <strong>{label}:</strong> {String(value)}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function TrainingConfigDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/training-config/overview`),
      axios.get(`${API_URL}/api/training-config/breakdown`),
      axios.get(`${API_URL}/api/training-config/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Training Config...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'diseases', 'features', 'models', 'training', 'definitions']
  const k = overview.summary || {}

  /* ── Overview Tab ── */
  const renderOverview = () => {
    const featDist = overview.feature_distribution || []
    const samplesDist = overview.samples_per_disease || []
    const modelsTable = overview.models_table || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Key Metrics" span={2}>
          <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
            <KPI label="Diseases" value={k.disease_count} />
            <KPI label="Models" value={k.n_models} sub={k.model_type} />
            <KPI label="Total Features" value={k.total_features} />
            <KPI label="Selected Features" value={k.selected_features} sub={k.feature_selection_method} />
            <KPI label="CV Folds" value={k.cv_folds} sub={k.stratified ? 'stratified' : ''} />
            <KPI label="Accuracy Target" value={`${(k.min_accuracy_target * 100).toFixed(0)}%`} />
          </div>
        </Card>

        <Card title="Feature Domain Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={featDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
                {featDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Samples per Disease">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={samplesDist} margin={{ left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="original" fill="#94a3b8" name="Original" stackId="a" />
              <Bar dataKey="augmented" fill="#3b82f6" name="Augmented" stackId="b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Ensemble Models" span={2}>
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Model</th>
                <th style={thStyle}>Key Parameters</th>
              </tr></thead>
              <tbody>
                {modelsTable.map((m, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{m.model}</td>
                    <td style={tdStyle}><code style={{ fontSize: 11, background: '#f1f5f9', padding: '1px 4px', borderRadius: 3 }}>{m.params}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  /* ── Diseases Tab ── */
  const renderDiseases = () => {
    const bd = breakdown || {}
    const diseases = bd.diseases || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 16 }}>
        {diseases.map((d, i) => (
          <Card key={i} title={d.name}>
            <div style={{ fontSize: 12, color: '#334155', lineHeight: 1.8 }}>
              <div><strong>Key:</strong> {d.key}</div>
              <div><strong>Path:</strong> <code style={{ background: '#f1f5f9', padding: '1px 4px', borderRadius: 3, fontSize: 11 }}>{d.path}</code></div>
              <div><strong>Original Samples:</strong> {d.original_samples}</div>
              <div><strong>Augmented Samples:</strong> {d.augmented_samples}</div>
              <div style={{ marginTop: 8 }}>
                <div style={{ background: '#e2e8f0', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                  <div style={{ background: '#3b82f6', height: '100%', width: `${Math.min(100, (d.original_samples / d.augmented_samples) * 100)}%`, borderRadius: 4 }} />
                </div>
                <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>
                  {d.original_samples} original / {d.augmented_samples} augmented ({((d.augmented_samples / d.original_samples - 1) * 100).toFixed(0)}% augmentation)
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    )
  }

  /* ── Features Tab ── */
  const renderFeatures = () => {
    const bd = breakdown || {}
    const domains = bd.feature_domains || {}
    const sel = bd.feature_selection || {}

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Feature Selection" span={2}>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 8 }}>
            <KPI label="Total Features" value={sel.total} />
            <KPI label="Selected" value={sel.selected} />
            <KPI label="Method" value={sel.method} />
          </div>
        </Card>

        {Object.entries(domains).map(([domain, feats]) => (
          <Card key={domain} title={domain.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) + ` (${feats.length})`}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {feats.map((f, i) => (
                <span key={i} style={{
                  background: '#eff6ff', color: '#1e40af', border: '1px solid #bfdbfe',
                  borderRadius: 4, padding: '3px 10px', fontSize: 11, fontWeight: 500
                }}>
                  {f}
                </span>
              ))}
            </div>
          </Card>
        ))}
      </div>
    )
  }

  /* ── Models Tab ── */
  const renderModels = () => {
    const bd = breakdown || {}
    const models = bd.models || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
        {models.map((m, i) => (
          <Card key={i} title={m.name}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
              {Object.entries(m.params || {}).map(([pk, pv]) => (
                <ParamBadge key={pk} label={pk} value={pv} />
              ))}
            </div>
          </Card>
        ))}
      </div>
    )
  }

  /* ── Training Tab ── */
  const renderTraining = () => {
    const bd = breakdown || {}
    const tr = bd.training || {}
    const val = bd.validation || {}
    const out = bd.output || {}
    const tgt = bd.targets || {}

    const renderSection = (title, obj) => (
      <Card title={title}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {Object.entries(obj).map(([k, v]) => (
            <ParamBadge key={k} label={k.replace(/_/g, ' ')} value={v} />
          ))}
        </div>
      </Card>
    )

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        {renderSection('Cross Validation', tr.cross_validation || {})}
        {renderSection('Regularization', tr.regularization || {})}
        {renderSection('Data Augmentation', tr.data_augmentation || {})}
        {renderSection('Oversampling', tr.oversampling || {})}
        {renderSection('Validation', val)}
        {renderSection('Output', out)}
        <Card title="Performance Targets" span={2}>
          <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
            <KPI label="Min Accuracy" value={`${(tgt.min_accuracy * 100).toFixed(0)}%`} />
            <KPI label="Min F1" value={`${(tgt.min_f1_score * 100).toFixed(0)}%`} />
            <KPI label="Max Train-Test Gap" value={`${(tgt.max_train_test_gap * 100).toFixed(0)}%`} />
            <KPI label="Max CV Std" value={`${(tgt.max_cv_std * 100).toFixed(0)}%`} />
            <KPI label="Overfit Threshold" value={tgt.overfitting_threshold} sub="risk score /100" />
          </div>
        </Card>
      </div>
    )
  }

  /* ── Definitions Tab ── */
  const renderDefinitions = () => {
    const d = defs || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Glossary" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={thStyle}>Term</th><th style={thStyle}>Definition</th></tr></thead>
            <tbody>
              {(d.glossary || []).map((g, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                  <td style={tdStyle}>{g.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        {(d.clinical_notes || []).length > 0 && (
          <Card title="Clinical Notes" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {d.clinical_notes.map((n, i) => <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{n}</li>)}
            </ul>
          </Card>
        )}

        {d.references && d.references.length > 0 && (
          <Card title="References" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {d.references.map((r, i) => <li key={i} style={{ fontSize: 12, color: '#3b82f6', marginBottom: 4 }}>{typeof r === 'string' ? r : `${r.ref} — ${r.detail}`}</li>)}
            </ul>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Training Configuration</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        {overview.title} v{overview.version} — {k.disease_count} diseases, {k.n_models} models, {k.selected_features}/{k.total_features} features, {k.cv_folds}-fold CV, {(k.min_accuracy_target * 100).toFixed(0)}% accuracy target
      </p>

      <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderBottom: '2px solid #e2e8f0' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', border: 'none', background: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t ? 600 : 400,
            color: tab === t ? '#3b82f6' : '#64748b',
            borderBottom: tab === t ? '2px solid #3b82f6' : '2px solid transparent',
            marginBottom: -2, textTransform: 'capitalize'
          }}>{t.replace(/-/g, ' ')}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'diseases' && renderDiseases()}
      {tab === 'features' && renderFeatures()}
      {tab === 'models' && renderModels()}
      {tab === 'training' && renderTraining()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
