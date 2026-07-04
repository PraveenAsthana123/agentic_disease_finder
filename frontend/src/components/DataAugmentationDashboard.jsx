import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(4)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

const TABS = ['Overview', 'Technique Comparison', 'Parameter Sweep', 'Class Balance', 'Methodology']

export default function DataAugmentationDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/data-augmentation/overview`),
      axios.get(`${API}/data-augmentation/breakdown`),
      axios.get(`${API}/data-augmentation/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Data Augmentation analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Data Augmentation Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        EEG feature augmentation techniques — real accuracy impact on {kpis.original_samples} samples, {kpis.n_features} features
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#3b82f6' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: tab === i ? 600 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} techniques={ov.techniques} classDist={ov.class_distribution} />}
      {tab === 1 && <ComparisonTab techniques={ov.techniques} baseline={kpis.baseline_accuracy} />}
      {tab === 2 && <ParameterTab data={bd} />}
      {tab === 3 && <BalanceTab classDist={ov.class_distribution} kpis={kpis} />}
      {tab === 4 && <MethodologyTab definitions={df} />}
    </div>
  )
}

function OverviewTab({ kpis, techniques, classDist }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Original Samples" value={kpis.original_samples} />
          <KPI label="Features" value={kpis.n_features} />
          <KPI label="Classes" value={kpis.n_classes} />
          <KPI label="Baseline Accuracy" value={fmtPct(kpis.baseline_accuracy)} color="#64748b" />
          <KPI label="Best Augmented" value={fmtPct(kpis.best_augmented_accuracy)} color="#10b981" />
          <KPI label="Best Technique" value={kpis.best_technique} color="#3b82f6" />
          <KPI label="Imbalance Before" value={kpis.imbalance_ratio_before + 'x'} color="#f59e0b" />
          <KPI label="Imbalance After SMOTE" value={kpis.imbalance_ratio_after_smote + 'x'} color="#10b981" />
        </div>
      </Card>

      <Card title="Accuracy by Technique">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={techniques} layout="vertical" margin={{ left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 1]} tickFormatter={v => fmtPct(v)} />
            <YAxis type="category" dataKey="name" width={80} />
            <Tooltip formatter={v => fmtPct(v)} />
            <Bar dataKey="accuracy" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Class Distribution (Original)">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={Object.entries(classDist).map(([name, value]) => ({ name, value }))}
              cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
              {Object.keys(classDist).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ComparisonTab({ techniques, baseline }) {
  const data = techniques.map(t => ({
    ...t,
    delta_pct: t.accuracy_delta * 100,
    baseline: baseline,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Accuracy Delta (% points vs baseline)" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data} margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis tickFormatter={v => v.toFixed(1) + '%'} />
            <Tooltip formatter={v => v.toFixed(2) + '% points'} />
            <Bar dataKey="delta_pct" name="Accuracy Δ" radius={[4, 4, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.accuracy_delta >= 0 ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Samples Generated per Technique">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="samples_generated" name="Generated" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Total Dataset Size After Augmentation">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="total_samples" name="Total Samples" fill="#06b6d4" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ParameterTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const results = data.results || []
  const summary = data.technique_summary || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Technique Summary — Best Parameters">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Technique</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Best Accuracy</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Best Parameter</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Variants Tested</th>
              </tr>
            </thead>
            <tbody>
              {summary.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.technique}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{fmtPct(s.best_accuracy)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{s.best_param}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{s.variants}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Full Parameter Sweep Results" span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Technique</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Parameter</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Augmented</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Total</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Accuracy</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Delta</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Label-Safe</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px' }}>{r.technique}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center', fontFamily: 'monospace' }}>{r.parameter}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.augmented_count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.total_count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600, color: '#1e293b' }}>{fmtPct(r.accuracy)}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center', color: r.delta >= 0 ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                    {r.delta >= 0 ? '+' : ''}{(r.delta * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                    {r.preserves_label ? '✓' : '✗'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function BalanceTab({ classDist, kpis }) {
  const classData = Object.entries(classDist).map(([name, count]) => ({ name, count }))
  const total = Object.values(classDist).reduce((a, b) => a + b, 0)
  const maxCount = Math.max(...Object.values(classDist))

  // Simulate SMOTE-balanced distribution
  const balancedData = classData.map(d => ({
    name: d.name,
    original: d.count,
    after_smote: maxCount,
    synthetic: maxCount - d.count,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Class Imbalance — Before vs After SMOTE" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={balancedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="original" name="Original" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            <Bar dataKey="after_smote" name="After SMOTE" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Imbalance Metrics">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, padding: '12px 0' }}>
          <KPI label="Before Augmentation" value={kpis.imbalance_ratio_before + 'x'} color="#f59e0b" sub="max/min class ratio" />
          <KPI label="After SMOTE" value={kpis.imbalance_ratio_after_smote + 'x'} color="#10b981" sub="balanced ratio" />
          <KPI label="Total Original" value={total} sub="samples" />
          <KPI label="Majority Class" value={maxCount} sub="samples in largest class" />
        </div>
      </Card>

      <Card title="Per-Class Synthetic Samples Needed">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Class</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Original</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Synthetic Needed</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>After SMOTE</th>
              </tr>
            </thead>
            <tbody>
              {balancedData.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{d.name}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{d.original}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: d.synthetic > 0 ? '#f59e0b' : '#10b981' }}>{d.synthetic}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{d.after_smote}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function MethodologyTab({ definitions }) {
  if (!definitions?.available) return <div style={{ color: '#f59e0b' }}>No definitions data</div>

  const techniques = definitions.techniques || []
  const bestPractices = definitions.best_practices || []
  const eegNotes = definitions.eeg_specific_notes || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {techniques.map((t, i) => (
        <Card key={i} title={t.name}>
          <p style={{ margin: '0 0 8px', fontSize: 13, color: '#475569' }}>{t.description}</p>
          <div style={{ margin: '8px 0', fontSize: 12, color: '#64748b' }}>
            <strong>Parameter:</strong> {t.parameter}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, margin: '12px 0' }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#10b981', marginBottom: 4 }}>Strengths</div>
              <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#475569' }}>
                {t.strengths.map((s, j) => <li key={j} style={{ marginBottom: 2 }}>{s}</li>)}
              </ul>
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#ef4444', marginBottom: 4 }}>Weaknesses</div>
              <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#475569' }}>
                {t.weaknesses.map((w, j) => <li key={j} style={{ marginBottom: 2 }}>{w}</li>)}
              </ul>
            </div>
          </div>
          <div style={{ fontSize: 12, color: '#3b82f6', marginTop: 8 }}>
            <strong>Clinical Relevance:</strong> {t.clinical_relevance}
          </div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{t.reference}</div>
        </Card>
      ))}

      <Card title="Best Practices">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {bestPractices.map((bp, i) => <li key={i} style={{ marginBottom: 6 }}>{bp}</li>)}
        </ul>
      </Card>

      <Card title="EEG-Specific Notes">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {eegNotes.map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
        </ul>
      </Card>
    </div>
  )
}
