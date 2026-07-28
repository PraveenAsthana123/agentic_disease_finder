import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  if (typeof v === 'number') return v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)
  return String(v)
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{typeof value === 'number' ? fmt(value) : value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s === 'completed' ? '#10b981' : s === 'failed' ? '#ef4444' : '#f59e0b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s || '--'}</span>
  )
}

function accColor(v) {
  if (v == null) return '#64748b'
  return v >= 0.9 ? '#10b981' : v >= 0.8 ? '#f59e0b' : '#ef4444'
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'models', label: 'All Models' },
  { id: 'bytype', label: 'By Model Type' },
  { id: 'bydataset', label: 'By Dataset' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ModelComparisonDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/model-comparison/overview`),
      axios.get(`${API_URL}/api/model-comparison/breakdown`),
      axios.get(`${API_URL}/api/model-comparison/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading model comparison data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Model Comparison Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        ML model performance analytics — {fmt(k.total_models)} runs, {fmt(k.distinct_model_types)} model types, {fmt(k.distinct_tasks)} tasks, {fmt(k.distinct_datasets)} datasets
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'models' && <ModelsTab models={breakdown?.models || []} />}
      {tab === 'bytype' && <ByModelTypeTab data={breakdown} />}
      {tab === 'bydataset' && <ByDatasetTab data={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const k = data.kpis || {}
  const successRate = k.total_models > 0 ? ((k.completed_count / k.total_models) * 100).toFixed(1) : 0

  const modelTypePie = (data.model_type_dist || []).map(d => ({ name: d.model_type, value: d.count }))
  const taskPie = (data.task_dist || []).map(d => ({ name: d.task, value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
          <KPI label="Total Runs" value={k.total_models} color="#3b82f6" />
          <KPI label="Avg Accuracy" value={k.avg_accuracy} color={accColor(k.avg_accuracy)} />
          <KPI label="Avg F1" value={k.avg_f1} color={accColor(k.avg_f1)} />
          <KPI label="Best Accuracy" value={k.best_accuracy} sub={k.best_accuracy_model} color="#10b981" />
          <KPI label="Success Rate" value={`${successRate}%`} sub={`${k.completed_count} / ${k.total_models}`} color="#8b5cf6" />
        </div>
      </Card>

      <Card title="Model Type Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={modelTypePie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {modelTypePie.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Task Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={taskPie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {taskPie.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Status Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.status_dist || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="status" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Runs" radius={[4, 4, 0, 0]}>
              {(data.status_dist || []).map((d, i) => {
                const c = d.status === 'completed' ? '#10b981' : d.status === 'failed' ? '#ef4444' : '#f59e0b'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Accuracy by Model Type" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data.accuracy_by_model_type || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model_type" angle={-20} textAnchor="end" height={70} tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v) => fmt(v)} />
            <Legend />
            <Bar dataKey="avg_accuracy" fill="#3b82f6" name="Accuracy" radius={[4, 4, 0, 0]} />
            <Bar dataKey="avg_f1" fill="#10b981" name="F1" radius={[4, 4, 0, 0]} />
            <Bar dataKey="avg_auc_roc" fill="#f59e0b" name="AUC-ROC" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Accuracy by Task" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data.accuracy_by_task || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="task" angle={-20} textAnchor="end" height={70} tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v) => fmt(v)} />
            <Legend />
            <Bar dataKey="avg_accuracy" fill="#3b82f6" name="Accuracy" radius={[4, 4, 0, 0]} />
            <Bar dataKey="avg_f1" fill="#10b981" name="F1" radius={[4, 4, 0, 0]} />
            <Bar dataKey="avg_auc_roc" fill="#f59e0b" name="AUC-ROC" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Trend" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data.monthly_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="left" allowDecimals={false} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
            <Tooltip formatter={(v, name) => name === 'Avg Accuracy' ? fmt(v) : v} />
            <Legend />
            <Bar yAxisId="left" dataKey="count" fill="#3b82f6" name="Runs" radius={[4, 4, 0, 0]} opacity={0.6} />
            <Line yAxisId="right" type="monotone" dataKey="avg_accuracy" stroke="#ef4444" name="Avg Accuracy" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Training Method Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.training_method_dist || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="trained_by" angle={-20} textAnchor="end" height={60} tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" name="Runs" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ModelsTab({ models }) {
  const [sortKey, setSortKey] = useState('accuracy')
  const [sortDir, setSortDir] = useState(-1)
  const [filter, setFilter] = useState('')

  const filtered = models.filter(m => {
    if (!filter) return true
    const f = filter.toLowerCase()
    return (m.model_name || '').toLowerCase().includes(f) ||
           (m.model_type || '').toLowerCase().includes(f) ||
           (m.task || '').toLowerCase().includes(f) ||
           (m.dataset || '').toLowerCase().includes(f) ||
           (m.status || '').toLowerCase().includes(f)
  })

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * sortDir
  })

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d * -1)
    else { setSortKey(key); setSortDir(-1) }
  }

  const hdr = (label, key) => (
    <th onClick={() => toggleSort(key)} style={{
      padding: '8px 10px', cursor: 'pointer', whiteSpace: 'nowrap', fontSize: 12,
      background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left',
      color: sortKey === key ? '#3b82f6' : '#475569'
    }}>{label} {sortKey === key ? (sortDir > 0 ? '▲' : '▼') : ''}</th>
  )

  return (
    <Card title={`All Model Runs (${filtered.length})`}>
      <input type="text" placeholder="Filter by model, type, task, dataset, or status..." value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8,
          fontSize: 13, marginBottom: 12, boxSizing: 'border-box' }} />
      <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0 }}>
            <tr>
              {hdr('Model', 'model_name')}
              {hdr('Type', 'model_type')}
              {hdr('Version', 'version')}
              {hdr('Task', 'task')}
              {hdr('Dataset', 'dataset')}
              {hdr('Accuracy', 'accuracy')}
              {hdr('F1', 'f1_score')}
              {hdr('AUC-ROC', 'auc_roc')}
              {hdr('Train Time', 'training_time_sec')}
              {hdr('Inference', 'inference_time_ms')}
              {hdr('Status', 'status')}
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 100).map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: m.status === 'failed' ? '#fef2f2' : undefined }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{m.model_name}</td>
                <td style={{ padding: '6px 10px' }}>{m.model_type}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{m.version || '--'}</td>
                <td style={{ padding: '6px 10px' }}>{m.task}</td>
                <td style={{ padding: '6px 10px' }}>{m.dataset}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600, color: accColor(m.accuracy) }}>{fmt(m.accuracy)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600, color: accColor(m.f1_score) }}>{fmt(m.f1_score)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600, color: accColor(m.auc_roc) }}>{fmt(m.auc_roc)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{m.training_time_sec != null ? `${fmt(m.training_time_sec)}s` : '--'}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{m.inference_time_ms != null ? `${fmt(m.inference_time_ms)}ms` : '--'}</td>
                <td style={{ padding: '6px 10px' }}><StatusBadge status={m.status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > 100 && <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing first 100 of {sorted.length} model runs</div>}
    </Card>
  )
}

function ByModelTypeTab({ data }) {
  if (!data) return null
  const types = data.by_model_type || []

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title={`Model Type Comparison (${types.length} types)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Model Type</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Count</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Accuracy</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg F1</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg AUC-ROC</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Best Accuracy</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Train Time</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Inference</th>
              </tr>
            </thead>
            <tbody>
              {types.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{t.model_type}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{t.count}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600, color: accColor(t.avg_accuracy) }}>{fmt(t.avg_accuracy)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600, color: accColor(t.avg_f1) }}>{fmt(t.avg_f1)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600, color: accColor(t.avg_auc_roc) }}>{fmt(t.avg_auc_roc)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600, color: accColor(t.best_accuracy) }}>{fmt(t.best_accuracy)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{t.avg_training_time != null ? `${fmt(t.avg_training_time)}s` : '--'}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{t.avg_inference_time != null ? `${fmt(t.avg_inference_time)}ms` : '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Model Type Performance Comparison">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={types}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model_type" angle={-20} textAnchor="end" height={70} tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v) => fmt(v)} />
            <Legend />
            <Bar dataKey="avg_accuracy" fill="#3b82f6" name="Avg Accuracy" radius={[4, 4, 0, 0]} />
            <Bar dataKey="avg_f1" fill="#10b981" name="Avg F1" radius={[4, 4, 0, 0]} />
            <Bar dataKey="avg_auc_roc" fill="#f59e0b" name="Avg AUC-ROC" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ByDatasetTab({ data }) {
  if (!data) return null
  const datasets = data.by_dataset || []

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title={`Dataset Comparison (${datasets.length} datasets)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Dataset</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Count</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Accuracy</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg F1</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg AUC-ROC</th>
                <th style={{ ...thStyle, textAlign: 'center', color: '#10b981' }}>Completed</th>
                <th style={{ ...thStyle, textAlign: 'center', color: '#ef4444' }}>Failed</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{d.dataset}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{d.count}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600, color: accColor(d.avg_accuracy) }}>{fmt(d.avg_accuracy)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600, color: accColor(d.avg_f1) }}>{fmt(d.avg_f1)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600, color: accColor(d.avg_auc_roc) }}>{fmt(d.avg_auc_roc)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{d.models_completed}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#ef4444', fontWeight: 600 }}>{d.models_failed}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Dataset Performance Comparison">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={datasets}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="dataset" angle={-20} textAnchor="end" height={70} tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v) => fmt(v)} />
            <Legend />
            <Bar dataKey="avg_accuracy" fill="#3b82f6" name="Avg Accuracy" radius={[4, 4, 0, 0]} />
            <Bar dataKey="avg_f1" fill="#10b981" name="Avg F1" radius={[4, 4, 0, 0]} />
            <Bar dataKey="avg_auc_roc" fill="#f59e0b" name="Avg AUC-ROC" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  if (!data) return <Card>No definitions available.</Card>
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
      <Card title={data.title || 'Definitions'} span={2}>
        {(data.concepts || []).map((c, i) => (
          <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < data.concepts.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{c.name}</div>
            <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{c.description}</div>
          </div>
        ))}
      </Card>

      {(data.data_sources || []).length > 0 && (
        <Card title="Data Sources">
          <ul style={{ margin: 0, padding: '0 0 0 20px', fontSize: 13, color: '#64748b' }}>
            {data.data_sources.map((src, i) => <li key={i} style={{ marginBottom: 4 }}>{src}</li>)}
          </ul>
        </Card>
      )}
    </div>
  )
}

const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569' }
const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }
