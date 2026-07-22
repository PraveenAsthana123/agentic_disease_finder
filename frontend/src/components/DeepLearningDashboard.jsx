import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981', '#06b6d4', '#ec4899', '#64748b']

const ACC_COLORS = { accuracy: '#3b82f6', sensitivity: '#ef4444', f1: '#10b981' }

export default function DeepLearningDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API}/api/deep-learning/overview`),
      axios.get(`${API}/api/deep-learning/breakdown`),
      axios.get(`${API}/api/deep-learning/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'architectures', label: 'Model Architectures' },
    { id: 'training', label: 'Training Metrics' },
    { id: 'patients', label: 'Patient Results' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Deep Learning data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const k = overview.kpis || {}

  /* -- Overview Tab ------------------------------------------------- */
  const renderOverview = () => {
    const methodsData = (overview.methods_comparison || []).map(m => ({
      name: m.method, accuracy: m.mean_accuracy_pct
    }))

    const patientData = (overview.per_patient_accuracy || []).map(p => ({
      name: p.subject, accuracy: p.accuracy_pct, sensitivity: p.sensitivity_pct, f1: p.f1_pct
    }))

    const archData = overview.arch_type_chart || []
    const diseaseData = overview.disease_chart || []

    return (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Trained Models" value={k.total_models} color="#3b82f6" /></Card>
          <Card><KPI label="Training Runs" value={k.total_training_runs} sub={`${k.successful_runs} successful`} color="#8b5cf6" /></Card>
          <Card><KPI label="Best Accuracy" value={`${k.best_accuracy_pct}%`} color="#10b981" /></Card>
          <Card><KPI label="Mean Accuracy" value={`${k.mean_accuracy_pct}%`} sub={`Sensitivity: ${k.mean_sensitivity_pct}%`} color="#06b6d4" /></Card>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Subjects Trained" value={k.total_patients_trained} color="#f59e0b" /></Card>
          <Card><KPI label="Architectures" value={k.architecture_count} color="#ec4899" /></Card>
          <Card><KPI label="Total Analyses" value={k.total_analyses} sub={`Avg conf: ${k.avg_confidence}%`} color="#64748b" /></Card>
          <Card><KPI label="Model Storage" value={`${k.total_model_size_mb} MB`} sub={`${k.features_count} features`} color="#3b82f6" /></Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          <Card title="Accuracy by Method">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={methodsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="accuracy" fill="#3b82f6" name="Mean Accuracy %" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Patient Accuracy">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={patientData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="accuracy" fill={ACC_COLORS.accuracy} name="Accuracy %" radius={[4, 4, 0, 0]} />
                <Bar dataKey="sensitivity" fill={ACC_COLORS.sensitivity} name="Sensitivity %" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" fill={ACC_COLORS.f1} name="F1 %" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Architecture Types">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={archData} cx="50%" cy="50%" outerRadius={80} dataKey="value"
                  label={({ name, value }) => `${name} (${value})`}>
                  {archData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Predicted Disease Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={diseaseData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#8b5cf6" name="Predictions" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </>
    )
  }

  /* -- Model Architectures Tab -------------------------------------- */
  const renderArchitectures = () => {
    const archs = breakdown?.architectures || []
    const modelFiles = breakdown?.model_files || []

    const modelClasses = archs.filter(a => a.architecture_type !== 'Utility')
    const utilClasses = archs.filter(a => a.architecture_type === 'Utility')

    return (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          {modelClasses.map((arch, i) => (
            <Card key={i}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#1e293b' }}>{arch.class_name}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{arch.title}</div>
                </div>
                <div style={{ display: 'flex', gap: 4 }}>
                  <Badge text={arch.architecture_type} color="#3b82f6" />
                  <Badge text={arch.target_disease} color="#8b5cf6" />
                </div>
              </div>
              {arch.description && (
                <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5, marginTop: 8 }}>
                  {arch.description}
                </div>
              )}
            </Card>
          ))}
        </div>

        {utilClasses.length > 0 && (
          <Card title="Utility Classes" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
              {utilClasses.map((u, i) => (
                <div key={i} style={{ padding: '8px 12px', borderRadius: 8, background: '#f8fafc' }}>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{u.class_name}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>{u.title}</div>
                </div>
              ))}
            </div>
          </Card>
        )}

        <Card title="Trained Model Files" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Filename', 'Disease', 'Size (KB)', 'Size (MB)'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {modelFiles.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, fontFamily: 'monospace', fontSize: 12 }}>{m.filename}</td>
                    <td style={{ padding: '8px 12px' }}>{m.disease}</td>
                    <td style={{ padding: '8px 12px' }}>{m.size_kb}</td>
                    <td style={{ padding: '8px 12px' }}>{m.size_mb}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* -- Training Metrics Tab ----------------------------------------- */
  const renderTraining = () => {
    const comparison = breakdown?.model_comparison || []
    const runs = breakdown?.training_detail || []
    const benchInfo = breakdown?.benchmark_info || {}

    const compData = comparison.map(c => ({
      name: c.method_label, accuracy: c.mean_accuracy_pct,
      min: Math.round(c.min_accuracy * 100), max: Math.round(c.max_accuracy * 100)
    }))

    return (
      <>
        <Card title="Benchmark Configuration" span={2}>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 13, color: '#475569' }}>
            <span>Benchmark: <strong>{benchInfo.benchmark || '—'}</strong></span>
            <span>Window: <strong>{benchInfo.window_seconds}s</strong></span>
            <span>Stride: <strong>{benchInfo.stride_seconds}s</strong></span>
            <span>Features: <strong>{benchInfo.features}</strong></span>
          </div>
          <div style={{ marginTop: 8, fontSize: 12, color: '#64748b' }}>
            Leakage prevention: {benchInfo.no_leakage || 'N/A'}
          </div>
        </Card>

        <Card title="Method Accuracy Comparison" span={2}>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-10} textAnchor="end" height={60} />
              <YAxis domain={[0, 100]} unit="%" />
              <Tooltip formatter={v => `${v}%`} />
              <Bar dataKey="accuracy" fill="#3b82f6" name="Mean Accuracy %" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Per-Method Fold Details" span={2}>
          {comparison.map((method, mi) => (
            <div key={mi} style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
                {method.method_label}
                <span style={{ marginLeft: 8 }}>
                  <Badge text={`Mean: ${method.mean_accuracy_pct}%`} color="#3b82f6" />
                </span>
              </div>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>{method.method_description}</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {(method.folds || []).map((fold, fi) => (
                  <div key={fi} style={{
                    padding: '6px 12px', borderRadius: 8, background: '#f8fafc',
                    border: '1px solid #e2e8f0', fontSize: 12
                  }}>
                    <strong>{fold.subject}</strong>: {fold.accuracy_pct}%
                    {fold.f1 !== fold.accuracy && <span style={{ color: '#64748b' }}> (F1: {Math.round(fold.f1 * 100)}%)</span>}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </Card>

        <Card title="Training Run History" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Script', 'Status', 'Exit Code', 'Duration (s)'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {runs.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, fontFamily: 'monospace', fontSize: 12 }}>{r.script}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <Badge text={r.ok ? 'SUCCESS' : 'FAILED'} color={r.ok ? '#10b981' : '#ef4444'} />
                    </td>
                    <td style={{ padding: '8px 12px' }}>{r.exit_code}</td>
                    <td style={{ padding: '8px 12px' }}>{r.seconds}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* -- Patient Results Tab ------------------------------------------ */
  const renderPatients = () => {
    const details = breakdown?.patient_details || []
    const dbProfiles = breakdown?.db_patient_profiles || []

    const chartData = details.map(d => ({
      name: d.subject,
      accuracy: d.accuracy_pct,
      sensitivity: d.sensitivity_pct,
      f1: d.f1_pct
    }))

    return (
      <>
        <Card title="Per-Subject Accuracy / Sensitivity / F1" span={2}>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} unit="%" />
              <Tooltip formatter={v => `${v}%`} />
              <Bar dataKey="accuracy" fill={ACC_COLORS.accuracy} name="Accuracy %" radius={[4, 4, 0, 0]} />
              <Bar dataKey="sensitivity" fill={ACC_COLORS.sensitivity} name="Sensitivity %" radius={[4, 4, 0, 0]} />
              <Bar dataKey="f1" fill={ACC_COLORS.f1} name="F1 %" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Patient-Specific Detailed Metrics" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Subject', 'Total Windows', 'Seizure Windows', 'Test Windows', 'Accuracy', 'Sensitivity', 'F1', 'Seizure Ratio'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {details.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{d.subject}</td>
                    <td style={{ padding: '8px 12px' }}>{d.n_total}</td>
                    <td style={{ padding: '8px 12px' }}>{d.n_seizure}</td>
                    <td style={{ padding: '8px 12px' }}>{d.n_test}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{ fontWeight: 700, color: d.accuracy_pct >= 95 ? '#10b981' : d.accuracy_pct >= 80 ? '#f59e0b' : '#ef4444' }}>
                        {d.accuracy_pct}%
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>{d.sensitivity_pct}%</td>
                    <td style={{ padding: '8px 12px' }}>{d.f1_pct}%</td>
                    <td style={{ padding: '8px 12px' }}>{d.seizure_ratio}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Clinical DB Patient Profiles (Top 20)" span={2}>
          <div style={{ overflowX: 'auto', maxHeight: 350, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#f8fafc' }}>
                  {['Patient', 'Name', 'Age', 'Gender', 'Disease', 'Analyses', 'Avg Confidence'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dbProfiles.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600, fontFamily: 'monospace', fontSize: 11 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 10px' }}>{p.name || '—'}</td>
                    <td style={{ padding: '8px 10px' }}>{p.age || '—'}</td>
                    <td style={{ padding: '8px 10px' }}>{p.gender || '—'}</td>
                    <td style={{ padding: '8px 10px' }}><Badge text={p.disease || 'Unknown'} color="#8b5cf6" /></td>
                    <td style={{ padding: '8px 10px' }}>{p.analysis_count}</td>
                    <td style={{ padding: '8px 10px' }}>{p.avg_confidence ? `${p.avg_confidence}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* -- Definitions Tab ---------------------------------------------- */
  const renderDefinitions = () => {
    const sections = definitions?.sections || []
    return (
      <>
        {sections.map((sec, si) => (
          <Card key={si} title={sec.title} span={2}>
            <div style={{ display: 'grid', gap: 10 }}>
              {(sec.items || []).map((item, ii) => (
                <div key={ii} style={{
                  padding: '10px 14px', borderRadius: 8, background: '#f8fafc',
                  borderLeft: '3px solid #3b82f6'
                }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.definition}</div>
                </div>
              ))}
            </div>
          </Card>
        ))}
      </>
    )
  }

  return (
    <div style={{ padding: '24px 20px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Deep Learning Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Model architectures, training history, and seizure detection accuracy metrics
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {tab === 'overview' && renderOverview()}
        {tab === 'architectures' && renderArchitectures()}
        {tab === 'training' && renderTraining()}
        {tab === 'patients' && renderPatients()}
        {tab === 'definitions' && renderDefinitions()}
      </div>
    </div>
  )
}
