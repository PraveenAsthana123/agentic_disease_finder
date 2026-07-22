import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = '/api'

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

const COLORS = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899', '#06b6d4', '#f97316', '#64748b']

export default function FineTuningDashboard() {
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
      axios.get(`${API}/fine-tuning/overview`),
      axios.get(`${API}/fine-tuning/breakdown`),
      axios.get(`${API}/fine-tuning/definitions`),
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
    { id: 'inventory', label: 'Model Inventory' },
    { id: 'training', label: 'Training Runs' },
    { id: 'pipeline', label: 'Pipeline Stages' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Fine-Tuning Pipeline dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No fine-tuning data available.'}</div>

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: '#1e293b' }}>Fine-Tuning Pipeline Dashboard</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Model inventory, training runs, accuracy distribution, and pipeline stage tracking
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          {/* KPI cards row 1 */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Total Models" value={overview.kpis.total_models} sub={`${overview.kpis.model_types} model types`} color="#3b82f6" /></Card>
            <Card><KPI label="Diseases Covered" value={overview.kpis.diseases_covered} sub={`${overview.kpis.total_analyses} analyses`} color="#10b981" /></Card>
            <Card><KPI label="Avg Accuracy" value={overview.kpis.avg_accuracy_pct != null ? `${overview.kpis.avg_accuracy_pct}%` : 'N/A'} sub="from model filenames" color="#8b5cf6" /></Card>
            <Card><KPI label="Training Runs" value={overview.kpis.total_training_runs} sub={`${overview.kpis.training_success_pct}% success`} color="#f59e0b" /></Card>
          </div>

          {/* KPI cards row 2 */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Avg Confidence" value={overview.kpis.avg_confidence ?? 'N/A'} sub={`${overview.kpis.patients_with_analyses} patients`} color="#06b6d4" /></Card>
            <Card><KPI label="Feedback Entries" value={overview.kpis.total_feedback} sub={overview.kpis.avg_feedback_rating ? `avg ${overview.kpis.avg_feedback_rating}/5` : 'awaiting feedback'} color="#ec4899" /></Card>
            <Card><KPI label="Pipeline Stages" value={`${overview.kpis.pipeline_stages_complete}/${overview.kpis.pipeline_stages_total}`} sub="stages complete" color="#10b981" /></Card>
            <Card><KPI label="Failed Runs" value={overview.kpis.failed_runs} sub={`of ${overview.kpis.total_training_runs}`} color={overview.kpis.failed_runs > 0 ? '#ef4444' : '#10b981'} /></Card>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Model Type Distribution">
              {(overview.model_type_distribution || []).length ? (
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={overview.model_type_distribution} dataKey="count" nameKey="type"
                      cx="50%" cy="50%" outerRadius={90} label={({ type, count }) => `${type} (${count})`}>
                      {overview.model_type_distribution.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No model data</div>}
            </Card>

            <Card title="Accuracy Distribution">
              {(overview.accuracy_distribution || []).length ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={overview.accuracy_distribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No accuracy data</div>}
            </Card>
          </div>

          {/* Disease distribution + Training timeline */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Disease Coverage">
              {(overview.disease_distribution || []).length ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={overview.disease_distribution} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" tick={{ fontSize: 10 }} />
                    <YAxis dataKey="disease" type="category" tick={{ fontSize: 10 }} width={80} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No disease data</div>}
            </Card>

            <Card title="Training Success Rate Over Time">
              {(overview.training_success_rate || []).length ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={overview.training_success_rate}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="rate_pct" stroke="#3b82f6" strokeWidth={2} name="Success %" />
                  </LineChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No training data yet</div>}
            </Card>
          </div>
        </>
      )}

      {/* ── Model Inventory Tab ── */}
      {tab === 'inventory' && breakdown && (
        <>
          {/* Best accuracy by disease */}
          <Card title="Best Model per Disease" span={2}>
            {(breakdown.best_by_disease || []).length ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12, marginBottom: 16 }}>
                {breakdown.best_by_disease.map((b, i) => (
                  <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, textAlign: 'center' }}>
                    <div style={{ fontSize: 11, color: '#64748b', textTransform: 'capitalize' }}>{b.disease}</div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: b.accuracy >= 80 ? '#10b981' : '#f59e0b' }}>{b.accuracy}%</div>
                    <div style={{ fontSize: 10, color: '#94a3b8' }}>{b.model_type}</div>
                  </div>
                ))}
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No accuracy data</div>}
          </Card>

          {/* Accuracy by model type */}
          <Card title="Accuracy by Model Type" span={2}>
            {(breakdown.accuracy_by_type || []).length ? (
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Model Type</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Count</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Avg Accuracy</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Max</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Min</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.accuracy_by_type.map((a, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.model_type}</td>
                        <td style={{ padding: '6px 10px' }}>{a.count}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={`${a.avg_accuracy}%`} color={a.avg_accuracy >= 60 ? '#10b981' : '#f59e0b'} />
                        </td>
                        <td style={{ padding: '6px 10px' }}>{a.max_accuracy}%</td>
                        <td style={{ padding: '6px 10px' }}>{a.min_accuracy}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No model type data</div>}
          </Card>

          {/* Full model inventory table */}
          <Card title={`Model Inventory (${(breakdown.model_inventory || []).length} models)`} span={2}>
            {(breakdown.model_inventory || []).length ? (
              <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Disease</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Model Type</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Accuracy</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Size (KB)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.model_inventory.map((m, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{m.disease}</td>
                        <td style={{ padding: '6px 10px' }}>{m.model_type}</td>
                        <td style={{ padding: '6px 10px', color: '#64748b', fontFamily: 'monospace', fontSize: 11 }}>{m.date || '-'}</td>
                        <td style={{ padding: '6px 10px' }}>
                          {m.accuracy != null ? (
                            <Badge text={`${m.accuracy}%`} color={m.accuracy >= 80 ? '#10b981' : m.accuracy >= 60 ? '#f59e0b' : '#ef4444'} />
                          ) : '-'}
                        </td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{m.size_kb}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No models found</div>}
          </Card>
        </>
      )}

      {/* ── Training Runs Tab ── */}
      {tab === 'training' && breakdown && (
        <>
          {/* Models by training date */}
          <Card title="Models Created by Date" span={2}>
            {(breakdown.models_by_date || []).length ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={breakdown.models_by_date}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Models" />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No date data</div>}
          </Card>

          {/* Training run detail */}
          <Card title="Training Run History" span={2}>
            {(breakdown.training_runs || []).length ? (
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Action</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Detail</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Actor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.training_runs.map((t, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{(t.date || '').slice(0, 16)}</td>
                        <td style={{ padding: '6px 10px' }}>{t.action}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={t.success ? 'success' : 'failed'} color={t.success ? '#10b981' : '#ef4444'} />
                        </td>
                        <td style={{ padding: '6px 10px' }}>{t.detail}</td>
                        <td style={{ padding: '6px 10px' }}>{t.actor}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No training runs yet</div>}
          </Card>

          {/* Dataset coverage */}
          <Card title="Dataset Coverage per Patient" span={2}>
            {(breakdown.patient_coverage || []).length ? (
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Analyses</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Avg Confidence</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Labels</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Signal Quality</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.patient_coverage.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{p.analysis_count}</td>
                        <td style={{ padding: '6px 10px' }}>{p.avg_confidence ?? '-'}</td>
                        <td style={{ padding: '6px 10px' }}>{p.labels || '-'}</td>
                        <td style={{ padding: '6px 10px' }}>
                          {p.signal_qualities && p.signal_qualities.split(',').map((q, qi) => (
                            <Badge key={qi} text={q.trim()} color={q.trim() === 'Good' ? '#10b981' : '#f59e0b'} />
                          ))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No patient coverage data</div>}
          </Card>
        </>
      )}

      {/* ── Pipeline Stages Tab ── */}
      {tab === 'pipeline' && (
        <>
          <Card title="Fine-Tuning Pipeline Stages" span={2}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 20 }}>
              {(overview.pipeline_stages || []).map((s, i) => (
                <React.Fragment key={i}>
                  <div style={{
                    padding: '16px 20px', borderRadius: 10, minWidth: 160, textAlign: 'center',
                    background: s.status === 'complete' ? '#f0fdf4' : s.status === 'partial' ? '#fffbeb' : '#f8fafc',
                    border: `1px solid ${s.status === 'complete' ? '#86efac' : s.status === 'partial' ? '#fde68a' : '#e2e8f0'}`
                  }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>{s.stage}</div>
                    <Badge text={s.status} color={s.status === 'complete' ? '#10b981' : s.status === 'partial' ? '#f59e0b' : '#94a3b8'} />
                    <div style={{ fontSize: 10, color: '#64748b', marginTop: 6 }}>{s.metric}</div>
                  </div>
                  {i < (overview.pipeline_stages || []).length - 1 && (
                    <div style={{ fontSize: 18, color: '#cbd5e1' }}>&#8594;</div>
                  )}
                </React.Fragment>
              ))}
            </div>
          </Card>

          {/* Stage descriptions */}
          <Card title="Stage Descriptions" span={2}>
            <div style={{ display: 'grid', gap: 8 }}>
              {(overview.pipeline_stages || []).map((s, i) => (
                <div key={i} style={{ padding: '10px 14px', background: '#f8fafc', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 12 }}>
                  <Badge text={s.status} color={s.status === 'complete' ? '#10b981' : s.status === 'partial' ? '#f59e0b' : '#94a3b8'} />
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155' }}>{s.stage}</div>
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{s.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Confidence distribution */}
          <Card title="Prediction Confidence Distribution" span={2}>
            {(overview.confidence_buckets || []).length ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.confidence_buckets}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No confidence data</div>}
          </Card>
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && definitions?.sections && (
        <div style={{ display: 'grid', gap: 16 }}>
          {definitions.sections.map((sec, si) => (
            <Card key={si} title={sec.title} span={2}>
              <div style={{ display: 'grid', gap: 8 }}>
                {(sec.items || []).map((item, ii) => (
                  <div key={ii} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 2 }}>{item.term}</div>
                    <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
