import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const STATUS_COLORS = { production: '#10b981', staging: '#f59e0b', not_deployed: '#94a3b8' }
const DRIFT_COLORS = { none: '#10b981', info: '#3b82f6', warning: '#f59e0b', critical: '#ef4444' }
const TRIGGER_COLORS = { scheduled: '#3b82f6', drift_detected: '#ef4444', manual: '#8b5cf6', new_data: '#10b981', latest: '#64748b' }

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

function fmt(v, digits = 4) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(digits)) : String(v)
}

function pct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

export default function ModelOpsDashboard() {
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
      axios.get(`${API}/model-ops/overview`),
      axios.get(`${API}/model-ops/breakdown`),
      axios.get(`${API}/model-ops/definitions`),
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
    { id: 'models', label: 'Model Registry' },
    { id: 'monitoring', label: 'Monitoring & Retrain' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Model Ops...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const kpis = overview?.kpis || {}
  const registry = overview?.registry || []
  const drift = overview?.drift || {}
  const consistency = overview?.consistency || {}
  const extVal = overview?.external_validation || {}
  const accDist = breakdown?.accuracy_distribution || []
  const sizeCmp = breakdown?.size_comparison || []
  const usageAct = breakdown?.usage_activity || {}
  const retrainHist = breakdown?.retrain_history || []
  const compUsage = usageAct.component_usage || []

  return (
    <div style={{ padding: '0 8px' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#1e40af' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          {/* KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12, marginBottom: 16 }}>
            <Card><KPI label="Total Models" value={kpis.total_models} /></Card>
            <Card><KPI label="In Production" value={kpis.deployed_production} color="#10b981" /></Card>
            <Card><KPI label="In Staging" value={kpis.deployed_staging} color="#f59e0b" /></Card>
            <Card><KPI label="Mean Accuracy" value={pct(kpis.mean_accuracy)} color="#3b82f6" /></Card>
            <Card><KPI label="Min Accuracy" value={pct(kpis.min_accuracy)} color={kpis.min_accuracy < 0.9 ? '#ef4444' : '#10b981'} /></Card>
            <Card><KPI label="Drift Alert" value={kpis.drift_alert?.toUpperCase()} color={DRIFT_COLORS[kpis.drift_alert] || '#64748b'} /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
            {/* Accuracy by Disease */}
            <Card title="Accuracy by Disease">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={accDist} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0.9, 1]} tickFormatter={v => pct(v)} />
                  <YAxis type="category" dataKey="disease" width={100} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={v => pct(v)} />
                  <Bar dataKey="accuracy" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Deploy Status Distribution */}
            <Card title="Deployment Status">
              {(() => {
                const statusCounts = registry.reduce((acc, m) => {
                  acc[m.deploy_status] = (acc[m.deploy_status] || 0) + 1
                  return acc
                }, {})
                const pieData = Object.entries(statusCounts).map(([k, v]) => ({ name: k, value: v }))
                return (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                    <ResponsiveContainer width="50%" height={200}>
                      <PieChart>
                        <Pie data={pieData} cx="50%" cy="50%" outerRadius={70} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                          {pieData.map((d, i) => <Cell key={i} fill={STATUS_COLORS[d.name] || COLORS[i]} />)}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                    <div>
                      {pieData.map(d => (
                        <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                          <div style={{ width: 12, height: 12, borderRadius: 3, background: STATUS_COLORS[d.name] || '#94a3b8' }} />
                          <span style={{ fontSize: 13 }}>{d.name}: <strong>{d.value}</strong></span>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })()}
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
            {/* Drift Status */}
            <Card title="Drift Monitoring">
              {drift.available ? (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 12 }}>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Verdict</span><div style={{ fontSize: 16, fontWeight: 700, color: DRIFT_COLORS[kpis.drift_alert] }}>{drift.verdict}</div></div>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Features Drifted</span><div style={{ fontSize: 16, fontWeight: 700 }}>{drift.n_high_drift}/{drift.n_features}</div></div>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Drift Fraction</span><div style={{ fontSize: 16, fontWeight: 700 }}>{pct(drift.frac_drifted)}</div></div>
                  </div>
                  {drift.top_drift?.length > 0 && (
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4, color: '#475569' }}>Top Drifting Features</div>
                      <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                        <thead><tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                          <th style={{ textAlign: 'left', padding: 4 }}>Feature</th>
                          <th style={{ textAlign: 'right', padding: 4 }}>PSI</th>
                          <th style={{ textAlign: 'right', padding: 4 }}>KS stat</th>
                        </tr></thead>
                        <tbody>
                          {drift.top_drift.map((f, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                              <td style={{ padding: 4 }}>{f.feature}</td>
                              <td style={{ padding: 4, textAlign: 'right', color: f.psi > 0.2 ? '#ef4444' : '#10b981' }}>{fmt(f.psi, 3)}</td>
                              <td style={{ padding: 4, textAlign: 'right' }}>{fmt(f.ks_stat, 3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>Last run: {drift.run_at}</div>
                </div>
              ) : <div style={{ color: '#94a3b8' }}>Drift monitoring not available</div>}
            </Card>

            {/* Consistency + External Validation */}
            <Card title="Quality Checks">
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Prediction Consistency</div>
                {consistency.available ? (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Verdict</span><div style={{ fontSize: 16, fontWeight: 700, color: consistency.verdict === 'PASS' ? '#10b981' : '#ef4444' }}>{consistency.verdict}</div></div>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Checked</span><div style={{ fontSize: 16, fontWeight: 700 }}>{consistency.n_checked}</div></div>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Mismatches</span><div style={{ fontSize: 16, fontWeight: 700, color: consistency.mismatches > 0 ? '#ef4444' : '#10b981' }}>{consistency.mismatches}</div></div>
                  </div>
                ) : <div style={{ color: '#94a3b8' }}>Not available</div>}
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>External Validation</div>
                {extVal.available ? (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Dataset</span><div style={{ fontSize: 14, fontWeight: 600 }}>{extVal.dataset}</div></div>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Accuracy</span><div style={{ fontSize: 16, fontWeight: 700, color: '#3b82f6' }}>{pct(extVal.accuracy)}</div></div>
                    <div><span style={{ fontSize: 12, color: '#64748b' }}>Samples</span><div style={{ fontSize: 16, fontWeight: 700 }}>{extVal.n_samples}</div></div>
                  </div>
                ) : <div style={{ color: '#94a3b8' }}>Not available</div>}
              </div>
            </Card>
          </div>
        </>
      )}

      {tab === 'models' && (
        <>
          {/* Model Size Comparison */}
          <Card title="Model Size (MB)" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sizeCmp}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="disease" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip formatter={v => `${v} MB`} />
                <Bar dataKey="size_mb" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Full Model Registry Table */}
          <Card title="Model Registry" span={2}>
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: 6 }}>Disease</th>
                    <th style={{ padding: 6 }}>Architecture</th>
                    <th style={{ padding: 6 }}>Version</th>
                    <th style={{ padding: 6 }}>Status</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>Accuracy</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>Precision</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>Recall</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>F1</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>Size</th>
                    <th style={{ padding: 6 }}>Last Trained</th>
                  </tr>
                </thead>
                <tbody>
                  {registry.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: 6, fontWeight: 600 }}>{m.name}</td>
                      <td style={{ padding: 6, color: '#64748b' }}>{m.architecture}</td>
                      <td style={{ padding: 6 }}><code style={{ background: '#f1f5f9', padding: '1px 6px', borderRadius: 4, fontSize: 11 }}>{m.version}</code></td>
                      <td style={{ padding: 6 }}><Badge text={m.deploy_status} color={STATUS_COLORS[m.deploy_status] || '#94a3b8'} /></td>
                      <td style={{ padding: 6, textAlign: 'right', fontWeight: 600, color: m.accuracy >= 0.95 ? '#10b981' : m.accuracy >= 0.9 ? '#f59e0b' : '#ef4444' }}>{pct(m.accuracy)}</td>
                      <td style={{ padding: 6, textAlign: 'right' }}>{pct(m.precision)}</td>
                      <td style={{ padding: 6, textAlign: 'right' }}>{pct(m.recall)}</td>
                      <td style={{ padding: 6, textAlign: 'right' }}>{pct(m.f1)}</td>
                      <td style={{ padding: 6, textAlign: 'right' }}>{m.size_mb > 0 ? `${m.size_mb} MB` : '--'}</td>
                      <td style={{ padding: 6, color: '#64748b', fontSize: 11 }}>{m.last_trained || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Per-model Metrics Radar */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12, marginTop: 12 }}>
            {registry.map((m, i) => (
              <Card key={i} title={m.name}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
                  <div><span style={{ color: '#64748b' }}>Architecture</span><div style={{ fontWeight: 600 }}>{m.architecture}</div></div>
                  <div><span style={{ color: '#64748b' }}>Version</span><div style={{ fontWeight: 600 }}>{m.version}</div></div>
                  <div><span style={{ color: '#64748b' }}>Accuracy</span><div style={{ fontWeight: 700, fontSize: 18, color: '#3b82f6' }}>{pct(m.accuracy)}</div></div>
                  <div><span style={{ color: '#64748b' }}>F1 Score</span><div style={{ fontWeight: 700, fontSize: 18, color: '#10b981' }}>{pct(m.f1)}</div></div>
                  <div><span style={{ color: '#64748b' }}>Samples</span><div style={{ fontWeight: 600 }}>{m.n_samples}</div></div>
                  <div><span style={{ color: '#64748b' }}>Size</span><div style={{ fontWeight: 600 }}>{m.size_mb > 0 ? `${m.size_mb} MB` : '--'}</div></div>
                </div>
                <div style={{ marginTop: 8 }}>
                  <Badge text={m.deploy_status} color={STATUS_COLORS[m.deploy_status] || '#94a3b8'} />
                  {m.exists && <span style={{ fontSize: 11, color: '#10b981', marginLeft: 8 }}>file present</span>}
                </div>
              </Card>
            ))}
          </div>
        </>
      )}

      {tab === 'monitoring' && (
        <>
          {/* Usage Activity */}
          {usageAct.daily_predictions?.length > 0 && (
            <Card title="Daily Prediction Activity (30 days)">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={usageAct.daily_predictions}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={4} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="predictions" stroke="#3b82f6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Component Usage */}
          {compUsage.length > 0 && (
            <Card title="Component Usage (30 days)">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={compUsage} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="component" width={120} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Retrain History */}
          <Card title="Retrain History">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: 6 }}>Date</th>
                    <th style={{ padding: 6 }}>Model</th>
                    <th style={{ padding: 6 }}>Trigger</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>Acc Before</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>Acc After</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>Improvement</th>
                    <th style={{ padding: 6, textAlign: 'right' }}>Samples Added</th>
                  </tr>
                </thead>
                <tbody>
                  {retrainHist.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', opacity: r.trigger === 'latest' ? 0.6 : 1 }}>
                      <td style={{ padding: 6, fontFamily: 'monospace', fontSize: 11 }}>{r.date}</td>
                      <td style={{ padding: 6, fontWeight: 600 }}>{r.label}</td>
                      <td style={{ padding: 6 }}><Badge text={r.trigger} color={TRIGGER_COLORS[r.trigger] || '#64748b'} /></td>
                      <td style={{ padding: 6, textAlign: 'right' }}>{r.accuracy_before != null ? pct(r.accuracy_before) : '--'}</td>
                      <td style={{ padding: 6, textAlign: 'right' }}>{r.accuracy_after != null ? pct(r.accuracy_after) : '--'}</td>
                      <td style={{ padding: 6, textAlign: 'right', color: r.improvement > 0 ? '#10b981' : '#64748b' }}>{r.improvement != null ? `+${pct(r.improvement)}` : '--'}</td>
                      <td style={{ padding: 6, textAlign: 'right' }}>{r.samples_added || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Retrain Trigger Distribution */}
          <Card title="Retrain Triggers Distribution">
            {(() => {
              const triggerCounts = retrainHist.filter(r => r.trigger !== 'latest').reduce((acc, r) => {
                acc[r.trigger] = (acc[r.trigger] || 0) + 1
                return acc
              }, {})
              const pieData = Object.entries(triggerCounts).map(([k, v]) => ({ name: k, value: v }))
              return (
                <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                  <ResponsiveContainer width="50%" height={200}>
                    <PieChart>
                      <Pie data={pieData} cx="50%" cy="50%" outerRadius={70} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                        {pieData.map((d, i) => <Cell key={i} fill={TRIGGER_COLORS[d.name] || COLORS[i]} />)}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                  <div>
                    {pieData.map(d => (
                      <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <div style={{ width: 12, height: 12, borderRadius: 3, background: TRIGGER_COLORS[d.name] || '#94a3b8' }} />
                        <span style={{ fontSize: 13 }}>{d.name}: <strong>{d.value}</strong></span>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })()}
          </Card>
        </>
      )}

      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gap: 12 }}>
          {definitions.sections?.map((sec, i) => (
            <Card key={i} title={sec.name}>
              <p style={{ fontSize: 13, color: '#64748b', marginTop: 0 }}>{sec.description}</p>
              {sec.fields && (
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <tbody>
                    {sec.fields.map((f, j) => (
                      <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, width: '30%' }}>{f.name}</td>
                        <td style={{ padding: '6px 8px', color: '#64748b' }}>{f.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {sec.items && (
                <ul style={{ margin: '4px 0 0', paddingLeft: 20, fontSize: 12, color: '#475569' }}>
                  {sec.items.map((item, j) => <li key={j} style={{ marginBottom: 4 }}>{item}</li>)}
                </ul>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
