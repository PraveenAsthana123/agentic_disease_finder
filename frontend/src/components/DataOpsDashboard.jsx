import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

function QualityBar({ label, score }) {
  const color = score >= 80 ? '#10b981' : score >= 60 ? '#f59e0b' : '#ef4444'
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ color: '#334155' }}>{label}</span>
        <span style={{ color, fontWeight: 600 }}>{fmt(score)}%</span>
      </div>
      <div style={{ height: 8, background: '#f1f5f9', borderRadius: 4 }}>
        <div style={{ height: '100%', width: `${Math.min(score, 100)}%`, background: color, borderRadius: 4 }} />
      </div>
    </div>
  )
}

export default function DataOpsDashboard() {
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
          axios.get(`${API_URL}/api/data-ops/overview`),
          axios.get(`${API_URL}/api/data-ops/breakdown`),
          axios.get(`${API_URL}/api/data-ops/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load data ops data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128450;</div>
      Loading data operations...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No data ops data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'pipeline', label: 'Pipeline Activity' },
    { id: 'quality', label: 'Data Quality' },
    { id: 'storage', label: 'Storage & Lineage' }
  ]

  const kpi = overview.kpis || {}
  const sigDist = Object.entries(overview.signal_quality_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const modCov = Object.entries(overview.modality_coverage || {}).map(([k, v]) => ({ modality: k, coverage: v }))
  const qualDimSum = Object.entries(overview.quality_dimensions_summary || {}).map(([k, v]) => ({ name: k, score: v }))
  const pipelineTop = overview.pipeline_top5 || []
  const vectorIngest = overview.vector_ingest || {}
  const pipelineActivity = (breakdown && breakdown.pipeline_activity) || []
  const dailyVolume = (breakdown && breakdown.daily_volume) || []
  const ingestionBd = (breakdown && breakdown.ingestion_breakdown) || []
  const storageInv = (breakdown && breakdown.storage_inventory) || []
  const qualDims = (breakdown && breakdown.quality_dimensions) || []
  const missingMatrix = (breakdown && breakdown.missing_matrix) || []
  const dataLineage = (breakdown && breakdown.data_lineage) || []
  const aiReadiness = (breakdown && breakdown.ai_readiness_components) || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Data Operations Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(kpi.total_patients)} patients | {fmt(kpi.total_uploads)} uploads | {fmt(kpi.total_txn_events)} pipeline events | AI grade: {kpi.ai_grade}
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Patients" value={fmt(kpi.total_patients)} color="#3b82f6" /></Card>
            <Card><KPI label="AI Readiness" value={`${fmt(kpi.ai_readiness)}%`} sub={kpi.ai_grade} color={kpi.ai_readiness >= 80 ? '#10b981' : '#f59e0b'} /></Card>
            <Card><KPI label="Avg Quality" value={`${fmt(kpi.avg_quality)}%`} color="#8b5cf6" /></Card>
            <Card><KPI label="Signal Good" value={`${fmt(kpi.signal_good_pct)}%`} color="#10b981" /></Card>
            <Card><KPI label="DB Size" value={`${fmt(kpi.db_size_mb)} MB`} sub={`Vector: ${fmt(kpi.vector_size_mb)} MB`} color="#06b6d4" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Signal Quality Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={sigDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {sigDist.map((d, i) => <Cell key={i} fill={d.name === 'Good' ? '#10b981' : '#ef4444'} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Quality Dimensions">
              {qualDimSum.map((q, i) => (
                <QualityBar key={i} label={q.name} score={q.score} />
              ))}
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Modality Coverage">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={modCov} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="modality" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} tickFormatter={v => v + '%'} />
                  <Tooltip formatter={v => v + '%'} />
                  <Bar dataKey="coverage" radius={[4, 4, 0, 0]} name="Coverage">
                    {modCov.map((d, i) => <Cell key={i} fill={d.coverage >= 80 ? '#10b981' : d.coverage >= 50 ? '#f59e0b' : '#ef4444'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Top Pipeline Components">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={pipelineTop} layout="vertical" margin={{ left: 80, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="component" width={75} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Events" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {tab === 'pipeline' && (
        <>
          <Card title="Daily Pipeline Volume" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={dailyVolume} margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={v => v.slice(5)} />
                <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} name="Events" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <Card title={`Pipeline Activity (${pipelineActivity.length})`}>
              <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '6px 10px' }}>Component</th>
                      <th style={{ padding: '6px 10px' }}>Action</th>
                      <th style={{ padding: '6px 10px' }}>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pipelineActivity.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.component}</td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{p.action}</td>
                        <td style={{ padding: '6px 10px' }}>{fmt(p.count)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card title="Ingestion Breakdown">
              <ResponsiveContainer width="100%" height={Math.max(200, ingestionBd.length * 32)}>
                <BarChart data={ingestionBd} layout="vertical" margin={{ left: 80, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="component" width={75} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Records" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {vectorIngest.status && (
            <Card title="Vector Ingest Status">
              <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 13 }}>
                <div>Status: <strong style={{ color: vectorIngest.status === 'ok' ? '#10b981' : '#ef4444' }}>{vectorIngest.status}</strong></div>
                <div>Records embedded: <strong>{vectorIngest.records_embedded}</strong></div>
                <div>Failed: <strong style={{ color: vectorIngest.records_failed > 0 ? '#ef4444' : '#10b981' }}>{vectorIngest.records_failed}</strong></div>
                <div>Collection: <strong>{vectorIngest.collection}</strong></div>
                <div>Size: <strong>{fmt(vectorIngest.db_size_mb)} MB</strong></div>
                <div style={{ color: '#64748b' }}>Last run: {(vectorIngest.last_run || '').slice(0, 16)}</div>
              </div>
            </Card>
          )}
        </>
      )}

      {tab === 'quality' && (
        <>
          <Card title="Quality Dimensions Detail">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Dimension</th>
                    <th style={{ padding: '8px 12px' }}>Score</th>
                    <th style={{ padding: '8px 12px' }}>Basis</th>
                  </tr>
                </thead>
                <tbody>
                  {qualDims.map((q, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{q.dimension}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{ color: q.score >= 80 ? '#10b981' : q.score >= 60 ? '#f59e0b' : '#ef4444', fontWeight: 700 }}>{fmt(q.score)}%</span>
                      </td>
                      <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{q.basis}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Missing Data Matrix" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Modality</th>
                    <th style={{ padding: '8px 12px' }}>Present</th>
                    <th style={{ padding: '8px 12px' }}>Missing</th>
                    <th style={{ padding: '8px 12px' }}>% Missing</th>
                    <th style={{ padding: '8px 12px' }}>Coverage</th>
                  </tr>
                </thead>
                <tbody>
                  {missingMatrix.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{m.modality}</td>
                      <td style={{ padding: '8px 12px', color: '#10b981' }}>{m.present}</td>
                      <td style={{ padding: '8px 12px', color: m.missing > 0 ? '#ef4444' : undefined }}>{m.missing}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(m.pct_missing)}%</td>
                      <td style={{ padding: '8px 12px', width: 120 }}>
                        <div style={{ height: 8, background: '#f1f5f9', borderRadius: 4 }}>
                          <div style={{ height: '100%', width: `${100 - m.pct_missing}%`, background: m.pct_missing > 50 ? '#ef4444' : '#10b981', borderRadius: 4 }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {Object.keys(aiReadiness).length > 0 && (
            <Card title="AI Readiness Components">
              {Object.entries(aiReadiness).map(([k, v], i) => (
                <QualityBar key={i} label={k} score={typeof v === 'number' ? v : 0} />
              ))}
            </Card>
          )}
        </>
      )}

      {tab === 'storage' && (
        <>
          <Card title={`Storage Inventory (${storageInv.length} tables)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Table</th>
                    <th style={{ padding: '8px 12px' }}>Rows</th>
                  </tr>
                </thead>
                <tbody>
                  {storageInv.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.table}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(s.rows)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {dataLineage.length > 0 && (
            <Card title={`Data Lineage (${dataLineage.length} flows)`}>
              {dataLineage.map((l, i) => (
                <div key={i} style={{
                  padding: '8px 14px', marginBottom: 8, background: '#f8fafc', borderRadius: 8,
                  fontSize: 12, display: 'flex', alignItems: 'center', gap: 8
                }}>
                  <span style={{ fontWeight: 700, color: '#1e293b' }}>{l.source || l.from}</span>
                  <span style={{ color: '#94a3b8' }}>&rarr;</span>
                  <span style={{ fontWeight: 700, color: '#3b82f6' }}>{l.target || l.to}</span>
                  {l.transform && <span style={{ color: '#64748b' }}>({l.transform})</span>}
                  {l.description && <span style={{ color: '#64748b' }}>— {l.description}</span>}
                </div>
              ))}
            </Card>
          )}
        </>
      )}

      {defs && (
        <div style={{ marginTop: 20, padding: 16, background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
          <strong>Definitions:</strong> {(defs.sections || []).map(s => s.title || s.name).join(', ')}
        </div>
      )}
    </div>
  )
}
