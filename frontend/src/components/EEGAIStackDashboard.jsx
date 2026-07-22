import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316', '#14b8a6', '#a855f7']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function StatusBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s === 'installed' ? '#3b82f6' : s === 'built' ? '#10b981' : s === 'external' ? '#f59e0b' : s === 'cataloged' ? '#8b5cf6' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(status || 'N/A')}</span>
  )
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const STATUS_COLORS = { installed: '#3b82f6', built: '#10b981', external: '#f59e0b', cataloged: '#8b5cf6' }

export default function EEGAIStackDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/eeg-ai-stack/overview`),
      axios.get(`${API_URL}/api/eeg-ai-stack/breakdown`),
      axios.get(`${API_URL}/api/eeg-ai-stack/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EEG AI Stack...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Stack data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'tools', label: 'All Tools' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = ov.kpis || {}
  const statusDist = (ov.status_distribution || []).map(d => ({ ...d, fill: STATUS_COLORS[d.status] || '#64748b' }))
  const layerData = (ov.layers || []).map((l, i) => ({ ...l, fill: COLORS[i % COLORS.length] }))

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: 600, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card title="EEG AI Tool Stack KPIs" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 16, padding: '8px 0' }}>
              <KPI label="Total Tools" value={kpis.total_tools} color="#1e293b" />
              <KPI label="Installed" value={kpis.installed} color="#3b82f6" />
              <KPI label="Built" value={kpis.built} color="#10b981" />
              <KPI label="External" value={kpis.external} color="#f59e0b" />
              <KPI label="Cataloged" value={kpis.cataloged} color="#8b5cf6" />
              <KPI label="Layers" value={kpis.layers} color="#06b6d4" />
              <KPI label="With Endpoints" value={kpis.with_endpoints} color="#ec4899" />
              <KPI label="EDC Tools" value={kpis.edc_tools} color="#64748b" />
            </div>
          </Card>

          {/* Status Distribution Pie */}
          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={statusDist} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={90} label={({ status, count }) => `${status}: ${count}`}>
                  {statusDist.map((d, i) => (
                    <Cell key={i} fill={d.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Layer Active % Bar */}
          <Card title="Active Tools per Layer (%)" span={2}>
            <ResponsiveContainer width="100%" height={Math.max(260, layerData.length * 28)}>
              <BarChart data={layerData} layout="vertical" margin={{ left: 180 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis type="category" dataKey="layer" tick={{ fontSize: 11 }} width={170} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="active_pct" radius={[0, 4, 4, 0]} name="Active %">
                  {layerData.map((d, i) => (
                    <Cell key={i} fill={d.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Layer Summary Table */}
          <Card title={`Layer Summary (${layerData.length} layers)`} span={3}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Layer', 'Total', 'Installed', 'Built', 'External', 'Cataloged', 'Active %'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {layerData.map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{l.layer}</td>
                      <td style={{ padding: '6px 10px' }}>{l.total}</td>
                      <td style={{ padding: '6px 10px', color: '#3b82f6', fontWeight: 600 }}>{l.installed}</td>
                      <td style={{ padding: '6px 10px', color: '#10b981', fontWeight: 600 }}>{l.built}</td>
                      <td style={{ padding: '6px 10px', color: '#f59e0b', fontWeight: 600 }}>{l.external}</td>
                      <td style={{ padding: '6px 10px', color: '#8b5cf6', fontWeight: 600 }}>{l.cataloged}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 700, color: l.active_pct >= 80 ? '#10b981' : l.active_pct >= 50 ? '#f59e0b' : '#ef4444' }}>{l.active_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Tools with Live Endpoints */}
          <Card title="Tools with Live API Endpoints" span={3}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Tool', 'Layer', 'Status', 'Endpoints'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(ov.tools || []).filter(t => (t.endpoints && t.endpoints.length > 0) || t.dashboard).map((t, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{t.name}</td>
                      <td style={{ padding: '6px 10px', color: '#3b82f6', fontWeight: 600, whiteSpace: 'nowrap' }}>{t.layer}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={t.status} /></td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12, color: '#64748b' }}>
                        {t.endpoints && t.endpoints.length > 0 ? t.endpoints.join(', ') : t.dashboard || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Recommended Pipeline */}
          {ov.recommended_pipeline && (
            <Card title="Recommended Pipeline" span={3}>
              <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6, fontFamily: 'monospace', background: '#f8fafc', padding: 12, borderRadius: 8 }}>{ov.recommended_pipeline}</p>
            </Card>
          )}

          {/* Honest Note */}
          {ov.honest_note && (
            <Card title="Honest Note" span={3}>
              <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{ov.honest_note}</p>
            </Card>
          )}
        </div>
      )}

      {tab === 'tools' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Full Tool Table */}
          <Card title={`All Tools (${bd.total})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Tool', 'Layer', 'Status', 'Use', 'Endpoints', 'Note'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.tools || []).map((t, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{t.name}</td>
                      <td style={{ padding: '6px 10px', color: '#3b82f6', fontWeight: 600, whiteSpace: 'nowrap', fontSize: 12 }}>{t.layer}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={t.status} /></td>
                      <td style={{ padding: '6px 10px', color: '#475569', maxWidth: 300 }}>{t.use || '--'}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11, color: '#64748b', maxWidth: 250 }}>
                        {t.endpoints && t.endpoints.length > 0 ? t.endpoints.join(', ') : t.dashboard || '--'}
                      </td>
                      <td style={{ padding: '6px 10px', color: '#94a3b8', fontSize: 12 }}>{t.note || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* EDC Assessment Tools */}
          {bd.edc_assessment_tools && bd.edc_assessment_tools.length > 0 && (
            <Card title={`EDC / Assessment Tools (${bd.edc_assessment_tools.length})`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      {['Tool', 'Use', 'Status', 'Endpoints'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {bd.edc_assessment_tools.map((e, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{e.name}</td>
                        <td style={{ padding: '6px 10px', color: '#475569' }}>{e.use || '--'}</td>
                        <td style={{ padding: '6px 10px' }}><StatusBadge status={e.status} /></td>
                        <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12, color: '#64748b' }}>
                          {e.endpoints && e.endpoints.length > 0 ? e.endpoints.join(', ') : '--'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Layer Descriptions */}
          <Card title="Layer Descriptions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Layer', 'Description'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.layers || []).map((l, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: '#1e293b', whiteSpace: 'nowrap' }}>{l.name}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', lineHeight: 1.5 }}>{l.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Status Legend */}
          <Card title="Status Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Status', 'Meaning'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.status_legend || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px' }}><StatusBadge status={s.status} /></td>
                    <td style={{ padding: '6px 10px', color: '#475569', lineHeight: 1.5 }}>{s.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Glossary */}
          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Term', 'Definition'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: '#1e293b', whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', lineHeight: 1.5 }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Clinical Notes */}
          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 13, color: '#475569', lineHeight: 1.7, marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          {/* References */}
          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6, marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </div>
      )}
    </div>
  )
}
