import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const SEVERITY_COLORS = {
  high: '#ef4444',
  moderate: '#f59e0b',
  low: '#64748b'
}

const EFFECT_SIZE_COLORS = {
  large: '#ef4444',
  medium: '#f59e0b',
  small: '#3b82f6'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)) : String(v)
}

function SeverityBadge({ severity }) {
  const color = SEVERITY_COLORS[severity] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(severity || '').replace(/_/g, ' ')}</span>
  )
}

function EffectSizeBadge({ size }) {
  const color = EFFECT_SIZE_COLORS[size] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(size || '').replace(/_/g, ' ')}</span>
  )
}

function DirectionBadge({ direction }) {
  const color = direction === 'positive' ? '#10b981' : direction === 'negative' ? '#ef4444' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(direction || '').replace(/_/g, ' ')}</span>
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

export default function CognitionLinkDashboard() {
  const [overview, setOverview] = useState(null)
  const [matrix, setMatrix] = useState(null)
  const [heatmap, setHeatmap] = useState(null)
  const [domains, setDomains] = useState(null)
  const [alerts, setAlerts] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, mx, hm, dm, al, df] = await Promise.all([
          axios.get(`${API_URL}/cognition-link/overview`),
          axios.get(`${API_URL}/cognition-link/matrix`),
          axios.get(`${API_URL}/cognition-link/heatmap`),
          axios.get(`${API_URL}/cognition-link/domains`),
          axios.get(`${API_URL}/cognition-link/alerts`),
          axios.get(`${API_URL}/cognition-link/definitions`)
        ])
        setOverview(ov.data)
        setMatrix(mx.data)
        setHeatmap(hm.data)
        setDomains(dm.data)
        setAlerts(al.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load CognitionLink data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading CognitionLink data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>CognitionLink data not available</div>

  const tabs = ['overview', 'matrix', 'domains', 'alerts', 'definitions']
  const topR = overview.top_correlations && overview.top_correlations.length > 0
    ? Math.max(...overview.top_correlations.map(c => Math.abs(c.r || 0)))
    : null

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>{overview.title || 'CognitionLink Dashboard'}</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        {overview.subtitle || 'EEG-Cognition correlation analysis'}
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#475569'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Pairs Tested" value={overview.total_pairs_tested} />
              <KPI label="Significant Pairs" value={overview.significant_pairs} color="#10b981" />
              <KPI label="EEG Features" value={overview.eeg_features_count} />
              <KPI label="Cognitive Tests" value={overview.cognitive_tests_count} />
              <KPI label="Top |r|" value={topR} color="#8b5cf6" />
            </div>
          </Card>

          <Card title="Domain Summary — Mean |r|" span={4}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview.domain_summary || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" fontSize={12} />
                <YAxis fontSize={12} />
                <Tooltip />
                <Bar dataKey="mean_abs_r" name="Mean |r|">
                  {(overview.domain_summary || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Top Correlations" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>EEG Feature</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Cognitive Test</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>r</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>p</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Direction</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.top_correlations || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px' }}>{c.eeg_feature}</td>
                      <td style={{ padding: '8px 12px' }}>{c.cognitive_test}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(c.r)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(c.p)}</td>
                      <td style={{ padding: '8px 12px' }}><DirectionBadge direction={c.r >= 0 ? 'positive' : 'negative'} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{c.note || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Matrix tab ── */}
      {tab === 'matrix' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Correlation Matrix (n=${matrix?.n || 0})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>EEG Feature</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Band</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Region</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Cognitive Test</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Domain</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>r</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>p</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Effect Size</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Direction</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Clinical Note</th>
                  </tr>
                </thead>
                <tbody>
                  {(matrix?.correlations || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px' }}>{c.eeg_feature}</td>
                      <td style={{ padding: '8px 12px' }}>{c.eeg_band}</td>
                      <td style={{ padding: '8px 12px' }}>{c.eeg_region}</td>
                      <td style={{ padding: '8px 12px' }}>{c.test_name}</td>
                      <td style={{ padding: '8px 12px' }}>{c.test_domain}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(c.r)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(c.p)}</td>
                      <td style={{ padding: '8px 12px' }}><EffectSizeBadge size={c.effect_size} /></td>
                      <td style={{ padding: '8px 12px' }}><DirectionBadge direction={c.direction} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{c.clinical_note || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Domains tab ── */}
      {tab === 'domains' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {(domains?.profiles || []).map((d, i) => (
            <Card key={i} title={d.domain}>
              <div style={{ marginBottom: 12 }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>Significant pairs: </span>
                <span style={{ fontWeight: 600 }}>{d.n_significant}</span>
                <span style={{ fontSize: 12, color: '#64748b', marginLeft: 16 }}>Strongest predictor: </span>
                <span style={{ fontWeight: 600 }}>{d.strongest_eeg_predictor}</span>
                <span style={{ fontSize: 12, color: '#64748b', marginLeft: 16 }}>r = </span>
                <span style={{ fontWeight: 600, color: '#8b5cf6' }}>{fmt(d.strongest_r)}</span>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>EEG Feature</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Band</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Test</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>r</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>p</th>
                  </tr>
                </thead>
                <tbody>
                  {(d.correlations || []).map((c, j) => (
                    <tr key={j} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 8px' }}>{c.eeg_feature}</td>
                      <td style={{ padding: '6px 8px' }}>{c.eeg_band}</td>
                      <td style={{ padding: '6px 8px' }}>{c.test}</td>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{fmt(c.r)}</td>
                      <td style={{ padding: '6px 8px' }}>{fmt(c.p)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </div>
      )}

      {/* ── Alerts tab ── */}
      {tab === 'alerts' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Clinical Alerts">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Severity</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>EEG Feature</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Cognitive Test</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>r</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>p</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Clinical Note</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Recommendation</th>
                  </tr>
                </thead>
                <tbody>
                  {(alerts?.alerts || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px' }}><SeverityBadge severity={a.severity} /></td>
                      <td style={{ padding: '8px 12px' }}>{a.eeg_feature}</td>
                      <td style={{ padding: '8px 12px' }}>{a.cognitive_test}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(a.r)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(a.p)}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{a.clinical_note || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12 }}>{a.recommendation || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Effect Size Thresholds">
            <div style={{ background: '#f8fafc', borderRadius: 8, padding: 16 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Size</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Min</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Max</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {defs?.effect_size_thresholds && Object.entries(defs.effect_size_thresholds).map(([key, val], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px' }}><EffectSizeBadge size={key} /></td>
                      <td style={{ padding: '8px 12px' }}>{fmt(val.min)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(val.max)}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{val.interpretation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="EEG Bands">
            <div style={{ background: '#f8fafc', borderRadius: 8, padding: 16 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Band</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Frequency (Hz)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Clinical Significance</th>
                  </tr>
                </thead>
                <tbody>
                  {defs?.eeg_bands && Object.entries(defs.eeg_bands).map(([key, val], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, textTransform: 'capitalize' }}>{key}</td>
                      <td style={{ padding: '8px 12px' }}>{val.range_hz}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{val.clinical}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Cognitive Domains">
            <div style={{ background: '#f8fafc', borderRadius: 8, padding: 16 }}>
              {defs?.cognitive_domains && Object.entries(defs.cognitive_domains).map(([key, val], i) => (
                <div key={i} style={{ marginBottom: 12 }}>
                  <span style={{ fontWeight: 600, textTransform: 'capitalize', fontSize: 13 }}>{key.replace(/_/g, ' ')}</span>
                  <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>{typeof val === 'string' ? val : JSON.stringify(val)}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="References">
            <div style={{ background: '#f8fafc', borderRadius: 8, padding: 16 }}>
              {(defs?.references || []).map((ref, i) => (
                <div key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>
                  {i + 1}. {typeof ref === 'string' ? ref : ref.text || ref.title || JSON.stringify(ref)}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
