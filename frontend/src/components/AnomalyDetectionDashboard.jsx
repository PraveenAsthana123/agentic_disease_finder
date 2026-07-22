import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ScatterChart, Scatter, ZAxis, LineChart, Line
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

const SEV_COLORS = {
  Normal: '#10b981', Mild: '#f59e0b', Moderate: '#f97316', Severe: '#ef4444'
}

const CAT_COLORS = {
  'Time-Domain': '#3b82f6', 'Spectral': '#8b5cf6', 'Complexity': '#f59e0b',
  'Hjorth': '#ec4899', 'Connectivity': '#06b6d4', 'Other': '#64748b'
}

export default function AnomalyDetectionDashboard() {
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
      axios.get(`${API}/api/anomaly-detection/overview`),
      axios.get(`${API}/api/anomaly-detection/breakdown`),
      axios.get(`${API}/api/anomaly-detection/definitions`),
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
    { id: 'patients', label: 'Patient Analysis' },
    { id: 'features', label: 'Feature Explorer' },
    { id: 'timeline', label: 'Anomaly Timeline' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Anomaly Detection dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No anomaly detection data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>EEG Anomaly Detection</h2>
        <Badge
          text={`${overview.kpis?.anomalous_analyses || 0} anomalous analyses`}
          color={overview.kpis?.anomalous_analyses > 10 ? '#ef4444' : overview.kpis?.anomalous_analyses > 5 ? '#f59e0b' : '#10b981'}
        />
        <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>Generated: {overview.generated_at}</span>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'patients' && renderPatients()}
      {tab === 'features' && renderFeatures()}
      {tab === 'timeline' && renderTimeline()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview.kpis || {}
    const kpis = [
      { label: 'Total Analyses', value: k.total_analyses, color: '#3b82f6' },
      { label: 'Patients', value: k.total_patients, color: '#8b5cf6' },
      { label: 'Features Monitored', value: k.total_features_monitored, color: '#06b6d4' },
      { label: 'Anomalous Analyses', value: k.anomalous_analyses, color: k.anomalous_analyses > 10 ? '#ef4444' : '#f59e0b' },
      { label: 'Anomaly Rate', value: k.anomaly_rate, color: '#f97316' },
      { label: 'Severe Anomalies', value: k.severe_anomalies, color: '#ef4444' },
      { label: 'Avg Anomalies/Analysis', value: k.mean_anomalies_per_analysis, color: '#64748b' },
      { label: 'Most Anomalous Feature', value: k.most_anomalous_feature, color: '#1e293b' },
    ]

    const catData = overview.anomaly_by_category || []
    const sevDist = overview.anomaly_severity_distribution || []
    const topFeatures = overview.top_anomalous_features || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Anomalies by Feature Category" span={2}>
          {catData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={catData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="anomaly_count" name="Anomalies" radius={[4, 4, 0, 0]}>
                  {catData.map((c, i) => <Cell key={i} fill={CAT_COLORS[c.category] || '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Severity Distribution" span={2}>
          {sevDist.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={sevDist} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={90} label={({ severity, count }) => `${severity}: ${count}`}>
                  {sevDist.map((s, i) => <Cell key={i} fill={SEV_COLORS[s.severity] || '#64748b'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Top Anomalous Features (by count)" span={4}>
          {topFeatures.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(200, topFeatures.length * 30)}>
              <BarChart data={topFeatures.slice(0, 15)} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={110} />
                <Tooltip formatter={(v) => typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v} />
                <Bar dataKey="anomaly_count" name="Anomaly Count" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No anomalous features detected.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderPatients() {
    const patients = breakdown?.per_patient_anomalies || []
    const sigQuality = breakdown?.signal_quality_analysis || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Anomaly Summary (${patients.length} patients)`}>
          {patients.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={patients.slice(0, 20)} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="total_anomalies" name="Total Anomalies" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="severe" name="Severe" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Patient Anomaly Detail" span={1}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Total Anomalies</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Severe</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Confidence</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Seizures</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Anomalous Features</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={p.total_anomalies} color={p.total_anomalies > 10 ? '#ef4444' : p.total_anomalies > 5 ? '#f59e0b' : '#10b981'} />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      {p.severe > 0 ? <Badge text={p.severe} color="#ef4444" /> : <span style={{ color: '#94a3b8' }}>0</span>}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {p.confidence != null ? (p.confidence * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      {p.has_seizures ? <Badge text="Yes" color="#ef4444" /> : <span style={{ color: '#94a3b8' }}>No</span>}
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>
                      {(p.anomalous_features || []).slice(0, 5).join(', ')}
                      {(p.anomalous_features || []).length > 5 && ` +${p.anomalous_features.length - 5} more`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {sigQuality.length > 0 && (
          <Card title="Signal Quality vs Anomalies">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={sigQuality} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="anomaly_count" name="Anomalies" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}
      </div>
    )
  }

  function renderFeatures() {
    const featureStats = breakdown?.per_feature_stats || []
    const catRadar = (overview?.anomaly_by_category || []).map(c => ({
      category: c.category,
      anomaly_rate: parseFloat(c.anomaly_rate) || 0
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {catRadar.length > 0 && (
          <Card title="Category Anomaly Rate Radar" span={1}>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={catRadar}>
                <PolarGrid />
                <PolarAngleAxis dataKey="category" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis tick={{ fontSize: 10 }} />
                <Radar dataKey="anomaly_rate" name="Anomaly Rate %" fill="#8b5cf6" fillOpacity={0.3} stroke="#8b5cf6" />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        )}

        <Card title="Feature Anomaly Correlation" span={1}>
          {(breakdown?.feature_correlation_anomalies || []).length > 0 ? (
            <div>
              {(breakdown?.feature_correlation_anomalies || []).slice(0, 10).map((fc, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '8px 0', borderBottom: i < 9 ? '1px solid #f1f5f9' : 'none'
                }}>
                  <div style={{ fontSize: 13, color: '#334155' }}>{fc.feature_pair}</div>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <span style={{ fontFamily: 'monospace', fontSize: 13, color: '#3b82f6' }}>r={typeof fc.correlation === 'number' ? fc.correlation.toFixed(3) : fc.correlation}</span>
                    <Badge text={`${fc.anomaly_overlap} overlap`} color="#f59e0b" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No correlated anomalies found.</p>
          )}
        </Card>

        <Card title={`Feature Statistics (${featureStats.length} features)`} span={2}>
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Category</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Mean</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Std</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Min</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Max</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Anomalies</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Outlier Patients</th>
                </tr>
              </thead>
              <tbody>
                {featureStats.map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: f.anomaly_count > 0 ? '#fef2f218' : undefined }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{f.feature}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={f.category} color={CAT_COLORS[f.category] || '#64748b'} />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.mean === 'number' ? f.mean.toFixed(4) : f.mean}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.std === 'number' ? f.std.toFixed(4) : f.std}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.min === 'number' ? f.min.toFixed(4) : f.min}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.max === 'number' ? f.max.toFixed(4) : f.max}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      {f.anomaly_count > 0 ? <Badge text={f.anomaly_count} color="#ef4444" /> : <span style={{ color: '#94a3b8' }}>0</span>}
                    </td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b' }}>
                      {(f.outlier_patients || []).slice(0, 3).join(', ')}
                      {(f.outlier_patients || []).length > 3 && ` +${f.outlier_patients.length - 3}`}
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

  function renderTimeline() {
    const timeline = breakdown?.anomaly_timeline || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Anomaly Count Over Analyses">
          {timeline.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={timeline} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="anomaly_count" name="Anomalies" stroke="#f59e0b" strokeWidth={2} dot={{ fill: '#f59e0b' }} />
                <Line type="monotone" dataKey="severe_count" name="Severe" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444' }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Anomaly Timeline Detail">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Analysis ID</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Date</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Anomalies</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Severe</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {timeline.map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>{t.analysis_id}</td>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{t.patient_id}</td>
                    <td style={{ padding: '10px 12px', fontSize: 12, color: '#64748b' }}>{t.created_at}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={t.anomaly_count} color={t.anomaly_count > 10 ? '#ef4444' : t.anomaly_count > 5 ? '#f59e0b' : '#10b981'} />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      {t.severe_count > 0 ? <Badge text={t.severe_count} color="#ef4444" /> : <span style={{ color: '#94a3b8' }}>0</span>}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {t.confidence != null ? (t.confidence * 100).toFixed(1) + '%' : 'N/A'}
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

  function renderDefinitions() {
    if (!definitions?.sections) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {definitions.sections.map((sec, si) => (
          <Card key={si} title={sec.title}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.definition}</div>
              </div>
            ))}
          </Card>
        ))}
      </div>
    )
  }
}
