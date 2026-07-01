import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, ScatterChart, Scatter, ZAxis
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

const DRIFT_COLORS = {
  high: '#ef4444', moderate: '#f59e0b', low: '#10b981'
}

const CAT_COLORS = {
  'Time-Domain': '#3b82f6', 'Spectral': '#8b5cf6', 'Complexity': '#f59e0b',
  'Hjorth': '#ec4899', 'Connectivity': '#06b6d4'
}

export default function DriftDetectionDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [categoryFilter, setCategoryFilter] = useState('All')

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API}/drift-detection/overview`),
      axios.get(`${API}/drift-detection/breakdown`),
      axios.get(`${API}/drift-detection/definitions`),
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
    { id: 'features', label: 'Feature Analysis' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'categories', label: 'Category Breakdown' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Drift Detection dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No drift detection data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>EEG Drift Detection</h2>
        <Badge
          text={overview?.verdict || 'Unknown'}
          color={overview?.verdict?.includes('SEVERE') ? '#ef4444' : overview?.verdict?.includes('MODERATE') ? '#f59e0b' : '#10b981'}
        />
        <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>Generated: {overview?.run_at}</span>
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
      {tab === 'features' && renderFeatures()}
      {tab === 'patients' && renderPatients()}
      {tab === 'categories' && renderCategories()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const kpiCards = overview?.kpi_cards || []
    const sevDist = overview?.severity_distribution || {}
    const sevData = Object.entries(sevDist).map(([k, v]) => ({ severity: k, count: v })).filter(d => d.count > 0)
    const topFeatures = (overview?.top_drifted_features || []).slice(0, 12)
    const psiDist = overview?.psi_distribution || []
    const timeline = overview?.drift_timeline || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpiCards.map((kp, i) => (
          <Card key={i}>
            <KPI label={kp.label} value={kp.value} sub={kp.detail} color={
              kp.label?.toLowerCase().includes('high') ? '#ef4444' :
              kp.label?.toLowerCase().includes('drift') ? '#f59e0b' : '#3b82f6'
            } />
          </Card>
        ))}

        <Card title="Severity Distribution" span={2}>
          {sevData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={sevData} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={90}
                  label={({ severity, count }) => `${severity}: ${count}`}>
                  {sevData.map((s, i) => <Cell key={i} fill={DRIFT_COLORS[s.severity] || '#64748b'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No severity data available.</p>
          )}
        </Card>

        <Card title="Top 12 Drifted Features (PSI)" span={2}>
          {topFeatures.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(250, topFeatures.length * 30)}>
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={110} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(2) : v} />
                <Bar dataKey="psi" name="PSI" radius={[0, 4, 4, 0]}>
                  {topFeatures.map((f, i) => <Cell key={i} fill={DRIFT_COLORS[f.severity] || '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No drifted features detected.</p>
          )}
        </Card>

        <Card title="PSI Distribution" span={2}>
          {psiDist.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={psiDist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Features" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Drift Timeline" span={2}>
          {timeline.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={timeline} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(2) : v} />
                <Line type="monotone" dataKey="frac_drifted" name="Fraction Drifted" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444' }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        {overview?.interpretation && (
          <Card title="Interpretation" span={4}>
            <p style={{ fontSize: 14, color: '#475569', lineHeight: 1.6, margin: 0 }}>{overview.interpretation}</p>
          </Card>
        )}
      </div>
    )
  }

  function renderFeatures() {
    const allFeatures = breakdown?.per_feature_drift || []
    const categories = ['All', ...new Set(allFeatures.map(f => f.category).filter(Boolean))]
    const filtered = categoryFilter === 'All' ? allFeatures : allFeatures.filter(f => f.category === categoryFilter)
    const sorted = [...filtered].sort((a, b) => (b.psi || 0) - (a.psi || 0))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Filter by Category">
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {categories.map(cat => (
              <button key={cat} onClick={() => setCategoryFilter(cat)} style={{
                padding: '4px 12px', borderRadius: 6, border: 'none', cursor: 'pointer',
                background: categoryFilter === cat ? '#1e293b' : '#f1f5f9',
                color: categoryFilter === cat ? '#fff' : '#475569', fontSize: 12, fontWeight: 500
              }}>{cat}</button>
            ))}
          </div>
        </Card>

        <Card title={`Feature Drift Analysis (${sorted.length} features)`}>
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Rank</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Category</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>PSI</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>KS Stat</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>KS p-value</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Severity</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: f.severity === 'high' ? '#fef2f218' : undefined }}>
                    <td style={{ padding: '8px 10px', textAlign: 'center', color: '#94a3b8' }}>{f.rank || i + 1}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{f.feature}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={f.category} color={CAT_COLORS[f.category] || '#64748b'} />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', fontWeight: 600 }}>
                      {typeof f.psi === 'number' ? f.psi.toFixed(2) : f.psi}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof f.ks_stat === 'number' ? f.ks_stat.toFixed(4) : f.ks_stat}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof f.ks_p === 'number' ? f.ks_p.toFixed(4) : f.ks_p}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <Badge text={f.severity} color={DRIFT_COLORS[f.severity] || '#64748b'} />
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

  function renderPatients() {
    const patients = breakdown?.per_patient_profiles || []
    const confVsDrift = breakdown?.confidence_vs_drift || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Confidence with Drift Overlay (${patients.length} patients)`}>
          {patients.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={patients.slice(0, 25)} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis yAxisId="left" tick={{ fontSize: 11 }} label={{ value: 'Confidence', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} label={{ value: 'High Drift Features', angle: 90, position: 'insideRight', style: { fontSize: 11 } }} />
                <Tooltip />
                <Bar yAxisId="left" dataKey="confidence" name="Confidence" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar yAxisId="right" dataKey="n_high_drift" name="High Drift" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Patient Drift Detail">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Name</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Disease</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Confidence</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>High Drift</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Moderate Drift</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '10px 12px' }}>{p.name || 'N/A'}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={p.disease || 'unknown'} color="#8b5cf6" />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {p.confidence != null ? (p.confidence * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      {p.n_high_drift > 0 ? <Badge text={p.n_high_drift} color="#ef4444" /> : <span style={{ color: '#94a3b8' }}>0</span>}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      {p.n_moderate_drift > 0 ? <Badge text={p.n_moderate_drift} color="#f59e0b" /> : <span style={{ color: '#94a3b8' }}>0</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Confidence vs Drifted Features">
          {confVsDrift.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="confidence" name="Confidence" tick={{ fontSize: 11 }}
                  label={{ value: 'Confidence', position: 'insideBottom', offset: -5, style: { fontSize: 11 } }} />
                <YAxis type="number" dataKey="n_drifted" name="Drifted Features" tick={{ fontSize: 11 }}
                  label={{ value: 'Drifted Features', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                <ZAxis range={[40, 200]} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(2) : v}
                  labelFormatter={() => ''} />
                <Scatter name="Patients" data={confVsDrift} fill="#8b5cf6" />
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No confidence vs drift data available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderCategories() {
    const catDrift = overview?.category_drift || []
    const perCat = breakdown?.per_category_summary || []
    const correlations = (breakdown?.feature_correlations || []).slice(0, 20)

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Average PSI by Category" span={2}>
          {catDrift.length > 0 && (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={catDrift} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(2) : v} />
                <Bar dataKey="avg_psi" name="Avg PSI" radius={[4, 4, 0, 0]}>
                  {catDrift.map((c, i) => <Cell key={i} fill={CAT_COLORS[c.category] || '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        {perCat.map((cat, ci) => (
          <Card key={ci} title={cat.category}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 12 }}>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Total Features</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#1e293b' }}>{cat.features?.length || 0}</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Max PSI</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#ef4444' }}>{typeof cat.max_psi === 'number' ? cat.max_psi.toFixed(2) : cat.max_psi}</div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <Badge text={`${cat.n_high || 0} high`} color="#ef4444" />
              <Badge text={`${cat.n_moderate || 0} moderate`} color="#f59e0b" />
              <Badge text={`${cat.n_low || 0} low`} color="#10b981" />
            </div>
            {cat.features?.length > 0 && (
              <div style={{ marginTop: 8, fontSize: 11, color: '#64748b' }}>
                Top: {cat.features.slice(0, 5).join(', ')}
                {cat.features.length > 5 && ` +${cat.features.length - 5} more`}
              </div>
            )}
          </Card>
        ))}

        <Card title="Feature Correlations (Top 20)" span={2}>
          {correlations.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature A</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature B</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Correlation</th>
                  </tr>
                </thead>
                <tbody>
                  {correlations.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.feature_a}</td>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.feature_b}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>
                        <span style={{ color: (c.correlation || 0) > 0.8 ? '#ef4444' : (c.correlation || 0) > 0.5 ? '#f59e0b' : '#10b981' }}>
                          {typeof c.correlation === 'number' ? c.correlation.toFixed(4) : c.correlation}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No feature correlation data available.</p>
          )}
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
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.description}</div>
              </div>
            ))}
          </Card>
        ))}
      </div>
    )
  }
}
