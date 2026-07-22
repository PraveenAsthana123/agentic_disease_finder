import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function RiskBadge({ significance }) {
  const s = String(significance || '')
  const color = s.includes('High') ? '#ef4444' : s.includes('Moderate') ? '#f59e0b' : '#10b981'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{s || '--'}</span>
  )
}

function EvidenceBadge({ level }) {
  const l = String(level || '')
  const color = l === '1A' ? '#10b981' : l === '1B' ? '#3b82f6' : l === '2A' ? '#f59e0b' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{l || '--'}</span>
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

export default function PharmacogenomicsDashboard() {
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
          axios.get(`${API_URL}/api/pharmacogenomics/overview`),
          axios.get(`${API_URL}/api/pharmacogenomics/breakdown`),
          axios.get(`${API_URL}/api/pharmacogenomics/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load pharmacogenomics data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading pharmacogenomics data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Pharmacogenomics data not available</div>

  const tabs = ['overview', 'alerts', 'patients', 'definitions']
  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Pharmacogenomics Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Pharmacogenomic testing and drug-gene interaction analytics — {fmt(kpis.total_tests)} tests across {fmt(kpis.total_patients)} patients
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Total Tests" value={kpis.total_tests} />
              <KPI label="Total Patients" value={kpis.total_patients} />
              <KPI label="Unique Genes" value={kpis.unique_genes} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="High Risk" value={kpis.high_risk_count} color="#ef4444" />
              <KPI label="Actionable Results" value={kpis.actionable_results} color="#10b981" />
              <KPI label="Poor Metabolizers" value={kpis.poor_metabolizer_count} color="#f59e0b" />
            </div>
          </Card>

          <Card title="Gene Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.gene_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="gene" type="category" tick={{ fontSize: 11 }} width={140} />
                <Tooltip />
                <Bar dataKey="count" name="Tests" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                  {(overview.gene_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Metabolizer Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.metabolizer_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metabolizer_status" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {(overview.metabolizer_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Evidence Level Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={overview.evidence_level_distribution || []} dataKey="count" nameKey="evidence_level"
                  cx="50%" cy="50%" outerRadius={100} label={({ evidence_level, count }) => `${evidence_level}: ${count}`}>
                  {(overview.evidence_level_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="High Risk by Gene" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.high_risk_by_gene || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gene" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="High Risk" fill="#ef4444" radius={[4, 4, 0, 0]}>
                  {(overview.high_risk_by_gene || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── Alerts tab ── */}
      {tab === 'alerts' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="High Risk Results">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#fef2f2' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Gene</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Variant</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Clinical Significance</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Affected Drugs</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Recommendation</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#991b1b' }}>Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.high_risk_results || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{row.gene || '--'}</td>
                      <td style={{ padding: '8px 12px' }}>{row.variant || '--'}</td>
                      <td style={{ padding: '8px 12px' }}>{row.clinical_significance || '--'}</td>
                      <td style={{ padding: '8px 12px', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.affected_drugs || '--'}</td>
                      <td style={{ padding: '8px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, color: '#64748b' }}>{row.recommendation || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><EvidenceBadge level={row.evidence_level} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Poor Metabolizers">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#fffbeb' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Gene</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Variant</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Metabolizer Status</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Affected Drugs</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Recommendation</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.poor_metabolizers || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{row.gene || '--'}</td>
                      <td style={{ padding: '8px 12px' }}>{row.variant || '--'}</td>
                      <td style={{ padding: '8px 12px' }}>{row.metabolizer_status || '--'}</td>
                      <td style={{ padding: '8px 12px', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.affected_drugs || '--'}</td>
                      <td style={{ padding: '8px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 12, color: '#64748b' }}>{row.recommendation || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Patients tab ── */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Tests</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Genes Tested</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>High Risk</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Actionable</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Sources</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.tests)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.genes_tested)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ flex: 1, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.min(((row.high_risk || 0) / Math.max(row.tests || 1, 1)) * 100, 100)}%`, height: '100%',
                              background: (row.high_risk || 0) > 0 ? '#ef4444' : '#10b981',
                              borderRadius: 4
                            }} />
                          </div>
                          <span style={{ fontSize: 12, color: (row.high_risk || 0) > 0 ? '#ef4444' : '#64748b', minWidth: 24 }}>{fmt(row.high_risk)}</span>
                        </div>
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#10b981' }}>{fmt(row.actionable)}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.sources || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Recent Tests">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Gene</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Variant</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Metabolizer Status</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Clinical Significance</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Evidence</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Test Date</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_tests || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{row.gene || '--'}</td>
                      <td style={{ padding: '8px 12px' }}>{row.variant || '--'}</td>
                      <td style={{ padding: '8px 12px' }}>{row.metabolizer_status || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><RiskBadge significance={row.clinical_significance} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><EvidenceBadge level={row.evidence_level} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.test_date || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {defs.gene_descriptions && (
            <Card title="Gene Descriptions">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.gene_descriptions).map(([key, desc]) => (
                  <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {defs.metabolizer_categories && (
            <Card title="Metabolizer Categories">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.metabolizer_categories).map(([key, desc]) => (
                  <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {defs.evidence_levels && (
            <Card title="Evidence Levels">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.evidence_levels).map(([key, desc]) => (
                  <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {defs.clinical_notes && (
            <Card title="Clinical Notes">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {(Array.isArray(defs.clinical_notes) ? defs.clinical_notes : Object.values(defs.clinical_notes)).map((note, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 8, lineHeight: 1.5 }}>{note}</li>
                ))}
              </ul>
            </Card>
          )}

          {defs.glossary && (
            <Card title="Glossary">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.glossary).map(([term, definition]) => (
                  <div key={term} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>{term}</div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
