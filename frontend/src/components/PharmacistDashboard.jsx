import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#ef4444', '#22c55e', '#f59e0b', '#7c4dff', '#ec4899', '#6366f1', '#14b8a6']
const fmt = v => (typeof v === 'number' ? v.toLocaleString() : v ?? '--')

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const badgeStyle = (color) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: color,
})

const sevColor = (sev) => {
  const s = (sev || '').toLowerCase()
  if (s === 'major' || s === 'contraindicated' || s === 'high' || s === 'low') return '#ef4444'
  if (s === 'moderate' || s === 'caution' || s === 'high_risk' || s === 'medium') return '#f59e0b'
  if (s === 'minor' || s === 'safe') return '#22c55e'
  return '#94a3b8'
}

const adherenceColor = (level) => {
  if (level === 'high') return '#22c55e'
  if (level === 'medium') return '#f59e0b'
  return '#ef4444'
}

const pregnancyColor = (level) => {
  const s = (level || '').toLowerCase()
  if (s === 'contraindicated') return '#ef4444'
  if (s === 'high_risk' || s === 'high risk') return '#f59e0b'
  if (s === 'caution') return '#7c4dff'
  return '#94a3b8'
}

const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }
const tdStyle = (i) => ({ padding: '8px 12px', borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' })

export default function PharmacistDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/clinical-pharmacist/overview`),
      axios.get(`${API_URL}/api/clinical-pharmacist/breakdown`),
      axios.get(`${API_URL}/api/clinical-pharmacist/definitions`),
    ]).then(([ov, bd, df]) => {
      setOverview(ov.data)
      setBreakdown(bd.data)
      setDefs(df.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Clinical Pharmacist Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40 }}>No data available</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'inventory', label: 'Medication Inventory' },
    { id: 'interactions', label: 'Drug Interactions' },
    { id: 'adherence', label: 'Adherence' },
    { id: 'adr', label: 'ADR Monitoring' },
    { id: 'pregnancy', label: 'Pregnancy Safety' },
    { id: 'tdm', label: 'TDM' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 8 }}>
        Clinical Pharmacist Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Medication reconciliation, drug interactions, TDM, ADR monitoring, adherence, and pregnancy safety
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#1e88e5' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Tab: Overview ── */}
      {tab === 'overview' && (
        <>
          {/* KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Total Patients" value={k.total_patients} color="#1e88e5" /></Card>
            <Card><KPI label="Medication Records" value={k.total_medication_records} color="#7c4dff" /></Card>
            <Card><KPI label="Drug Interactions" value={k.total_interactions} color="#ef4444" /></Card>
            <Card><KPI label="Major Interactions" value={k.major_interactions} color="#ef4444" /></Card>
            <Card><KPI label="ASM Catalog" value={k.asm_catalog_size} color="#14b8a6" /></Card>
            <Card><KPI label="Interaction KB" value={k.interaction_kb_size} color="#6366f1" /></Card>
            <Card><KPI label="Low Adherence" value={k.low_adherence_patients} color="#f59e0b" /></Card>
            <Card><KPI label="Avg MMAS-8" value={k.avg_mmas8} sub={`MPR: ${fmt(k.avg_mpr)}`} color="#22c55e" /></Card>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
            {/* Adherence Pie */}
            <Card title="Adherence Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={overview.adherence_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                    <Cell fill="#22c55e" />
                    <Cell fill="#f59e0b" />
                    <Cell fill="#ef4444" />
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Drug Class Bar */}
            <Card title="Drug Class Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={overview.drug_class_distribution || []} layout="vertical" margin={{ left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#1e88e5" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Interaction Severity */}
            <Card title="Interaction Severity Breakdown">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={overview.interaction_severity || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    <Cell fill="#ef4444" />
                    <Cell fill="#f59e0b" />
                    <Cell fill="#22c55e" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Pregnancy Risk */}
            <Card title="Pregnancy Risk Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={overview.pregnancy_risk_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                    {(overview.pregnancy_risk_distribution || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ── Tab: Medication Inventory ── */}
      {tab === 'inventory' && (
        <Card title={`Medication Inventory (${(breakdown?.medication_inventory || []).length} patients)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Drug</th>
                  <th style={thStyle}>Brand</th>
                  <th style={thStyle}>Class</th>
                  <th style={thStyle}>Dose (mg)</th>
                  <th style={thStyle}>Frequency</th>
                  <th style={thStyle}>Therapeutic Range</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.medication_inventory || []).flatMap((pat, pi) =>
                  (pat.medications || []).map((med, mi) => (
                    <tr key={`${pi}-${mi}`}>
                      <td style={{ ...tdStyle(pi), fontWeight: 600 }}>{mi === 0 ? pat.patient_id : ''}</td>
                      <td style={tdStyle(pi)}>{med.drug || '--'}</td>
                      <td style={tdStyle(pi)}>{med.brand || '--'}</td>
                      <td style={tdStyle(pi)}>{med.drug_class || '--'}</td>
                      <td style={tdStyle(pi)}>{fmt(med.dose_mg)}</td>
                      <td style={tdStyle(pi)}>{med.frequency || '--'}</td>
                      <td style={tdStyle(pi)}>
                        {Array.isArray(med.therapeutic_range)
                          ? `${med.therapeutic_range[0]}-${med.therapeutic_range[1]} mcg/mL`
                          : '--'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Tab: Drug Interactions ── */}
      {tab === 'interactions' && (
        <>
          {(breakdown?.interaction_details || []).map((pat, pi) => (
            <Card key={pi} title={`Patient ${pat.patient_id} (${pat.interactions?.length || 0} interactions)`}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
                Drugs: {(pat.drugs_checked || []).join(', ')}
              </div>

              {(pat.interactions || []).length > 0 ? (
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Drug A</th>
                      <th style={thStyle}>Drug B</th>
                      <th style={thStyle}>Severity</th>
                      <th style={thStyle}>Mechanism</th>
                      <th style={thStyle}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pat.interactions.map((ix, i) => (
                      <tr key={i}>
                        <td style={tdStyle(i)}>{ix.drug_a}</td>
                        <td style={tdStyle(i)}>{ix.drug_b}</td>
                        <td style={tdStyle(i)}>
                          <span style={badgeStyle(sevColor(ix.severity))}>{ix.severity}</span>
                        </td>
                        <td style={{ ...tdStyle(i), fontSize: 12, maxWidth: 300 }}>{ix.mechanism}</td>
                        <td style={{ ...tdStyle(i), fontSize: 12, maxWidth: 250 }}>{ix.recommended_action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={{ color: '#22c55e', fontWeight: 600, fontSize: 13 }}>No interactions detected</div>
              )}

              {Object.keys(pat.cyp_enzyme_overlaps || {}).length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>CYP Enzyme Overlaps</div>
                  {Object.entries(pat.cyp_enzyme_overlaps).map(([enzyme, drugs], i) => (
                    <div key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 2 }}>
                      <span style={{ fontWeight: 600, color: '#7c4dff' }}>{enzyme}:</span> {drugs.join(', ')}
                    </div>
                  ))}
                </div>
              )}
              <div style={{ marginBottom: 16 }} />
            </Card>
          ))}
        </>
      )}

      {/* ── Tab: Adherence ── */}
      {tab === 'adherence' && (
        <Card title="Per-Patient Adherence Assessment">
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Medications</th>
                  <th style={thStyle}>MPR</th>
                  <th style={thStyle}>MMAS-8</th>
                  <th style={thStyle}>Level</th>
                  <th style={thStyle}>Seizure Gap</th>
                  <th style={thStyle}>Coaching Notes</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.adherence_details || []).map((pat, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{pat.patient_id}</td>
                    <td style={{ ...tdStyle(i), fontSize: 12 }}>{(pat.medications || []).join(', ')}</td>
                    <td style={tdStyle(i)}>{fmt(pat.mpr_estimate)}</td>
                    <td style={tdStyle(i)}>{fmt(pat.mmas8_proxy_score)}</td>
                    <td style={tdStyle(i)}>
                      <span style={badgeStyle(adherenceColor(pat.mmas8_level))}>{pat.mmas8_level || '--'}</span>
                    </td>
                    <td style={tdStyle(i)}>
                      {pat.seizure_gap_flag
                        ? <span style={badgeStyle('#ef4444')}>Yes</span>
                        : <span style={{ color: '#22c55e', fontWeight: 600, fontSize: 12 }}>No</span>}
                    </td>
                    <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 300 }}>
                      {(pat.coaching_notes || []).length > 0
                        ? <ul style={{ margin: 0, paddingLeft: 14 }}>{pat.coaching_notes.map((n, ni) => <li key={ni}>{n}</li>)}</ul>
                        : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Tab: ADR Monitoring ── */}
      {tab === 'adr' && (
        <>
          {(breakdown?.adr_profiles || []).map((pat, pi) => (
            <Card key={pi} title={`Patient ${pat.patient_id} (${pat.total_unique_adrs} unique ADRs)`}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Drug</th>
                    <th style={thStyle}>Brand</th>
                    <th style={thStyle}>Known ADRs</th>
                  </tr>
                </thead>
                <tbody>
                  {(pat.medications || []).map((med, mi) => (
                    <tr key={mi}>
                      <td style={tdStyle(mi)}>{med.drug}</td>
                      <td style={tdStyle(mi)}>{med.brand}</td>
                      <td style={{ ...tdStyle(mi), fontSize: 12 }}>{(med.known_adrs || []).join(', ') || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {Object.keys(pat.overlapping_adrs || {}).length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#ef4444', marginBottom: 4 }}>
                    Overlapping ADRs (additive risk)
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {Object.entries(pat.overlapping_adrs).map(([adr, count], i) => (
                      <span key={i} style={badgeStyle('#ef4444')}>{adr} ({count}x)</span>
                    ))}
                  </div>
                </div>
              )}

              {(pat.high_risk_flags || []).length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#dc2626' }}>
                    High-risk flags: {pat.high_risk_flags.join(', ')}
                  </div>
                </div>
              )}
              <div style={{ marginBottom: 16 }} />
            </Card>
          ))}
        </>
      )}

      {/* ── Tab: Pregnancy Safety ── */}
      {tab === 'pregnancy' && (
        <Card title="Pregnancy Safety Flags">
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Drug</th>
                  <th style={thStyle}>Brand</th>
                  <th style={thStyle}>Category</th>
                  <th style={thStyle}>Risk Level</th>
                  <th style={thStyle}>Guidance</th>
                  <th style={thStyle}>Alerts</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.pregnancy_details || []).flatMap((pat, pi) =>
                  (pat.medications || []).map((med, mi) => (
                    <tr key={`${pi}-${mi}`}>
                      <td style={{ ...tdStyle(pi), fontWeight: 600 }}>{mi === 0 ? pat.patient_id : ''}</td>
                      <td style={tdStyle(pi)}>{med.drug}</td>
                      <td style={tdStyle(pi)}>{med.brand}</td>
                      <td style={tdStyle(pi)}>
                        <span style={{ fontWeight: 700, color: pregnancyColor(med.risk_level) }}>{med.pregnancy_category}</span>
                      </td>
                      <td style={tdStyle(pi)}>
                        <span style={badgeStyle(pregnancyColor(med.risk_level))}>{med.risk_level}</span>
                      </td>
                      <td style={{ ...tdStyle(pi), fontSize: 12, maxWidth: 300 }}>{med.guidance || '--'}</td>
                      {mi === 0 && (
                        <td style={tdStyle(pi)} rowSpan={pat.medications.length}>
                          {pat.urgent_alert && <div style={badgeStyle('#ef4444')}>Contraindicated drug!</div>}
                          {pat.folate_supplementation_needed && <div style={{ ...badgeStyle('#f59e0b'), marginTop: 4 }}>Folate needed</div>}
                          {!pat.urgent_alert && !pat.folate_supplementation_needed && <span style={{ color: '#22c55e', fontSize: 12 }}>Clear</span>}
                        </td>
                      )}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Tab: TDM ── */}
      {tab === 'tdm' && (
        <Card title="Therapeutic Drug Monitoring">
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Drug</th>
                  <th style={thStyle}>Brand</th>
                  <th style={thStyle}>Dose (mg)</th>
                  <th style={thStyle}>Frequency</th>
                  <th style={thStyle}>Therapeutic Range</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Recommendation</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.tdm_details || []).flatMap((pat, pi) =>
                  (pat.tdm || []).map((med, mi) => (
                    <tr key={`${pi}-${mi}`}>
                      <td style={{ ...tdStyle(pi), fontWeight: 600 }}>{mi === 0 ? pat.patient_id : ''}</td>
                      <td style={tdStyle(pi)}>{med.drug || '--'}</td>
                      <td style={tdStyle(pi)}>{med.brand || '--'}</td>
                      <td style={tdStyle(pi)}>{fmt(med.dose_mg)}</td>
                      <td style={tdStyle(pi)}>{med.frequency || '--'}</td>
                      <td style={tdStyle(pi)}>
                        {Array.isArray(med.therapeutic_range_mcg_ml)
                          ? `${med.therapeutic_range_mcg_ml[0]}-${med.therapeutic_range_mcg_ml[1]} mcg/mL`
                          : '--'}
                      </td>
                      <td style={tdStyle(pi)}>
                        <span style={badgeStyle(med.monitoring_status === 'range_available' ? '#22c55e' : '#f59e0b')}>
                          {med.monitoring_status === 'range_available' ? 'Monitored' : 'No Range'}
                        </span>
                      </td>
                      <td style={{ ...tdStyle(pi), fontSize: 11, maxWidth: 280 }}>{med.recommendation || '--'}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Tab: Definitions ── */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Pharmacology Concepts">
            {(defs.concepts || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{c.term}</div>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>{c.definition}</div>
              </div>
            ))}
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Quality Metrics">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Metric</th>
                  <th style={thStyle}>Description</th>
                  <th style={thStyle}>Target</th>
                </tr>
              </thead>
              <tbody>
                {(defs.quality_metrics || []).map((m, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{m.metric}</td>
                    <td style={tdStyle(i)}>{m.description}</td>
                    <td style={tdStyle(i)}><span style={badgeStyle('#22c55e')}>{m.target}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Compliance References">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Standard</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.compliance_references || []).map((r, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{r.standard}</td>
                    <td style={tdStyle(i)}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Remediation Strategies">
            {(defs.remediation_strategies || []).map((s, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.remediation_strategies.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{s.strategy}</div>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>{s.description}</div>
              </div>
            ))}
          </Card>
        </>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 32, paddingBottom: 24 }}>
        Source: pharmacist_module &middot; ASM catalog ({fmt(k.asm_catalog_size)}) &middot; Interaction KB ({fmt(k.interaction_kb_size)})
      </div>
    </div>
  )
}
