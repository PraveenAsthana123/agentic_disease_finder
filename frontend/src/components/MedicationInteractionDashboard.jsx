import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const SEVERITY_COLORS = {
  major: '#ef4444',
  moderate: '#f97316',
  minor: '#f59e0b',
  none: '#10b981',
}

const PSYCH_COLORS = {
  normal: '#10b981',
  mild: '#f59e0b',
  moderate: '#f97316',
  severe: '#ef4444',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'medications', label: 'Medication Inventory' },
  { id: 'interactions', label: 'Interaction Checks' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'adr', label: 'ADR Flags' },
  { id: 'psych', label: 'Psychiatric Cross-Ref' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function MedicationInteractionDashboard() {
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
      axios.get(`${API_URL}/api/medication-interaction/overview`),
      axios.get(`${API_URL}/api/medication-interaction/breakdown`),
      axios.get(`${API_URL}/api/medication-interaction/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Medication Interaction Checker...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No data'}</div>

  const k = overview.kpis || {}

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="KPI Summary" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 12 }}>
          <KPI label="Prescriptions" value={k.total_prescriptions} />
          <KPI label="Unique Drugs" value={k.unique_drugs} />
          <KPI label="Patients on Meds" value={k.patients_on_meds} />
          <KPI label="Interactions Found" value={k.interactions_detected} color={k.interactions_detected > 0 ? '#ef4444' : '#10b981'} />
          <KPI label="Major Interactions" value={k.major_interactions} color={k.major_interactions > 0 ? '#ef4444' : '#10b981'} />
          <KPI label="Patients at Risk" value={k.patients_at_risk} color={k.patients_at_risk > 0 ? '#f97316' : '#10b981'} />
          <KPI label="Polytherapy" value={k.polytherapy_patients} sub=">=2 AEDs" />
          <KPI label="ADR Flags" value={k.adr_flags} color={k.adr_flags > 0 ? '#f59e0b' : '#10b981'} />
        </div>
      </Card>

      <Card title="Drug Frequency" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.drug_frequency} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="drug" type="category" width={120} tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Polytherapy Distribution" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.polytherapy_distribution} dataKey="count" nameKey="tier"
              cx="50%" cy="50%" outerRadius={80} label={({ tier, count }) => `${tier}: ${count}`}>
              {(overview.polytherapy_distribution || []).map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Interaction Severity" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.severity_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {(overview.severity_distribution || []).map((entry, i) => (
                <Cell key={i} fill={entry.color || COLORS[i]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Prescription Activity" span={4}>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={overview.daily_activity}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )

  const renderMedications = () => {
    const inv = breakdown?.medication_inventory || []
    return (
      <Card title={`Medication Inventory (${inv.length} prescriptions)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Drug</th>
                <th style={{ padding: '8px 12px' }}>Dose (mg)</th>
                <th style={{ padding: '8px 12px' }}>Frequency</th>
                <th style={{ padding: '8px 12px' }}>ADR Risk</th>
                <th style={{ padding: '8px 12px' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {inv.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{m.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{m.drug_name}</td>
                  <td style={{ padding: '8px 12px' }}>{m.dose_mg}</td>
                  <td style={{ padding: '8px 12px' }}>{m.frequency}</td>
                  <td style={{ padding: '8px 12px' }}>
                    {m.has_adr_risk
                      ? <Badge text={m.adr_type} color="#f59e0b" />
                      : <span style={{ color: '#94a3b8' }}>—</span>}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{(m.created_at || '').slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )
  }

  const renderInteractions = () => {
    const ix = breakdown?.interaction_results || []
    const kb = breakdown?.interaction_knowledge_base || []
    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title={`Detected Interactions (${ix.length} found in patient data)`}>
          {ix.length === 0
            ? <div style={{ color: '#10b981', padding: 20, textAlign: 'center' }}>No interactions detected in current prescriptions</div>
            : <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 12px' }}>Patient</th>
                      <th style={{ padding: '8px 12px' }}>Drug A</th>
                      <th style={{ padding: '8px 12px' }}>Drug B</th>
                      <th style={{ padding: '8px 12px' }}>Severity</th>
                      <th style={{ padding: '8px 12px' }}>Mechanism</th>
                      <th style={{ padding: '8px 12px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ix.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: r.severity === 'major' ? '#fef2f2' : undefined }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.patient_id}</td>
                        <td style={{ padding: '8px 12px' }}>{r.drug_a}</td>
                        <td style={{ padding: '8px 12px' }}>{r.drug_b}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <Badge text={r.severity} color={SEVERITY_COLORS[r.severity] || '#64748b'} />
                        </td>
                        <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 300 }}>{r.mechanism}</td>
                        <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 250, color: '#1e40af' }}>{r.action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
          }
        </Card>

        <Card title="AED Interaction Knowledge Base (reference)">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Drug A</th>
                  <th style={{ padding: '8px 12px' }}>Drug B</th>
                  <th style={{ padding: '8px 12px' }}>Severity</th>
                  <th style={{ padding: '8px 12px' }}>Mechanism</th>
                  <th style={{ padding: '8px 12px' }}>Action</th>
                  <th style={{ padding: '8px 12px' }}>Reference</th>
                </tr>
              </thead>
              <tbody>
                {kb.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px' }}>{r.drug_a}</td>
                    <td style={{ padding: '8px 12px' }}>{r.drug_b}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <Badge text={r.severity} color={SEVERITY_COLORS[r.severity] || '#64748b'} />
                    </td>
                    <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 300 }}>{r.mechanism}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 250 }}>{r.action}</td>
                    <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{r.reference}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  const renderPatients = () => {
    const profiles = breakdown?.patient_profiles || []
    return (
      <Card title={`Patient Medication Profiles (${profiles.length})`}>
        <div style={{ display: 'grid', gap: 16 }}>
          {profiles.map((p, i) => (
            <div key={i} style={{
              border: '1px solid #e2e8f0', borderRadius: 10, padding: 16,
              background: p.worst_interaction === 'major' ? '#fef2f2' : '#fff'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <div>
                  <span style={{ fontWeight: 700, fontSize: 15, marginRight: 10 }}>{p.patient_id}</span>
                  <Badge text={p.therapy_tier} color={p.therapy_tier === 'monotherapy' ? '#10b981' : '#f59e0b'} />
                  {p.worst_interaction !== 'none' && (
                    <Badge text={`${p.worst_interaction} interaction`} color={SEVERITY_COLORS[p.worst_interaction]} />
                  )}
                </div>
                <div style={{ fontSize: 12, color: '#64748b' }}>
                  {p.interaction_count} interaction(s) | {p.seizure_entries} seizure entries
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                {p.dose_details.map((d, j) => (
                  <span key={j} style={{
                    display: 'inline-block', padding: '4px 10px', borderRadius: 6,
                    fontSize: 12, background: '#f1f5f9', color: '#334155'
                  }}>
                    {d.drug} {d.dose_mg}mg {d.frequency}
                  </span>
                ))}
              </div>
              {p.adr_risks.length > 0 && (
                <div style={{ fontSize: 12, color: '#f59e0b', marginTop: 4 }}>
                  ADR risks: {p.adr_risks.join(', ')}
                </div>
              )}
              {p.psych_assessments.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 4 }}>Psychiatric Assessments:</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {p.psych_assessments.map((a, j) => (
                      <Badge key={j} text={`${a.instrument} ${a.score}/${a.max_score} (${a.level})`}
                        color={PSYCH_COLORS[a.level] || '#64748b'} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
    )
  }

  const renderADR = () => {
    const adr = breakdown?.adr_detail || []
    return (
      <Card title={`Adverse Drug Reaction (ADR) Flags (${adr.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Drug</th>
                <th style={{ padding: '8px 12px' }}>Dose (mg)</th>
                <th style={{ padding: '8px 12px' }}>Risk Type</th>
                <th style={{ padding: '8px 12px' }}>Detail</th>
                <th style={{ padding: '8px 12px' }}>Frequency</th>
              </tr>
            </thead>
            <tbody>
              {adr.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{r.drug}</td>
                  <td style={{ padding: '8px 12px' }}>{r.dose_mg}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <Badge text={r.risk_type} color="#f59e0b" />
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 400 }}>{r.detail}</td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>{r.frequency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )
  }

  const renderPsych = () => {
    const alerts = breakdown?.psych_alerts || []
    return (
      <Card title={`Psychiatric Cross-Reference — Patients Needing AED-Psychotropic Review (${alerts.length} alerts)`}>
        <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
          Patients with moderate/severe PHQ-9, GAD-7, or C-SSRS scores may require psychotropic medications
          that interact with AEDs or lower seizure threshold.
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Instrument</th>
                <th style={{ padding: '8px 12px' }}>Score</th>
                <th style={{ padding: '8px 12px' }}>Level</th>
                <th style={{ padding: '8px 12px' }}>Alert</th>
                <th style={{ padding: '8px 12px' }}>On AEDs?</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((a, i) => (
                <tr key={i} style={{
                  borderBottom: '1px solid #f1f5f9',
                  background: a.alert ? '#fef2f2' : undefined
                }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{a.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{a.instrument}</td>
                  <td style={{ padding: '8px 12px' }}>{a.score}/{a.max_score}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <Badge text={a.level} color={PSYCH_COLORS[a.level] || '#64748b'} />
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12, color: '#ef4444' }}>{a.alert || '—'}</td>
                  <td style={{ padding: '8px 12px' }}>
                    {a.on_aeds ? <Badge text="Yes" color="#ef4444" /> : <span style={{ color: '#94a3b8' }}>No</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )
  }

  const renderPipeline = () => {
    const events = breakdown?.pipeline_events || []
    return (
      <Card title={`Pipeline Events (${events.length})`}>
        {events.length === 0
          ? <div style={{ color: '#64748b', textAlign: 'center', padding: 20 }}>No medication-related pipeline events yet</div>
          : <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '6px 10px' }}>Time</th>
                    <th style={{ padding: '6px 10px' }}>Component</th>
                    <th style={{ padding: '6px 10px' }}>Action</th>
                    <th style={{ padding: '6px 10px' }}>Actor</th>
                    <th style={{ padding: '6px 10px' }}>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{(e.ts_local || e.ts_utc || '').slice(0, 19)}</td>
                      <td style={{ padding: '6px 10px' }}>{e.component}</td>
                      <td style={{ padding: '6px 10px' }}>{e.action}</td>
                      <td style={{ padding: '6px 10px' }}>{e.actor}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 300, fontSize: 11 }}>{(e.detail || '').slice(0, 120)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
        }
      </Card>
    )
  }

  const renderDefinitions = () => {
    if (!definitions) return null
    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title="Medication Interaction Concepts">
          <div style={{ display: 'grid', gap: 10 }}>
            {(definitions.concepts || []).map((c, i) => (
              <div key={i} style={{ padding: '10px 14px', background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{c.term}</div>
                <div style={{ fontSize: 13, color: '#475569' }}>{c.definition}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Quality Metrics">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Metric</th>
                  <th style={{ padding: '8px 12px' }}>Target</th>
                  <th style={{ padding: '8px 12px' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(definitions.quality_metrics || []).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{m.metric}</td>
                    <td style={{ padding: '8px 12px' }}><Badge text={m.target} color="#3b82f6" /></td>
                    <td style={{ padding: '8px 12px', fontSize: 12 }}>{m.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="AED Drug Classes">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Class</th>
                  <th style={{ padding: '8px 12px' }}>Examples</th>
                </tr>
              </thead>
              <tbody>
                {(definitions.drug_classes || []).map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{c.class}</td>
                    <td style={{ padding: '8px 12px' }}>{c.examples}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Compliance References">
          <div style={{ display: 'grid', gap: 8 }}>
            {(definitions.compliance_references || []).map((r, i) => (
              <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 13 }}>
                <span style={{ fontWeight: 600, color: '#1e293b' }}>{r.ref}</span>
                <span style={{ color: '#64748b', marginLeft: 8 }}>— {r.note}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Remediation Strategies">
          <div style={{ display: 'grid', gap: 8 }}>
            {(definitions.remediation_strategies || []).map((r, i) => (
              <div key={i} style={{ padding: '10px 14px', background: '#fffbeb', borderRadius: 8, fontSize: 13 }}>
                <div style={{ fontWeight: 600, color: '#92400e', marginBottom: 4 }}>{r.issue}</div>
                <div style={{ color: '#78350f' }}>{r.remedy}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Medication Interaction Checker</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          AED polytherapy risk screening, drug-drug interaction detection, psychiatric comorbidity cross-reference
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'medications' && renderMedications()}
      {tab === 'interactions' && renderInteractions()}
      {tab === 'patients' && renderPatients()}
      {tab === 'adr' && renderADR()}
      {tab === 'psych' && renderPsych()}
      {tab === 'pipeline' && renderPipeline()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
