import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f97316', '#22c55e', '#ef4444', '#eab308', '#ec4899']

const BIOMARKER_COLORS = {
  High: '#ef4444', Elevated: '#f97316', Present: '#eab308',
  Normal: '#22c55e', Moderate: '#3b82f6', Absent: '#94a3b8', Reduced: '#06b6d4'
}

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

function BiomarkerBadge({ status }) {
  const color = BIOMARKER_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status || '--'}</span>
  )
}

export default function NeurologistWorkbenchDashboard() {
  const [data, setData] = useState({ overview: null, breakdown: null, definitions: null })
  const [tab, setTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/neurologist-workbench/overview`),
      axios.get(`${API_URL}/api/neurologist-workbench/breakdown`),
      axios.get(`${API_URL}/api/neurologist-workbench/definitions`)
    ])
      .then(([ov, bd, df]) => {
        setData({ overview: ov.data, breakdown: bd.data, definitions: df.data })
        setLoading(false)
      })
      .catch(err => { setError(err.message || 'Failed to load neurologist workbench data'); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading neurologist workbench data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!data.overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No data available</div>

  const tabs = ['overview', 'breakdown', 'definitions']
  const ov = data.overview
  const bd = data.breakdown
  const df = data.definitions

  const ps = ov.patient_summary || {}
  const ai = ov.ai_findings || {}
  const expl = ov.explainability || []
  const bio = ov.biomarkers || []
  const loc = ov.localization || []
  const mri = ov.mri_correlation || []
  const meds = ov.medications || []
  const audit = ov.audit || {}

  // Breakdown data
  const bdPs = bd ? (bd.patient_summary || {}) : {}
  const bdExpl = bd ? (bd.explainability || []) : []
  const bdBio = bd ? (bd.biomarkers || []) : []
  const bdLoc = bd ? (bd.localization || []) : []
  const bdMri = bd ? (bd.mri_correlation || []) : []
  const bdMeds = bd ? (bd.medications || []) : []
  const bdAudit = bd ? (bd.audit || {}) : {}

  // Definitions data
  const dfPs = df ? (df.patient_summary || {}) : {}
  const dfExpl = df ? (df.explainability || []) : []
  const dfBio = df ? (df.biomarkers || []) : []
  const dfLoc = df ? (df.localization || []) : []

  const thStyle = { padding: '8px 12px', textAlign: 'left', fontSize: 12, fontWeight: 600, color: '#475569', background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }
  const tdStyle = { padding: '8px 12px', fontSize: 13, color: '#334155', borderBottom: '1px solid #e2e8f0' }

  return (
    <div style={{ padding: '24px 20px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', margin: '0 0 4px' }}>Neurologist Workbench</h2>
      <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 20px' }}>
        Clinical decision-support dashboard — patient profile, AI findings, biomarkers, localization, and audit trail
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#f1f5f9', color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI row */}
          <Card title="Patient Summary" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
              <KPI label="Age" value={fmt(ps.age)} />
              <KPI label="Gender" value={ps.gender || '--'} />
              <KPI label="Diagnosis" value={ps.diagnosis || '--'} />
              <KPI label="Duration (yrs)" value={fmt(ps.duration_years)} />
              <KPI label="Seizure Freq" value={ps.seizure_frequency || '--'} color="#ef4444" />
              <KPI label="Last Seizure" value={ps.last_seizure_days != null ? `${ps.last_seizure_days}d ago` : '--'} />
            </div>
            {ps.current_medication && (
              <div style={{ marginTop: 10, fontSize: 13, color: '#64748b' }}>
                Current Medication: <strong style={{ color: '#334155' }}>{ps.current_medication}</strong>
                {ps.demo && <span style={{ marginLeft: 8, padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600, background: '#fef3c7', color: '#92400e' }}>DEMO</span>}
              </div>
            )}
          </Card>

          {/* AI Findings */}
          <Card title="AI Findings">
            {ai.available ? (
              <div>
                <div style={{ fontSize: 13, marginBottom: 6 }}>Predicted: <strong>{ai.predicted}</strong></div>
                <div style={{ fontSize: 13, marginBottom: 6 }}>Confidence: <strong>{fmt(ai.confidence)}%</strong></div>
                <div style={{ fontSize: 13 }}>Signal Quality: <strong>{ai.signal_quality}</strong></div>
              </div>
            ) : (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>
                AI analysis not yet available for this patient
              </div>
            )}
          </Card>

          {/* Explainability — horizontal bar chart */}
          <Card title="Feature Importance (XAI)" span={2}>
            {expl.length > 0 ? (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={expl} layout="vertical" margin={{ left: 80, right: 20, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} width={80} />
                  <Tooltip formatter={v => `${v}%`} />
                  <Bar dataKey="pct" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No explainability data</div>}
          </Card>

          {/* Biomarkers table */}
          <Card title="EEG Biomarkers" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Marker</th>
                  <th style={thStyle}>Status</th>
                </tr>
              </thead>
              <tbody>
                {bio.map((b, i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{b.marker}</td>
                    <td style={tdStyle}><BiomarkerBadge status={b.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Localization — horizontal bar */}
          <Card title="Seizure Localization">
            {loc.length > 0 ? (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={loc} layout="vertical" margin={{ left: 60, right: 20, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="region" tick={{ fontSize: 12 }} width={60} />
                  <Tooltip formatter={v => `${v}%`} />
                  <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                    {loc.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No localization data</div>}
          </Card>

          {/* MRI Correlation */}
          <Card title="MRI Correlation">
            {mri.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Finding</th>
                    <th style={thStyle}>Match</th>
                  </tr>
                </thead>
                <tbody>
                  {mri.map((m, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{m.fields_json}</td>
                      <td style={tdStyle}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                          background: m.match === 'Match' ? '#dcfce7' : '#fef2f2',
                          color: m.match === 'Match' ? '#166534' : '#991b1b'
                        }}>{m.match}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No MRI data</div>}
          </Card>

          {/* Medications */}
          <Card title="Current Medications">
            {meds.length > 0 ? (
              <ul style={{ margin: 0, padding: '0 0 0 16px' }}>
                {meds.map((m, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#334155', marginBottom: 4 }}>{m.fields_json}</li>
                ))}
              </ul>
            ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No medications listed</div>}
          </Card>

          {/* Audit trail */}
          <Card title="Audit Trail" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              <div style={{ fontSize: 13 }}>Model Version: <strong>{audit.model_version || '--'}</strong></div>
              <div style={{ fontSize: 13 }}>Training Dataset: <strong>{audit.training_dataset || '--'}</strong></div>
              <div style={{ fontSize: 13 }}>Date: <strong>{audit.date || '--'}</strong></div>
              <div style={{ fontSize: 13 }}>Reviewer: <strong>{audit.reviewer || '--'}</strong></div>
            </div>
          </Card>

          {ov.note && (
            <Card span={3}>
              <div style={{ fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>{ov.note}</div>
            </Card>
          )}
        </div>
      )}

      {tab === 'breakdown' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* Breakdown patient summary */}
          <Card title="Patient Profile (Breakdown View)" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
              <KPI label="Age" value={fmt(bdPs.age)} />
              <KPI label="Gender" value={bdPs.gender || '--'} />
              <KPI label="Diagnosis" value={bdPs.diagnosis || '--'} />
              <KPI label="Duration (yrs)" value={fmt(bdPs.duration_years)} />
              <KPI label="Seizure Freq" value={bdPs.seizure_frequency || '--'} color="#ef4444" />
              <KPI label="Last Seizure" value={bdPs.last_seizure_days != null ? `${bdPs.last_seizure_days}d ago` : '--'} />
            </div>
            {bdPs.current_medication && (
              <div style={{ marginTop: 10, fontSize: 13, color: '#64748b' }}>
                Medication: <strong style={{ color: '#334155' }}>{bdPs.current_medication}</strong>
                {bdPs.demo && <span style={{ marginLeft: 8, padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600, background: '#fef3c7', color: '#92400e' }}>DEMO</span>}
              </div>
            )}
          </Card>

          {/* Breakdown explainability */}
          <Card title="Feature Importance (Breakdown)" span={2}>
            {bdExpl.length > 0 ? (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={bdExpl} layout="vertical" margin={{ left: 80, right: 20, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} width={80} />
                  <Tooltip formatter={v => `${v}%`} />
                  <Bar dataKey="pct" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No data</div>}
          </Card>

          {/* Breakdown localization */}
          <Card title="Localization (Breakdown)">
            {bdLoc.length > 0 ? (
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={bdLoc} layout="vertical" margin={{ left: 60, right: 20, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="region" tick={{ fontSize: 12 }} width={60} />
                  <Tooltip formatter={v => `${v}%`} />
                  <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                    {bdLoc.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No data</div>}
          </Card>

          {/* Breakdown biomarkers */}
          <Card title="EEG Biomarkers (Breakdown)" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Marker</th>
                  <th style={thStyle}>Status</th>
                </tr>
              </thead>
              <tbody>
                {bdBio.map((b, i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{b.marker}</td>
                    <td style={tdStyle}><BiomarkerBadge status={b.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Breakdown MRI + Meds */}
          <Card title="MRI & Medications">
            <h4 style={{ fontSize: 13, color: '#475569', margin: '0 0 8px' }}>MRI Correlation</h4>
            {bdMri.length > 0 ? bdMri.map((m, i) => (
              <div key={i} style={{ fontSize: 13, marginBottom: 4 }}>
                {m.fields_json} — <span style={{
                  padding: '1px 6px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                  background: m.match === 'Match' ? '#dcfce7' : '#fef2f2',
                  color: m.match === 'Match' ? '#166534' : '#991b1b'
                }}>{m.match}</span>
              </div>
            )) : <div style={{ fontSize: 12, color: '#94a3b8' }}>No MRI data</div>}
            <h4 style={{ fontSize: 13, color: '#475569', margin: '16px 0 8px' }}>Medications</h4>
            {bdMeds.length > 0 ? bdMeds.map((m, i) => (
              <div key={i} style={{ fontSize: 13, color: '#334155', marginBottom: 4 }}>{m.fields_json}</div>
            )) : <div style={{ fontSize: 12, color: '#94a3b8' }}>No medications</div>}
          </Card>

          {/* Breakdown audit */}
          <Card title="Audit Trail (Breakdown)" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              <div style={{ fontSize: 13 }}>Model: <strong>{bdAudit.model_version || '--'}</strong></div>
              <div style={{ fontSize: 13 }}>Dataset: <strong>{bdAudit.training_dataset || '--'}</strong></div>
              <div style={{ fontSize: 13 }}>Date: <strong>{bdAudit.date || '--'}</strong></div>
              <div style={{ fontSize: 13 }}>Reviewer: <strong>{bdAudit.reviewer || '--'}</strong></div>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && df && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Patient summary card */}
          <Card title="Sample Patient Profile" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
              <KPI label="Age" value={fmt(dfPs.age)} />
              <KPI label="Gender" value={dfPs.gender || '--'} />
              <KPI label="Diagnosis" value={dfPs.diagnosis || '--'} />
              <KPI label="Duration (yrs)" value={fmt(dfPs.duration_years)} />
              <KPI label="Seizure Freq" value={dfPs.seizure_frequency || '--'} />
              <KPI label="Last Seizure" value={dfPs.last_seizure_days != null ? `${dfPs.last_seizure_days}d ago` : '--'} />
            </div>
          </Card>

          {/* Biomarker definitions */}
          <Card title="Biomarker Status Definitions">
            <div style={{ display: 'grid', gap: 8 }}>
              {Object.entries(BIOMARKER_COLORS).map(([status, color]) => (
                <div key={status} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <BiomarkerBadge status={status} />
                  <span style={{ fontSize: 12, color: '#64748b' }}>
                    {status === 'High' && '— Above normal range, clinical attention needed'}
                    {status === 'Elevated' && '— Slightly above normal, monitor closely'}
                    {status === 'Present' && '— Detected in recording'}
                    {status === 'Normal' && '— Within expected range'}
                    {status === 'Moderate' && '— Intermediate level'}
                    {status === 'Absent' && '— Not detected in recording'}
                    {status === 'Reduced' && '— Below expected range'}
                  </span>
                </div>
              ))}
            </div>
          </Card>

          {/* Feature importance definitions */}
          <Card title="Explainability Features">
            <div style={{ display: 'grid', gap: 8 }}>
              {dfExpl.map((f, i) => (
                <div key={i} style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: 8 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{f.feature}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>Contribution: {f.pct}%</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Brain region definitions */}
          <Card title="Brain Region Localization">
            <div style={{ display: 'grid', gap: 8 }}>
              {dfLoc.map((l, i) => (
                <div key={i} style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: 8 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{l.region} Lobe</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>
                    Probability: {l.prob}%
                    <div style={{ marginTop: 4, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ width: `${l.prob}%`, height: '100%', background: COLORS[i % COLORS.length], borderRadius: 3 }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Clinical glossary */}
          <Card title="Clinical Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
              {[
                { term: 'EEG', def: 'Electroencephalogram — records electrical brain activity via scalp electrodes' },
                { term: 'HFO', def: 'High-Frequency Oscillation — fast ripples (250-500 Hz) associated with epileptogenic zones' },
                { term: 'XAI', def: 'Explainable AI — methods that make model predictions interpretable to clinicians' },
                { term: 'Ictal', def: 'During a seizure event' },
                { term: 'Interictal', def: 'Between seizure events — baseline state' },
                { term: 'Spike', def: 'Transient sharp EEG deflection (20-70ms) indicating abnormal neuronal discharge' },
                { term: 'Sharp Wave', def: 'Similar to spike but longer duration (70-200ms)' },
                { term: 'Theta Band', def: '4-8 Hz frequency range, elevated in temporal lobe epilepsy' },
                { term: 'Delta Band', def: '0.5-4 Hz frequency range, associated with slow-wave abnormalities' },
                { term: 'Beta Band', def: '13-30 Hz frequency range, related to cortical arousal and medication effects' }
              ].map((g, i) => (
                <div key={i} style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: 8 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{g.term}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{g.def}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Clinical notes */}
          <Card title="Clinical Notes" span={2}>
            <div style={{ display: 'grid', gap: 8 }}>
              {[
                'The neurologist workbench integrates EEG AI predictions with clinical context for decision support.',
                'Biomarker status badges reflect automated EEG feature extraction; always verify against raw signal.',
                'Localization probabilities are model-derived — correlate with semiology and imaging before surgical planning.',
                'MRI correlation shows concordance between structural imaging and EEG-based localization.',
                'Audit trail captures model version, training data provenance, and reviewer sign-off for regulatory compliance.'
              ].map((n, i) => (
                <div key={i} style={{ background: '#f8fafc', padding: '8px 12px', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
                  {n}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
