import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316', '#14b8a6', '#a855f7']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function DrugBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s.includes('resistant') ? '#ef4444' : s.includes('responsive') ? '#10b981' : s.includes('partial') ? '#f59e0b' : '#64748b'
  const label = s.includes('resistant') ? 'RESISTANT' : s.includes('responsive') ? 'RESPONSIVE' : s.includes('partial') ? 'PARTIAL' : 'PENDING'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{label}</span>
  )
}

function SurgeryBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s.includes('strong') ? '#10b981' : s.includes('possible') ? '#3b82f6' : s.includes('no') ? '#ef4444' : s.includes('further') ? '#f59e0b' : '#64748b'
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

export default function SeizureMetadataDashboard() {
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
      axios.get(`${API_URL}/seizure-metadata/overview`),
      axios.get(`${API_URL}/seizure-metadata/breakdown`),
      axios.get(`${API_URL}/seizure-metadata/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading seizure metadata...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Seizure metadata not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Patient Detail' },
    { id: 'surgery', label: 'Surgery Candidates' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = ov.kpis || {}

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
          <Card title="ILAE Classification Summary" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, padding: '8px 0' }}>
              <KPI label="Total Patients" value={k.total_patients} color="#1e293b" />
              <KPI label="Focal Epilepsy" value={k.focal_epilepsy} sub={`${k.focal_pct}%`} color="#3b82f6" />
              <KPI label="Generalized" value={k.generalized_epilepsy} sub={`${(100 - (k.focal_pct || 0)).toFixed(1)}%`} color="#8b5cf6" />
              <KPI label="Drug-Resistant" value={k.drug_resistant} sub={`${k.drug_resistant_pct}%`} color="#ef4444" />
              <KPI label="Surgery Candidates" value={k.surgery_candidates} color="#f59e0b" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, padding: '8px 0', marginTop: 8 }}>
              <KPI label="Avg Age at Onset" value={k.avg_age_at_onset} sub="years" color="#06b6d4" />
              <KPI label="Seizure-Free" value={k.seizure_free} sub=">12 months" color="#10b981" />
              <KPI label="Onset Zones" value={(ov.onset_zone_distribution || []).length} color="#64748b" />
              <KPI label="Seizure Types" value={(ov.seizure_type_frequency || []).length} color="#64748b" />
            </div>
          </Card>

          <Card title="Onset Zone Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={ov.onset_zone_distribution || []} layout="vertical" margin={{ left: 150 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="zone" width={140} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Focal vs Generalized">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={[
                  { name: 'Focal', value: k.focal_epilepsy || 0 },
                  { name: 'Generalized', value: k.generalized_epilepsy || 0 },
                ]} cx="50%" cy="50%" outerRadius={90} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  <Cell fill="#3b82f6" />
                  <Cell fill="#8b5cf6" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="ILAE Seizure Type Frequency" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={ov.seizure_type_frequency || []} layout="vertical" margin={{ left: 180 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="type" width={170} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Age at Onset">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={ov.age_at_onset_histogram || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Etiology Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={ov.etiology_distribution || []} layout="vertical" margin={{ left: 200 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="etiology" width={190} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Drug Responsiveness">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={ov.drug_responsiveness_distribution || []} cx="50%" cy="50%" outerRadius={90} dataKey="count" nameKey="status" label={({ status, percent }) => `${(status || '').split('—')[0].trim()} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.drug_responsiveness_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Syndrome Distribution" span={2}>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={ov.syndrome_distribution || []} layout="vertical" margin={{ left: 250 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="syndrome" width={240} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Seizure Frequency">
            <ResponsiveContainer width="100%" height={320}>
              <PieChart>
                <Pie data={ov.frequency_distribution || []} cx="50%" cy="50%" outerRadius={90} dataKey="count" nameKey="frequency" label={({ frequency, percent }) => `${(frequency || '').split('(')[0].trim()} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.frequency_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="EEG Pattern Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={ov.eeg_distribution || []} layout="vertical" margin={{ left: 200 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="pattern" width={190} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#06b6d4" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="MRI Findings">
            <div style={{ overflowY: 'auto', maxHeight: 300 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Finding</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.mri_distribution || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px' }}>{r.finding}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{r.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'breakdown' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Semiology Feature Frequency" span={1}>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={bd.semiology_frequency || []} layout="vertical" margin={{ left: 220 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={210} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#ec4899" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Patient Classification">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient', 'Onset Zone', 'Seizure Types', 'Syndrome', 'Etiology', 'Drug Response', 'Surgery', 'Frequency', 'Age Onset'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.per_patient || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{r.onset_zone}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.seizure_types}</td>
                      <td style={{ padding: '6px 10px' }}>{r.syndrome}</td>
                      <td style={{ padding: '6px 10px' }}>{r.etiology}</td>
                      <td style={{ padding: '6px 10px' }}><DrugBadge status={r.drug_responsiveness} /></td>
                      <td style={{ padding: '6px 10px' }}><SurgeryBadge status={r.surgery_candidacy} /></td>
                      <td style={{ padding: '6px 10px' }}>{r.seizure_frequency}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.age_at_onset}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Drug-Resistant Patients" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#fef2f2' }}>
                    {['Patient', 'Syndrome', 'Etiology', 'Seizure Types', 'Frequency', 'Surgery'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #fecaca', color: '#991b1b', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.drug_resistant || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{r.syndrome}</td>
                      <td style={{ padding: '6px 10px' }}>{r.etiology}</td>
                      <td style={{ padding: '6px 10px' }}>{r.seizure_types}</td>
                      <td style={{ padding: '6px 10px' }}>{r.seizure_frequency}</td>
                      <td style={{ padding: '6px 10px' }}><SurgeryBadge status={r.surgery_candidacy} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Onset Zone x Etiology Cross-Tab">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Onset Zone</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Etiology</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.zone_etiology_crosstab || []).slice(0, 25).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px' }}>{r.onset_zone}</td>
                      <td style={{ padding: '6px 10px' }}>{r.etiology}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{r.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'surgery' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Surgery Candidates (${(bd.surgery_candidates || []).length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#fffbeb' }}>
                    {['Patient', 'Onset Zone', 'Lateralization', 'MRI Finding', 'EEG Pattern', 'Drug Response', 'Candidacy'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #fde68a', color: '#92400e', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.surgery_candidates || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{r.onset_zone}</td>
                      <td style={{ padding: '6px 10px' }}>{r.lateralization}</td>
                      <td style={{ padding: '6px 10px' }}>{r.mri_finding}</td>
                      <td style={{ padding: '6px 10px' }}>{r.eeg_pattern}</td>
                      <td style={{ padding: '6px 10px' }}><DrugBadge status={r.drug_responsiveness} /></td>
                      <td style={{ padding: '6px 10px' }}><SurgeryBadge status={r.surgery_candidacy} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="ILAE Seizure Type Definitions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 220 }}>Seizure Type</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(defs.seizure_type_definitions || {}).map(([k, v], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{k}</td>
                    <td style={{ padding: '6px 10px', lineHeight: 1.5 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Onset Zone Descriptions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 220 }}>Zone</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(defs.onset_zone_descriptions || {}).map(([k, v], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{k}</td>
                    <td style={{ padding: '6px 10px', lineHeight: 1.5 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Etiology Categories">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 180 }}>Category</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(defs.etiology_categories || {}).map(([k, v], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{k}</td>
                    <td style={{ padding: '6px 10px', lineHeight: 1.5 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 8, color: '#334155' }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 180 }}>Term</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(defs.glossary || {}).map(([k, v], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{k}</td>
                    <td style={{ padding: '6px 10px', lineHeight: 1.5 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}
