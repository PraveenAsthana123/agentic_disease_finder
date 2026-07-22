import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const STATUS_COLORS = { grounded: '#10b981', partial: '#f59e0b', ungrounded: '#ef4444', verified: '#10b981', unverified: '#ef4444', pending: '#f59e0b' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'Patient Grounding' },
  { id: 'claims', label: 'Claim Traces' },
  { id: 'verification', label: 'Verification Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function GroundingDashboard() {
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
      axios.get(`${API}/api/grounding/overview`),
      axios.get(`${API}/api/grounding/breakdown`),
      axios.get(`${API}/api/grounding/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Grounding data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const k = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Grounding Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Source verification and citation mapping for AI-generated clinical outputs
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab k={k} overview={overview} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'claims' && <ClaimsTab breakdown={breakdown} />}
      {tab === 'verification' && <VerificationTab breakdown={breakdown} overview={overview} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ k, overview }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI Row */}
      <Card><KPI label="Total Claims" value={k.total_claims} sub="AI-generated assertions" color="#3b82f6" /></Card>
      <Card><KPI label="Grounded Claims" value={k.grounded_claims} sub="backed by source data" color="#10b981" /></Card>
      <Card><KPI label="Grounding Rate" value={`${k.grounding_rate}%`} sub="claims with evidence" color={k.grounding_rate >= 80 ? '#10b981' : '#f59e0b'} /></Card>
      <Card><KPI label="Citation Coverage" value={`${k.citation_coverage}%`} sub="analyses with patient links" color="#8b5cf6" /></Card>
      <Card><KPI label="Source Types" value={k.source_types_used} sub="distinct data sources" color="#06b6d4" /></Card>
      <Card><KPI label="Verification Checks" value={k.verification_checks} sub="expert + clinical reviews" color="#f59e0b" /></Card>
      <Card><KPI label="Avg Confidence" value={k.avg_confidence} sub="analysis confidence" color="#ec4899" /></Card>
      <Card><KPI label="Patients Covered" value={k.patients_covered} sub="with grounding data" color="#3b82f6" /></Card>

      {/* Grounding by Source Bar */}
      <Card title="Grounding Rate by Source Type" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={overview.grounding_by_source || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="source" width={120} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Bar dataKey="rate" fill="#3b82f6" radius={[0, 4, 4, 0]}>
              {(overview.grounding_by_source || []).map((d, i) => (
                <Cell key={i} fill={d.rate >= 80 ? '#10b981' : d.rate >= 50 ? '#f59e0b' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Verification Summary Pie */}
      <Card title="Verification Status" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={[
                { name: 'Verified', value: overview.verification_summary?.verified || 0 },
                { name: 'Unverified', value: overview.verification_summary?.unverified || 0 },
                { name: 'Pending', value: overview.verification_summary?.pending || 0 },
              ].filter(d => d.value > 0)}
              dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80}
              label={({ name, value }) => `${name}: ${value}`}
            >
              <Cell fill="#10b981" />
              <Cell fill="#ef4444" />
              <Cell fill="#f59e0b" />
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Confidence Distribution */}
      <Card title="Analysis Confidence Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.confidence_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Citation Map Table */}
      <Card title="Citation Map — Patient Source Coverage" span={2}>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Claim Type</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Source</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Records</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Grounded</th>
              </tr>
            </thead>
            <tbody>
              {(overview.citation_map || []).slice(0, 50).map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{row.patient_id}</td>
                  <td style={{ padding: 8, color: '#64748b' }}>{row.claim_type}</td>
                  <td style={{ padding: 8 }}>{row.source_table}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{row.source_count}</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>
                    <Badge text={row.grounded ? 'Yes' : 'No'} color={row.grounded ? '#10b981' : '#ef4444'} />
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

function PatientsTab({ breakdown }) {
  const patients = breakdown?.per_patient_grounding || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Grounding Score Bar Chart */}
      <Card title="Per-Patient Grounding Score" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={patients.slice(0, 40)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} interval={0} angle={-45} textAnchor="end" height={60} />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="grounding_score" radius={[4, 4, 0, 0]}>
              {patients.slice(0, 40).map((d, i) => (
                <Cell key={i} fill={d.grounding_score >= 0.7 ? '#10b981' : d.grounding_score >= 0.4 ? '#f59e0b' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Patient Detail Table */}
      <Card title="Patient Grounding Detail" span={2}>
        <div style={{ maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Name</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Disease</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Sources</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Claims</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Citations</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Score</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: 8 }}>{p.name}</td>
                  <td style={{ padding: 8, color: '#64748b' }}>{p.disease}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{p.sources_present?.length || 0}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{p.claims}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{p.citations}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontWeight: 600, color: p.grounding_score >= 0.7 ? '#10b981' : p.grounding_score >= 0.4 ? '#f59e0b' : '#ef4444' }}>
                    {p.grounding_score}
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

function ClaimsTab({ breakdown }) {
  const traces = breakdown?.claim_traces || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="AI Claim Traces — Source Verification">
        <div style={{ maxHeight: 600, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Trace ID</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Preview</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Claimed</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Verified</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {traces.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{t.trace_id}</td>
                  <td style={{ padding: 8, color: '#64748b', maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.text_preview}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{t.claimed_sources}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{t.verified_sources}</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>
                    <Badge text={t.grounding_status} color={STATUS_COLORS[t.grounding_status] || '#64748b'} />
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

function VerificationTab({ breakdown, overview }) {
  const verLog = breakdown?.source_verification_log || []
  const expertVerif = breakdown?.expert_verification || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Expert Verification */}
      <Card title="Expert Verification Reviews" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Role</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Expert</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Finding</th>
              <th style={{ textAlign: 'center', padding: 8 }}>Agrees with AI</th>
            </tr>
          </thead>
          <tbody>
            {expertVerif.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{r.patient_id}</td>
                <td style={{ padding: 8 }}><Badge text={r.role} color={COLORS[i % COLORS.length]} /></td>
                <td style={{ padding: 8 }}>{r.expert}</td>
                <td style={{ padding: 8, color: '#64748b', maxWidth: 300 }}>{r.finding}</td>
                <td style={{ padding: 8, textAlign: 'center' }}>
                  <Badge text={r.agree_with_ai ? 'Yes' : 'No'} color={r.agree_with_ai ? '#10b981' : '#ef4444'} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Source Verification Log */}
      <Card title="Source Verification Event Log" span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Timestamp</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Component</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Action</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Actor</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Detail</th>
              </tr>
            </thead>
            <tbody>
              {verLog.map((ev, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{ev.ts_utc}</td>
                  <td style={{ padding: 8, fontWeight: 600 }}>{ev.patient_id || '—'}</td>
                  <td style={{ padding: 8 }}>{ev.component}</td>
                  <td style={{ padding: 8 }}><Badge text={ev.action} color={COLORS[i % COLORS.length]} /></td>
                  <td style={{ padding: 8, color: '#64748b' }}>{ev.actor}</td>
                  <td style={{ padding: 8, color: '#94a3b8', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ev.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {(definitions.sections || []).map((sec, si) => (
        <Card key={si} title={sec.title}>
          <div style={{ display: 'grid', gap: 12 }}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
