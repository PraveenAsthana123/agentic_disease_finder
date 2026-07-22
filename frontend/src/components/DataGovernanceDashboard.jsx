import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#6366f1', '#64748b', '#06b6d4', '#ec4899', '#8b5cf6']
const STATUS_COLORS = { active: '#22c55e', partial: '#f59e0b', tracked: '#6366f1', missing: '#ef4444' }

export default function DataGovernanceDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/api/data-manager/governance`)
        setData(res.data)
      } catch (err) {
        setError(err.message || 'Failed to load governance report')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🏛️</div>
      Running data governance analysis on clinical database...
    </div>
  )
  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca',
      borderRadius: 8, color: '#991b1b' }}>Error: {error}</div>
  )
  if (!data?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a',
      borderRadius: 8, color: '#92400e' }}>No governance data available.</div>
  )

  const summary = data.summary || {}
  const consent = data.consent_status || {}
  const irb = data.irb_tracking || {}
  const deid = data.deidentification || {}
  const encryption = data.encryption_status || {}
  const audit = data.access_audit || {}

  const card = { background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 16 }
  const kpiCard = { ...card, textAlign: 'center', flex: 1, minWidth: 130 }
  const tabStyle = (t) => ({
    padding: '8px 16px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
    borderBottom: tab === t ? '2px solid #2563eb' : '2px solid transparent',
    color: tab === t ? '#2563eb' : '#64748b', fontSize: 13, background: 'none', border: 'none'
  })

  // Governance score radar
  const radarData = [
    { metric: 'Consent', value: summary.consent_rate_pct || 0 },
    { metric: 'De-ID', value: summary.deidentification_score || 0 },
    { metric: 'Oversight', value: summary.oversight_ratio_pct || 0 },
    { metric: 'Audit', value: Math.min(summary.audit_coverage_pct || 0, 100) }
  ]

  // Consent pie
  const consentPie = [
    { name: 'Documented', value: consent.documented || 0 },
    { name: 'Incomplete', value: consent.incomplete_records || 0 }
  ].filter(d => d.value > 0)

  // Access by action bar
  const actionBar = Object.entries(audit.by_action || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([action, count]) => ({ name: action, count }))

  // Access by component bar
  const componentBar = Object.entries(audit.by_component || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([comp, count]) => ({ name: comp.length > 16 ? comp.slice(0, 14) + '..' : comp, count }))

  // PII column data
  const piiCols = deid.patients_columns || []

  // IRB checklist
  const irbChecklist = irb.irb_checklist || {}

  // Encryption inventory (top 12 by size)
  const inventory = (encryption.data_at_rest_inventory || [])
    .filter(t => t.rows > 0)
    .sort((a, b) => b.rows - a.rows)
    .slice(0, 12)

  const scoreColor = (s) => s >= 70 ? '#22c55e' : s >= 40 ? '#f59e0b' : '#ef4444'

  const renderOverview = () => (
    <>
      {/* KPI tiles */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: scoreColor(summary.overall_governance_score) }}>
            {summary.overall_governance_score}%
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Governance Score</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: scoreColor(summary.consent_rate_pct) }}>
            {summary.consent_rate_pct}%
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Consent Rate</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: scoreColor(summary.deidentification_score) }}>
            {summary.deidentification_score}%
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>De-ID Score</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#6366f1' }}>{summary.total_patients}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Total Patients</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: summary.anomalies_detected > 0 ? '#ef4444' : '#22c55e' }}>
            {summary.anomalies_detected}
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Anomalies</div>
        </div>
      </div>

      {/* Radar + Consent pie */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Governance Score Breakdown</h3>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Radar name="Score" dataKey="value" stroke="#2563eb" fill="#2563eb" fillOpacity={0.25} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Consent Status</h3>
          <p style={{ margin: '0 0 4px', fontSize: 12, color: '#64748b' }}>
            {consent.documented}/{consent.total_patients} patients fully documented
          </p>
          {consentPie.length > 0 ? (
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={consentPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={65} label={({ name, value }) => `${name}: ${value}`}>
                  {consentPie.map((_, i) => <Cell key={i} fill={i === 0 ? '#22c55e' : '#ef4444'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No consent data</p>}
          {consent.note && (
            <p style={{ margin: '8px 0 0', fontSize: 11, color: '#94a3b8', fontStyle: 'italic' }}>
              {consent.note}
            </p>
          )}
        </div>
      </div>

      {/* IRB checklist */}
      <div style={card}>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>IRB Protocol Checklist</h3>
        <p style={{ margin: '0 0 8px', fontSize: 12, color: '#64748b' }}>
          {irb.distinct_diseases_studied} disease(s) studied: {(irb.disease_list || []).join(', ')}
          {' '}| {irb.expert_reviews} expert reviews | {irb.hitl_reviews} HITL reviews
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 8 }}>
          {Object.entries(irbChecklist).map(([key, val]) => (
            <div key={key} style={{
              padding: '8px 12px', borderRadius: 6,
              border: `1px solid ${STATUS_COLORS[val.status] || '#e5e7eb'}`,
              background: val.status === 'active' ? '#f0fdf4' : val.status === 'partial' ? '#fffbeb' : '#f5f3ff'
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: STATUS_COLORS[val.status] || '#64748b', textTransform: 'capitalize' }}>
                {val.status === 'active' ? '✅' : val.status === 'partial' ? '⚠️' : '📋'} {key.replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{val.evidence}</div>
            </div>
          ))}
        </div>
        {irb.note && (
          <p style={{ margin: '8px 0 0', fontSize: 11, color: '#94a3b8', fontStyle: 'italic' }}>{irb.note}</p>
        )}
      </div>
    </>
  )

  const renderDeidentification = () => (
    <>
      <div style={card}>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>
          De-Identification Audit — Score: {deid.deidentification_score}%
        </h3>
        <p style={{ margin: '0 0 8px', fontSize: 12, color: '#64748b' }}>
          {deid.safe_columns}/{deid.total_columns} columns are PII-free |
          Direct identifiers: {(deid.direct_identifiers || []).join(', ') || 'none'} |
          Quasi identifiers: {(deid.quasi_identifiers || []).join(', ') || 'none'}
        </p>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e5e7eb', background: '#f8fafc' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px' }}>Column</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>PII?</th>
              <th style={{ textAlign: 'left', padding: '6px 8px' }}>Risk Flags</th>
              <th style={{ textAlign: 'right', padding: '6px 8px' }}>Non-Null</th>
            </tr>
          </thead>
          <tbody>
            {piiCols.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9',
                background: c.pii_exposed ? '#fef2f2' : 'transparent' }}>
                <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{c.column}</td>
                <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                  {c.pii_exposed ? '🔴' : '🟢'}
                </td>
                <td style={{ padding: '4px 8px', fontSize: 11 }}>
                  {(c.risk_flags || []).map(f => (
                    <span key={f} style={{
                      display: 'inline-block', padding: '1px 6px', borderRadius: 4, fontSize: 10,
                      background: f === 'direct_identifier' ? '#fee2e2' : f === 'quasi_identifier' ? '#fef3c7' : '#ede9fe',
                      color: f === 'direct_identifier' ? '#991b1b' : f === 'quasi_identifier' ? '#92400e' : '#5b21b6',
                      marginRight: 4
                    }}>{f.replace(/_/g, ' ')}</span>
                  ))}
                </td>
                <td style={{ padding: '4px 8px', textAlign: 'right' }}>{c.non_null}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {deid.recommendation && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: '#eff6ff',
            border: '1px solid #bfdbfe', borderRadius: 6, fontSize: 12, color: '#1e40af' }}>
            💡 {deid.recommendation}
          </div>
        )}
      </div>

      <div style={card}>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>
          PII-Sensitive Tables ({(deid.pii_sensitive_tables || []).length})
        </h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {(deid.pii_sensitive_tables || []).map(t => (
            <span key={t} style={{
              padding: '4px 10px', borderRadius: 4, fontSize: 11, fontFamily: 'monospace',
              background: '#fef2f2', color: '#991b1b', border: '1px solid #fecaca'
            }}>{t}</span>
          ))}
        </div>
      </div>

      {/* Encryption status */}
      <div style={card}>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Encryption & Storage Posture</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 12 }}>
          <div style={{ padding: 10, background: '#f8fafc', borderRadius: 6 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 4 }}>Database</div>
            <div style={{ fontSize: 11, color: '#64748b' }}>
              Size: {encryption.database?.size_kb} KB |
              Permissions: {encryption.database?.permissions} |
              Encrypted: {encryption.database?.encrypted_at_rest ? '✅ Yes' : '❌ No'}
            </div>
            {encryption.database?.world_readable && (
              <div style={{ fontSize: 11, color: '#ef4444', marginTop: 2 }}>
                ⚠️ World-readable ({encryption.database?.permissions})
              </div>
            )}
            {encryption.database?.encryption_note && (
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2, fontStyle: 'italic' }}>
                {encryption.database?.encryption_note}
              </div>
            )}
          </div>
          <div style={{ padding: 10, background: '#f8fafc', borderRadius: 6 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 4 }}>Transport Security</div>
            <div style={{ fontSize: 11, color: '#64748b' }}>
              API: {encryption.transport_security?.api_endpoint} |
              TLS Required: {encryption.transport_security?.tls_required ? '✅ Yes' : '❌ No'}
            </div>
            {encryption.transport_security?.tls_note && (
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2, fontStyle: 'italic' }}>
                {encryption.transport_security?.tls_note}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )

  const renderAudit = () => (
    <>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#2563eb' }}>{audit.total_access_events}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Access Events</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#6366f1' }}>{audit.distinct_patients_accessed}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Patients Accessed</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 20, fontWeight: 600, color: '#64748b', fontFamily: 'monospace' }}>
            {audit.most_recent_access ? new Date(audit.most_recent_access).toLocaleDateString() : '-'}
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Last Access</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 16 }}>
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Access by Action Type</h3>
          {actionBar.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={actionBar} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={90} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1">
                  {actionBar.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No action data</p>}
        </div>

        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Access by Component</h3>
          {componentBar.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={componentBar} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={100} />
                <Tooltip />
                <Bar dataKey="count" fill="#2563eb">
                  {componentBar.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No component data</p>}
        </div>
      </div>

      {/* Anomalies */}
      {(audit.anomalies || []).length > 0 && (
        <div style={{ ...card, border: '1px solid #fecaca', background: '#fef2f2' }}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#991b1b' }}>
            ⚠️ Access Anomalies ({audit.anomalies.length})
          </h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #fecaca' }}>
                <th style={{ textAlign: 'left', padding: '4px 8px' }}>Actor</th>
                <th style={{ textAlign: 'left', padding: '4px 8px' }}>Component</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>Count</th>
                <th style={{ textAlign: 'left', padding: '4px 8px' }}>Severity</th>
              </tr>
            </thead>
            <tbody>
              {audit.anomalies.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #fecaca' }}>
                  <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{a.actor}</td>
                  <td style={{ padding: '4px 8px' }}>{a.component}</td>
                  <td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600 }}>{a.count}</td>
                  <td style={{ padding: '4px 8px' }}>
                    <span style={{
                      padding: '1px 6px', borderRadius: 4, fontSize: 10,
                      background: a.severity === 'high' ? '#fee2e2' : '#fef3c7',
                      color: a.severity === 'high' ? '#991b1b' : '#92400e'
                    }}>{a.severity || 'medium'}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Data-at-rest inventory */}
      <div style={card}>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>
          Data-at-Rest Inventory (top {inventory.length} tables by size)
        </h3>
        {inventory.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={inventory.map(t => ({
              name: t.table.length > 16 ? t.table.slice(0, 14) + '..' : t.table,
              Rows: t.rows, pii: t.pii_risk
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="Rows" fill="#6366f1">
                {inventory.map((t, i) => <Cell key={i} fill={t.pii_risk ? '#ef4444' : '#22c55e'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No inventory data</p>}
        <p style={{ margin: '8px 0 0', fontSize: 11, color: '#64748b' }}>
          Red = PII-risk table, Green = no PII risk
        </p>
      </div>
    </>
  )

  const renderDefinitions = () => (
    <div style={card}>
      <h3 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Governance Definitions</h3>
      {[
        { term: 'Consent Rate', def: 'Percentage of patients with complete demographic documentation (age, gender, disease, department all non-null). A proxy for informed-consent readiness.' },
        { term: 'De-Identification Score', def: 'Percentage of patient-table columns free of personally identifiable information. 100% = fully de-identified; 0% = all columns carry PII risk.' },
        { term: 'Direct Identifier', def: 'A column (e.g. name) that uniquely identifies an individual without combining with other data.' },
        { term: 'Quasi Identifier', def: 'A column (e.g. patient_id) that can identify individuals when combined with external datasets.' },
        { term: 'IRB Checklist', def: 'Evidence-based tracking of Institutional Review Board requirements: data collection, expert oversight, informed consent, adverse-event monitoring, and decision auditing.' },
        { term: 'Access Anomaly', def: 'An actor-component pair exceeding 50 transaction_log events, indicating unusually high access frequency.' },
        { term: 'Oversight Ratio', def: 'Ratio of oversight events (expert reviews + HITL reviews + clinical decisions) to total patients × 100.' },
        { term: 'Audit Coverage', def: 'Ratio of access events to total patients × 100; above 100% means high audit density.' },
        { term: 'Governance Score', def: 'Average of consent rate, de-identification score, oversight ratio, and audit coverage (capped at 100 each).' }
      ].map((d, i) => (
        <div key={i} style={{ marginBottom: 10 }}>
          <div style={{ fontWeight: 600, fontSize: 12, color: '#1e293b' }}>{d.term}</div>
          <div style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>{d.def}</div>
        </div>
      ))}
    </div>
  )

  return (
    <div style={{ padding: 16 }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, color: '#1e293b' }}>
        🏛️ Clinical Data Manager — Governance
      </h2>
      <p style={{ margin: '0 0 12px', color: '#64748b', fontSize: 13 }}>
        Consent tracking, IRB protocol, de-identification, encryption, access audit
        ({consent.total_patients} patients, {audit.total_access_events} events)
      </p>

      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #e5e7eb', marginBottom: 16 }}>
        {[
          ['overview', 'Overview'],
          ['deid', 'De-ID & Encryption'],
          ['audit', 'Access Audit'],
          ['definitions', 'Definitions']
        ].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={tabStyle(id)}>{label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'deid' && renderDeidentification()}
      {tab === 'audit' && renderAudit()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
