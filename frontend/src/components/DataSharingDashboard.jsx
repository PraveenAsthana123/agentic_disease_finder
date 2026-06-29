import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#6366f1', '#64748b', '#06b6d4', '#ec4899']
const TIER_COLORS = { full: '#22c55e', 'de-identified': '#f59e0b', aggregate: '#6366f1', none: '#ef4444' }

export default function DataSharingDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/data-manager/data-sharing`)
        setData(res.data)
      } catch (err) {
        setError(err.message || 'Failed to load data sharing report')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🔗</div>
      Running data sharing analysis on clinical database...
    </div>
  )
  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca',
      borderRadius: 8, color: '#991b1b' }}>Error: {error}</div>
  )
  if (!data?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a',
      borderRadius: 8, color: '#92400e' }}>No data available for sharing analysis.</div>
  )

  const summary = data.summary || {}
  const pii = data.pii_scan || {}
  const audit = data.audit_summary || {}
  const exportR = data.export_readiness || {}
  const policies = data.access_policy || []
  const duas = data.data_use_agreements || {}

  const card = { background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 16 }
  const kpiCard = { ...card, textAlign: 'center', flex: 1, minWidth: 140 }

  // PII pie
  const piiPie = [
    { name: 'PII Columns', value: pii.pii_columns_found || 0 },
    { name: 'Safe Columns', value: Math.max(0, (pii.total_columns || 0) - (pii.pii_columns_found || 0)) }
  ].filter(d => d.value > 0)

  // Export readiness bar
  const exportBar = Object.entries(exportR.tables || {}).map(([table, info]) => ({
    name: table.length > 14 ? table.slice(0, 12) + '..' : table,
    Records: info.row_count || 0,
    ready: info.export_ready ? 1 : 0
  })).filter(d => d.Records > 0)

  // Access anomalies
  const anomalies = audit.anomalies || []

  return (
    <div style={{ padding: 16 }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, color: '#1e293b' }}>
        🔗 Clinical Data Manager — Data Sharing
      </h2>
      <p style={{ margin: '0 0 16px', color: '#64748b', fontSize: 13 }}>
        Secure sharing readiness: PII scan, access policies, audit trail, export checks ({summary.total_tables} tables, {summary.total_records_shareable} records)
      </p>

      {/* KPI tiles */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#2563eb' }}>{summary.total_tables}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Total Tables</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#22c55e' }}>{summary.tables_with_data}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Tables with Data</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#ef4444' }}>{summary.pii_tables}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>PII Tables</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#16a34a' }}>{summary.export_ready_tables}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Export-Ready</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#6366f1' }}>{summary.total_records_shareable}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Total Records</div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        {/* PII scan pie */}
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>PII Column Distribution</h3>
          <p style={{ margin: '0 0 8px', fontSize: 12, color: '#64748b' }}>
            De-ID readiness: {pii.deidentification_readiness_pct}%
          </p>
          {piiPie.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={piiPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                  {piiPie.map((_, i) => <Cell key={i} fill={i === 0 ? '#ef4444' : '#22c55e'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No column data</p>}
          {(pii.columns || []).length > 0 && (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 8 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                  <th style={{ textAlign: 'left', padding: '4px 8px' }}>Column</th>
                  <th style={{ textAlign: 'center', padding: '4px 8px' }}>Has PII</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px' }}>Type</th>
                  <th style={{ textAlign: 'right', padding: '4px 8px' }}>Non-Null</th>
                </tr>
              </thead>
              <tbody>
                {pii.columns.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9',
                    background: c.has_pii ? '#fef2f2' : 'transparent' }}>
                    <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{c.column}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                      {c.has_pii ? '🔴' : '🟢'}
                    </td>
                    <td style={{ padding: '4px 8px', fontSize: 11 }}>{c.pii_type || '-'}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{c.non_null_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Export readiness bar */}
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Export Readiness by Table</h3>
          {exportBar.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={exportBar}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Records" fill="#6366f1">
                  {exportBar.map((d, i) => (
                    <Cell key={i} fill={d.ready ? '#22c55e' : '#f59e0b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No export data</p>}
          <p style={{ margin: '8px 0 0', fontSize: 11, color: '#64748b' }}>
            Green = export-ready (no PII redaction needed), Amber = needs de-identification first
          </p>
        </div>
      </div>

      {/* Access Policy Matrix */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Role-Based Access Policy Matrix</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Table</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Clinician</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Researcher</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>External</th>
              </tr>
            </thead>
            <tbody>
              {policies.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontWeight: 600 }}>{p.table}</td>
                  {['clinician', 'researcher', 'external'].map(role => {
                    const tier = p[role] || 'none'
                    return (
                      <td key={role} style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 4,
                          fontSize: 11, fontWeight: 600,
                          background: TIER_COLORS[tier] + '20', color: TIER_COLORS[tier]
                        }}>{tier}</span>
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Audit Summary */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Data Access Audit Summary</h3>
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 13, marginBottom: 12 }}>
          <div><strong>Total accesses:</strong> {audit.total_accesses}</div>
          <div><strong>Transaction log:</strong> {audit.transaction_log_entries}</div>
          <div><strong>Conversation log:</strong> {audit.conversation_log_entries}</div>
          <div><strong>Last access:</strong> {audit.most_recent_access || 'N/A'}</div>
        </div>
        {anomalies.length > 0 && (
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 6, padding: 10, marginBottom: 12 }}>
            <strong style={{ color: '#991b1b', fontSize: 12 }}>Access Anomalies ({anomalies.length})</strong>
            <ul style={{ margin: '4px 0 0', paddingLeft: 18, fontSize: 12, color: '#991b1b' }}>
              {anomalies.map((a, i) => (
                <li key={i}>{a.actor} / {a.component}: {a.count} accesses (threshold: {a.threshold})</li>
              ))}
            </ul>
          </div>
        )}
        {(audit.by_action || []).length > 0 && (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: '4px 8px' }}>Action Type</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {audit.by_action.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '4px 8px' }}>{a.action}</td>
                  <td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600 }}>{a.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Data Use Agreements */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Data Use Agreement Tiers</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
          {Object.entries(duas).map(([tier, info]) => (
            <div key={tier} style={{
              border: '1px solid #e5e7eb', borderRadius: 8, padding: 12,
              borderLeft: `4px solid ${TIER_COLORS[tier] || '#64748b'}`
            }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6, textTransform: 'capitalize' }}>{tier}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>{info.description}</div>
              <div style={{ fontSize: 11 }}>
                <strong>Consent:</strong> {info.consent_required}
              </div>
              <div style={{ fontSize: 11, marginTop: 4 }}>
                <strong>Restrictions:</strong>
                <ul style={{ margin: '2px 0 0', paddingLeft: 16 }}>
                  {(info.restrictions || []).map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Note */}
      {data.note && (
        <div style={{ ...card, background: '#f0fdf4', borderColor: '#bbf7d0' }}>
          <p style={{ margin: 0, fontSize: 12, color: '#166534' }}>{data.note}</p>
        </div>
      )}
    </div>
  )
}
