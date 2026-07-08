import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const STATUS_COLORS = {
  paid: '#10b981', approved: '#3b82f6', submitted: '#06b6d4',
  denied: '#ef4444', 'partially_approved': '#f59e0b', appealed: '#8b5cf6', write_off: '#64748b'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function currency(v) {
  if (v == null) return '--'
  return '$' + Number(v).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(status || '').replace(/_/g, ' ')}</span>
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

export default function BillingClaimsDashboard() {
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
          axios.get(`${API_URL}/billing-claims/overview`),
          axios.get(`${API_URL}/billing-claims/breakdown`),
          axios.get(`${API_URL}/billing-claims/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load billing data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128176;</div>
      Loading billing & claims data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview || overview.available === false) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No billing data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'claims', label: 'Claims & Denials' },
    { id: 'aging', label: 'Aging & Trends' },
    { id: 'payers', label: 'Payer Analysis' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const kpi = overview.kpis || {}
  const collColor = kpi.collection_rate >= 85 ? '#10b981' : kpi.collection_rate >= 70 ? '#f59e0b' : '#ef4444'
  const denialColor = kpi.denial_rate <= 5 ? '#10b981' : kpi.denial_rate <= 10 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Billing & Claims Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(kpi.total_claims)} claims | {currency(kpi.total_billed)} billed | {fmt(kpi.unique_patients)} patients
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} kpi={kpi} collColor={collColor} denialColor={denialColor} />}
      {tab === 'claims' && <ClaimsTab breakdown={breakdown} />}
      {tab === 'aging' && <AgingTab overview={overview} breakdown={breakdown} />}
      {tab === 'payers' && <PayersTab overview={overview} breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview, kpi, collColor, denialColor }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Claims" value={kpi.total_claims} /></Card>
      <Card><KPI label="Collection Rate" value={`${fmt(kpi.collection_rate)}%`} color={collColor} sub="Target: >85%" /></Card>
      <Card><KPI label="Denial Rate" value={`${fmt(kpi.denial_rate)}%`} color={denialColor} sub="Target: <10%" /></Card>
      <Card><KPI label="Avg Days to Payment" value={fmt(kpi.avg_days_to_payment)} sub="days" /></Card>

      <Card><KPI label="Total Billed" value={currency(kpi.total_billed)} color="#3b82f6" /></Card>
      <Card><KPI label="Total Collected" value={currency(kpi.total_collected)} color="#10b981" /></Card>
      <Card><KPI label="Outstanding" value={currency(kpi.outstanding_balance)} color="#f59e0b" /></Card>
      <Card><KPI label="Unique Patients" value={kpi.unique_patients} /></Card>

      <Card title="Claim Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.status_distribution || []} dataKey="count" nameKey="status"
              cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${(status || '').replace(/_/g, ' ')} (${count})`}>
              {(overview.status_distribution || []).map((e, i) => (
                <Cell key={i} fill={STATUS_COLORS[e.status] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Top Services by Claims" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.top_services || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="service" width={160} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => currency(v)} />
            <Bar dataKey="claims" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Claims" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ClaimsTab({ breakdown }) {
  const denied = breakdown?.denied_claims || []
  const recent = breakdown?.recent_claims || []
  const cptData = breakdown?.cpt_analysis || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Denied Claims">
        {denied.length === 0 ? <div style={{ color: '#94a3b8', fontSize: 13 }}>No denied claims</div> : (
          <div style={{ maxHeight: 300, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Claim ID</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Patient</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Service</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Billed</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Denial Reason</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Submitted</th>
              </tr></thead>
              <tbody>{denied.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6, fontFamily: 'monospace', fontSize: 11 }}>{r.claim_id}</td>
                  <td style={{ padding: 6 }}>{r.patient_id}</td>
                  <td style={{ padding: 6 }}>{(r.service_type || '').replace(/_/g, ' ')}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{currency(r.amount_billed)}</td>
                  <td style={{ padding: 6, color: '#ef4444' }}>{r.denial_reason}</td>
                  <td style={{ padding: 6, fontSize: 11 }}>{r.submitted_at}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="CPT Code Analysis">
        {cptData.length === 0 ? <div style={{ color: '#94a3b8', fontSize: 13 }}>No CPT data</div> : (
          <div style={{ maxHeight: 300, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>CPT</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Description</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Claims</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Avg Billed</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Avg Paid</th>
              </tr></thead>
              <tbody>{cptData.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6, fontFamily: 'monospace' }}>{r.cpt_code}</td>
                  <td style={{ padding: 6 }}>{r.cpt_desc}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{r.claims}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{currency(r.avg_billed)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{currency(r.avg_paid)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="Recent Claims">
        <div style={{ maxHeight: 350, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Claim ID</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Patient</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Service</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Billed</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Status</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Submitted</th>
            </tr></thead>
            <tbody>{recent.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6, fontFamily: 'monospace', fontSize: 11 }}>{r.claim_id}</td>
                <td style={{ padding: 6 }}>{r.patient_id}</td>
                <td style={{ padding: 6 }}>{(r.service_type || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: 6, textAlign: 'right' }}>{currency(r.amount_billed)}</td>
                <td style={{ padding: 6 }}><StatusBadge status={r.status} /></td>
                <td style={{ padding: 6, fontSize: 11 }}>{r.submitted_at}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function AgingTab({ overview, breakdown }) {
  const aging = breakdown?.aging_buckets || []
  const monthly = overview?.monthly_trend || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Accounts Receivable Aging">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={aging}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="left" orientation="left" />
            <YAxis yAxisId="right" orientation="right" tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
            <Tooltip formatter={(v, name) => name === 'amount' ? currency(v) : v} />
            <Bar yAxisId="left" dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Claims" />
            <Bar yAxisId="right" dataKey="amount" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Amount" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Revenue Trend">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={monthly}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 10 }} />
            <YAxis tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
            <Tooltip formatter={(v) => currency(v)} />
            <Line type="monotone" dataKey="billed" stroke="#3b82f6" strokeWidth={2} name="Billed" dot={false} />
            <Line type="monotone" dataKey="collected" stroke="#10b981" strokeWidth={2} name="Collected" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Patient Balances (Top 20)">
        <div style={{ maxHeight: 300, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Patient</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Claims</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Total Billed</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Total Paid</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Balance</th>
            </tr></thead>
            <tbody>{(breakdown?.per_patient || []).map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}>{r.patient_id}</td>
                <td style={{ padding: 6, textAlign: 'right' }}>{r.claims}</td>
                <td style={{ padding: 6, textAlign: 'right' }}>{currency(r.total_billed)}</td>
                <td style={{ padding: 6, textAlign: 'right', color: '#10b981' }}>{currency(r.total_paid)}</td>
                <td style={{ padding: 6, textAlign: 'right', color: r.balance > 0 ? '#ef4444' : '#10b981' }}>{currency(r.balance)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function PayersTab({ overview, breakdown }) {
  const insurers = overview?.insurance_breakdown || []
  const payerMix = breakdown?.payer_mix || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Payer Mix" span={1}>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={payerMix} dataKey="pct" nameKey="insurer"
              cx="50%" cy="50%" outerRadius={80} label={({ insurer, pct }) => `${insurer} (${pct}%)`}>
              {payerMix.map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Insurance Performance" span={1}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={insurers} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tickFormatter={v => `$${(v/1000).toFixed(0)}k`} />
            <YAxis type="category" dataKey="insurer" width={130} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => currency(v)} />
            <Bar dataKey="billed" fill="#3b82f6" name="Billed" />
            <Bar dataKey="collected" fill="#10b981" name="Collected" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Insurer Detail" span={2}>
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead><tr style={{ borderBottom: '1px solid #e2e8f0' }}>
            <th style={{ textAlign: 'left', padding: 6 }}>Insurer</th>
            <th style={{ textAlign: 'right', padding: 6 }}>Claims</th>
            <th style={{ textAlign: 'right', padding: 6 }}>Total Billed</th>
            <th style={{ textAlign: 'right', padding: 6 }}>Total Collected</th>
            <th style={{ textAlign: 'right', padding: 6 }}>Collection %</th>
          </tr></thead>
          <tbody>{insurers.map((r, i) => {
            const collPct = r.billed > 0 ? ((r.collected / r.billed) * 100).toFixed(1) : 0
            return (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6, fontWeight: 600 }}>{r.insurer}</td>
                <td style={{ padding: 6, textAlign: 'right' }}>{r.claims}</td>
                <td style={{ padding: 6, textAlign: 'right' }}>{currency(r.billed)}</td>
                <td style={{ padding: 6, textAlign: 'right', color: '#10b981' }}>{currency(r.collected)}</td>
                <td style={{ padding: 6, textAlign: 'right', color: collPct >= 85 ? '#10b981' : '#f59e0b' }}>{collPct}%</td>
              </tr>
            )
          })}</tbody>
        </table>
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs) return null
  const sections = [
    { key: 'statuses', title: 'Claim Statuses' },
    { key: 'service_types', title: 'Service Types' }
  ]
  const notes = ['aging_note', 'icd10_note', 'cpt_note', 'collection_rate_formula']

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {sections.map(s => (
        <Card key={s.key} title={s.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>{Object.entries(defs[s.key] || {}).map(([k, v]) => (
              <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600, color: '#334155', whiteSpace: 'nowrap', verticalAlign: 'top', width: 200 }}>
                  {k.replace(/_/g, ' ')}
                </td>
                <td style={{ padding: 8, color: '#64748b' }}>{v}</td>
              </tr>
            ))}</tbody>
          </table>
        </Card>
      ))}
      <Card title="Notes">
        {notes.map(n => defs[n] ? (
          <p key={n} style={{ margin: '4px 0', fontSize: 13, color: '#64748b' }}>
            <strong>{n.replace(/_/g, ' ')}:</strong> {defs[n]}
          </p>
        ) : null)}
      </Card>
    </div>
  )
}
