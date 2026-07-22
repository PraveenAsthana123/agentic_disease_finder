import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const REFILL_STATUS_COLORS = {
  on_time: '#10b981',
  gap: '#ef4444'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function RefillStatusBadge({ status }) {
  const color = REFILL_STATUS_COLORS[status] || '#64748b'
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

export default function MedicationRefillDashboard() {
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
          axios.get(`${API_URL}/api/medication-refills/overview`),
          axios.get(`${API_URL}/api/medication-refills/breakdown`),
          axios.get(`${API_URL}/api/medication-refills/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load medication refill data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128138;</div>
      Loading medication refill data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview || overview.available === false) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No medication refill data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'drugs', label: 'Drug Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'gaps', label: 'Gap Analysis' }
  ]

  const kpi = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Medication Refill Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(kpi.total_refills)} refills | {fmt(kpi.total_drugs)} drugs | {fmt(kpi.total_patients)} patients
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

      {tab === 'overview' && <OverviewTab overview={overview} kpi={kpi} />}
      {tab === 'drugs' && <DrugAnalysisTab breakdown={breakdown} />}
      {tab === 'patients' && <PatientDetailTab breakdown={breakdown} />}
      {tab === 'gaps' && <GapAnalysisTab breakdown={breakdown} defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview, kpi }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Refills" value={kpi.total_refills} /></Card>
      <Card><KPI label="Total Patients" value={kpi.total_patients} /></Card>
      <Card><KPI label="Total Drugs" value={kpi.total_drugs} /></Card>
      <Card><KPI label="Avg Quantity" value={fmt(kpi.avg_quantity)} /></Card>

      <Card><KPI label="Avg Days Supply" value={fmt(kpi.avg_days_supply)} sub="days" /></Card>
      <Card><KPI label="Auto-Refill %" value={`${fmt(kpi.auto_refill_pct)}%`} color="#10b981" /></Card>
      <Card><KPI label="Total Pharmacies" value={kpi.total_pharmacies} /></Card>

      <Card title="Drug Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.drug_distribution || []} dataKey="count" nameKey="drug"
              cx="50%" cy="50%" outerRadius={80} label={({ drug, count }) => `${drug} (${count})`}>
              {(overview.drug_distribution || []).map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Pharmacy Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.pharmacy_distribution || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="pharmacy" width={160} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Refills" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Refill Trend" span={4}>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={overview.monthly_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="refills" stroke="#3b82f6" strokeWidth={2} name="Refills" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DrugAnalysisTab({ breakdown }) {
  const drugs = breakdown?.drug_details || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Refills per Drug">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={drugs}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="drug_name" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="total_refills" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Refills" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Drug Details">
        {drugs.length === 0 ? <div style={{ color: '#94a3b8', fontSize: 13 }}>No drug data</div> : (
          <div style={{ maxHeight: 350, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Drug Name</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Total Refills</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Unique Patients</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Avg Quantity</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Avg Days Supply</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Auto-Refill %</th>
              </tr></thead>
              <tbody>{drugs.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6, fontWeight: 600 }}>{r.drug_name}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.total_refills)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.unique_patients)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.avg_quantity)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.avg_days_supply)}</td>
                  <td style={{ padding: 6, textAlign: 'right', color: '#10b981' }}>{fmt(r.auto_refill_pct)}%</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

function PatientDetailTab({ breakdown }) {
  const patients = breakdown?.per_patient || []
  const recent = breakdown?.recent_refills || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Per-Patient Summary">
        {patients.length === 0 ? <div style={{ color: '#94a3b8', fontSize: 13 }}>No patient data</div> : (
          <div style={{ maxHeight: 350, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Patient ID</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Refills</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Drugs</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Avg Quantity</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Avg Days Supply</th>
              </tr></thead>
              <tbody>{patients.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{r.patient_id}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.refills)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.drugs)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.avg_quantity)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.avg_days_supply)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="Recent Refills (Last 20)">
        {recent.length === 0 ? <div style={{ color: '#94a3b8', fontSize: 13 }}>No recent refills</div> : (
          <div style={{ maxHeight: 350, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Patient ID</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Drug</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Quantity</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Days Supply</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Pharmacy</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Date</th>
              </tr></thead>
              <tbody>{recent.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{r.patient_id}</td>
                  <td style={{ padding: 6 }}>{r.drug_name}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.quantity)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.days_supply)}</td>
                  <td style={{ padding: 6 }}>{r.pharmacy}</td>
                  <td style={{ padding: 6, fontSize: 11 }}>{r.refill_date}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

function GapAnalysisTab({ breakdown, defs }) {
  const gaps = breakdown?.gap_analysis || []
  const gapNote = defs?.gap_analysis_note

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {gapNote && (
        <Card title="Gap Analysis Note">
          <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>{gapNote}</p>
        </Card>
      )}

      <Card title="Refill Gap Analysis">
        {gaps.length === 0 ? <div style={{ color: '#94a3b8', fontSize: 13 }}>No gap analysis data</div> : (
          <div style={{ maxHeight: 450, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Patient ID</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Drug</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Days Supply</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Gap Days</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Status</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Last Refill</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Next Expected</th>
              </tr></thead>
              <tbody>{gaps.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{r.patient_id}</td>
                  <td style={{ padding: 6 }}>{r.drug_name}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.days_supply)}</td>
                  <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.gap_days)}</td>
                  <td style={{ padding: 6 }}><RefillStatusBadge status={r.status} /></td>
                  <td style={{ padding: 6, fontSize: 11 }}>{r.last_refill}</td>
                  <td style={{ padding: 6, fontSize: 11 }}>{r.next_expected}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
