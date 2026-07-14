import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const RISK_BG = {
  'High': { bg: '#fee2e2', text: '#991b1b', border: '#fca5a5' },
  'Markedly': { bg: '#fee2e2', text: '#991b1b', border: '#fca5a5' },
  'Moderate': { bg: '#fef3c7', text: '#92400e', border: '#fcd34d' },
  'Reduced': { bg: '#fef3c7', text: '#92400e', border: '#fcd34d' },
}

function getRiskStyle(sig) {
  for (const [key, style] of Object.entries(RISK_BG)) {
    if (sig && sig.includes(key)) return style
  }
  return { bg: '#f0fdf4', text: '#166534', border: '#86efac' }
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'alerts', label: 'Risk Alerts' },
  { id: 'profiles', label: 'Patient Profiles' },
  { id: 'genes', label: 'Gene-Variant Matrix' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PharmacogenomicsDashboard() {
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
      axios.get(`${API_URL}/api/pharmacogenomics/overview`),
      axios.get(`${API_URL}/api/pharmacogenomics/breakdown`),
      axios.get(`${API_URL}/api/pharmacogenomics/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Pharmacogenomics data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Pharmacogenomics Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          PGx-guided AED prescribing — HLA screening, CYP enzyme panels, drug-gene interactions (CPIC/PharmGKB)
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} trend={breakdown?.testing_trend} />}
      {tab === 'alerts' && overview && <AlertsTab alerts={overview.high_risk_alerts} interactions={overview.drug_gene_interactions} />}
      {tab === 'profiles' && breakdown && <ProfilesTab profiles={breakdown.patient_profiles} crossref={breakdown.med_pgx_crossref} />}
      {tab === 'genes' && breakdown && <GenesTab matrix={breakdown.gene_variant_matrix} hla={breakdown.hla_screening} cyp={breakdown.cyp_panel} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data, trend }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total PGx Tests"><KPI label="Tests performed" value={data.total_tests} /></Card>
      <Card title="Patients Tested"><KPI label="Unique patients" value={data.total_patients} /></Card>
      <Card title="Genes Analyzed"><KPI label="Gene panels" value={data.total_genes} /></Card>
      <Card title="Actionable Results"><KPI label={`${data.actionable_pct}% of all tests`} value={data.actionable_results} color="#ef4444" /></Card>

      <Card title="Gene Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.gene_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="gene" fontSize={12} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Metabolizer Status" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={data.metabolizer_distribution} dataKey="count" nameKey="status" cx="50%" cy="50%"
              outerRadius={90} label={({ status, count }) => `${status} (${count})`} labelLine={false}
              fontSize={10}>
              {data.metabolizer_distribution.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Evidence Level Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data.evidence_distribution} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={12} />
            <YAxis dataKey="level" type="category" fontSize={12} width={40} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Guideline Sources" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={data.source_distribution} dataKey="count" nameKey="source" cx="50%" cy="50%"
              outerRadius={70} label={({ source, count }) => `${source} (${count})`}>
              {data.source_distribution.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {trend && trend.length > 0 && (
        <Card title="Monthly Testing Trend" span={4}>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="tests" stroke="#3b82f6" name="Tests" strokeWidth={2} />
              <Line type="monotone" dataKey="patients" stroke="#10b981" name="Patients" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}
    </div>
  )
}

function AlertsTab({ alerts, interactions }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`High-Risk Alerts (${alerts.length})`} span={1}>
        {alerts.length === 0 ? (
          <p style={{ color: '#10b981', fontWeight: 600 }}>No high-risk PGx alerts</p>
        ) : (
          <div style={{ maxHeight: 500, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Gene</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Variant</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Significance</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Affected Drugs</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Recommendation</th>
                </tr>
              </thead>
              <tbody>
                {alerts.map((a, i) => {
                  const rs = getRiskStyle(a.clinical_significance)
                  return (
                    <tr key={i} style={{ background: rs.bg }}>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{a.patient_id}</td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{a.gene}</td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>{a.variant}</td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', color: rs.text, fontWeight: 600 }}>{a.clinical_significance}</td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12 }}>{a.affected_drugs}</td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12 }}>{a.recommendation}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="Drug-Gene Interaction Summary">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Affected Drugs</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Patients</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Actionable</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Rate</th>
            </tr>
          </thead>
          <tbody>
            {interactions.map((d, i) => (
              <tr key={i}>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{d.drugs}</td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{d.patients}</td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center',
                  color: d.actionable > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{d.actionable}</td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>
                  {d.patients > 0 ? `${Math.round(d.actionable / d.patients * 100)}%` : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function ProfilesTab({ profiles, crossref }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Patient PGx Profiles (${profiles.length})`}>
        <div style={{ maxHeight: 500, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Tests</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Actionable</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Genes Tested</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Test Date</th>
              </tr>
            </thead>
            <tbody>
              {profiles.map((p, i) => (
                <tr key={i} style={{ background: p.actionable > 2 ? '#fef2f2' : undefined }}>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{p.total_tests}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center',
                    color: p.actionable > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{p.actionable}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12 }}>{p.genes_tested}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{p.test_date}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {crossref && crossref.length > 0 && (
        <Card title="Medication-PGx Cross-Reference">
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            Patients currently on AEDs with relevant pharmacogenomic findings
          </p>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Current Drug</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Gene</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Recommendation</th>
              </tr>
            </thead>
            <tbody>
              {crossref.map((r, i) => (
                <tr key={i}>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{r.drug_name}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{r.gene} {r.variant}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{r.metabolizer_status}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12 }}>{r.recommendation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}

function GenesTab({ matrix, hla, cyp }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="HLA Allele Screening">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Gene</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Variant</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Status</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Patients</th>
            </tr>
          </thead>
          <tbody>
            {hla.map((r, i) => {
              const isCarrier = r.metabolizer_status === 'Carrier'
              return (
                <tr key={i} style={{ background: isCarrier ? '#fee2e2' : undefined }}>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{r.gene}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>{r.variant}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                    color: isCarrier ? '#991b1b' : '#166534', fontWeight: 600 }}>{r.metabolizer_status}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{r.patients}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Card>

      <Card title="CYP Enzyme Panel">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Gene</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Variant</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Metabolizer Status</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Patients</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Affected Drugs</th>
            </tr>
          </thead>
          <tbody>
            {cyp.map((r, i) => {
              const isPM = r.metabolizer_status?.includes('Poor')
              return (
                <tr key={i} style={{ background: isPM ? '#fef2f2' : undefined }}>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{r.gene}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>{r.variant}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                    color: isPM ? '#991b1b' : '#334155', fontWeight: isPM ? 600 : 400 }}>{r.metabolizer_status}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{r.patients}</td>
                  <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12 }}>{r.affected_drugs}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Card>

      <Card title="Full Gene-Variant Matrix">
        <div style={{ maxHeight: 500, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Gene</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Variant</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Patients</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Significance</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Recommendation</th>
              </tr>
            </thead>
            <tbody>
              {matrix.map((r, i) => {
                const rs = getRiskStyle(r.clinical_significance)
                return (
                  <tr key={i}>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{r.gene}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>{r.variant}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>{r.metabolizer_status}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', textAlign: 'center' }}>{r.patients}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', color: rs.text, fontSize: 12 }}>{r.clinical_significance}</td>
                    <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0', fontSize: 12 }}>{r.recommendation}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={data.title}>
        {data.terms.map((t, i) => (
          <div key={i} style={{ marginBottom: 16, paddingBottom: 16, borderBottom: i < data.terms.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <strong style={{ color: '#1e293b', fontSize: 14 }}>{t.term}</strong>
            <p style={{ margin: '4px 0 0', fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{t.definition}</p>
          </div>
        ))}
      </Card>
      <Card title="Data Sources">
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {data.data_sources.map((s, i) => (
            <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 6 }}>{s}</li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
