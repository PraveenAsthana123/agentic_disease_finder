import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(4)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

const TABS = ['Overview', 'Risk Analysis', 'Patient Breakdown', 'Biomarkers', 'Methodology']

export default function SeizurePredictionDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/seizure-prediction/overview`),
      axios.get(`${API}/api/seizure-prediction/breakdown`),
      axios.get(`${API}/api/seizure-prediction/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Seizure Prediction analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Seizure Prediction Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Real-time seizure risk prediction — {fmt(kpis.total_windows)} analysis windows, {fmtPct(kpis.sensitivity)} sensitivity
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#3b82f6' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: tab === i ? 600 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} ov={ov} />}
      {tab === 1 && <RiskAnalysisTab bd={bd} />}
      {tab === 2 && <PatientBreakdownTab bd={bd} />}
      {tab === 3 && <BiomarkersTab bd={bd} />}
      {tab === 4 && <MethodologyTab definitions={df} />}
    </div>
  )
}

function OverviewTab({ kpis, ov }) {
  const riskDist = ov.risk_distribution || []
  const temporalPattern = ov.temporal_pattern || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Total Windows" value={fmt(kpis.total_windows)} />
          <KPI label="Sensitivity" value={fmtPct(kpis.sensitivity)} color="#10b981" />
          <KPI label="False Alarm Rate (/hr)" value={fmt(kpis.false_alarm_rate_hr)} color="#ef4444" />
          <KPI label="Mean Prediction Horizon" value={fmt(kpis.mean_prediction_horizon)} color="#3b82f6" sub="minutes" />
        </div>
      </Card>

      <Card title="Risk Score Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={riskDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" name="Windows" fill="#3b82f6" radius={[4, 4, 0, 0]}>
              {riskDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Temporal Risk Trends (Hourly Average)">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={temporalPattern}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="mean_risk" name="Mean Risk" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function RiskAnalysisTab({ bd }) {
  if (!bd?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const thresholds = bd.threshold_analysis || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Risk Threshold Analysis">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Threshold</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Sensitivity</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Specificity</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>PPV</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>F1</th>
              </tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600 }}>{fmt(t.threshold)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{fmtPct(t.sensitivity)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{fmtPct(t.specificity)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmtPct(t.ppv)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#8b5cf6', fontWeight: 600 }}>{fmtPct(t.f1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Sensitivity vs Specificity by Threshold">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={thresholds} margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="threshold" />
            <YAxis domain={[0, 1]} tickFormatter={v => fmtPct(v)} />
            <Tooltip formatter={v => fmtPct(v)} />
            <Legend />
            <Bar dataKey="sensitivity" name="Sensitivity" fill="#10b981" radius={[4, 4, 0, 0]} />
            <Bar dataKey="specificity" name="Specificity" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function PatientBreakdownTab({ bd }) {
  if (!bd?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const patients = bd.patient_breakdown || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Per-Patient Prediction Performance">
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient ID</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Seizure Count</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Mean Risk Score</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Detection Accuracy</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Mean Confidence</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{p.seizure_count}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#f59e0b', fontWeight: 600 }}>{fmt(p.mean_risk_score)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{fmtPct(p.detection_accuracy)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmtPct(p.mean_confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Per-Patient Mean Risk Score">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={patients} margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="patient_id" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="mean_risk_score" name="Mean Risk Score" fill="#f59e0b" radius={[4, 4, 0, 0]}>
              {patients.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BiomarkersTab({ bd }) {
  if (!bd?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const biomarkers = bd.preictal_biomarkers || []
  const correlations = bd.feature_correlations || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Pre-ictal Biomarker Comparison" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={biomarkers} margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="biomarker" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="seizure_day" name="Seizure Day" fill="#ef4444" radius={[4, 4, 0, 0]} />
            <Bar dataKey="non_seizure_day" name="Non-Seizure Day" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Feature Correlation with Risk" span={2}>
        <ResponsiveContainer width="100%" height={Math.max(220, correlations.length * 32)}>
          <BarChart data={correlations} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[-1, 1]} />
            <YAxis type="category" dataKey="feature" width={120} />
            <Tooltip formatter={v => fmt(v)} />
            <Bar dataKey="correlation" name="Correlation" radius={[0, 4, 4, 0]}>
              {correlations.map((c, i) => (
                <Cell key={i} fill={c.correlation >= 0 ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function MethodologyTab({ definitions }) {
  if (!definitions?.available) return <div style={{ color: '#f59e0b' }}>No definitions data</div>

  const defs = definitions.definitions || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {defs.map((d, i) => (
        <Card key={i} title={d.title}>
          <p style={{ margin: 0, fontSize: 13, color: '#475569' }}>{d.description}</p>
        </Card>
      ))}
    </div>
  )
}
