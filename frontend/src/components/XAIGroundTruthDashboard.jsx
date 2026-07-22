import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, LineChart, Line, AreaChart, Area
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4']
const BLUE = '#3b82f6'
const GREEN = '#10b981'
const ORANGE = '#f59e0b'
const PURPLE = '#8b5cf6'
const RED = '#ef4444'

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)) : String(v)
}

function pct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

function concordanceColor(score) {
  if (score >= 0.8) return GREEN
  if (score >= 0.6) return ORANGE
  return RED
}

export default function XAIGroundTruthDashboard() {
  const [overview, setOverview] = useState(null)
  const [concordance, setConcordance] = useState(null)
  const [features, setFeatures] = useState(null)
  const [patients, setPatients] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, co, fe, pa, de] = await Promise.all([
          axios.get(`${API}/api/xai-groundtruth/overview`),
          axios.get(`${API}/api/xai-groundtruth/concordance`),
          axios.get(`${API}/api/xai-groundtruth/features`),
          axios.get(`${API}/api/xai-groundtruth/patients`),
          axios.get(`${API}/api/xai-groundtruth/definitions`),
        ])
        setOverview(ov.data)
        setConcordance(co.data)
        setFeatures(fe.data)
        setPatients(pa.data)
        setDefinitions(de.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading XAI Ground-Truth Comparison data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No XAI ground-truth data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'concordance', label: 'Concordance' },
    { id: 'features', label: 'Features' },
    { id: 'patients', label: 'Patients' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* ── Overview Tab ── */
  function renderOverview() {
    const ov = overview
    const diseases = ov.per_disease || ov.diseases || []
    const chartData = diseases.map(d => ({
      name: d.disease || d.name || d.label,
      concordance: d.concordance != null ? +(d.concordance * 100).toFixed(1) : 0,
      matched: d.matched_features || d.matched || 0,
      unmatched: d.unmatched_features || d.unmatched || 0,
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        <Card>
          <KPI label="Diseases Analyzed" value={ov.total_diseases || ov.diseases_analyzed || diseases.length} color={BLUE} />
        </Card>
        <Card>
          <KPI label="Avg Concordance" value={pct(ov.avg_concordance || ov.mean_concordance)} color={GREEN} sub="AI vs Expert" />
        </Card>
        <Card>
          <KPI label="XAI Method" value={ov.method || 'SHAP'} color={PURPLE} sub={ov.model_type || 'TreeSHAP'} />
        </Card>
        <Card>
          <KPI label="Expert Annotations" value={fmt(ov.total_annotations || ov.expert_annotations || '--')} color={ORANGE} />
        </Card>

        <Card title="Per-Disease Concordance (%)" span={3}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} unit="%" />
              <Tooltip formatter={(v) => v + '%'} />
              <Bar dataKey="concordance" fill={BLUE} radius={[4, 4, 0, 0]} name="Concordance %" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Summary">
          <div style={{ fontSize: 13, lineHeight: 1.9, color: '#475569' }}>
            <div><strong>Method:</strong> {ov.method || 'SHAP'}</div>
            <div><strong>Framework:</strong> {ov.framework || 'shap 0.43'}</div>
            <div><strong>Reference:</strong> {ov.reference || 'Lundberg & Lee 2017'}</div>
            <div style={{ marginTop: 8 }}>
              <Badge
                text={ov.avg_concordance >= 0.7 ? 'HIGH AGREEMENT' : ov.avg_concordance >= 0.5 ? 'MODERATE' : 'LOW AGREEMENT'}
                color={ov.avg_concordance >= 0.7 ? GREEN : ov.avg_concordance >= 0.5 ? ORANGE : RED}
              />
            </div>
          </div>
        </Card>

        <Card title="Disease Concordance Summary" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Disease</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Concordance</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Matched</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Unmatched</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {diseases.map((d, i) => {
                  const score = d.concordance || 0
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{d.disease || d.name || d.label}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600, color: concordanceColor(score) }}>
                        {pct(score)}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{d.matched_features || d.matched || 0}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{d.unmatched_features || d.unmatched || 0}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <Badge
                          text={score >= 0.8 ? 'STRONG' : score >= 0.6 ? 'MODERATE' : 'WEAK'}
                          color={concordanceColor(score)}
                        />
                      </td>
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

  /* ── Concordance Tab ── */
  function renderConcordance() {
    const data = concordance
    const diseases = data.diseases || data.per_disease || data.results || []
    const distributionData = (data.distribution || data.histogram || []).map((d, i) => ({
      bin: d.bin || d.range || d.label || `${(i * 0.1).toFixed(1)}-${((i + 1) * 0.1).toFixed(1)}`,
      count: d.count || d.frequency || 0,
    }))

    const barData = diseases.map(d => ({
      name: d.disease || d.name,
      concordance: +(((d.concordance || 0) * 100).toFixed(1)),
      matched: d.matched_features || d.matched || 0,
      unmatched: d.unmatched_features || d.unmatched || 0,
      top_k_overlap: d.top_k_overlap != null ? +(d.top_k_overlap * 100).toFixed(1) : null,
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        <Card>
          <KPI label="Mean Concordance" value={pct(data.mean_concordance || data.avg_concordance)} color={GREEN} />
        </Card>
        <Card>
          <KPI label="Median Concordance" value={pct(data.median_concordance || data.median)} color={BLUE} />
        </Card>
        <Card>
          <KPI label="Std Deviation" value={fmt(data.std_concordance || data.std || data.sd)} color={ORANGE} />
        </Card>

        <Card title="Matched vs Unmatched Features by Disease" span={2}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={barData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="matched" fill={GREEN} radius={[4, 4, 0, 0]} name="Matched" stackId="stack" />
              <Bar dataKey="unmatched" fill={RED} radius={[4, 4, 0, 0]} name="Unmatched" stackId="stack" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Concordance Distribution">
          {distributionData.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill={PURPLE} radius={[4, 4, 0, 0]} name="Count" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', paddingTop: 40 }}>
              No distribution data available
            </div>
          )}
        </Card>

        <Card title="Detailed Concordance Scores" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Disease</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Concordance</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Top-K Overlap</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Matched</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Unmatched</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Rank Corr</th>
                </tr>
              </thead>
              <tbody>
                {diseases.map((d, i) => {
                  const score = d.concordance || 0
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{d.disease || d.name}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600, color: concordanceColor(score) }}>
                        {pct(score)}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{d.top_k_overlap != null ? pct(d.top_k_overlap) : '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', color: GREEN }}>{d.matched_features || d.matched || 0}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', color: RED }}>{d.unmatched_features || d.unmatched || 0}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{d.rank_correlation != null ? fmt(d.rank_correlation) : '--'}</td>
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

  /* ── Features Tab ── */
  function renderFeatures() {
    const data = features
    const comparisons = data.comparisons || data.features || data.rankings || []
    const bands = data.band_analysis || data.bands || []

    const groupedData = comparisons.slice(0, 15).map(f => ({
      name: f.feature || f.name,
      ai_rank: f.ai_rank || f.shap_rank || 0,
      expert_rank: f.expert_rank || f.ground_truth_rank || 0,
    }))

    const radarData = bands.map(b => ({
      band: b.band || b.name,
      ai: b.ai_importance != null ? +(b.ai_importance * 100).toFixed(1) : (b.ai_score || 0),
      expert: b.expert_importance != null ? +(b.expert_importance * 100).toFixed(1) : (b.expert_score || 0),
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card>
          <KPI label="Total Features Compared" value={data.total_features || comparisons.length} color={BLUE} />
        </Card>
        <Card>
          <KPI label="Rank Correlation (Spearman)" value={fmt(data.spearman_rho || data.rank_correlation)} color={GREEN}
            sub={data.p_value != null ? `p = ${data.p_value.toExponential(2)}` : undefined} />
        </Card>

        <Card title="AI Rank vs Expert Rank (Top 15 Features)" span={2}>
          <ResponsiveContainer width="100%" height={380}>
            <BarChart data={groupedData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={80} />
              <YAxis tick={{ fontSize: 11 }} reversed label={{ value: 'Rank (1 = most important)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
              <Tooltip />
              <Bar dataKey="ai_rank" fill={BLUE} name="AI (SHAP) Rank" radius={[4, 4, 0, 0]} />
              <Bar dataKey="expert_rank" fill={ORANGE} name="Expert Rank" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {radarData.length > 0 && (
          <Card title="Band-Level Importance: AI vs Expert" span={1}>
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="band" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis tick={{ fontSize: 10 }} />
                <Radar name="AI (SHAP)" dataKey="ai" stroke={BLUE} fill={BLUE} fillOpacity={0.2} />
                <Radar name="Expert" dataKey="expert" stroke={ORANGE} fill={ORANGE} fillOpacity={0.2} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        )}

        <Card title={radarData.length > 0 ? 'Agreement Summary' : 'Agreement Summary'} span={radarData.length > 0 ? 1 : 2}>
          <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.8 }}>
            <div><strong>Agreed (top-5):</strong> {data.agreed_top5 != null ? data.agreed_top5 : '--'} features</div>
            <div><strong>Disagreed (top-5):</strong> {data.disagreed_top5 != null ? data.disagreed_top5 : '--'} features</div>
            <div><strong>Kendall Tau:</strong> {fmt(data.kendall_tau)}</div>
            <div><strong>Weighted Kappa:</strong> {fmt(data.weighted_kappa)}</div>
            <div style={{ marginTop: 10 }}>
              <Badge
                text={data.spearman_rho >= 0.7 ? 'STRONG CORRELATION' : data.spearman_rho >= 0.4 ? 'MODERATE' : 'WEAK CORRELATION'}
                color={data.spearman_rho >= 0.7 ? GREEN : data.spearman_rho >= 0.4 ? ORANGE : RED}
              />
            </div>
          </div>
        </Card>

        <Card title="Feature-Level Agreement / Disagreement" span={2}>
          <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Feature</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>AI Rank</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Expert Rank</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Rank Diff</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Agreement</th>
                </tr>
              </thead>
              <tbody>
                {comparisons.map((f, i) => {
                  const aiR = f.ai_rank || f.shap_rank || 0
                  const exR = f.expert_rank || f.ground_truth_rank || 0
                  const diff = Math.abs(aiR - exR)
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{f.feature || f.name}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', color: BLUE, fontWeight: 600 }}>{aiR}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', color: ORANGE, fontWeight: 600 }}>{exR}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{diff}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <Badge
                          text={diff <= 1 ? 'AGREE' : diff <= 3 ? 'CLOSE' : 'DISAGREE'}
                          color={diff <= 1 ? GREEN : diff <= 3 ? ORANGE : RED}
                        />
                      </td>
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

  /* ── Patients Tab ── */
  function renderPatients() {
    const data = patients
    const patientList = data.patients || data.records || data.audit || []
    const summary = data.summary || {}

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        <Card>
          <KPI label="Total Patients" value={data.total_patients || patientList.length} color={BLUE} />
        </Card>
        <Card>
          <KPI label="Avg Patient Concordance" value={pct(summary.avg_concordance || data.avg_concordance)} color={GREEN} />
        </Card>
        <Card>
          <KPI label="High Agreement" value={summary.high_agreement || data.high_agreement || '--'} color={GREEN} sub=">= 80%" />
        </Card>
        <Card>
          <KPI label="Low Agreement" value={summary.low_agreement || data.low_agreement || '--'} color={RED} sub="< 50%" />
        </Card>

        {patientList.length > 0 && (
          <Card title="Patient Concordance Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={patientList.map((p, i) => ({
                idx: i + 1,
                concordance: +((p.concordance || 0) * 100).toFixed(1),
              })).sort((a, b) => b.concordance - a.concordance)} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="idx" tick={{ fontSize: 11 }} label={{ value: 'Patient (sorted)', position: 'insideBottom', offset: -2, style: { fontSize: 11 } }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} unit="%" />
                <Tooltip formatter={(v) => v + '%'} />
                <Area type="monotone" dataKey="concordance" stroke={BLUE} fill={BLUE} fillOpacity={0.15} name="Concordance %" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        )}

        {patientList.length > 0 && (
          <Card title="Agreement Breakdown" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'High (>=80%)', value: patientList.filter(p => (p.concordance || 0) >= 0.8).length },
                    { name: 'Moderate (50-80%)', value: patientList.filter(p => (p.concordance || 0) >= 0.5 && (p.concordance || 0) < 0.8).length },
                    { name: 'Low (<50%)', value: patientList.filter(p => (p.concordance || 0) < 0.5).length },
                  ]}
                  cx="50%" cy="50%" outerRadius={100} dataKey="value" label={({ name, value }) => `${name}: ${value}`}
                >
                  <Cell fill={GREEN} />
                  <Cell fill={ORANGE} />
                  <Cell fill={RED} />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        )}

        <Card title="Patient-Level Explainability Audit" span={4}>
          <div style={{ overflowX: 'auto', maxHeight: 450, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Patient ID</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Disease</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Concordance</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Top AI Feature</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Top Expert Feature</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Matched</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {patientList.map((p, i) => {
                  const score = p.concordance || 0
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12 }}>{p.patient_id || p.id || `P-${i + 1}`}</td>
                      <td style={{ padding: '8px 12px' }}>{p.disease || p.diagnosis || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', fontWeight: 600, color: concordanceColor(score) }}>
                        {pct(score)}
                      </td>
                      <td style={{ padding: '8px 12px', fontSize: 12 }}>{p.top_ai_feature || p.top_shap_feature || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12 }}>{p.top_expert_feature || p.top_ground_truth_feature || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{p.matched_count || p.matched || '--'}/{p.total_features || p.total || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <Badge
                          text={score >= 0.8 ? 'PASS' : score >= 0.5 ? 'REVIEW' : 'FLAG'}
                          color={score >= 0.8 ? GREEN : score >= 0.5 ? ORANGE : RED}
                        />
                      </td>
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

  /* ── Definitions Tab ── */
  function renderDefinitions() {
    const data = definitions
    const methods = data.methods || data.definitions || []
    const references = data.references || data.citations || []

    const definitionCards = [
      {
        title: 'SHAP (SHapley Additive exPlanations)',
        desc: data.shap_definition || 'SHAP assigns each feature an importance value for a particular prediction using Shapley values from cooperative game theory. It provides locally consistent, additive feature attributions that sum to the model output difference from the baseline.',
        color: BLUE,
      },
      {
        title: 'Concordance Score',
        desc: data.concordance_definition || 'Measures overlap between AI-derived feature importance rankings and expert neurologist ground-truth annotations. Computed as the weighted intersection of top-K features normalized by the maximum possible overlap. Range: 0 (no overlap) to 1 (perfect agreement).',
        color: GREEN,
      },
      {
        title: 'Ground-Truth Annotations',
        desc: data.ground_truth_definition || 'Expert neurologist annotations identifying the most diagnostically relevant EEG features for each condition. Annotations follow IFCN/ACNS standards and include spatial (channel), temporal (segment), and spectral (frequency band) attributes.',
        color: ORANGE,
      },
      {
        title: 'EU AI Act — Article 86 (Explainability)',
        desc: data.eu_ai_act_definition || 'The EU AI Act (2024) mandates that high-risk AI systems provide meaningful explanations of decisions. Article 86 requires that affected persons can obtain clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken. Clinical AI systems must demonstrate concordance between model explanations and domain expert interpretations.',
        color: PURPLE,
      },
    ]

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        {definitionCards.map((card, i) => (
          <Card key={i} title={card.title}>
            <div style={{
              borderLeft: `3px solid ${card.color}`, paddingLeft: 14, fontSize: 13,
              color: '#475569', lineHeight: 1.7
            }}>
              {card.desc}
            </div>
          </Card>
        ))}

        {methods.length > 0 && (
          <Card title="Methods & Metrics" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Method</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Description</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Type</th>
                  </tr>
                </thead>
                <tbody>
                  {methods.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{m.name || m.method}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{m.description || m.desc}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <Badge text={m.type || m.category || 'metric'} color={PURPLE} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        <Card title="References" span={2}>
          <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.9 }}>
            {references.length > 0 ? (
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                {references.map((r, i) => (
                  <li key={i} style={{ marginBottom: 6 }}>
                    {r.citation || r.text || r}
                    {(r.doi || r.url) && (
                      <span style={{ marginLeft: 6 }}>
                        <a href={r.doi || r.url} target="_blank" rel="noopener noreferrer"
                          style={{ color: BLUE, fontSize: 12 }}>[link]</a>
                      </span>
                    )}
                  </li>
                ))}
              </ol>
            ) : (
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                <li>Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.</li>
                <li>Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). Why should I trust you? Explaining the predictions of any classifier. KDD.</li>
                <li>European Parliament (2024). EU Artificial Intelligence Act. Regulation (EU) 2024/1689.</li>
                <li>Tjoa, E. & Guan, C. (2021). A survey on explainable artificial intelligence: Toward medical XAI. IEEE TNNLS.</li>
                <li>Ahmad, M. A. et al. (2018). Interpretable machine learning in healthcare. IEEE ICHI.</li>
              </ol>
            )}
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>XAI Ground-Truth Comparison</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          AI SHAP explanations vs expert neurologist ground-truth annotations
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 2 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'concordance' && renderConcordance()}
      {tab === 'features' && renderFeatures()}
      {tab === 'patients' && renderPatients()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
