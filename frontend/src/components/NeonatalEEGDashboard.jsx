import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
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

const GA_COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981']
const ETIOLOGY_COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#06b6d4', '#94a3b8']
const SEIZURE_COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#8b5cf6', '#10b981', '#06b6d4']
const BG_COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#dc2626', '#7f1d1d', '#94a3b8']

export default function NeonatalEEGDashboard() {
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
      axios.get(`${API}/api/neonatal-eeg/overview`),
      axios.get(`${API}/api/neonatal-eeg/breakdown`),
      axios.get(`${API}/api/neonatal-eeg/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patterns', label: 'EEG Patterns' },
    { id: 'seizures', label: 'Seizure Types' },
    { id: 'performance', label: 'AI Performance' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Neonatal EEG data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const k = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Neonatal EEG Dashboard (Helsinki)
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        {k.total_neonates} neonates | {k.channels}-channel EEG @ {k.sampling_rate_hz} Hz |
        {' '}{k.total_seizure_events_annotated} annotated seizures | GA {k.gestational_age_range_weeks} wk
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(overview, k)}
      {tab === 'patterns' && renderPatterns(overview, breakdown)}
      {tab === 'seizures' && renderSeizures(overview, breakdown)}
      {tab === 'performance' && renderPerformance(overview, breakdown)}
      {tab === 'definitions' && renderDefinitions(definitions)}
    </div>
  )
}

function renderOverview(ov, k) {
  const gaData = ov.gestational_age_distribution || []
  const readiness = ov.model_readiness || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Neonates" value={k.total_neonates} color="#3b82f6" /></Card>
      <Card><KPI label="Seizure+" value={k.seizure_positive_neonates} sub={`${Math.round(k.seizure_positive_neonates / k.total_neonates * 100)}% of cohort`} color="#ef4444" /></Card>
      <Card><KPI label="Annotated Seizures" value={k.total_seizure_events_annotated} color="#f59e0b" /></Card>
      <Card><KPI label="Inter-rater Kappa" value={k.annotation_agreement_kappa} color="#10b981" /></Card>
      <Card><KPI label="Channels" value={k.channels} sub="Modified 10-20" color="#8b5cf6" /></Card>
      <Card><KPI label="Sampling Rate" value={`${k.sampling_rate_hz} Hz`} color="#06b6d4" /></Card>
      <Card><KPI label="Median Recording" value={`${k.median_recording_duration_hours}h`} color="#64748b" /></Card>
      <Card><KPI label="GA Range" value={k.gestational_age_range_weeks} sub="weeks" color="#3b82f6" /></Card>

      {/* GA Distribution */}
      <Card title="Gestational Age Distribution" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={gaData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="group" tick={{ fontSize: 10 }} angle={-15} />
            <YAxis />
            <Tooltip formatter={(v, name) => [v, name === 'n' ? 'Neonates' : name]} />
            <Bar dataKey="n" name="Neonates" radius={[4, 4, 0, 0]}>
              {gaData.map((_, i) => <Cell key={i} fill={GA_COLORS[i % GA_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Adult vs Neonate comparison */}
      <Card title="Neonatal vs Adult Detection" span={2}>
        {(() => {
          const bench = ov.detection_performance_benchmark || {}
          const radarData = [
            { metric: 'Sensitivity', neonatal: bench.neonatal_sensitivity_pct || 0, adult: bench.adult_sensitivity_pct || 0 },
            { metric: 'Specificity', neonatal: bench.neonatal_specificity_pct || 0, adult: bench.adult_specificity_pct || 0 },
            { metric: 'AUC x100', neonatal: (bench.neonatal_auc || 0) * 100, adult: (bench.adult_auc || 0) * 100 },
          ]
          return (
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                <Radar name="Neonatal" dataKey="neonatal" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} />
                <Radar name="Adult" dataKey="adult" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          )
        })()}
      </Card>

      {/* Differences from Adult EEG */}
      <Card title="Neonatal vs Adult EEG Differences" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Dimension</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Adult</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Neonate</th>
              </tr>
            </thead>
            <tbody>
              {(ov.differences_from_adult_eeg || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{d.dimension}</td>
                  <td style={{ padding: 8, color: '#64748b' }}>{d.adult}</td>
                  <td style={{ padding: 8 }}>{d.neonate}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Model Readiness */}
      <Card title="Model Readiness" span={4}>
        <div style={{ padding: 16, background: '#fef3c7', borderRadius: 8, border: '1px solid #fcd34d' }}>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
            <Badge text={readiness.status === 'gap' ? 'GAP' : readiness.status} color={readiness.status === 'gap' ? '#f59e0b' : '#10b981'} />
            <span style={{ fontSize: 13 }}>Data in clinical.db: <strong>{readiness.data_in_clinical_db ? 'Yes' : 'No'}</strong></span>
            <span style={{ fontSize: 13 }}>Specialist model trained: <strong>{readiness.specialist_model_trained ? 'Yes' : 'No'}</strong></span>
            <span style={{ fontSize: 13 }}>Transfer candidate: <code style={{ fontSize: 12, background: '#f1f5f9', padding: '2px 6px', borderRadius: 4 }}>{readiness.transfer_learning_candidate}</code></span>
          </div>
          <p style={{ fontSize: 12, color: '#92400e', marginTop: 10, lineHeight: 1.5 }}>{readiness.next_step}</p>
        </div>
      </Card>
    </div>
  )
}

function renderPatterns(ov, bd) {
  const bgChars = ov.eeg_background_characteristics || []
  const bgClass = bd.eeg_background_classification || []
  const gaProfiles = bd.gestational_age_pattern_profiles || []
  const aeeg = bd.aeeg_patterns || {}
  const montage = bd.montage_comparison || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Background Characteristics */}
      <Card title="Neonatal EEG Background Patterns" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
          {bgChars.map((p, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{p.pattern}</div>
              <div style={{ fontSize: 11, color: '#475569', marginTop: 6, lineHeight: 1.5 }}>{p.description}</div>
              <div style={{ fontSize: 11, color: '#8b5cf6', marginTop: 6, fontStyle: 'italic' }}>{p.clinical_significance}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Background Classification Distribution */}
      <Card title="Background Classification (Helsinki)" span={1}>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={bgClass} dataKey="pct_in_helsinki" nameKey="class" cx="50%" cy="50%"
                 outerRadius={90} label={({ payload }) => `${payload.class}: ${payload.pct_in_helsinki}%`}>
              {bgClass.map((_, i) => <Cell key={i} fill={BG_COLORS[i % BG_COLORS.length]} />)}
            </Pie>
            <Tooltip formatter={(v) => `${v}%`} />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Background Classification Table */}
      <Card title="Background Class Details" span={1}>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Class</th>
                <th style={{ textAlign: 'right', padding: 6 }}>%</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Prognosis</th>
              </tr>
            </thead>
            <tbody>
              {bgClass.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{c.class}</td>
                  <td style={{ padding: 6, textAlign: 'right', fontWeight: 600 }}>{c.pct_in_helsinki}%</td>
                  <td style={{ padding: 6 }}>
                    <Badge text={c.prognosis_category}
                           color={c.prognosis_category === 'Normal' ? '#10b981' :
                                  c.prognosis_category === 'Mildly abnormal' ? '#3b82f6' :
                                  c.prognosis_category === 'Moderately abnormal' ? '#f59e0b' : '#ef4444'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* GA Pattern Profiles */}
      <Card title="Gestational Age Pattern Profiles" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Group</th>
                <th style={{ textAlign: 'left', padding: 8 }}>GA</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Background</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Amplitude</th>
                <th style={{ textAlign: 'left', padding: 8 }}>IBI</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Delta Brushes</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Seizure Risk</th>
              </tr>
            </thead>
            <tbody>
              {gaProfiles.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{p.group}</td>
                  <td style={{ padding: 8 }}>{p.ga_range}</td>
                  <td style={{ padding: 8, fontSize: 10 }}>{p.background}</td>
                  <td style={{ padding: 8 }}>{p.amplitude_range_uv} uV</td>
                  <td style={{ padding: 8 }}>{p.typical_ibi_sec} s</td>
                  <td style={{ padding: 8 }}>{p.delta_brushes}</td>
                  <td style={{ padding: 8 }}>
                    <Badge text={p.seizure_risk}
                           color={p.seizure_risk.includes('High') ? '#ef4444' :
                                  p.seizure_risk.includes('Moderate') ? '#f59e0b' : '#10b981'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Montage Comparison */}
      <Card title="Montage: Neonatal vs Adult" span={1}>
        <div style={{ display: 'grid', gap: 12 }}>
          {[montage.neonatal_modified, montage.adult_standard].filter(Boolean).map((m, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: '#1e293b' }}>{m.name}</div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                {m.electrode_count} electrodes | {m.inter_electrode_distance_cm} cm spacing | {m.electrode_cap_size_cm}
              </div>
              <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>{m.notes}</div>
            </div>
          ))}
          {montage.spatial_resolution_impact && (
            <div style={{ fontSize: 11, color: '#475569', fontStyle: 'italic', lineHeight: 1.5 }}>
              {montage.spatial_resolution_impact}
            </div>
          )}
        </div>
      </Card>

      {/* aEEG Patterns */}
      <Card title="Amplitude-Integrated EEG (aEEG)" span={1}>
        <p style={{ fontSize: 11, color: '#475569', lineHeight: 1.5, marginBottom: 12 }}>{aeeg.description}</p>
        <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 4 }}>Class</th>
              <th style={{ textAlign: 'center', padding: 4 }}>Upper</th>
              <th style={{ textAlign: 'center', padding: 4 }}>Lower</th>
              <th style={{ textAlign: 'left', padding: 4 }}>Prognosis</th>
            </tr>
          </thead>
          <tbody>
            {(aeeg.background_pattern_classes || []).map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 4, fontSize: 10 }}>{c.class}</td>
                <td style={{ padding: 4, textAlign: 'center' }}>{c.upper_margin_uv}</td>
                <td style={{ padding: 4, textAlign: 'center' }}>{c.lower_margin_uv}</td>
                <td style={{ padding: 4 }}>
                  <Badge text={c.prognosis}
                         color={c.prognosis === 'Normal' ? '#10b981' :
                                c.prognosis.includes('Mildly') ? '#3b82f6' : '#ef4444'} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ marginTop: 10, padding: 8, background: '#fef3c7', borderRadius: 6, fontSize: 11, color: '#92400e' }}>
          Seizure sensitivity: {aeeg.sensitivity_for_seizure_pct}% — {aeeg.note}
        </div>
      </Card>
    </div>
  )
}

function renderSeizures(ov, bd) {
  const seizureTypes = ov.neonatal_seizure_types || []
  const etiologies = bd.seizure_etiology_distribution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Seizure Type Distribution */}
      <Card title="Neonatal Seizure Types" span={1}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={seizureTypes} dataKey="pct_of_neonatal_seizures" nameKey="type"
                 cx="50%" cy="50%" outerRadius={100}
                 label={({ type, pct_of_neonatal_seizures }) => `${type}: ${pct_of_neonatal_seizures}%`}>
              {seizureTypes.map((_, i) => <Cell key={i} fill={SEIZURE_COLORS[i % SEIZURE_COLORS.length]} />)}
            </Pie>
            <Tooltip formatter={(v) => `${v}%`} />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Seizure Etiology */}
      <Card title="Seizure Etiology Distribution" span={1}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={etiologies} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" unit="%" />
            <YAxis type="category" dataKey="etiology" width={160} tick={{ fontSize: 10 }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Bar dataKey="pct" name="%" radius={[0, 4, 4, 0]}>
              {etiologies.map((_, i) => <Cell key={i} fill={ETIOLOGY_COLORS[i % ETIOLOGY_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Seizure Types Detail */}
      <Card title="Seizure Type Details" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
                <th style={{ textAlign: 'right', padding: 8 }}>%</th>
                <th style={{ textAlign: 'left', padding: 8 }}>EEG Correlate</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Note</th>
              </tr>
            </thead>
            <tbody>
              {seizureTypes.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{s.type}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{s.pct_of_neonatal_seizures}%</td>
                  <td style={{ padding: 8, fontSize: 11, color: '#475569' }}>{s.eeg_correlate}</td>
                  <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{s.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Etiology Detail */}
      <Card title="Seizure Etiology: EEG Patterns & Prognosis" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
          {etiologies.map((e, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: 12 }}>{e.etiology}</span>
                <Badge text={`${e.pct}%`} color={e.pct >= 20 ? '#ef4444' : e.pct >= 10 ? '#f59e0b' : '#64748b'} />
              </div>
              <div style={{ fontSize: 11, color: '#475569', marginTop: 6 }}><strong>EEG:</strong> {e.eeg_pattern}</div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}><strong>Prognosis:</strong> {e.prognosis}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function renderPerformance(ov, bd) {
  const comp = bd.detection_performance_comparison || {}
  const neo = comp.neonatal_model || {}
  const adult = comp.adult_model || {}
  const gap = comp.performance_gap || {}

  const perfBars = [
    { metric: 'Sensitivity %', neonatal: neo.sensitivity_pct || 0, adult: adult.sensitivity_pct || 0 },
    { metric: 'Specificity %', neonatal: neo.specificity_pct || 0, adult: adult.specificity_pct || 0 },
    { metric: 'AUC x100', neonatal: (neo.auc || 0) * 100, adult: (adult.auc || 0) * 100 },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Performance Comparison Bar */}
      <Card title="Detection Performance: Neonatal vs Adult" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={perfBars}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Bar dataKey="neonatal" name="Neonatal" fill="#ef4444" radius={[4, 4, 0, 0]} />
            <Bar dataKey="adult" name="Adult" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Neonatal Model Details */}
      <Card title="Neonatal Seizure Detector" span={1}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 16 }}>
          <KPI label="Sensitivity" value={`${neo.sensitivity_pct}%`} color="#ef4444" />
          <KPI label="Specificity" value={`${neo.specificity_pct}%`} color="#f59e0b" />
          <KPI label="AUC" value={neo.auc} color="#3b82f6" />
          <KPI label="FPR/hour" value={neo.false_positive_rate_per_hour} color="#8b5cf6" />
        </div>
        <p style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>Dataset: {neo.dataset}</p>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 6 }}>Key Challenges:</div>
        <ul style={{ margin: 0, paddingLeft: 16 }}>
          {(neo.key_challenges || []).map((c, i) => (
            <li key={i} style={{ fontSize: 11, color: '#475569', marginBottom: 4, lineHeight: 1.4 }}>{c}</li>
          ))}
        </ul>
      </Card>

      {/* Adult Model Details */}
      <Card title="Adult Seizure Detector (Reference)" span={1}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 16 }}>
          <KPI label="Sensitivity" value={`${adult.sensitivity_pct}%`} color="#10b981" />
          <KPI label="Specificity" value={`${adult.specificity_pct}%`} color="#10b981" />
          <KPI label="AUC" value={adult.auc} color="#10b981" />
          <KPI label="FPR/hour" value={adult.false_positive_rate_per_hour} color="#10b981" />
        </div>
        <p style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>Dataset: {adult.dataset}</p>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 6 }}>Key Advantages:</div>
        <ul style={{ margin: 0, paddingLeft: 16 }}>
          {(adult.key_advantages || []).map((a, i) => (
            <li key={i} style={{ fontSize: 11, color: '#475569', marginBottom: 4, lineHeight: 1.4 }}>{a}</li>
          ))}
        </ul>
      </Card>

      {/* Performance Gap Analysis */}
      <Card title="Performance Gap Analysis" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
          <KPI label="Sensitivity Gap" value={`${gap.sensitivity_gap_pp} pp`} color="#ef4444" />
          <KPI label="Specificity Gap" value={`${gap.specificity_gap_pp} pp`} color="#f59e0b" />
          <KPI label="AUC Gap" value={gap.auc_gap} color="#8b5cf6" />
        </div>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Root Causes:</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {(gap.root_causes || []).map((c, i) => (
            <div key={i} style={{ padding: 10, background: '#fef2f2', borderRadius: 6, border: '1px solid #fecaca',
                                   fontSize: 11, color: '#991b1b', lineHeight: 1.4 }}>
              {c}
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function renderDefinitions(defs) {
  if (!defs || !defs.sections) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {defs.sections.map((sec, si) => (
        <Card key={si} title={sec.title}>
          <div style={{ display: 'grid', gap: 12 }}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#475569', marginTop: 4, lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
