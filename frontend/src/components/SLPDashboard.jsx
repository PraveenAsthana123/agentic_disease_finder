import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#ef4444', '#22c55e', '#f59e0b', '#7c4dff', '#ec4899', '#6366f1', '#14b8a6']
const fmt = v => (typeof v === 'number' ? v.toLocaleString() : v ?? '—')

const cardStyle = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
}

const sectionHeadingStyle = {
  fontSize: 16,
  fontWeight: 600,
  color: '#1e293b',
  marginBottom: 12,
  marginTop: 24,
}

const badgeStyle = (color) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: color,
})

const riskColor = (level) => {
  const s = (level || '').toLowerCase()
  if (s === 'high') return '#ef4444'
  if (s === 'moderate') return '#f59e0b'
  if (s === 'low') return '#22c55e'
  return '#94a3b8'
}

const severityColor = (sev) => {
  const s = (sev || '').toLowerCase()
  if (s === 'severe') return '#ef4444'
  if (s === 'moderate') return '#f59e0b'
  if (s === 'mild') return '#22c55e'
  return '#94a3b8'
}

export default function SLPDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    axios.get(`${API_URL}/api/slp`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Speech-Language Pathology...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data?.available) return <div style={{ padding: 40 }}>No data available</div>

  const summary = data.summary || {}
  const language = data.language_assessment || {}
  const swallowing = data.swallowing || {}
  const aedEffects = data.aed_speech_effects || {}
  const cognitive = data.cognitive_communication || {}
  const goals = data.therapy_goals || {}
  const definitions = data.definitions || []

  // Language assessment bar chart data
  const languageBarData = (language.test_scores || [])

  // Swallowing risk pie chart
  const dysphagiaDistrib = (swallowing.risk_distribution || []).filter(d => d.value > 0)

  // AED speech effects bar chart
  const aedBarData = (aedEffects.medication_rates || [])

  // Cognitive-communication radar data
  const radarData = (cognitive.domain_scores || [])

  // KPI tiles
  const kpis = [
    { label: 'Language Laterality Index', value: summary.language_laterality_index, color: '#1e88e5' },
    { label: 'Mean Naming Score', value: summary.mean_naming_score, color: '#7c4dff' },
    { label: 'Patients with Dysphagia Risk', value: summary.patients_with_dysphagia_risk, color: '#ef4444', urgent: (summary.patients_with_dysphagia_risk || 0) > 0 },
    { label: 'AED Speech Side Effects Rate', value: summary.aed_speech_side_effects_rate, color: '#f59e0b' },
  ]

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          {data.title || 'Speech-Language Pathology'}
        </h1>
        <p style={{ color: '#64748b', marginTop: 4, fontSize: 14 }}>
          {data.subtitle || `Laterality ${fmt(summary.language_laterality_index)} · Mean naming ${fmt(summary.mean_naming_score)} · ${fmt(summary.patients_with_dysphagia_risk)} dysphagia risk · AED effects ${fmt(summary.aed_speech_side_effects_rate)}`}
        </p>
      </div>

      {/* KPI Tiles */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
        {kpis.map((kpi, i) => (
          <div key={i} style={{
            ...cardStyle,
            flex: '1 1 180px',
            borderLeft: `4px solid ${kpi.color}`,
            ...(kpi.urgent ? { border: '2px solid #ef4444', borderLeft: '4px solid #ef4444' } : {}),
          }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{fmt(kpi.value)}</div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{kpi.label}</div>
          </div>
        ))}
      </div>

      {/* Language Assessment */}
      <h2 style={sectionHeadingStyle}>Language Assessment</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {languageBarData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Test Scores by Patient</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={languageBarData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="patient_id" type="category" tick={{ fontSize: 12 }} width={80} />
                <Tooltip />
                <Bar dataKey="boston_naming" name="Boston Naming" fill="#1e88e5" radius={[0, 4, 4, 0]} />
                <Bar dataKey="token_test" name="Token Test" fill="#7c4dff" radius={[0, 4, 4, 0]} />
                <Bar dataKey="verbal_fluency" name="Verbal Fluency" fill="#22c55e" radius={[0, 4, 4, 0]} />
                <Legend />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {(language.lateralization || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Lateralization Results</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Patient</th>
                  <th style={{ padding: '6px 10px' }}>Laterality Index</th>
                  <th style={{ padding: '6px 10px' }}>Dominance</th>
                  <th style={{ padding: '6px 10px' }}>Wada Concordance</th>
                </tr>
              </thead>
              <tbody>
                {language.lateralization.map((pt, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.laterality_index)}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={badgeStyle(pt.dominance === 'Left' ? '#1e88e5' : pt.dominance === 'Right' ? '#ef4444' : '#f59e0b')}>
                        {pt.dominance || '—'}
                      </span>
                    </td>
                    <td style={{ padding: '6px 10px' }}>{pt.wada_concordance || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Swallowing / Dysphagia */}
      <h2 style={sectionHeadingStyle}>Swallowing / Dysphagia</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {dysphagiaDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Risk Level Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={dysphagiaDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {dysphagiaDistrib.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {(swallowing.patients || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Patient Dysphagia Risk Factors</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Patient</th>
                  <th style={{ padding: '6px 10px' }}>Risk Level</th>
                  <th style={{ padding: '6px 10px' }}>Risk Factors</th>
                </tr>
              </thead>
              <tbody>
                {swallowing.patients.map((pt, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={badgeStyle(riskColor(pt.risk_level))}>{pt.risk_level || '—'}</span>
                    </td>
                    <td style={{ padding: '6px 10px', fontSize: 12, color: '#64748b' }}>
                      {(pt.risk_factors || []).slice(0, 4).join(', ') || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* AED Speech Effects */}
      <h2 style={sectionHeadingStyle}>AED Speech Effects</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {aedBarData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Medication vs Speech Side Effect Rate</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={aedBarData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="medication" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="effect_rate" name="Effect Rate (%)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {(aedEffects.details || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Medication Speech Side Effects</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Medication</th>
                  <th style={{ padding: '6px 10px' }}>Effect</th>
                  <th style={{ padding: '6px 10px' }}>Severity</th>
                </tr>
              </thead>
              <tbody>
                {aedEffects.details.map((item, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{item.medication}</td>
                    <td style={{ padding: '6px 10px' }}>{item.effect || '—'}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={badgeStyle(severityColor(item.severity))}>{item.severity || '—'}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Cognitive-Communication */}
      <h2 style={sectionHeadingStyle}>Cognitive-Communication</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {radarData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', minHeight: 320 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Domain Scores</div>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="domain" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar name="Score" dataKey="score" stroke="#7c4dff" fill="#7c4dff" fillOpacity={0.3} />
                <Tooltip />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}

        {(cognitive.patients || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Patient Cognitive-Communication Profiles</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Patient</th>
                  <th style={{ padding: '6px 10px' }}>Word-Finding</th>
                  <th style={{ padding: '6px 10px' }}>Naming</th>
                  <th style={{ padding: '6px 10px' }}>Discourse</th>
                  <th style={{ padding: '6px 10px' }}>Pragmatics</th>
                  <th style={{ padding: '6px 10px' }}>Reading</th>
                  <th style={{ padding: '6px 10px' }}>Writing</th>
                </tr>
              </thead>
              <tbody>
                {cognitive.patients.map((pt, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.word_finding)}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.naming)}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.discourse)}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.pragmatics)}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.reading)}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.writing)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Therapy Goals */}
      <h2 style={sectionHeadingStyle}>Therapy Goals</h2>
      {(goals.items || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Goal</th>
                <th style={{ padding: '8px 12px' }}>Progress</th>
                <th style={{ padding: '8px 12px' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {goals.items.map((g, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{g.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{g.goal || '—'}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{
                        flex: 1,
                        height: 8,
                        background: '#e2e8f0',
                        borderRadius: 4,
                        overflow: 'hidden',
                        minWidth: 80,
                      }}>
                        <div style={{
                          width: `${Math.min(g.progress || 0, 100)}%`,
                          height: '100%',
                          background: (g.progress || 0) >= 80 ? '#22c55e' : (g.progress || 0) >= 50 ? '#f59e0b' : '#ef4444',
                          borderRadius: 4,
                        }} />
                      </div>
                      <span style={{ fontSize: 12, color: '#64748b', minWidth: 36 }}>{fmt(g.progress)}%</span>
                    </div>
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={badgeStyle(
                      (g.status || '').toLowerCase() === 'completed' ? '#22c55e'
                        : (g.status || '').toLowerCase() === 'in progress' ? '#1e88e5'
                        : (g.status || '').toLowerCase() === 'not started' ? '#94a3b8'
                        : '#f59e0b'
                    )}>
                      {g.status || '—'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '\u25BE Hide' : '\u25B8 Show'} SLP Definitions & Methodology
        </button>
        {showDefs && definitions.length > 0 && (
          <div style={{ marginTop: 16, maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Term</th>
                  <th style={{ padding: '6px 10px' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {definitions.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, whiteSpace: 'nowrap' }}>{d.term}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{d.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
