import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const TIER_COLORS = { excellent: '#10b981', good: '#3b82f6', adequate: '#f59e0b', needs_improvement: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function ResponseQualityDashboard() {
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
          axios.get(`${API_URL}/api/response-quality/overview`),
          axios.get(`${API_URL}/api/response-quality/breakdown`),
          axios.get(`${API_URL}/api/response-quality/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load response quality data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9733;</div>
      Loading response quality data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Response quality data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const tierDist = overview.quality_tier_distribution || []
  const confDist = overview.confidence_distribution || []
  const timeline = overview.daily_quality_timeline || []
  const compActivity = overview.component_activity || []
  const recentResp = breakdown?.recent_responses || []
  const diseaseQuality = breakdown?.disease_analysis_quality || []
  const compReliability = breakdown?.component_reliability || []
  const ragCov = breakdown?.rag_coverage || {}
  const feedbackDetail = breakdown?.feedback_detail || []

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'responses', label: 'Responses' },
    { id: 'analysis', label: 'Analysis Quality' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const card = (title, children, span) => (
    <div style={{ background: '#fff', borderRadius: 10, padding: 18, boxShadow: '0 1px 4px rgba(0,0,0,.08)', gridColumn: span || 'auto' }}>
      <div style={{ fontWeight: 600, fontSize: 13, color: '#475569', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '.5px' }}>{title}</div>
      {children}
    </div>
  )

  const kpi = (label, value, sub) => (
    <div style={{ textAlign: 'center', padding: '8px 0' }}>
      <div style={{ fontSize: 26, fontWeight: 700, color: '#0f172a' }}>{fmt(value)}</div>
      <div style={{ fontSize: 11, color: '#64748b' }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {card('Quality KPIs', (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
          {kpi('Avg Quality', s.avg_quality_score, '/100')}
          {kpi('Median Quality', s.median_quality_score, '/100')}
          {kpi('Total Responses', s.total_responses)}
          {kpi('Total Queries', s.total_queries)}
        </div>
      ), 'span 2')}

      {card('Content Metrics', (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
          {kpi('Avg Words', s.avg_word_count, 'per response')}
          {kpi('Structure', s.structure_rate_pct + '%', 'formatted')}
          {kpi('Data Rate', s.data_inclusion_rate_pct + '%', 'with numbers')}
          {kpi('Citations', s.citation_rate_pct + '%', 'referenced')}
        </div>
      ), 'span 2')}

      {card('Quality Tier Distribution', (
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={tierDist.filter(d => d.count > 0)} dataKey="count" nameKey="tier" cx="50%" cy="50%" outerRadius={75} label={({ tier, count }) => `${tier}: ${count}`}>
              {tierDist.filter(d => d.count > 0).map((d, i) => (
                <Cell key={i} fill={TIER_COLORS[d.tier] || COLORS[i]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      ))}

      {card('Confidence Distribution', confDist.length > 0 ? (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={confDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      ) : <div style={{ color: '#94a3b8', fontSize: 12, padding: 20 }}>No confidence data</div>)}

      {card('Daily Quality Timeline', timeline.length > 0 ? (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="avg_quality" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Avg Quality" />
          </LineChart>
        </ResponsiveContainer>
      ) : <div style={{ color: '#94a3b8', fontSize: 12, padding: 20 }}>No timeline data</div>, 'span 2')}

      {card('Component Activity', (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={compActivity} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="component" type="category" tick={{ fontSize: 10 }} width={100} />
            <Tooltip />
            <Bar dataKey="transactions" fill="#06b6d4" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      ))}

      {card('Analysis & Feedback', (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
          {kpi('Analyses', s.total_analyses)}
          {kpi('Avg Confidence', s.avg_confidence)}
          {kpi('Feedback', s.feedback_count, s.feedback_avg_rating ? `Avg: ${s.feedback_avg_rating}/5` : 'No ratings')}
        </div>
      ))}
    </div>
  )

  const renderResponses = () => (
    <div style={{ display: 'grid', gap: 16 }}>
      {card('Recent Responses (Quality Scored)', (
        <div style={{ maxHeight: 500, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Time</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Score</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Tier</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Words</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Struct</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Data</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Preview</th>
              </tr>
            </thead>
            <tbody>
              {recentResp.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', whiteSpace: 'nowrap', color: '#64748b' }}>{r.timestamp ? r.timestamp.slice(0, 16).replace('T', ' ') : '--'}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{r.quality_score}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 600, background: (TIER_COLORS[r.tier] || '#94a3b8') + '20', color: TIER_COLORS[r.tier] || '#64748b' }}>
                      {r.tier?.replace('_', ' ')}
                    </span>
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.word_count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.has_structure ? '\u2713' : '\u2717'}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.has_data ? '\u2713' : '\u2717'}</td>
                  <td style={{ padding: '6px 10px', color: '#475569', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.preview}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      {card('Component Reliability', (
        <div style={{ maxHeight: 400, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Component</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Total Txns</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Top Actions</th>
              </tr>
            </thead>
            <tbody>
              {compReliability.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{c.component}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{c.total}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>
                    {(c.actions || []).map(a => `${a.action}(${a.count})`).join(', ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}

      {ragCov.available && card('RAG Vector DB Coverage', (
        <div>
          <div style={{ display: 'flex', gap: 24, marginBottom: 12 }}>
            {kpi('Collections', ragCov.total_collections)}
            {kpi('Embeddings', ragCov.total_embeddings)}
          </div>
          {(ragCov.collections || []).map((c, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
              <span style={{ color: '#334155' }}>{c.collection}</span>
              <span style={{ fontWeight: 600 }}>{c.documents} docs</span>
            </div>
          ))}
        </div>
      ))}

      {feedbackDetail.length > 0 && card('Feedback Detail', (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Role</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Rating</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Correction?</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Date</th>
            </tr>
          </thead>
          <tbody>
            {feedbackDetail.map((f, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px' }}>{f.patient_id}</td>
                <td style={{ padding: '6px 10px' }}>{f.reviewer_role}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{f.rating || '--'}/5</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{f.has_correction ? '\u2713' : '\u2717'}</td>
                <td style={{ padding: '6px 10px', color: '#64748b' }}>{f.created_at || '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </div>
  )

  const renderAnalysis = () => (
    <div style={{ display: 'grid', gap: 16 }}>
      {card('Per-Disease Analysis Quality', diseaseQuality.length > 0 ? (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Disease</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Count</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Avg Confidence</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Min</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Max</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Signal Quality</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Labels</th>
            </tr>
          </thead>
          <tbody>
            {diseaseQuality.map((d, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 500 }}>{d.disease}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{d.count}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{d.avg_confidence}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', color: '#64748b' }}>{d.min_confidence}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', color: '#64748b' }}>{d.max_confidence}</td>
                <td style={{ padding: '6px 10px' }}>{Object.entries(d.signal_quality || {}).map(([k, v]) => `${k}(${v})`).join(', ')}</td>
                <td style={{ padding: '6px 10px' }}>{Object.entries(d.labels || {}).map(([k, v]) => `${k}(${v})`).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <div style={{ color: '#94a3b8', fontSize: 12 }}>No analysis data</div>)}
    </div>
  )

  const renderDefinitions = () => (
    <div style={{ display: 'grid', gap: 16 }}>
      {card('Response Quality Metric Definitions', (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Metric</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Description</th>
              <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Range</th>
            </tr>
          </thead>
          <tbody>
            {(defs?.metrics || []).map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, whiteSpace: 'nowrap' }}>{m.name}</td>
                <td style={{ padding: '6px 10px', color: '#475569' }}>{m.description}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', color: '#64748b', whiteSpace: 'nowrap' }}>{m.range}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </div>
  )

  return (
    <div style={{ padding: 20 }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', margin: 0 }}>Response Quality Dashboard</h2>
        <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0 0' }}>
          Real conversation analytics — {s.total_responses} responses, {s.total_analyses} analyses, {s.feedback_count} feedback entries
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 18, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', fontSize: 12, fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'responses' && renderResponses()}
      {tab === 'analysis' && renderAnalysis()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
