import React, { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, Treemap
} from 'recharts'

const API_URL = '/api'

const ONSET_COLORS = {
  focal: '#1e88e5',
  generalized: '#ef4444',
  unknown: '#94a3b8',
}

const SUBTYPE_COLORS = [
  '#1e88e5', '#ef4444', '#22c55e', '#f59e0b', '#7c4dff',
  '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#64748b',
]

const CONFIDENCE_COLORS = { high: '#22c55e', moderate: '#f59e0b', low: '#ef4444' }

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

export default function ILAEClassificationDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedSubject, setSelectedSubject] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/ilae-classification`)
        setData(res.data)
      } catch (e) {
        setError(e.message || 'Failed to load ILAE classification data')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  // Derive KPIs
  const kpis = useMemo(() => {
    if (!data || !data.available) return null
    const focalPct = data.onset_distribution?.find(d => d.type === 'focal')?.percent || 0
    const genPct = data.onset_distribution?.find(d => d.type === 'generalized')?.percent || 0
    const highConf = data.confidence_distribution?.find(d => d.level === 'high')?.count || 0
    const totalConf = data.confidence_distribution?.reduce((s, d) => s + d.count, 0) || 1
    return {
      totalSeizures: data.total_seizures,
      totalSubjects: data.total_subjects,
      focalPct,
      genPct,
      highConfPct: Math.round(100 * highConf / totalConf),
      topSubtype: data.subtype_distribution?.[0]?.subtype || '-',
    }
  }, [data])

  // Per-subject bar chart data
  const subjectChart = useMemo(() => {
    if (!data?.subjects) return []
    return data.subjects.map(s => ({
      subject: s.subject.replace('chb', 'CHB-'),
      subjectId: s.subject,
      focal: s.onset_type_counts?.focal || 0,
      generalized: s.onset_type_counts?.generalized || 0,
      unknown: s.onset_type_counts?.unknown || 0,
      total: s.total_seizures,
    }))
  }, [data])

  // Selected subject's seizures
  const selectedSeizures = useMemo(() => {
    if (!selectedSubject || !data?.classifications) return []
    return data.classifications.filter(c => c.subject === selectedSubject)
  }, [selectedSubject, data])

  if (loading) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>
      Loading ILAE Seizure Classification...
    </div>
  )

  if (error) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>
      Error: {error}
    </div>
  )

  if (!data?.available) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>
      {data?.error || 'ILAE classification data not available'}
    </div>
  )

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          ILAE 2017 Seizure Classification
        </h2>
        <p style={{ color: '#64748b', fontSize: 13, marginTop: 4 }}>
          {data.reference} &bull; {data.data_source} &bull; {data.total_seizures} seizures from {data.total_subjects} subjects
        </p>
      </div>

      {/* KPI Tiles */}
      {kpis && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 24 }}>
          {[
            { label: 'Total Seizures', value: kpis.totalSeizures, color: '#1e88e5', icon: '⚡' },
            { label: 'Subjects', value: kpis.totalSubjects, color: '#7c4dff', icon: '👤' },
            { label: 'Focal Onset', value: `${kpis.focalPct}%`, color: '#1e88e5', icon: '🎯' },
            { label: 'Generalized', value: `${kpis.genPct}%`, color: '#ef4444', icon: '🌐' },
            { label: 'High Confidence', value: `${kpis.highConfPct}%`, color: '#22c55e', icon: '✓' },
            { label: 'Top Subtype', value: kpis.topSubtype, color: '#f59e0b', icon: '📊' },
          ].map((kpi, i) => (
            <div key={i} style={{
              ...cardStyle,
              borderLeft: `4px solid ${kpi.color}`,
              display: 'flex', alignItems: 'center', gap: 12,
            }}>
              <span style={{ fontSize: 28 }}>{kpi.icon}</span>
              <div>
                <div style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase' }}>{kpi.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#0f172a' }}>{kpi.value}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Row: Onset Distribution Pie + Subtype Bar */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        {/* Onset Type Pie */}
        <div style={cardStyle}>
          <h3 style={sectionHeadingStyle}>Onset Type Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={data.onset_distribution || []}
                dataKey="count"
                nameKey="label"
                cx="50%"
                cy="50%"
                outerRadius={100}
                innerRadius={50}
                label={({ label, percent }) => `${label} (${percent}%)`}
              >
                {(data.onset_distribution || []).map((d, i) => (
                  <Cell key={i} fill={ONSET_COLORS[d.type] || '#94a3b8'} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Subtype Distribution Bar */}
        <div style={cardStyle}>
          <h3 style={sectionHeadingStyle}>Seizure Subtype Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.subtype_distribution || []} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="subtype" type="category" width={120} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#7c4dff" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Row: Confidence Distribution + ILAE Taxonomy */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        {/* Confidence */}
        <div style={cardStyle}>
          <h3 style={sectionHeadingStyle}>Classification Confidence</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data.confidence_distribution || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {(data.confidence_distribution || []).map((d, i) => (
                  <Cell key={i} fill={CONFIDENCE_COLORS[d.level] || '#94a3b8'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Taxonomy Reference */}
        <div style={cardStyle}>
          <h3 style={sectionHeadingStyle}>ILAE 2017 Classification Framework</h3>
          <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.6 }}>
            {data.taxonomy && Object.entries(data.taxonomy).map(([key, val]) => (
              <div key={key} style={{ marginBottom: 12 }}>
                <div style={{
                  fontWeight: 700,
                  color: ONSET_COLORS[key] || '#0f172a',
                  fontSize: 13,
                }}>{val.label}</div>
                <div style={{ color: '#64748b', marginBottom: 4 }}>{val.description}</div>
                {val.motor_subtypes && (
                  <div><strong>Motor:</strong> {val.motor_subtypes.join(', ')}</div>
                )}
                {val.non_motor_subtypes && (
                  <div><strong>Non-motor:</strong> {val.non_motor_subtypes.join(', ')}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Per-Subject Stacked Bar Chart */}
      <div style={{ ...cardStyle, marginBottom: 24 }}>
        <h3 style={sectionHeadingStyle}>Per-Subject Seizure Classification</h3>
        <p style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
          Click a bar to view individual seizure details for that subject.
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={subjectChart}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="subject" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
            <YAxis label={{ value: 'Seizure Count', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar
              dataKey="focal" stackId="a" fill={ONSET_COLORS.focal} name="Focal"
              onClick={(d) => d && setSelectedSubject(d.subjectId)}
              cursor="pointer"
            />
            <Bar
              dataKey="generalized" stackId="a" fill={ONSET_COLORS.generalized} name="Generalized"
              onClick={(d) => d && setSelectedSubject(d.subjectId)}
              cursor="pointer"
            />
            <Bar
              dataKey="unknown" stackId="a" fill={ONSET_COLORS.unknown} name="Unknown"
              onClick={(d) => d && setSelectedSubject(d.subjectId)}
              cursor="pointer"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Selected Subject Detail Table */}
      {selectedSubject && selectedSeizures.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 24 }}>
          <h3 style={sectionHeadingStyle}>
            Seizure Details — {selectedSubject.replace('chb', 'CHB-')}
            <span
              style={{ fontSize: 12, color: '#1e88e5', marginLeft: 12, cursor: 'pointer' }}
              onClick={() => setSelectedSubject(null)}
            >
              ✕ close
            </span>
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>File</th>
                  <th style={{ padding: '8px 12px' }}>Start (s)</th>
                  <th style={{ padding: '8px 12px' }}>Duration</th>
                  <th style={{ padding: '8px 12px' }}>Onset</th>
                  <th style={{ padding: '8px 12px' }}>Level 2</th>
                  <th style={{ padding: '8px 12px' }}>Subtype</th>
                  <th style={{ padding: '8px 12px' }}>Confidence</th>
                  <th style={{ padding: '8px 12px' }}>Ch. Spread</th>
                  <th style={{ padding: '8px 12px' }}>Lat. Index</th>
                  <th style={{ padding: '8px 12px' }}>Reasoning</th>
                </tr>
              </thead>
              <tbody>
                {selectedSeizures.map((sz, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace' }}>{sz.file}</td>
                    <td style={{ padding: '8px 12px' }}>{sz.start_sec}</td>
                    <td style={{ padding: '8px 12px' }}>{sz.duration_sec}s</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                        background: ONSET_COLORS[sz.classification.onset_type] + '20',
                        color: ONSET_COLORS[sz.classification.onset_type],
                      }}>
                        {sz.classification.onset_type}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>{sz.classification.level2 || '-'}</td>
                    <td style={{ padding: '8px 12px' }}>{sz.classification.level3_subtype || '-'}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 12, fontSize: 11,
                        background: CONFIDENCE_COLORS[sz.classification.confidence] + '20',
                        color: CONFIDENCE_COLORS[sz.classification.confidence],
                      }}>
                        {sz.classification.confidence}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>{sz.features.channel_spread}</td>
                    <td style={{ padding: '8px 12px' }}>{sz.features.lateralization_index}</td>
                    <td style={{ padding: '8px 12px', maxWidth: 250, fontSize: 11, color: '#64748b' }}>
                      {sz.classification.reasoning}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
