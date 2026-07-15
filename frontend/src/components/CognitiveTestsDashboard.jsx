import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316', '#84cc16', '#14b8a6', '#a855f7']
const DOMAIN_COLORS = {
  'Executive function': '#3b82f6',
  'Working memory': '#8b5cf6',
  'Attention': '#10b981',
  'Processing speed': '#f59e0b',
  'Impulse control': '#ef4444',
  'Sustained attention': '#06b6d4',
  'Visuospatial': '#ec4899',
  'Memory': '#f97316',
  'Language': '#84cc16'
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function CognitiveTestsDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/cognitive-tests/overview`),
      axios.get(`${API_URL}/api/cognitive-tests/breakdown`),
      axios.get(`${API_URL}/api/cognitive-tests/definitions`),
    ]).then(([ov, br, df]) => {
      setOverview(ov.data)
      setBreakdown(br.data)
      setDefinitions(df.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading cognitive tests data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: '#1e293b' }}>Digital Cognitive Tests</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        Neuropsychological test battery results: Stroop, Trail Making, Digit Span, WCST, N-Back, Go/No-Go, CPT, Clock Drawing, RAVLT, Verbal Fluency
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderBottom: '2px solid #e2e8f0' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 20px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#fff' : 'transparent',
            color: tab === t.id ? '#2563eb' : '#64748b',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const testDist = Object.entries(data.test_distribution || {}).map(([k, v]) => ({ name: k, count: v }))
  const domainDist = Object.entries(data.domain_distribution || {}).map(([k, v]) => ({ name: k, count: v }))
  const adminDist = Object.entries(data.admin_distribution || {}).map(([k, v]) => ({ name: k, count: v }))
  const domainAccuracy = (data.domain_accuracy || []).map(d => ({ ...d, fullMark: 100 }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI row */}
      <Card><KPI label="Total Tests Administered" value={data.total_tests} color="#3b82f6" /></Card>
      <Card><KPI label="Patients Tested" value={data.total_patients} color="#8b5cf6" /></Card>
      <Card><KPI label="Test Types" value={data.total_test_types} color="#10b981" /></Card>
      <Card><KPI label="Avg Accuracy %" value={`${data.avg_accuracy}%`} color={data.avg_accuracy >= 70 ? '#10b981' : '#f59e0b'} /></Card>

      {/* Domain accuracy radar */}
      <Card title="Domain Accuracy (Radar)" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={domainAccuracy}>
            <PolarGrid />
            <PolarAngleAxis dataKey="domain" tick={{ fontSize: 10 }} />
            <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9 }} />
            <Radar name="Avg Accuracy" dataKey="avg_accuracy" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      {/* Test performance bar chart */}
      <Card title="Average Accuracy by Test" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data.test_performance} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="test_name" tick={{ fontSize: 11 }} width={120} />
            <Tooltip />
            <Bar dataKey="avg_accuracy" name="Accuracy %">
              {(data.test_performance || []).map((entry, i) => (
                <Cell key={i} fill={DOMAIN_COLORS[entry.domain] || COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Test distribution pie */}
      <Card title="Tests Administered (Distribution)" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={testDist} dataKey="count" nameKey="name" cx="50%" cy="50%"
              outerRadius={100} label={({ name, percent }) => `${name.split(' ')[0]} ${(percent * 100).toFixed(0)}%`}
              labelLine={{ strokeWidth: 1 }} style={{ fontSize: 10 }}>
              {testDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Domain distribution */}
      <Card title="Tests by Cognitive Domain" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={domainDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10, angle: -30 }} height={60} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" name="Tests">
              {domainDist.map((entry, i) => (
                <Cell key={i} fill={DOMAIN_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Monthly volume trend */}
      <Card title="Monthly Testing Volume" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.monthly_volume}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Tests" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Administrator distribution */}
      <Card title="Tests by Administrator" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={adminDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" name="Tests Administered" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Test performance table */}
      <Card title="Test Performance Summary" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Test</th>
                <th style={{ padding: '8px 12px', textAlign: 'right' }}>Avg Score</th>
                <th style={{ padding: '8px 12px', textAlign: 'right' }}>Max Score</th>
                <th style={{ padding: '8px 12px', textAlign: 'right' }}>Avg Accuracy</th>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Performance</th>
              </tr>
            </thead>
            <tbody>
              {(data.test_performance || []).map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 500 }}>{t.test_name}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right' }}>{t.avg_score}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', color: '#64748b' }}>{t.max_score}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600,
                    color: t.avg_accuracy >= 75 ? '#10b981' : t.avg_accuracy >= 60 ? '#f59e0b' : '#ef4444'
                  }}>{t.avg_accuracy}%</td>
                  <td style={{ padding: '8px 12px' }}>
                    <div style={{ background: '#e2e8f0', borderRadius: 4, height: 8, width: 120 }}>
                      <div style={{
                        background: t.avg_accuracy >= 75 ? '#10b981' : t.avg_accuracy >= 60 ? '#f59e0b' : '#ef4444',
                        borderRadius: 4, height: 8, width: `${Math.min(t.avg_accuracy, 100) * 1.2}px`
                      }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* Per-patient summary */}
      <Card title="Per-Patient Summary" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Patient</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Total Tests</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Types Taken</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Avg Accuracy</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Performance</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>First Test</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Last Test</th>
              </tr>
            </thead>
            <tbody>
              {(data.per_patient || []).map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.total}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.tests_taken}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600,
                    color: p.avg_accuracy >= 75 ? '#10b981' : p.avg_accuracy >= 60 ? '#f59e0b' : '#ef4444'
                  }}>{p.avg_accuracy}%</td>
                  <td style={{ padding: '6px 10px' }}>
                    <div style={{ background: '#e2e8f0', borderRadius: 4, height: 8, width: 100 }}>
                      <div style={{
                        background: p.avg_accuracy >= 75 ? '#10b981' : p.avg_accuracy >= 60 ? '#f59e0b' : '#ef4444',
                        borderRadius: 4, height: 8, width: `${Math.min(p.avg_accuracy, 100)}px`
                      }} />
                    </div>
                  </td>
                  <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 11 }}>{p.first_test}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 11 }}>{p.last_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Per-test detail */}
      <Card title="Per-Test Detail" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Test</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Domain</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Administrations</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Avg Accuracy</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Min</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Max</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Avg RT (ms)</th>
              </tr>
            </thead>
            <tbody>
              {(data.test_detail || []).map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{t.test_name}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={{
                      background: (DOMAIN_COLORS[t.domain] || '#64748b') + '20',
                      color: DOMAIN_COLORS[t.domain] || '#64748b',
                      padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600
                    }}>{t.domain}</span>
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{t.administrations}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600,
                    color: t.avg_accuracy >= 75 ? '#10b981' : t.avg_accuracy >= 60 ? '#f59e0b' : '#ef4444'
                  }}>{t.avg_accuracy}%</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#ef4444' }}>{t.min_accuracy}%</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#10b981' }}>{t.max_accuracy}%</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#64748b' }}>
                    {t.avg_rt_ms ? `${t.avg_rt_ms}` : '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Low performers */}
      {(data.low_performers || []).length > 0 && (
        <Card title="Low Performers (Avg Accuracy < 55%)" span={2}>
          <div style={{ background: '#fef2f2', borderRadius: 8, padding: 12 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Patient</th>
                  <th style={{ padding: '4px 8px', textAlign: 'right' }}>Avg Accuracy</th>
                  <th style={{ padding: '4px 8px', textAlign: 'right' }}>Tests</th>
                </tr>
              </thead>
              <tbody>
                {data.low_performers.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #fecaca' }}>
                    <td style={{ padding: '4px 8px', fontWeight: 500 }}>{p.patient_id}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right', color: '#ef4444', fontWeight: 600 }}>{p.avg_accuracy}%</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{p.total}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* High performers */}
      {(data.high_performers || []).length > 0 && (
        <Card title="High Performers (Avg Accuracy >= 80%)" span={2}>
          <div style={{ background: '#f0fdf4', borderRadius: 8, padding: 12 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Patient</th>
                  <th style={{ padding: '4px 8px', textAlign: 'right' }}>Avg Accuracy</th>
                  <th style={{ padding: '4px 8px', textAlign: 'right' }}>Tests</th>
                </tr>
              </thead>
              <tbody>
                {data.high_performers.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #bbf7d0' }}>
                    <td style={{ padding: '4px 8px', fontWeight: 500 }}>{p.patient_id}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right', color: '#10b981', fontWeight: 600 }}>{p.avg_accuracy}%</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{p.total}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Recent tests */}
      <Card title="Recent Test Results" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Date</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Patient</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Test</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Domain</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Score</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Accuracy</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>RT (ms)</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>By</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Notes</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent_tests || []).map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '6px 8px', color: '#64748b', fontSize: 11 }}>{t.administered_at}</td>
                  <td style={{ padding: '6px 8px', fontWeight: 500 }}>{t.patient_id}</td>
                  <td style={{ padding: '6px 8px' }}>{t.test_name}</td>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={{
                      background: (DOMAIN_COLORS[t.domain] || '#64748b') + '20',
                      color: DOMAIN_COLORS[t.domain] || '#64748b',
                      padding: '1px 6px', borderRadius: 8, fontSize: 10, fontWeight: 600
                    }}>{t.domain}</span>
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}>{t.score}/{t.max_score}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', fontWeight: 600,
                    color: t.accuracy_pct >= 75 ? '#10b981' : t.accuracy_pct >= 60 ? '#f59e0b' : '#ef4444'
                  }}>{t.accuracy_pct}%</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: '#64748b' }}>
                    {t.reaction_time_ms || '--'}
                  </td>
                  <td style={{ padding: '6px 8px', color: '#64748b', fontSize: 11 }}>{t.administered_by}</td>
                  <td style={{ padding: '6px 8px', color: '#94a3b8', fontSize: 11, fontStyle: t.notes ? 'italic' : 'normal' }}>
                    {t.notes || '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Clinical notes */}
      {(data.clinical_notes || []).length > 0 && (
        <Card title="Clinical Observations (from test notes)" span={4}>
          <div style={{ background: '#fffbeb', borderRadius: 8, padding: 12 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Date</th>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Patient</th>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Test</th>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Note</th>
                </tr>
              </thead>
              <tbody>
                {data.clinical_notes.map((n, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #fde68a' }}>
                    <td style={{ padding: '4px 8px', color: '#64748b', fontSize: 11 }}>{n.administered_at}</td>
                    <td style={{ padding: '4px 8px', fontWeight: 500 }}>{n.patient_id}</td>
                    <td style={{ padding: '4px 8px' }}>{n.test_name}</td>
                    <td style={{ padding: '4px 8px', fontStyle: 'italic' }}>{n.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* Test descriptions */}
      <Card title="Cognitive Test Battery" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {(data.tests || []).map((t, i) => (
            <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{t.name}</div>
              <span style={{
                background: (DOMAIN_COLORS[t.domain] || '#64748b') + '20',
                color: DOMAIN_COLORS[t.domain] || '#64748b',
                padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600
              }}>{t.domain}</span>
              <p style={{ fontSize: 12, color: '#475569', margin: '8px 0 4px' }}>{t.description}</p>
              <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}><strong>Scoring:</strong> {t.scoring}</p>
              <p style={{ fontSize: 11, color: '#8b5cf6', margin: '2px 0' }}><strong>Clinical:</strong> {t.clinical_relevance}</p>
            </div>
          ))}
        </div>
      </Card>

      {/* Domains */}
      <Card title="Cognitive Domains" span={2}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {Object.entries(data.domains || {}).map(([domain, desc], i) => (
            <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 6, borderLeft: `3px solid ${DOMAIN_COLORS[domain] || '#64748b'}` }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: DOMAIN_COLORS[domain] || '#334155' }}>{domain}</div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{desc}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Glossary */}
      <Card title="Glossary" span={2}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {(data.glossary || []).map((g, i) => (
            <div key={i} style={{ padding: '6px 10px', background: i % 2 === 0 ? '#f8fafc' : '#fff', borderRadius: 4 }}>
              <span style={{ fontWeight: 700, fontSize: 12, color: '#334155' }}>{g.term}</span>
              <span style={{ fontSize: 12, color: '#64748b' }}> — {g.definition}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Clinical notes */}
      <Card title="AED Cognitive Effects" span={2}>
        <div style={{ background: '#fef2f2', borderRadius: 8, padding: 14 }}>
          <p style={{ fontSize: 12, color: '#991b1b', margin: 0 }}>{data.clinical_notes?.aed_effects}</p>
        </div>
      </Card>

      <Card title="Testing Protocol Notes" span={2}>
        <div style={{ background: '#eff6ff', borderRadius: 8, padding: 14 }}>
          <p style={{ fontSize: 12, color: '#1e40af', margin: '0 0 8px' }}><strong>Timing:</strong> {data.clinical_notes?.testing_timing}</p>
          <p style={{ fontSize: 12, color: '#1e40af', margin: 0 }}><strong>Epilepsy Patterns:</strong> {data.clinical_notes?.epilepsy_patterns}</p>
        </div>
      </Card>
    </div>
  )
}
