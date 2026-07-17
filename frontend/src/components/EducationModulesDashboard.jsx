import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line
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

function Badge({ text, color }) {
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
      background: color + '20', color: color
    }}>{text}</span>
  )
}

function completionColor(pct) {
  if (pct >= 80) return '#10b981'
  if (pct >= 50) return '#f59e0b'
  if (pct >= 25) return '#3b82f6'
  return '#ef4444'
}

function quizColor(score) {
  if (score >= 90) return '#10b981'
  if (score >= 70) return '#3b82f6'
  if (score >= 60) return '#f59e0b'
  return '#ef4444'
}

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const FORMAT_COLORS = { video: '#3b82f6', article: '#10b981', quiz: '#f59e0b', interactive: '#8b5cf6' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EducationModulesDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/education-modules/overview`),
      axios.get(`${API_URL}/api/education-modules/breakdown`),
      axios.get(`${API_URL}/api/education-modules/definitions`),
    ]).then(([o, b, d]) => {
      setOv(o.data); setBd(b.data); setDefs(d.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading education modules data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Education Modules Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Patient education analytics — {ov?.total_enrollments} enrollments, {ov?.total_patients} patients, {ov?.total_modules} modules
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t.id ? '#1e293b' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && ov && renderOverview(ov)}
      {tab === 'breakdown' && bd && renderBreakdown(bd)}
      {tab === 'definitions' && defs && renderDefinitions(defs)}
    </div>
  )
}

function renderOverview(ov) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {/* KPI Row */}
      <Card title="Enrollments" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
          <KPI label="Total Enrollments" value={ov.total_enrollments} />
          <KPI label="Patients" value={ov.total_patients} />
          <KPI label="Modules" value={ov.total_modules} />
          <KPI label="Completed" value={ov.completed_count} color="#10b981" />
          <KPI label="In Progress" value={ov.in_progress} color="#3b82f6" />
          <KPI label="Not Started" value={ov.not_started} color="#ef4444" />
          <KPI label="Completion Rate" value={`${ov.completion_rate}%`} color={completionColor(ov.completion_rate)} />
          <KPI label="Avg Completion" value={`${ov.avg_completion}%`} />
          <KPI label="Avg Time" value={`${ov.avg_time_minutes} min`} />
          <KPI label="Total Time" value={`${ov.total_time_hours} hrs`} />
          <KPI label="Avg Quiz Score" value={ov.avg_quiz_score} color={quizColor(ov.avg_quiz_score)} />
          <KPI label="Quiz Pass Rate" value={`${ov.quiz_pass_rate}%`} sub={`${ov.quiz_taken} quizzes taken`} />
        </div>
      </Card>

      {/* Format Distribution Pie */}
      <Card title="Format Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={ov.format_distribution} dataKey="cnt" nameKey="format" cx="50%" cy="50%" outerRadius={80} label={({ format, cnt }) => `${format}: ${cnt}`}>
              {ov.format_distribution.map((f, i) => (
                <Cell key={i} fill={FORMAT_COLORS[f.format] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Completion Distribution Bar */}
      <Card title="Completion Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.completion_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="cnt" name="Count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Quiz Score Distribution */}
      <Card title="Quiz Score Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={ov.quiz_score_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="cnt" name="Count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Module Distribution Bar */}
      <Card title="Enrollments by Module" span={2}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={ov.module_distribution} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="module_name" type="category" width={160} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="cnt" name="Enrolled" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            <Bar dataKey="completed" name="Completed" fill="#10b981" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Monthly Trend */}
      <Card title="Monthly Enrollment Trend" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={ov.monthly_trend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="enrollments" stroke="#3b82f6" name="Enrollments" strokeWidth={2} />
            <Line yAxisId="left" type="monotone" dataKey="completed" stroke="#10b981" name="Completed" strokeWidth={2} />
            <Line yAxisId="right" type="monotone" dataKey="avg_completion" stroke="#f59e0b" name="Avg Completion %" strokeWidth={2} strokeDasharray="5 5" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Module Performance Table */}
      <Card title="Module Performance Summary" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Module', 'Enrolled', 'Completed', 'Avg Completion', 'Avg Time (min)', 'Format Avg Completion'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ov.module_distribution.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{m.module_name}</td>
                  <td style={{ padding: '6px 10px' }}>{m.cnt}</td>
                  <td style={{ padding: '6px 10px' }}>{m.completed}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={`${m.avg_comp}%`} color={completionColor(m.avg_comp)} /></td>
                  <td style={{ padding: '6px 10px' }}>{m.avg_time}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <div style={{ background: '#e2e8f0', borderRadius: 4, height: 8, width: 100 }}>
                      <div style={{ background: completionColor(m.avg_comp), borderRadius: 4, height: 8, width: `${m.avg_comp}%` }} />
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

function renderBreakdown(bd) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {/* Per-Patient Summary */}
      <Card title="Per-Patient Progress" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient', 'Modules', 'Completed', 'Avg Completion', 'Avg Time', 'Total Time', 'Avg Quiz'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bd.per_patient.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500, fontFamily: 'monospace' }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 10px' }}>{p.total_modules}</td>
                  <td style={{ padding: '6px 10px' }}>{p.completed}/{p.total_modules}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={`${p.avg_completion}%`} color={completionColor(p.avg_completion)} /></td>
                  <td style={{ padding: '6px 10px' }}>{p.avg_time} min</td>
                  <td style={{ padding: '6px 10px' }}>{p.total_time} min</td>
                  <td style={{ padding: '6px 10px' }}>
                    {p.avg_quiz != null ? <Badge text={p.avg_quiz} color={quizColor(p.avg_quiz)} /> : <span style={{ color: '#94a3b8' }}>--</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Per-Module Completion Rates */}
      <Card title="Module Completion Rates" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Module', 'Enrolled', 'Completed', 'Rate', 'Avg Progress', 'Avg Time', 'Avg Quiz', 'Quizzes'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bd.per_module.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{m.module_name}</td>
                  <td style={{ padding: '6px 10px' }}>{m.enrolled}</td>
                  <td style={{ padding: '6px 10px' }}>{m.completed}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={`${m.completion_rate}%`} color={completionColor(m.completion_rate)} /></td>
                  <td style={{ padding: '6px 10px' }}>{m.avg_progress}%</td>
                  <td style={{ padding: '6px 10px' }}>{m.avg_time} min</td>
                  <td style={{ padding: '6px 10px' }}>
                    {m.avg_quiz != null ? <Badge text={m.avg_quiz} color={quizColor(m.avg_quiz)} /> : <span style={{ color: '#94a3b8' }}>--</span>}
                  </td>
                  <td style={{ padding: '6px 10px' }}>{m.quiz_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Low Engagement Alert */}
      {bd.low_engagement.length > 0 && (
        <Card title="Low Engagement Alert">
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: 12 }}>
            <div style={{ fontWeight: 600, color: '#991b1b', fontSize: 12, marginBottom: 8 }}>
              Patients with &lt;30% avg completion
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  {['Patient', 'Modules', 'Avg Completion', 'Not Started'].map(h => (
                    <th key={h} style={{ padding: '4px 8px', textAlign: 'left', borderBottom: '1px solid #fecaca', fontWeight: 600, color: '#991b1b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {bd.low_engagement.map((p, i) => (
                  <tr key={i}>
                    <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{p.patient_id}</td>
                    <td style={{ padding: '4px 8px' }}>{p.modules}</td>
                    <td style={{ padding: '4px 8px' }}><Badge text={`${p.avg_completion}%`} color="#ef4444" /></td>
                    <td style={{ padding: '4px 8px' }}>{p.not_started}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Top Quiz Performers */}
      <Card title="Top Quiz Performers" span={bd.low_engagement.length > 0 ? 1 : 1}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Patient', 'Module', 'Score', 'Format'].map(h => (
                <th key={h} style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {bd.top_quiz_performers.map((q, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{q.patient_id}</td>
                <td style={{ padding: '4px 8px' }}>{q.module_name}</td>
                <td style={{ padding: '4px 8px' }}><Badge text={q.quiz_score?.toFixed(1)} color={quizColor(q.quiz_score)} /></td>
                <td style={{ padding: '4px 8px' }}><Badge text={q.format} color={FORMAT_COLORS[q.format] || '#64748b'} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Recent Enrollments */}
      <Card title="Recent Enrollments" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient', 'Module', 'Format', 'Completion', 'Quiz Score', 'Time (min)', 'Started', 'Completed'].map(h => (
                  <th key={h} style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bd.recent_enrollments.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '4px 10px', fontFamily: 'monospace' }}>{r.patient_id}</td>
                  <td style={{ padding: '4px 10px' }}>{r.module_name}</td>
                  <td style={{ padding: '4px 10px' }}><Badge text={r.format} color={FORMAT_COLORS[r.format] || '#64748b'} /></td>
                  <td style={{ padding: '4px 10px' }}><Badge text={`${r.completion_pct}%`} color={completionColor(r.completion_pct)} /></td>
                  <td style={{ padding: '4px 10px' }}>
                    {r.quiz_score != null ? <Badge text={r.quiz_score.toFixed(1)} color={quizColor(r.quiz_score)} /> : <span style={{ color: '#94a3b8' }}>--</span>}
                  </td>
                  <td style={{ padding: '4px 10px' }}>{r.time_spent_minutes}</td>
                  <td style={{ padding: '4px 10px', fontSize: 11 }}>{r.started_at?.substring(0, 10) || '--'}</td>
                  <td style={{ padding: '4px 10px', fontSize: 11 }}>{r.completed_at?.substring(0, 10) || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderDefinitions(defs) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title={defs.title} span={3}>
        <p style={{ color: '#64748b', fontSize: 13 }}>Clinical glossary, module descriptions, format descriptions, and evidence-based notes on patient education in epilepsy.</p>
      </Card>

      {/* Glossary */}
      <Card title="Clinical Glossary" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569', width: 160 }}>Term</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {defs.glossary.map((g, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{g.term}</td>
                <td style={{ padding: '6px 10px', color: '#475569' }}>{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Format Descriptions */}
      <Card title="Delivery Formats">
        {defs.format_descriptions.map((f, i) => (
          <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.format_descriptions.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <Badge text={f.format} color={FORMAT_COLORS[f.format] || '#64748b'} />
            <p style={{ margin: '6px 0 0', fontSize: 12, color: '#475569' }}>{f.description}</p>
          </div>
        ))}
      </Card>

      {/* Module Descriptions */}
      <Card title="Module Descriptions" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
          {defs.module_descriptions.map((m, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{m.module}</div>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{m.description}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Clinical Notes */}
      <Card title="Clinical Notes" span={2}>
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {defs.clinical_notes.map((n, i) => (
            <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 8, lineHeight: 1.5 }}>{n}</li>
          ))}
        </ul>
      </Card>

      {/* Data Sources */}
      <Card title="Data Sources">
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {defs.data_sources.map((s, i) => (
            <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{s}</li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
