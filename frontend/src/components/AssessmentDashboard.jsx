import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (window._env_ && window._env_.API_URL) || '/api'
const COLORS = ['#3b82f6','#22c55e','#f97316','#ef4444','#8b5cf6','#ec4899','#14b8a6','#eab308','#6366f1','#f43f5e','#06b6d4','#84cc16','#d946ef','#0ea5e9','#a855f7','#10b981']
const LEVEL_COLORS = {
  normal: '#22c55e', mild: '#3b82f6', moderate: '#eab308',
  severe: '#ef4444', critical: '#7f1d1d', none: '#94a3b8',
  low: '#eab308', high: '#ef4444',
  'Average': '#22c55e', 'Low Average': '#3b82f6', 'High Average': '#14b8a6',
  'Superior': '#8b5cf6', 'Borderline': '#f97316',
  'Extremely Low': '#ef4444', 'Extremely High': '#d946ef'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: (color || '#94a3b8') + '22', color: color || '#94a3b8'
    }}>{text}</span>
  )
}

export default function AssessmentDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/assessment-dashboard`)
        setData(res.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading assessment data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 40, color: '#64748b' }}>No assessment data available.</div>

  const tabs = ['overview', 'instruments', 'severity', 'trends', 'recent']
  const tabLabels = { overview: 'Overview', instruments: 'Instruments', severity: 'Severity', trends: 'Trends', recent: 'Recent' }

  const typeData = Object.entries(data.by_type || {}).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count)
  const levelData = Object.entries(data.by_level || {}).filter(([k]) => k).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count)
  const diseaseData = Object.entries(data.by_disease || {}).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count)
  const sourceData = Object.entries(data.by_source || {}).map(([name, count]) => ({ name, count }))
  const dateData = Object.entries(data.by_date || {}).map(([date, count]) => ({ date, count }))
  const avgScoreData = Object.entries(data.avg_score || {}).map(([name, score]) => ({ name, score })).sort((a, b) => b.score - a.score)
  const userCounts = Object.entries(data.by_user || {}).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count)

  const renderOverview = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14 }}>
        <Card><KPI label="Total Assessments" value={fmt(data.total)} color="#3b82f6" /></Card>
        <Card><KPI label="Instrument Types" value={typeData.length} color="#22c55e" /></Card>
        <Card><KPI label="Active Alerts" value={fmt(data.alerts)} color={data.alerts > 0 ? '#ef4444' : '#22c55e'} /></Card>
        <Card><KPI label="Examiners" value={userCounts.length} color="#8b5cf6" /></Card>
        <Card><KPI label="Diseases Covered" value={diseaseData.length} color="#f97316" /></Card>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <Card title="Assessments by Instrument">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={typeData} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={75} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {typeData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Severity Distribution">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={levelData} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {levelData.map((d, i) => <Cell key={i} fill={LEVEL_COLORS[d.name] || COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <Card title="By Disease">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={diseaseData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Data Source">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={sourceData} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, count }) => `${name}: ${count}`}>
                {sourceData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  )

  const renderInstruments = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14 }}>
        <Card><KPI label="Total Instruments" value={typeData.length} color="#3b82f6" /></Card>
        <Card><KPI label="Most Used" value={typeData[0]?.name || '--'} sub={`${typeData[0]?.count || 0} assessments`} color="#22c55e" /></Card>
        <Card><KPI label="Least Used" value={typeData[typeData.length - 1]?.name || '--'} sub={`${typeData[typeData.length - 1]?.count || 0} assessments`} color="#eab308" /></Card>
      </div>

      <Card title="Average Score by Instrument">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={avgScoreData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
            <YAxis />
            <Tooltip formatter={(v) => v.toFixed(1)} />
            <Bar dataKey="score" radius={[4, 4, 0, 0]}>
              {avgScoreData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Assessment Count by Instrument">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Instrument</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Count</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Score</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Share</th>
              </tr>
            </thead>
            <tbody>
              {typeData.map((d, i) => {
                const pct = Math.round(d.count / data.total * 100)
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{d.name}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>{d.count}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(data.avg_score?.[d.name])}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ background: '#f1f5f9', borderRadius: 4, height: 8, flex: 1, maxWidth: 120 }}>
                          <div style={{ width: `${pct}%`, background: COLORS[i % COLORS.length], borderRadius: 4, height: '100%' }} />
                        </div>
                        <span style={{ fontSize: 11, color: '#64748b' }}>{pct}%</span>
                      </div>
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

  const renderSeverity = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 14 }}>
        {levelData.slice(0, 6).map((d, i) => (
          <Card key={i}>
            <KPI label={d.name} value={d.count} color={LEVEL_COLORS[d.name] || '#64748b'} sub={`${Math.round(d.count / data.total * 100)}% of total`} />
          </Card>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <Card title="Severity Level Breakdown">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={levelData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {levelData.map((d, i) => <Cell key={i} fill={LEVEL_COLORS[d.name] || COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Examiner Workload">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={userCounts} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={75} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#14b8a6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card title="Alert Summary">
        <div style={{ display: 'flex', alignItems: 'center', gap: 20, padding: 10 }}>
          <div style={{ fontSize: 48, fontWeight: 700, color: data.alerts > 0 ? '#ef4444' : '#22c55e' }}>{data.alerts}</div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#334155' }}>Active Clinical Alerts</div>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
              {data.alerts > 0
                ? `${data.alerts} assessment${data.alerts > 1 ? 's' : ''} flagged for clinical attention`
                : 'No assessments currently flagged'}
            </div>
          </div>
        </div>
      </Card>
    </div>
  )

  const renderTrends = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card title="Assessments Over Time" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={dateData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <Card title="Disease Distribution">
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={diseaseData} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {diseaseData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Data Provenance">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: 10 }}>
            {sourceData.map((d, i) => {
              const pct = Math.round(d.count / data.total * 100)
              return (
                <div key={i}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
                    <span style={{ fontWeight: 600 }}>{d.name}</span>
                    <span>{d.count} ({pct}%)</span>
                  </div>
                  <div style={{ background: '#f1f5f9', borderRadius: 4, height: 10 }}>
                    <div style={{ width: `${pct}%`, background: COLORS[i % COLORS.length], borderRadius: 4, height: '100%' }} />
                  </div>
                </div>
              )
            })}
          </div>
        </Card>
      </div>
    </div>
  )

  const renderRecent = () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <Card title={`Recent Assessments (${(data.recent || []).length})`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569' }}>Patient</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569' }}>Instrument</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569' }}>Score</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569' }}>Level</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569' }}>Interpretation</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569' }}>Examiner</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569' }}>Date</th>
                <th style={{ textAlign: 'center', padding: '8px 10px', color: '#475569' }}>Alert</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent || []).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: r.alert ? '#fef2f2' : undefined }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '8px 10px' }}><Badge text={r.instrument} color="#3b82f6" /></td>
                  <td style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600 }}>
                    {r.score != null ? `${fmt(r.score)}/${fmt(r.max_score)}` : '--'}
                  </td>
                  <td style={{ padding: '8px 10px' }}>
                    <Badge text={r.level || 'n/a'} color={LEVEL_COLORS[r.level] || '#94a3b8'} />
                  </td>
                  <td style={{ padding: '8px 10px', fontSize: 12, color: '#475569' }}>{r.interpretation || '--'}</td>
                  <td style={{ padding: '8px 10px', fontSize: 12 }}>{r.examiner || '--'}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b' }}>{r.created_at ? r.created_at.slice(0, 16).replace('T', ' ') : '--'}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                    {r.alert ? <Badge text="ALERT" color="#ef4444" /> : <span style={{ color: '#cbd5e1' }}>--</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>Assessment Analytics Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Clinical assessment battery — {data.total} assessments across {typeData.length} instruments
        </p>
      </div>

      <div style={{ display: 'flex', gap: 6, marginBottom: 18, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#f1f5f9', color: tab === t ? '#fff' : '#475569',
            transition: 'all 0.15s'
          }}>{tabLabels[t]}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'instruments' && renderInstruments()}
      {tab === 'severity' && renderSeverity()}
      {tab === 'trends' && renderTrends()}
      {tab === 'recent' && renderRecent()}
    </div>
  )
}
