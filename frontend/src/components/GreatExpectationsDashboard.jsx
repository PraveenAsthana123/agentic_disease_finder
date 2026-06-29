import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#22c55e', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#64748b']
const PASS_COLOR = '#22c55e'
const FAIL_COLOR = '#ef4444'

const card = {
  background: '#ffffff', borderRadius: 12, padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.08)', marginBottom: 16,
}
const kpiCard = (color) => ({
  ...card, borderLeft: `4px solid ${color}`, flex: '1 1 180px', minWidth: 160,
})

export default function GreatExpectationsDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState(0)
  const [filterStatus, setFilterStatus] = useState('all') // all | pass | fail

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/great-expectations`)
        setData(res.data)
      } catch (err) {
        setError(err.message || 'Failed to load Great Expectations report')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>📊</div>
      Running Great Expectations data quality validation on EEG datasets...
      <div style={{ fontSize: 12, marginTop: 8, color: '#94a3b8' }}>
        (Validating column types, ranges, nulls, and distributions)
      </div>
    </div>
  )
  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca',
      borderRadius: 8, color: '#991b1b' }}>Error: {error}</div>
  )
  if (!data?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a',
      borderRadius: 8, color: '#92400e' }}>
      Great Expectations unavailable: {data?.error || 'No data found'}
    </div>
  )

  const datasets = data.per_dataset || []
  const current = datasets[selectedDataset] || {}
  const allResults = current.results || []
  const filtered = filterStatus === 'all' ? allResults
    : filterStatus === 'pass' ? allResults.filter(r => r.success)
    : allResults.filter(r => !r.success)

  // Per-dataset bar chart data
  const barData = datasets.map(ds => ({
    name: `${ds.disease} / ${ds.file.replace('.csv', '').substring(0, 20)}`,
    Passed: ds.passed,
    Failed: ds.failed,
    pass_rate: ds.pass_rate,
  }))

  // Expectation type pie chart
  const typeSummary = data.expectation_type_summary || {}
  const pieData = Object.entries(typeSummary).map(([type, counts]) => ({
    name: type.replace('expect_', '').replace(/_/g, ' ').substring(0, 30),
    value: counts.passed + counts.failed,
    passed: counts.passed,
    failed: counts.failed,
  }))

  // Column stats for selected dataset
  const colStats = current.column_stats || {}
  const colStatsArr = Object.entries(colStats).map(([col, stats]) => ({
    column: col, ...stats
  }))

  // Overall pass/fail pie
  const overallPie = [
    { name: 'Passed', value: data.total_passed },
    { name: 'Failed', value: data.total_failed },
  ]

  return (
    <div style={{ background: '#f8fafc', minHeight: '100vh', padding: 24 }}>
      {/* Header */}
      <div style={{ ...card, background: 'linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%)',
        color: '#fff', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <span style={{ fontSize: 28 }}>📊</span>
          <div>
            <h2 style={{ margin: 0, fontSize: 22 }}>Great Expectations Data Quality</h2>
            <div style={{ fontSize: 13, opacity: 0.85, marginTop: 4 }}>
              Real great_expectations v{data.version} validation on {data.datasets_validated} EEG
              feature datasets &mdash; {data.total_expectations} expectations evaluated
            </div>
          </div>
        </div>
      </div>

      {/* KPI Tiles */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <div style={kpiCard('#3b82f6')}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Datasets Validated</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#1e3a5f' }}>{data.datasets_validated}</div>
        </div>
        <div style={kpiCard('#8b5cf6')}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Total Expectations</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#1e3a5f' }}>{data.total_expectations}</div>
        </div>
        <div style={kpiCard(PASS_COLOR)}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Passed</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: PASS_COLOR }}>{data.total_passed}</div>
        </div>
        <div style={kpiCard(FAIL_COLOR)}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Failed</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: data.total_failed > 0 ? FAIL_COLOR : '#64748b' }}>
            {data.total_failed}
          </div>
        </div>
        <div style={kpiCard('#f59e0b')}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Overall Pass Rate</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: data.overall_pass_rate >= 90 ? PASS_COLOR : FAIL_COLOR }}>
            {data.overall_pass_rate}%
          </div>
        </div>
      </div>

      {/* Row: Per-dataset bar chart + Overall pie */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 20 }}>
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#1e3a5f' }}>
            Pass / Fail by Dataset
          </h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="Passed" fill={PASS_COLOR} radius={[4, 4, 0, 0]} />
              <Bar dataKey="Failed" fill={FAIL_COLOR} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#1e3a5f' }}>
            Overall Results
          </h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={overallPie} cx="50%" cy="50%" innerRadius={50} outerRadius={90}
                dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                <Cell fill={PASS_COLOR} />
                <Cell fill={data.total_failed > 0 ? FAIL_COLOR : '#e2e8f0'} />
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Expectation Type Breakdown */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#1e3a5f' }}>
          Expectations by Type
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={pieData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="name" type="category" width={200} tick={{ fontSize: 10 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="passed" stackId="a" fill={PASS_COLOR} />
            <Bar dataKey="failed" stackId="a" fill={FAIL_COLOR} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Dataset selector */}
      <div style={{ ...card, marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h3 style={{ margin: 0, fontSize: 16, color: '#1e3a5f' }}>Dataset Details</h3>
          <select value={selectedDataset} onChange={e => setSelectedDataset(+e.target.value)}
            style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid #cbd5e1',
              fontSize: 13, background: '#fff' }}>
            {datasets.map((ds, i) => (
              <option key={i} value={i}>{ds.disease} / {ds.file} ({ds.pass_rate}%)</option>
            ))}
          </select>
          <div style={{ display: 'flex', gap: 6 }}>
            {['all', 'pass', 'fail'].map(s => (
              <button key={s} onClick={() => setFilterStatus(s)}
                style={{ padding: '4px 12px', borderRadius: 6, fontSize: 12, cursor: 'pointer',
                  border: filterStatus === s ? '2px solid #2563eb' : '1px solid #cbd5e1',
                  background: filterStatus === s ? '#eff6ff' : '#fff',
                  color: filterStatus === s ? '#2563eb' : '#64748b', fontWeight: 500 }}>
                {s === 'all' ? `All (${allResults.length})` :
                 s === 'pass' ? `Pass (${allResults.filter(r => r.success).length})` :
                 `Fail (${allResults.filter(r => !r.success).length})`}
              </button>
            ))}
          </div>
        </div>

        {/* Dataset info */}
        <div style={{ display: 'flex', gap: 16, marginTop: 12, fontSize: 13, color: '#475569' }}>
          <span>Rows: <strong>{current.rows}</strong></span>
          <span>Columns: <strong>{current.columns}</strong></span>
          <span>Passed: <strong style={{ color: PASS_COLOR }}>{current.passed}</strong></span>
          <span>Failed: <strong style={{ color: current.failed > 0 ? FAIL_COLOR : '#64748b' }}>
            {current.failed}</strong></span>
          <span>Pass Rate: <strong style={{ color: current.pass_rate >= 90 ? PASS_COLOR : FAIL_COLOR }}>
            {current.pass_rate}%</strong></span>
        </div>
      </div>

      {/* Expectation Results Table */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#1e3a5f' }}>
          Expectation Results — {current.disease} / {current.file}
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={th}>Status</th>
                <th style={th}>Expectation</th>
                <th style={th}>Column</th>
                <th style={th}>Observed</th>
                <th style={th}>Elements</th>
                <th style={th}>Unexpected %</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #e2e8f0',
                  background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={td}>
                    <span style={{ display: 'inline-block', width: 18, height: 18,
                      borderRadius: '50%', textAlign: 'center', lineHeight: '18px',
                      fontSize: 11, fontWeight: 700,
                      background: r.success ? '#dcfce7' : '#fef2f2',
                      color: r.success ? PASS_COLOR : FAIL_COLOR }}>
                      {r.success ? '✓' : '✗'}
                    </span>
                  </td>
                  <td style={{ ...td, maxWidth: 300 }}>
                    {r.expectation_type.replace('expect_', '').replace(/_/g, ' ')}
                  </td>
                  <td style={{ ...td, fontFamily: 'monospace', fontSize: 11 }}>{r.column || '—'}</td>
                  <td style={{ ...td, fontFamily: 'monospace' }}>
                    {r.observed_value !== null && r.observed_value !== undefined
                      ? String(r.observed_value) : '—'}
                  </td>
                  <td style={td}>{r.element_count ?? '—'}</td>
                  <td style={td}>
                    {r.unexpected_percent !== null && r.unexpected_percent !== undefined
                      ? `${r.unexpected_percent.toFixed(1)}%` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Column Statistics */}
      {colStatsArr.length > 0 && (
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#1e3a5f' }}>
            Column Statistics — {current.file}
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  <th style={th}>Column</th>
                  <th style={th}>Mean</th>
                  <th style={th}>Std</th>
                  <th style={th}>Min</th>
                  <th style={th}>Max</th>
                  <th style={th}>Null %</th>
                </tr>
              </thead>
              <tbody>
                {colStatsArr.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0',
                    background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...td, fontFamily: 'monospace', fontSize: 11 }}>{c.column}</td>
                    <td style={td}>{c.mean ?? '—'}</td>
                    <td style={td}>{c.std ?? '—'}</td>
                    <td style={td}>{c.min ?? '—'}</td>
                    <td style={td}>{c.max ?? '—'}</td>
                    <td style={td}>
                      <span style={{ color: c.null_pct > 5 ? FAIL_COLOR : '#64748b' }}>
                        {c.null_pct}%
                      </span>
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

const th = { padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 11 }
const td = { padding: '6px 10px', color: '#334155' }
