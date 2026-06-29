import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#607d8b']

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: decimals }) : String(v)
}

export default function DatabaseOpsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/database-ops/overview`),
          axios.get(`${API_URL}/database-ops/breakdown`),
          axios.get(`${API_URL}/database-ops/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load database ops data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading database ops data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Database ops data not available.'}
    </div>
  )

  const summary = overview.summary || {}
  const tables = overview.tables || []
  const backups = overview.backups || {}
  const tableDetails = Array.isArray(breakdown?.tables) ? breakdown.tables : []

  const topTables = tables.slice(0, 10).map(t => ({ name: t.name, rows: t.rows }))
  const colDistribution = tables.map(t => ({ name: t.name, columns: t.columns })).slice(0, 10)

  const integrityColor = summary.integrity === 'ok' ? '#4caf50' : '#f44336'
  const fragColor = summary.fragmentation_pct > 10 ? '#ff9800' : '#4caf50'

  const kpiItems = [
    { label: 'DB Size', value: `${fmt(summary.db_size_kb, 1)} KB`, color: COLORS[0] },
    { label: 'Tables', value: summary.total_tables, color: COLORS[1] },
    { label: 'Total Rows', value: summary.total_rows, color: COLORS[2] },
    { label: 'Indexes', value: summary.total_indexes, color: COLORS[3] },
    { label: 'Integrity', value: summary.integrity, color: integrityColor },
    { label: 'Journal', value: summary.journal_mode?.toUpperCase(), color: COLORS[5] },
    { label: 'Fragmentation', value: `${fmt(summary.fragmentation_pct, 1)}%`, color: fragColor },
    { label: 'WAL Size', value: `${fmt(summary.wal_size_kb, 1)} KB`, color: COLORS[6] },
  ]

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 120
  })

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Database Ops Dashboard</h2>
        <button
          onClick={() => setShowDefs(!showDefs)}
          style={{ background: '#f1f5f9', border: '1px solid #cbd5e1', borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}
        >
          {showDefs ? 'Hide' : 'Show'} Definitions
        </button>
      </div>

      {showDefs && defs?.definitions && (
        <div style={{ ...cardStyle, background: '#f0f9ff', border: '1px solid #bae6fd' }}>
          <h4 style={{ margin: '0 0 10px', color: '#0369a1' }}>Metric Definitions</h4>
          {defs.definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <strong style={{ color: '#0c4a6e' }}>{d.term}:</strong>{' '}
              <span style={{ color: '#334155', fontSize: 13 }}>{d.meaning}</span>
            </div>
          ))}
        </div>
      )}

      {/* KPI cards */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
        {kpiItems.map((k, i) => (
          <div key={i} style={kpiStyle(k.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>
              {typeof k.value === 'number' ? fmt(k.value) : (k.value || '--')}
            </div>
          </div>
        ))}
      </div>

      {/* Backup status */}
      <div style={cardStyle}>
        <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Backup Status</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Backup Count</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: COLORS[0] }}>{fmt(backups.count)}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Latest Backup</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{backups.latest || '--'}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Total Backup Size</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: COLORS[3] }}>{fmt(backups.total_size_kb, 1)} KB</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Retention</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: COLORS[2] }}>{backups.retention_days} days</div>
          </div>
        </div>
      </div>

      {/* Top tables by rows + column distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
        {topTables.length > 0 && (
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Top Tables by Row Count</h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={topTables} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={130} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="rows" fill={COLORS[0]} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {colDistribution.length > 0 && (
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Schema Complexity (columns per table)</h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={colDistribution} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={130} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="columns" fill={COLORS[1]} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Storage breakdown pie */}
      {tables.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Row Distribution Across Tables</h4>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={tables.filter(t => t.rows > 0).slice(0, 12)}
                dataKey="rows"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {tables.filter(t => t.rows > 0).slice(0, 12).map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Table detail */}
      {tableDetails.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Table Detail</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>Table</th>
                  <th style={{ padding: '8px 10px' }}>Rows</th>
                  <th style={{ padding: '8px 10px' }}>Columns</th>
                  <th style={{ padding: '8px 10px' }}>Indexes</th>
                  <th style={{ padding: '8px 10px' }}>Last Activity</th>
                </tr>
              </thead>
              <tbody>
                {tableDetails.map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{t.name}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(t.rows)}</td>
                    <td style={{ padding: '8px 10px' }}>{t.columns}</td>
                    <td style={{ padding: '8px 10px' }}>
                      {t.index_count > 0 ? (
                        <span style={{ color: '#4caf50' }}>{t.index_count}</span>
                      ) : (
                        <span style={{ color: '#94a3b8' }}>0</span>
                      )}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#64748b', fontSize: 12 }}>
                      {t.last_activity ? t.last_activity.slice(0, 16).replace('T', ' ') : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Page/storage info */}
      <div style={cardStyle}>
        <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Storage Internals</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, fontSize: 13 }}>
          <div><span style={{ color: '#64748b' }}>Page Size:</span> <strong>{fmt(summary.page_size)}</strong> bytes</div>
          <div><span style={{ color: '#64748b' }}>Page Count:</span> <strong>{fmt(summary.page_count)}</strong></div>
          <div><span style={{ color: '#64748b' }}>Freelist Pages:</span> <strong>{fmt(summary.freelist_pages)}</strong></div>
          <div><span style={{ color: '#64748b' }}>Triggers:</span> <strong>{fmt(summary.total_triggers || 0)}</strong></div>
        </div>
      </div>

      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        Generated: {overview.generated_at || '--'}
      </div>
    </div>
  )
}
