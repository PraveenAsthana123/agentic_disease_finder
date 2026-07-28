import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function CategoryBadge({ category }) {
  const colors = {
    Security: '#ef4444', AI: '#8b5cf6', Data: '#3b82f6',
    Notification: '#f59e0b', Performance: '#10b981', General: '#64748b'
  }
  const c = colors[category] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: c + '22', color: c, fontWeight: 600, fontSize: 12
    }}>{category}</span>
  )
}

function ValueBadge({ value }) {
  const v = String(value).toLowerCase()
  if (v === 'true') return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 12, background: '#10b98122', color: '#10b981', fontWeight: 600, fontSize: 12 }}>true</span>
  if (v === 'false') return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 12, background: '#ef444422', color: '#ef4444', fontWeight: 600, fontSize: 12 }}>false</span>
  return <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{value}</span>
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'configs', label: 'All Configs' },
  { id: 'category', label: 'By Category' },
  { id: 'audit', label: 'Audit Trail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SystemConfigDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [search, setSearch] = useState('')
  const [sortCol, setSortCol] = useState('key')
  const [sortDir, setSortDir] = useState('asc')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/system-config/overview`),
      axios.get(`${API_URL}/api/system-config/breakdown`),
      axios.get(`${API_URL}/api/system-config/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading system configuration…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const o = overview || {}
  const b = breakdown || {}

  const sorted = (rows) => {
    if (!rows) return []
    const filtered = rows.filter(r =>
      !search || Object.values(r).some(v => String(v).toLowerCase().includes(search.toLowerCase()))
    )
    return [...filtered].sort((a, b) => {
      const av = a[sortCol], bv = b[sortCol]
      if (av == null) return 1
      if (bv == null) return -1
      const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
      return sortDir === 'asc' ? cmp : -cmp
    })
  }

  const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', cursor: 'pointer', fontSize: 12, color: '#64748b', userSelect: 'none' }
  const tdStyle = { padding: '7px 10px', borderBottom: '1px solid #f1f5f9', fontSize: 13 }
  const sortIcon = (col) => sortCol === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''
  const onSort = (col) => { setSortDir(sortCol === col && sortDir === 'asc' ? 'desc' : 'asc'); setSortCol(col) }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 8 }}>System Configuration Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 14, marginBottom: 20 }}>
        Platform configuration registry — {o.total_configs} settings across {o.total_categories} categories, managed by {o.total_updaters} administrators
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Total Configs" value={o.total_configs} color="#3b82f6" />
              <KPI label="Categories" value={o.total_categories} color="#8b5cf6" />
              <KPI label="Administrators" value={o.total_updaters} color="#10b981" />
            </div>
          </Card>

          <Card title="Config Types">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Boolean (true)" value={o.boolean_true} color="#10b981" />
              <KPI label="Boolean (false)" value={o.boolean_false} color="#ef4444" />
              <KPI label="Numeric" value={o.numeric_configs} color="#3b82f6" />
            </div>
          </Card>

          <Card title="Category Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={o.category_distribution} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={80} label={({ category, count }) => `${category} (${count})`}>
                  {(o.category_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Updates by Administrator">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={o.updater_distribution} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="updater" tick={{ fontSize: 11 }} width={80} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Monthly Update Activity" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={o.monthly_updates}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Updates" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Security Settings">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Setting</th>
                <th style={thStyle}>Value</th>
              </tr></thead>
              <tbody>
                {(o.security_configs || []).map((c, i) => (
                  <tr key={i}>
                    <td style={tdStyle}><span style={{ fontWeight: 500 }}>{c.key.replace(/_/g, ' ')}</span><br /><span style={{ fontSize: 11, color: '#94a3b8' }}>{c.description}</span></td>
                    <td style={tdStyle}><ValueBadge value={c.value} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="AI Model Settings">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Setting</th>
                <th style={thStyle}>Value</th>
              </tr></thead>
              <tbody>
                {(o.ai_configs || []).map((c, i) => (
                  <tr key={i}>
                    <td style={tdStyle}><span style={{ fontWeight: 500 }}>{c.key.replace(/_/g, ' ')}</span><br /><span style={{ fontSize: 11, color: '#94a3b8' }}>{c.description}</span></td>
                    <td style={tdStyle}><ValueBadge value={c.value} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Performance Tuning">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Setting</th>
                <th style={thStyle}>Value</th>
              </tr></thead>
              <tbody>
                {(o.performance_configs || []).map((c, i) => (
                  <tr key={i}>
                    <td style={tdStyle}><span style={{ fontWeight: 500 }}>{c.key.replace(/_/g, ' ')}</span><br /><span style={{ fontSize: 11, color: '#94a3b8' }}>{c.description}</span></td>
                    <td style={tdStyle}><ValueBadge value={c.value} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Recent Changes" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Config</th>
                <th style={thStyle}>Category</th>
                <th style={thStyle}>Value</th>
                <th style={thStyle}>Updated</th>
                <th style={thStyle}>By</th>
              </tr></thead>
              <tbody>
                {(o.recent_changes || []).map((c, i) => (
                  <tr key={i}>
                    <td style={tdStyle}><span style={{ fontWeight: 500 }}>{c.key}</span><br /><span style={{ fontSize: 11, color: '#94a3b8' }}>{c.description}</span></td>
                    <td style={tdStyle}><CategoryBadge category={c.category} /></td>
                    <td style={tdStyle}><ValueBadge value={c.value} /></td>
                    <td style={tdStyle}><span style={{ fontSize: 12, color: '#64748b' }}>{c.updated_at}</span></td>
                    <td style={tdStyle}>{c.updated_by}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {tab === 'configs' && (
        <Card title="All Configuration Settings">
          <input
            type="text" placeholder="Search configs…" value={search} onChange={e => setSearch(e.target.value)}
            style={{ padding: '8px 14px', borderRadius: 8, border: '1px solid #e2e8f0', width: 300, marginBottom: 12, fontSize: 13 }}
          />
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle} onClick={() => onSort('config_id')}>ID{sortIcon('config_id')}</th>
                <th style={thStyle} onClick={() => onSort('key')}>Key{sortIcon('key')}</th>
                <th style={thStyle} onClick={() => onSort('value')}>Value{sortIcon('value')}</th>
                <th style={thStyle} onClick={() => onSort('category')}>Category{sortIcon('category')}</th>
                <th style={thStyle} onClick={() => onSort('description')}>Description{sortIcon('description')}</th>
                <th style={thStyle} onClick={() => onSort('updated_at')}>Updated{sortIcon('updated_at')}</th>
                <th style={thStyle} onClick={() => onSort('updated_by')}>By{sortIcon('updated_by')}</th>
              </tr></thead>
              <tbody>
                {sorted(b.configs).map((c, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><span style={{ fontSize: 11, color: '#94a3b8' }}>{c.config_id}</span></td>
                    <td style={tdStyle}><span style={{ fontWeight: 600 }}>{c.key}</span></td>
                    <td style={tdStyle}><ValueBadge value={c.value} /></td>
                    <td style={tdStyle}><CategoryBadge category={c.category} /></td>
                    <td style={tdStyle}><span style={{ fontSize: 12, color: '#64748b' }}>{c.description}</span></td>
                    <td style={tdStyle}><span style={{ fontSize: 12, color: '#64748b' }}>{c.updated_at}</span></td>
                    <td style={tdStyle}>{c.updated_by}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'category' && (
        <div style={{ display: 'grid', gap: 16 }}>
          {(b.by_category || []).map((cat, ci) => (
            <Card key={ci} title={`${cat.category} (${cat.count} configs)`}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={thStyle}>Key</th>
                  <th style={thStyle}>Value</th>
                  <th style={thStyle}>Description</th>
                  <th style={thStyle}>Last Updated</th>
                  <th style={thStyle}>By</th>
                </tr></thead>
                <tbody>
                  {cat.items.map((item, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}><span style={{ fontWeight: 600 }}>{item.key}</span></td>
                      <td style={tdStyle}><ValueBadge value={item.value} /></td>
                      <td style={tdStyle}><span style={{ fontSize: 12, color: '#64748b' }}>{item.description}</span></td>
                      <td style={tdStyle}><span style={{ fontSize: 12, color: '#64748b' }}>{item.updated_at}</span></td>
                      <td style={tdStyle}>{item.updated_by}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </div>
      )}

      {tab === 'audit' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
          <Card title="Configuration Managers">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Administrator</th>
                <th style={thStyle}>Configs Managed</th>
                <th style={thStyle}>Last Update</th>
              </tr></thead>
              <tbody>
                {(b.by_updater || []).map((u, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><span style={{ fontWeight: 600 }}>{u.updater}</span></td>
                    <td style={tdStyle}>{u.configs_managed}</td>
                    <td style={tdStyle}><span style={{ fontSize: 12, color: '#64748b' }}>{u.last_update}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Update Timeline">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={o.monthly_updates}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} name="Config Changes" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Latest Update" span={2}>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              <div><span style={{ color: '#64748b', fontSize: 12 }}>Most Recent:</span><br /><span style={{ fontWeight: 600 }}>{o.latest_update}</span></div>
              <div><span style={{ color: '#64748b', fontSize: 12 }}>Oldest:</span><br /><span style={{ fontWeight: 600 }}>{o.oldest_update}</span></div>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card title={defs.title}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Term</th>
                <th style={thStyle}>Definition</th>
              </tr></thead>
              <tbody>
                {(defs.concepts || []).map((c, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, minWidth: 150 }}>{c.name}</td>
                    <td style={{ ...tdStyle, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{c.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <Card title="Configuration Categories">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Category</th>
                <th style={thStyle}>Description</th>
              </tr></thead>
              <tbody>
                {(defs.categories || []).map((c, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><CategoryBadge category={c.category} /></td>
                    <td style={{ ...tdStyle, fontSize: 13, color: '#475569' }}>{c.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <Card title="Data Sources">
            <ul style={{ margin: 0, paddingLeft: 20, color: '#475569', fontSize: 13, lineHeight: 1.8 }}>
              {(defs.data_sources || []).map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
