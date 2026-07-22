import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#64748b', '#0ea5e9']

const card = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  marginBottom: 16,
}

const badge = (bg) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: bg,
})

export default function TransactionLogDashboard() {
  const [tab, setTab] = useState('all')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [selectedComponent, setSelectedComponent] = useState(null)
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    if (tab === 'all' || tab === 'summary' || tab === 'component') {
      axios.get(API + '/api/transaction-log/overview')
        .then(r => { setOverview(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    } else if (tab === 'patient') {
      const params = selectedPatient ? `?patient_id=${encodeURIComponent(selectedPatient)}&limit=50` : '?limit=50'
      axios.get(API + '/api/transaction-log/breakdown' + params)
        .then(r => { setBreakdown(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    } else if (tab === 'definitions') {
      axios.get(API + '/api/transaction-log/definitions')
        .then(r => { setDefinitions(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    }
  }, [tab, selectedPatient])

  useEffect(() => {
    if (tab === 'component' && selectedComponent) {
      setLoading(true)
      axios.get(API + '/api/transaction-log/breakdown?component=' + encodeURIComponent(selectedComponent) + '&limit=50')
        .then(r => { setBreakdown(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    }
  }, [tab, selectedComponent])

  const tabs = [
    { id: 'all', label: 'Recent Transactions' },
    { id: 'summary', label: 'Summary' },
    { id: 'component', label: 'By Component' },
    { id: 'patient', label: 'By Patient' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const renderTxn = (t) => (
    <div key={t.id} style={{ ...card, borderLeft: '4px solid #6366f1' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={badge('#6366f1')}>{t.component}</span>
          <span style={badge('#10b981')}>{t.action}</span>
          {t.actor && <span style={badge('#94a3b8')}>{t.actor}</span>}
        </div>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>#{t.id}</span>
      </div>
      <div style={{ fontSize: 13, color: '#334155', marginBottom: 6 }}>{t.detail || '—'}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', display: 'flex', gap: 16 }}>
        <span>Patient: {t.patient_id}</span>
        {t.ref_id && <span>Ref: {t.ref_id}</span>}
        <span>{t.ts_local || t.ts_utc}</span>
      </div>
    </div>
  )

  const renderAll = () => {
    if (!overview) return null
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 20 }}>
          <div style={card}><div style={{ fontSize: 28, fontWeight: 700 }}>{overview.total}</div><div style={{ color: '#64748b', fontSize: 13 }}>Total Transactions</div></div>
          <div style={card}><div style={{ fontSize: 28, fontWeight: 700, color: '#6366f1' }}>{Object.keys(overview.by_component || {}).length}</div><div style={{ color: '#64748b', fontSize: 13 }}>Components</div></div>
          <div style={card}><div style={{ fontSize: 28, fontWeight: 700, color: '#10b981' }}>{Object.keys(overview.by_action || {}).length}</div><div style={{ color: '#64748b', fontSize: 13 }}>Action Types</div></div>
          <div style={card}><div style={{ fontSize: 28, fontWeight: 700, color: '#f59e0b' }}>{Object.keys(overview.by_patient || {}).length}</div><div style={{ color: '#64748b', fontSize: 13 }}>Patients/Actors</div></div>
        </div>
        <h4 style={{ marginBottom: 12 }}>Recent 50 Transactions</h4>
        <div style={{ maxHeight: 600, overflowY: 'auto' }}>
          {overview.recent?.map(renderTxn)}
        </div>
      </div>
    )
  }

  const renderSummary = () => {
    if (!overview) return null
    const compData = Object.entries(overview.by_component || {}).slice(0, 10).map(([k, v], i) => ({ name: k, value: v, fill: COLORS[i % COLORS.length] }))
    const actionData = Object.entries(overview.by_action || {}).map(([k, v], i) => ({ name: k, value: v, fill: COLORS[i % COLORS.length] }))
    const hourlyData = overview.hourly_activity || []
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div style={card}>
            <h4 style={{ marginTop: 0 }}>By Component (Top 10)</h4>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={compData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {compData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <h4 style={{ marginTop: 0 }}>By Action</h4>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={actionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-30} textAnchor="end" height={60} fontSize={11} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value">{actionData.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div style={card}>
          <h4 style={{ marginTop: 0 }}>Hourly Activity (UTC)</h4>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" label={{ value: 'Hour (UTC)', position: 'insideBottom', offset: -5 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} dot={{ fill: '#6366f1' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={card}>
          <h4 style={{ marginTop: 0 }}>By Actor</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left' }}>Actor</th>
                <th style={{ padding: '8px 12px', textAlign: 'right' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(overview.by_actor || {}).map(([k, v]) => (
                <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px' }}>{k}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  const renderComponent = () => {
    if (!overview) return null
    const components = Object.keys(overview.by_component || {})
    return (
      <div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
          {components.map((c, i) => (
            <button key={c} onClick={() => setSelectedComponent(c)}
              style={{ padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 12,
                background: selectedComponent === c ? COLORS[i % COLORS.length] : '#f1f5f9',
                color: selectedComponent === c ? '#fff' : '#334155' }}>
              {c} ({overview.by_component[c]})
            </button>
          ))}
        </div>
        {selectedComponent && breakdown && (
          <div>
            <div style={card}>
              <strong>{breakdown.total}</strong> transactions for component <span style={badge('#6366f1')}>{selectedComponent}</span>
            </div>
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              {breakdown.items?.map(renderTxn)}
            </div>
          </div>
        )}
        {!selectedComponent && <div style={card}>Select a component above to view its transactions.</div>}
      </div>
    )
  }

  const renderPatient = () => {
    if (!overview && !breakdown) return null
    const patients = overview ? Object.keys(overview.by_patient || {}) : []
    return (
      <div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
          <button onClick={() => setSelectedPatient(null)}
            style={{ padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 12,
              background: !selectedPatient ? '#6366f1' : '#f1f5f9', color: !selectedPatient ? '#fff' : '#334155' }}>
            All
          </button>
          {patients.map((p, i) => (
            <button key={p} onClick={() => setSelectedPatient(p)}
              style={{ padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 12,
                background: selectedPatient === p ? COLORS[i % COLORS.length] : '#f1f5f9',
                color: selectedPatient === p ? '#fff' : '#334155' }}>
              {p} ({overview?.by_patient?.[p] || '?'})
            </button>
          ))}
        </div>
        {breakdown && (
          <div>
            <div style={card}>
              <strong>{breakdown.total}</strong> transactions
              {selectedPatient && <> for patient <span style={badge('#f59e0b')}>{selectedPatient}</span></>}
            </div>
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              {breakdown.items?.map(renderTxn)}
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderDefinitions = () => {
    if (!definitions) return null
    return (
      <div>
        {Object.entries(definitions).map(([section, entries]) => (
          <div key={section} style={card}>
            <h4 style={{ marginTop: 0, textTransform: 'capitalize' }}>{section.replace(/_/g, ' ')}</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(entries || {}).map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, width: '25%', fontFamily: 'monospace', fontSize: 13 }}>{k}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginTop: 0 }}>Transaction Log</h2>
      <p style={{ color: '#64748b', marginBottom: 20 }}>
        Audit trail of all system transactions — model predictions, data processing, consistency checks, and user actions — sourced from clinical.db transaction_log
      </p>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{ padding: '8px 20px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600,
              background: tab === t.id ? '#6366f1' : '#f1f5f9', color: tab === t.id ? '#fff' : '#334155' }}>
            {t.label}
          </button>
        ))}
      </div>
      {loading && <div style={card}>Loading...</div>}
      {error && <div style={{ ...card, borderLeft: '4px solid #ef4444', color: '#ef4444' }}>Error: {error}</div>}
      {!loading && !error && tab === 'all' && renderAll()}
      {!loading && !error && tab === 'summary' && renderSummary()}
      {!loading && !error && tab === 'component' && renderComponent()}
      {!loading && !error && tab === 'patient' && renderPatient()}
      {!loading && !error && tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
