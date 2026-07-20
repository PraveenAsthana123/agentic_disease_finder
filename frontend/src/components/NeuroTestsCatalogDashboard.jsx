import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const bg = status === 'built' ? '#22c55e' : status === 'partial' ? '#f97316' : '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {status}
    </span>
  )
}

function EegLinkBadge({ category, index }) {
  const bg = COLORS[index % COLORS.length]
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {category}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function NeuroTestsCatalogDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/neuro-tests-catalog/overview`),
      axios.get(`${API_URL}/neuro-tests-catalog/breakdown`),
      axios.get(`${API_URL}/neuro-tests-catalog/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Neuro Tests Catalog...</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#ef4444' }}>neuro_tests_catalog.json not found</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'by-eeg-link', label: 'By EEG Linkage' },
    { id: 'by-role', label: 'By Role' },
    { id: 'case-data', label: 'Case Data' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const s = overview.summary

  /* Build a lookup for EEG link category colors */
  const eegLinkCategories = (overview.eeg_link_distribution || []).map(e => e.name)
  const eegLinkColorMap = {}
  eegLinkCategories.forEach((cat, i) => { eegLinkColorMap[cat] = COLORS[i % COLORS.length] })

  return (
    <div style={{ padding: '20px 24px', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>Neuro Tests Catalog Dashboard</h2>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          {overview.title} &middot; Updated: {overview.updated_at}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', fontSize: 12, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#fff', color: tab === t.id ? '#fff' : '#475569',
            border: `1px solid ${tab === t.id ? '#3b82f6' : '#e2e8f0'}`, borderRadius: 6, cursor: 'pointer'
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 16 }}>
              <KPI label="Total Tests" value={s.total_tests} />
              <KPI label="Built" value={s.built_count} />
              <KPI label="EEG Link Categories" value={s.eeg_link_categories} />
              <KPI label="Unique Roles" value={s.unique_roles} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.status_distribution || []).map((entry, i) => {
                      const color = entry.name === 'built' ? '#22c55e' : entry.name === 'partial' ? '#f97316' : COLORS[i % COLORS.length]
                      return <Cell key={i} fill={color} />
                    })}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="EEG Linkage Category Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.eeg_link_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.eeg_link_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Role Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.role_distribution} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Tests Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>ID</th>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Purpose</th>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Output</th>
                    <th style={thStyle}>EEG Link</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.tests_table || []).map((t, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontFamily: 'monospace', fontWeight: 600 }}>{t.id}</td>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{t.name}</td>
                      <td style={tdStyle}>{t.purpose}</td>
                      <td style={tdStyle}>{t.role}</td>
                      <td style={tdStyle}>{t.output}</td>
                      <td style={tdStyle}>
                        <EegLinkBadge category={t.eeg_link} index={eegLinkCategories.indexOf(t.eeg_link)} />
                      </td>
                      <td style={tdStyle}><StatusBadge status={t.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'by-eeg-link' && breakdown && (
        <>
          {(breakdown.by_eeg_link || []).map((group, gi) => (
            <Card key={gi} title={group.category}>
              <div style={{ marginBottom: 12 }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>{group.tests.length} test{group.tests.length !== 1 ? 's' : ''} in this category</span>
              </div>
              {group.tests.map((t, ti) => (
                <div key={ti} style={{
                  background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6,
                  padding: 12, marginBottom: 8
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: '#334155' }}>{t.name}</span>
                    <StatusBadge status={t.status || 'built'} />
                    <EegLinkBadge category={t.eeg_link} index={eegLinkCategories.indexOf(t.eeg_link)} />
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>
                    <strong>Purpose:</strong> {t.purpose}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>
                    <strong>Role:</strong> {t.role}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569' }}>
                    <strong>Output:</strong> {t.output}
                  </div>
                </div>
              ))}
            </Card>
          ))}

          {breakdown.eeg_linkage_summary && (
            <Card title="EEG Linkage Summary">
              <div style={{ background: '#eff6ff', borderRadius: 6, padding: 16 }}>
                {typeof breakdown.eeg_linkage_summary === 'string' ? (
                  <div style={{ fontSize: 12, color: '#1e40af', lineHeight: 1.6 }}>{breakdown.eeg_linkage_summary}</div>
                ) : (
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={thStyle}>Category</th>
                        <th style={thStyle}>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(Array.isArray(breakdown.eeg_linkage_summary) ? breakdown.eeg_linkage_summary : Object.entries(breakdown.eeg_linkage_summary).map(([k, v]) => ({ category: k, description: v }))).map((item, i) => (
                        <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>{item.category}</td>
                          <td style={tdStyle}>{item.description}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </Card>
          )}
        </>
      )}

      {tab === 'by-role' && breakdown && (
        <>
          {(breakdown.by_role || []).map((group, gi) => (
            <Card key={gi} title={group.role}>
              <div style={{ marginBottom: 12 }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>{group.tests.length} test{group.tests.length !== 1 ? 's' : ''} with this role</span>
              </div>
              {group.tests.map((t, ti) => (
                <div key={ti} style={{
                  background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6,
                  padding: 12, marginBottom: 8
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: '#334155' }}>{t.name}</span>
                    <StatusBadge status={t.status || 'built'} />
                    <EegLinkBadge category={t.eeg_link} index={eegLinkCategories.indexOf(t.eeg_link)} />
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>
                    <strong>Purpose:</strong> {t.purpose}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569' }}>
                    <strong>Output:</strong> {t.output}
                  </div>
                </div>
              ))}
            </Card>
          ))}
        </>
      )}

      {tab === 'case-data' && breakdown && (
        <>
          <Card title="Tests with Case Data">
            {(breakdown.tests_with_case_data || []).map((t, i) => (
              <div key={i} style={{
                background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6,
                padding: 12, marginBottom: 8
              }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#334155', marginBottom: 8 }}>{t.name}</div>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 6 }}>Case Data Files:</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {(t.case_data || []).map((file, j) => (
                    <span key={j} style={{
                      background: '#dbeafe', color: '#1d4ed8', borderRadius: 4,
                      padding: '3px 10px', fontSize: 11, fontWeight: 500, fontFamily: 'monospace'
                    }}>
                      {file}
                    </span>
                  ))}
                </div>
              </div>
            ))}
            {(!breakdown.tests_with_case_data || breakdown.tests_with_case_data.length === 0) && (
              <div style={{ fontSize: 12, color: '#94a3b8', textAlign: 'center', padding: 24 }}>No tests with case data found.</div>
            )}
          </Card>
        </>
      )}

      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.status_legend || []).map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}>
                      <span style={{
                        background: `${r.color || '#22c55e'}22`, color: r.color || '#22c55e',
                        border: `1px solid ${r.color || '#22c55e'}55`,
                        borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
                      }}>
                        {r.status || r.value}
                      </span>
                    </td>
                    <td style={tdStyle}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="EEG Link Type Descriptions">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.eeg_link_descriptions || []).map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}>
                      <EegLinkBadge category={r.type || r.category} index={i} />
                    </td>
                    <td style={tdStyle}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Term</th>
                  <th style={thStyle}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
