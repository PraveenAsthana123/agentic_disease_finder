import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function pct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

export default function ReleaseDashboard() {
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
          axios.get(`${API_URL}/api/release/overview`),
          axios.get(`${API_URL}/api/release/breakdown`),
          axios.get(`${API_URL}/api/release/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load release data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8, animation: 'spin 1.5s linear infinite' }}>&#9881;</div>
      Loading release management data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Release data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const modelInv = overview.model_inventory || []
  const diseaseCov = overview.disease_coverage || []
  const changeActionDist = overview.change_action_distribution || []
  const dailyTrend = overview.daily_trend || []
  const actorDist = overview.actor_distribution || []

  const uploadLog = breakdown?.upload_log || []
  const expertLog = breakdown?.expert_log || []
  const hitlLog = breakdown?.hitl_log || []
  const decisionLog = breakdown?.decision_log || []
  const trainingLog = breakdown?.training_log || []
  const changeLog = breakdown?.change_log || []
  const componentChanges = breakdown?.component_changes || []
  const hourlyPattern = breakdown?.hourly_pattern || []
  const definitions = defs?.metrics || []

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'models', label: 'Models & Data' },
    { id: 'approvals', label: 'Approvals & Reviews' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const cardStyle = {
    background: '#f8fafc', border: '1px solid #e2e8f0',
    borderRadius: 8, padding: 16, textAlign: 'center'
  }
  const bigNum = { fontSize: 28, fontWeight: 700, color: '#1e293b' }
  const label = { fontSize: 12, color: '#64748b', marginTop: 4 }
  const tableStyle = {
    width: '100%', borderCollapse: 'collapse', fontSize: 13
  }
  const th = {
    textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0',
    background: '#f1f5f9', fontWeight: 600, fontSize: 12
  }
  const td = {
    padding: '6px 10px', borderBottom: '1px solid #f1f5f9'
  }

  return (
    <div style={{ padding: 20, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ marginBottom: 4 }}>Release Management Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        Dataset versions, model artefacts, approval workflow, and change log from real clinical.db data
      </p>

      {/* Tab nav */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#334155', fontWeight: 600, fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, marginBottom: 24 }}>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.total_models)}</div>
              <div style={label}>Model Artefacts</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.total_uploads)}</div>
              <div style={label}>Dataset Uploads</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.total_analyses)}</div>
              <div style={label}>Inference Runs</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.total_approvals)}</div>
              <div style={label}>Total Approvals</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{pct(s.agreement_rate)}</div>
              <div style={label}>Expert Agreement</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.training_runs)}</div>
              <div style={label}>Training Runs</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.total_changes)}</div>
              <div style={label}>Total Changes</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.signoff_count)}</div>
              <div style={label}>Sign-offs</div>
            </div>
          </div>

          {/* Daily change trend */}
          {dailyTrend.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h4 style={{ marginBottom: 8 }}>Daily Change Trend</h4>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={dailyTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="changes" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Change action distribution + Actor distribution side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 }}>
            {changeActionDist.length > 0 && (
              <div>
                <h4 style={{ marginBottom: 8 }}>Change Action Distribution</h4>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={changeActionDist}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="action" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" radius={[4,4,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {actorDist.length > 0 && (
              <div>
                <h4 style={{ marginBottom: 8 }}>Actor Activity</h4>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={actorDist} dataKey="changes" nameKey="actor" cx="50%" cy="50%"
                      outerRadius={80} label={({ actor, changes }) => `${actor}: ${changes}`}>
                      {actorDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Hourly pattern */}
          {hourlyPattern.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h4 style={{ marginBottom: 8 }}>Hourly Change Pattern</h4>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={hourlyPattern}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="changes" fill="#10b981" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* ── MODELS & DATA TAB ── */}
      {tab === 'models' && (
        <div>
          {/* Model inventory */}
          <h4 style={{ marginBottom: 8 }}>Model Artefact Inventory ({modelInv.length})</h4>
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={th}>Disease</th>
                  <th style={th}>File</th>
                  <th style={th}>Size (KB)</th>
                  <th style={th}>Last Modified</th>
                </tr>
              </thead>
              <tbody>
                {modelInv.map((m, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={td}>{m.disease}</td>
                    <td style={{ ...td, fontFamily: 'monospace', fontSize: 12 }}>{m.file}</td>
                    <td style={td}>{fmt(m.size_kb)}</td>
                    <td style={{ ...td, fontSize: 12 }}>{m.last_modified?.slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Disease coverage */}
          {diseaseCov.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h4 style={{ marginBottom: 8 }}>Disease Analysis Coverage</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={diseaseCov}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="analyses" fill="#8b5cf6" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Upload log */}
          <h4 style={{ marginBottom: 8 }}>Dataset Upload Log ({uploadLog.length})</h4>
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={th}>Patient</th>
                  <th style={th}>File</th>
                  <th style={th}>Disease</th>
                  <th style={th}>Department</th>
                  <th style={th}>Uploaded</th>
                </tr>
              </thead>
              <tbody>
                {uploadLog.slice(0, 25).map((u, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={td}>{u.patient_id}</td>
                    <td style={{ ...td, fontFamily: 'monospace', fontSize: 12 }}>{u.file}</td>
                    <td style={td}>{u.disease}</td>
                    <td style={td}>{u.department}</td>
                    <td style={{ ...td, fontSize: 12 }}>{u.uploaded_at?.slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Component changes */}
          {componentChanges.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h4 style={{ marginBottom: 8 }}>Component Change Counts</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={componentChanges} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis dataKey="component" type="category" tick={{ fontSize: 11 }} width={120} />
                  <Tooltip />
                  <Bar dataKey="changes" fill="#f59e0b" radius={[0,4,4,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* ── APPROVALS & REVIEWS TAB ── */}
      {tab === 'approvals' && (
        <div>
          {/* Approval KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 24 }}>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.expert_reviews)}</div>
              <div style={label}>Expert Reviews</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{pct(s.agreement_rate)}</div>
              <div style={label}>Agreement Rate</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.hitl_reviews)}</div>
              <div style={label}>HITL Reviews</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{pct(s.override_rate)}</div>
              <div style={label}>Override Rate</div>
            </div>
            <div style={cardStyle}>
              <div style={bigNum}>{fmt(s.clinical_decisions)}</div>
              <div style={label}>Clinical Decisions</div>
            </div>
          </div>

          {/* Expert review log */}
          <h4 style={{ marginBottom: 8 }}>Expert Review Log</h4>
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={th}>Patient</th>
                  <th style={th}>Role</th>
                  <th style={th}>Expert</th>
                  <th style={th}>Finding</th>
                  <th style={th}>Agreement</th>
                  <th style={th}>Note</th>
                  <th style={th}>Date</th>
                </tr>
              </thead>
              <tbody>
                {expertLog.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={td}>{r.patient_id}</td>
                    <td style={td}>{r.role}</td>
                    <td style={td}>{r.expert}</td>
                    <td style={{ ...td, maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.finding}</td>
                    <td style={td}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                        background: r.agreement === 'agree' ? '#dcfce7' : '#fef2f2',
                        color: r.agreement === 'agree' ? '#166534' : '#991b1b'
                      }}>{r.agreement}</span>
                    </td>
                    <td style={td}>{r.note}</td>
                    <td style={{ ...td, fontSize: 12 }}>{r.reviewed_at?.slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* HITL override log */}
          <h4 style={{ marginBottom: 8 }}>HITL Override Log</h4>
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={th}>Patient</th>
                  <th style={th}>AI Prediction</th>
                  <th style={th}>Decision</th>
                  <th style={th}>Human Decision</th>
                  <th style={th}>Reason</th>
                  <th style={th}>Date</th>
                </tr>
              </thead>
              <tbody>
                {hitlLog.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={td}>{r.patient_id}</td>
                    <td style={td}>{r.ai_prediction}</td>
                    <td style={td}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                        background: r.decision === 'override' ? '#fef2f2' : '#dcfce7',
                        color: r.decision === 'override' ? '#991b1b' : '#166534'
                      }}>{r.decision}</span>
                    </td>
                    <td style={td}>{r.human_decision}</td>
                    <td style={td}>{r.reason_code}</td>
                    <td style={{ ...td, fontSize: 12 }}>{r.reviewed_at?.slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Clinical decision log */}
          <h4 style={{ marginBottom: 8 }}>Clinical Decision Log</h4>
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={th}>Patient</th>
                  <th style={th}>AI Prediction</th>
                  <th style={th}>Confidence</th>
                  <th style={th}>Neuro Agreement</th>
                  <th style={th}>Final Decision</th>
                  <th style={th}>Reviewer</th>
                  <th style={th}>Date</th>
                </tr>
              </thead>
              <tbody>
                {decisionLog.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={td}>{r.patient_id}</td>
                    <td style={td}>{r.ai_prediction}</td>
                    <td style={td}>{r.ai_confidence}</td>
                    <td style={td}>{r.neurologist_agreement}</td>
                    <td style={td}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                        background: '#dcfce7', color: '#166534'
                      }}>{r.final_decision}</span>
                    </td>
                    <td style={td}>{r.reviewer}</td>
                    <td style={{ ...td, fontSize: 12 }}>{r.decided_at?.slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Training log */}
          <h4 style={{ marginBottom: 8 }}>Training Schedule Log</h4>
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={th}>Patient</th>
                  <th style={th}>Actor</th>
                  <th style={th}>Detail</th>
                  <th style={th}>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {trainingLog.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={td}>{r.patient_id}</td>
                    <td style={td}>{r.actor}</td>
                    <td style={{ ...td, maxWidth: 300, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.detail}</td>
                    <td style={{ ...td, fontSize: 12 }}>{r.timestamp?.slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Recent change log */}
          <h4 style={{ marginBottom: 8 }}>Recent Change Log (last 50)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={th}>Component</th>
                  <th style={th}>Action</th>
                  <th style={th}>Actor</th>
                  <th style={th}>Patient</th>
                  <th style={th}>Detail</th>
                  <th style={th}>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {changeLog.slice(0, 30).map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={td}>{r.component}</td>
                    <td style={td}>
                      <span style={{
                        padding: '2px 6px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                        background: r.action === 'sign-off' ? '#dbeafe' : '#f1f5f9',
                        color: r.action === 'sign-off' ? '#1e40af' : '#334155'
                      }}>{r.action}</span>
                    </td>
                    <td style={td}>{r.actor}</td>
                    <td style={td}>{r.patient_id}</td>
                    <td style={{ ...td, maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: 12 }}>{r.detail}</td>
                    <td style={{ ...td, fontSize: 12 }}>{r.timestamp?.slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && (
        <div>
          <h4 style={{ marginBottom: 8 }}>Metric Definitions</h4>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={th}>Metric</th>
                <th style={th}>Definition</th>
              </tr>
            </thead>
            <tbody>
              {definitions.map((d, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ ...td, fontWeight: 600, whiteSpace: 'nowrap' }}>{d.metric}</td>
                  <td style={td}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
