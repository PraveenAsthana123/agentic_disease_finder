import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const RISK_COLORS = {
  Low: '#22c55e',
  Moderate: '#eab308',
  High: '#ef4444',
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
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

function RiskBadge({ level }) {
  const color = RISK_COLORS[level] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 700,
      background: color + '22', color, border: `1px solid ${color}44`, letterSpacing: '0.02em'
    }}>{level || 'Unknown'}</span>
  )
}

function InfoBox({ icon, children, color }) {
  return (
    <div style={{
      background: (color || '#3b82f6') + '0d', border: `1px solid ${color || '#3b82f6'}33`,
      borderRadius: 10, padding: '12px 16px', display: 'flex', gap: 10, alignItems: 'flex-start'
    }}>
      {icon && <span style={{ fontSize: 18, flexShrink: 0 }}>{icon}</span>}
      <div style={{ fontSize: 13, color: '#334155', lineHeight: 1.55 }}>{children}</div>
    </div>
  )
}

const TABS = ['Overview', 'My Results', 'Medications', 'Next Steps', 'Glossary']

export default function PatientFacingReportDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/patient-facing-report/overview`),
      axios.get(`${API}/patient-facing-report/breakdown`),
      axios.get(`${API}/patient-facing-report/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 32, marginBottom: 12 }}>...</div>
      Loading your report — please wait a moment...
    </div>
  )
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>
      {ov?.error || 'Your report is not yet available. Please check back later.'}
    </div>
  )

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f4c81' }}>Your Health Report</h2>
        <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>
          A friendly summary of your brain health data — reviewed by your care team.
          This report does not provide a diagnosis.
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#0f4c81' : '#f1f5f9',
            color: tab === i ? '#fff' : '#475569',
            fontWeight: tab === i ? 600 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} riskDist={ov.risk_distribution} recentReports={ov.recent_reports} />}
      {tab === 1 && <MyResultsTab data={bd} />}
      {tab === 2 && <MedicationsTab data={bd} />}
      {tab === 3 && <NextStepsTab data={bd} />}
      {tab === 4 && <GlossaryTab definitions={df} />}
    </div>
  )
}

/* ─── Tab 1: Overview ─────────────────────────────────────────────────── */
function OverviewTab({ kpis, riskDist, recentReports }) {
  const pieData = riskDist
    ? Object.entries(riskDist).map(([k, v]) => ({ name: k, value: v, color: RISK_COLORS[k] || '#94a3b8' }))
    : []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      {/* KPIs */}
      <Card span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16 }}>
          <KPI label="Total Reports" value={fmt(kpis.total_reports)} sub="on file" />
          <KPI label="Patients with Follow-Up" value={fmt(kpis.patients_with_followup)} color="#0f4c81" />
          <KPI label="Average Risk Score" value={fmt(kpis.avg_risk_score)} sub="lower is better" color={kpis.avg_risk_score >= 0.7 ? '#ef4444' : kpis.avg_risk_score >= 0.4 ? '#eab308' : '#22c55e'} />
          <KPI label="Reports This Month" value={fmt(kpis.reports_this_month)} sub="recent activity" />
          <KPI label="Patients Monitored" value={fmt(kpis.patients_monitored)} sub="active" color="#10b981" />
        </div>
      </Card>

      {/* Risk Distribution Pie */}
      <Card title="Risk Level Overview">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name"
              cx="50%" cy="50%" innerRadius={45} outerRadius={80} paddingAngle={3}>
              {pieData.map((d, i) => <Cell key={i} fill={d.color} />)}
            </Pie>
            <Tooltip formatter={(v, n) => [`${v} patients`, n]} />
          </PieChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 8, flexWrap: 'wrap' }}>
          {pieData.map(d => (
            <span key={d.name} style={{ fontSize: 12, color: '#475569', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 5, background: d.color }} />
              {d.name}: {d.value}
            </span>
          ))}
        </div>
      </Card>

      {/* Recent Reports Table */}
      <Card title="Recent Reports" span={2}>
        <div style={{ maxHeight: 280, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', background: '#f8fafc' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left', color: '#64748b' }}>Patient</th>
                <th style={{ padding: '6px 10px', textAlign: 'left', color: '#64748b' }}>Date</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', color: '#64748b' }}>Risk Level</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', color: '#64748b' }}>Follow-Up</th>
                <th style={{ padding: '6px 10px', textAlign: 'left', color: '#64748b' }}>Summary</th>
              </tr>
            </thead>
            <tbody>
              {(recentReports || []).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id || `Patient ${i + 1}`}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{r.date || '--'}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}><RiskBadge level={r.risk_level} /></td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                    <span style={{ fontSize: 11, color: r.follow_up ? '#10b981' : '#94a3b8' }}>
                      {r.follow_up ? 'Scheduled' : 'Pending'}
                    </span>
                  </td>
                  <td style={{ padding: '6px 10px', color: '#475569' }}>{r.summary || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {(!recentReports || recentReports.length === 0) && (
            <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>
              No recent reports found.
            </div>
          )}
        </div>
      </Card>

      {/* Encouragement banner */}
      <Card span={3}>
        <InfoBox icon="💙" color="#3b82f6">
          Your care team is monitoring your data and will reach out if anything needs attention.
          These results are a helpful guide — not a final verdict. Keep attending your scheduled visits
          and reach out anytime you have questions.
        </InfoBox>
      </Card>
    </div>
  )
}

/* ─── Tab 2: My Results ───────────────────────────────────────────────── */
function MyResultsTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b', padding: 16 }}>Your results are being prepared. Please check back soon.</div>

  const patients = data.patient_biomarkers || []
  const trendData = data.risk_trend || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Friendly intro */}
      <Card span={1}>
        <InfoBox icon="📋" color="#10b981">
          Below you'll find a simplified summary of your brain health measurements.
          Each marker is explained in plain language. The "risk badge" reflects patterns
          your care team has noted — it is NOT a diagnosis.
        </InfoBox>
      </Card>

      {/* Per-patient biomarker summaries */}
      {patients.map((p, i) => (
        <Card key={i} title={`Patient ${p.patient_id || i + 1} — Health Summary`}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <span style={{ fontSize: 13, color: '#64748b' }}>Last updated: {p.last_updated || '--'}</span>
            <RiskBadge level={p.risk_level} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            {(p.biomarkers || []).map((b, j) => {
              const barColor = b.status === 'Normal' ? '#22c55e' : b.status === 'Borderline' ? '#eab308' : '#ef4444'
              const pct = b.value != null && b.max != null ? Math.min(100, Math.max(0, (b.value / b.max) * 100)) : 50
              return (
                <div key={j} style={{ background: '#f8fafc', borderRadius: 8, padding: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>{b.label}</span>
                    <span style={{ fontSize: 11, color: barColor, fontWeight: 600 }}>{b.status || '--'}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{b.plain_description || b.description || ''}</div>
                  <div style={{ background: '#e2e8f0', borderRadius: 4, height: 6 }}>
                    <div style={{ width: `${pct}%`, background: barColor, borderRadius: 4, height: '100%', transition: 'width 0.4s' }} />
                  </div>
                  {b.value != null && (
                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                      Value: {fmt(b.value)}{b.unit ? ` ${b.unit}` : ''}
                      {b.reference_range ? ` (normal: ${b.reference_range})` : ''}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          {p.care_note && (
            <div style={{ marginTop: 12 }}>
              <InfoBox icon="💬" color="#8b5cf6">{p.care_note}</InfoBox>
            </div>
          )}
        </Card>
      ))}

      {patients.length === 0 && (
        <Card>
          <div style={{ textAlign: 'center', padding: 20, color: '#94a3b8', fontSize: 13 }}>
            No biomarker data found. Your care team will update this section after your next visit.
          </div>
        </Card>
      )}

      {/* Risk trend chart */}
      {trendData.length > 0 && (
        <Card title="Your Risk Score Over Time — Trending in the Right Direction!">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <Tooltip formatter={v => [`${(v * 100).toFixed(1)}%`, 'Risk Score']} />
              <Line type="monotone" dataKey="risk_score" stroke="#3b82f6" strokeWidth={2.5} dot={{ r: 4 }} name="Risk Score" />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
            A lower score means lower detected risk. Speak with your care team about any changes.
          </div>
        </Card>
      )}
    </div>
  )
}

/* ─── Tab 3: Medications ──────────────────────────────────────────────── */
function MedicationsTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b', padding: 16 }}>Medication data is not yet available.</div>

  const meds = data.medications || []
  const adherenceTips = data.adherence_tips || []
  const adherenceData = data.adherence_history || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card span={2}>
        <InfoBox icon="💊" color="#10b981">
          Taking your medications as prescribed is one of the most important things you can do for your brain health.
          If you are having trouble with any medication, please tell your care team — there are always options.
        </InfoBox>
      </Card>

      {/* Medication list */}
      <Card title="Your Current Medications" span={2}>
        {meds.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
            {meds.map((m, i) => (
              <div key={i} style={{
                background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 10, padding: 14
              }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#15803d', marginBottom: 4 }}>{m.name}</div>
                <div style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{m.purpose || ''}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                  {m.dosage && <span style={{ fontSize: 11, color: '#64748b' }}>Dose: <strong>{m.dosage}</strong></span>}
                  {m.frequency && <span style={{ fontSize: 11, color: '#64748b' }}>When: <strong>{m.frequency}</strong></span>}
                  {m.route && <span style={{ fontSize: 11, color: '#64748b' }}>How: <strong>{m.route}</strong></span>}
                  {m.prescriber && <span style={{ fontSize: 11, color: '#64748b' }}>Prescribed by: <strong>{m.prescriber}</strong></span>}
                </div>
                {m.reminder && (
                  <div style={{ marginTop: 8, fontSize: 11, color: '#0369a1', background: '#e0f2fe', borderRadius: 6, padding: '4px 8px' }}>
                    Reminder: {m.reminder}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div style={{ color: '#94a3b8', fontSize: 13, padding: 12 }}>No medications on record yet.</div>
        )}
      </Card>

      {/* Adherence chart */}
      {adherenceData.length > 0 && (
        <Card title="Medication Adherence — Keep It Up!">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={adherenceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Tooltip formatter={v => [`${v}%`, 'Adherence']} />
              <Bar dataKey="adherence_pct" name="Adherence" radius={[4, 4, 0, 0]}>
                {adherenceData.map((d, i) => (
                  <Cell key={i} fill={d.adherence_pct >= 80 ? '#22c55e' : d.adherence_pct >= 60 ? '#eab308' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
            80% or above is excellent — aim for green every week!
          </div>
        </Card>
      )}

      {/* Adherence tips */}
      {adherenceTips.length > 0 && (
        <Card title="Tips for Taking Your Medications" span={adherenceData.length > 0 ? 1 : 2}>
          <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: 13, color: '#475569' }}>
            {adherenceTips.map((tip, i) => (
              <li key={i} style={{ marginBottom: 8, lineHeight: 1.55 }}>{tip}</li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}

/* ─── Tab 4: Next Steps ───────────────────────────────────────────────── */
function NextStepsTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b', padding: 16 }}>Next steps are being prepared by your care team.</div>

  const schedule = data.followup_schedule || []
  const lifestyle = data.lifestyle_guidance || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card span={2}>
        <InfoBox icon="🌟" color="#10b981">
          Small, consistent steps make a big difference. Your care team has put together
          a personalised plan to help you stay well. You are doing great by staying engaged with your health!
        </InfoBox>
      </Card>

      {/* Follow-up schedule */}
      <Card title="Your Upcoming Appointments" span={2}>
        {schedule.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {schedule.map((s, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 14,
                background: '#f8fafc', borderRadius: 10, padding: '12px 16px',
                borderLeft: `4px solid ${s.priority === 'Urgent' ? '#ef4444' : s.priority === 'Soon' ? '#eab308' : '#22c55e'}`
              }}>
                <div style={{ minWidth: 80, textAlign: 'center' }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#1e293b' }}>{s.date || '--'}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>{s.time || ''}</div>
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#334155' }}>{s.appointment_type || 'Follow-up visit'}</div>
                  {s.location && <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{s.location}</div>}
                  {s.notes && <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{s.notes}</div>}
                </div>
                {s.priority && (
                  <span style={{
                    fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6,
                    background: s.priority === 'Urgent' ? '#fef2f2' : s.priority === 'Soon' ? '#fefce8' : '#f0fdf4',
                    color: s.priority === 'Urgent' ? '#ef4444' : s.priority === 'Soon' ? '#ca8a04' : '#15803d'
                  }}>{s.priority}</span>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div style={{ color: '#94a3b8', fontSize: 13, padding: 12 }}>
            No upcoming appointments scheduled yet. Your care team will be in touch shortly.
          </div>
        )}
      </Card>

      {/* Lifestyle guidance cards */}
      {lifestyle.length > 0 && lifestyle.map((g, i) => (
        <Card key={i} title={g.category || 'Lifestyle Tip'}>
          <div style={{ marginBottom: 10 }}>
            <InfoBox icon={g.icon || '🌱'} color={COLORS[i % COLORS.length]}>
              {g.headline || g.description}
            </InfoBox>
          </div>
          {g.tips && g.tips.length > 0 && (
            <ul style={{ margin: '8px 0 0', padding: '0 0 0 18px', fontSize: 13, color: '#475569' }}>
              {g.tips.map((tip, j) => (
                <li key={j} style={{ marginBottom: 6, lineHeight: 1.55 }}>{tip}</li>
              ))}
            </ul>
          )}
          {g.goal && (
            <div style={{ marginTop: 10, fontSize: 12, fontWeight: 600, color: COLORS[i % COLORS.length] }}>
              Goal: {g.goal}
            </div>
          )}
        </Card>
      ))}

      {lifestyle.length === 0 && (
        <Card span={2} title="General Lifestyle Guidance">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            {[
              { icon: '😴', title: 'Sleep Well', tip: 'Aim for 7–9 hours of quality sleep each night.' },
              { icon: '🚶', title: 'Stay Active', tip: 'Even a 30-minute daily walk supports brain health.' },
              { icon: '🥦', title: 'Eat Mindfully', tip: 'A balanced diet with plenty of vegetables and omega-3s helps.' },
              { icon: '🧘', title: 'Manage Stress', tip: 'Try mindfulness, breathing exercises, or time in nature.' },
            ].map((c, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 10, padding: 14 }}>
                <div style={{ fontSize: 22, marginBottom: 6 }}>{c.icon}</div>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>{c.title}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.55 }}>{c.tip}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

/* ─── Tab 5: Glossary ─────────────────────────────────────────────────── */
function GlossaryTab({ definitions }) {
  if (!definitions?.available) return <div style={{ color: '#f59e0b', padding: 16 }}>Glossary is loading...</div>

  const terms = definitions.terms || []
  const disclaimer = definitions.disclaimer || ''
  const aiNote = definitions.ai_report_note || ''

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Friendly intro */}
      <Card>
        <InfoBox icon="📖" color="#3b82f6">
          Not sure what a term means? This glossary explains the words used in your report
          in plain, everyday language. If anything is still unclear, please ask your care team — no question is too small.
        </InfoBox>
      </Card>

      {/* Glossary terms */}
      <Card title="Terms Explained">
        {terms.length > 0 ? (
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>
              {terms.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{
                    padding: '10px 14px', fontWeight: 700, color: '#0f4c81',
                    verticalAlign: 'top', whiteSpace: 'nowrap', width: 220
                  }}>{t.term}</td>
                  <td style={{ padding: '10px 14px', color: '#475569', lineHeight: 1.6 }}>{t.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ color: '#94a3b8', fontSize: 13, padding: 12 }}>No glossary terms available yet.</div>
        )}
      </Card>

      {/* AI report note */}
      {aiNote && (
        <Card title="About AI-Assisted Reports">
          <InfoBox icon="🤖" color="#8b5cf6">{aiNote}</InfoBox>
        </Card>
      )}

      {/* Disclaimer */}
      <Card title="Important Disclaimer">
        <div style={{
          background: '#fafafa', border: '1px solid #e2e8f0', borderRadius: 8,
          padding: 14, fontSize: 12, color: '#64748b', lineHeight: 1.7
        }}>
          {disclaimer || (
            <>
              <strong>This report is for informational purposes only.</strong> It does not constitute a medical diagnosis,
              medical advice, or a substitute for professional clinical evaluation. All findings have been generated
              using AI-assisted analysis of brain signal data and must be interpreted by a qualified healthcare
              professional. If you have any concerns about your health, please contact your doctor or care team directly.
              In an emergency, call emergency services immediately.
            </>
          )}
        </div>
      </Card>
    </div>
  )
}
