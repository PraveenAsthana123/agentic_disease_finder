import { useState, useEffect } from 'react'
import axios from 'axios'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8010'

const TABS = [
  { id: 'overview', label: 'App Overview' },
  { id: 'diary', label: 'Seizure Diary & Meds' },
  { id: 'definitions', label: 'Feature Guide' },
]

const badge = (val, ok, warn) => {
  const color = val >= ok ? '#16a34a' : val >= warn ? '#d97706' : '#dc2626'
  return <span style={{ background: color, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700 }}>{val}%</span>
}

const sosColor = status => ({
  sent: '#2563eb', acknowledged: '#16a34a', escalated: '#dc2626', resolved: '#6b7280',
}[status] || '#9ca3af')

export default function PatientMobileAppDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [diary, setDiary] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState(null)

  useEffect(() => {
    setLoading(true)
    setErr(null)
    const calls = {
      overview: axios.get(`${API}/api/patient-mobile/overview`),
      diary: axios.get(`${API}/api/patient-mobile/diary`),
      defs: axios.get(`${API}/api/patient-mobile/definitions`),
    }
    Promise.all([calls.overview, calls.diary, calls.defs])
      .then(([o, d, df]) => {
        setOverview(o.data)
        setDiary(d.data)
        setDefs(df.data)
      })
      .catch(e => setErr(e.message))
      .finally(() => setLoading(false))
  }, [])

  const card = (label, value, sub, color = '#2563eb') => (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 150, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 800, color }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  )

  const renderOverview = () => {
    if (!overview) return null
    const k = overview.kpis
    return (
      <div>
        {/* KPIs */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 20 }}>
          {card('Active Patients', k.active_patients, 'enrolled', '#6366f1')}
          {card('Seizures (7d)', k.seizures_last_7d, 'across cohort', '#dc2626')}
          {card('Adherence', `${k.medication_adherence_pct}%`, 'medication', k.medication_adherence_pct >= 90 ? '#16a34a' : k.medication_adherence_pct >= 75 ? '#d97706' : '#dc2626')}
          {card('Missed Doses (7d)', k.missed_doses_7d, 'doses skipped', '#f59e0b')}
          {card('Open SOS', k.open_sos_events, 'events', k.open_sos_events === 0 ? '#16a34a' : '#dc2626')}
          {card('Offline Queue', k.offline_records_pending_sync, 'awaiting sync', '#7c3aed')}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {/* Adherence Trend */}
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Medication Adherence — 14-Day Trend</div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 80 }}>
              {overview.adherence_trend_14d.map((d, i) => (
                <div key={i} title={`${d.date}: ${d.adherence_pct}%`} style={{
                  flex: 1, height: `${d.adherence_pct}%`,
                  background: d.adherence_pct >= 90 ? '#16a34a' : d.adherence_pct >= 75 ? '#d97706' : '#dc2626',
                  borderRadius: 2, opacity: 0.85,
                }} />
              ))}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b', marginTop: 4 }}>
              <span>14 days ago</span><span>Today</span>
            </div>
          </div>

          {/* Seizure Type Breakdown */}
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Seizure Type Breakdown</div>
            {overview.seizure_type_breakdown.slice(0, 6).map((t, i) => {
              const max = overview.seizure_type_breakdown[0].count
              return (
                <div key={i} style={{ marginBottom: 6 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>
                    <span>{t.type}</span><span>{t.count}</span>
                  </div>
                  <div style={{ background: '#334155', borderRadius: 3, height: 6 }}>
                    <div style={{ width: `${(t.count / max) * 100}%`, height: '100%', background: '#6366f1', borderRadius: 3 }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {/* Trigger Breakdown */}
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Reported Seizure Triggers</div>
            {overview.trigger_breakdown.slice(0, 7).map((t, i) => {
              const max = overview.trigger_breakdown[0].count
              return (
                <div key={i} style={{ marginBottom: 6 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>
                    <span>{t.trigger}</span><span>{t.count}</span>
                  </div>
                  <div style={{ background: '#334155', borderRadius: 3, height: 6 }}>
                    <div style={{ width: `${(t.count / max) * 100}%`, height: '100%', background: '#f59e0b', borderRadius: 3 }} />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Device Pairing Summary */}
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Paired Device Distribution</div>
            {overview.device_pairing_summary.map((d, i) => {
              const max = overview.device_pairing_summary[0].count
              return (
                <div key={i} style={{ marginBottom: 6 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>
                    <span>{d.device}</span><span>{d.count}</span>
                  </div>
                  <div style={{ background: '#334155', borderRadius: 3, height: 6 }}>
                    <div style={{ width: `${(d.count / max) * 100}%`, height: '100%', background: '#0ea5e9', borderRadius: 3 }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Recent SOS Events */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Recent SOS Events</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                {['Event ID', 'Patient', 'Trigger', 'Triggered At', 'Resp Time', 'Status', 'Escalated'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.sos_recent.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{s.event_id}</td>
                  <td style={{ padding: '5px 8px', color: '#e2e8f0' }}>{s.patient_id}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{s.trigger_source.replace(/_/g, ' ')}</td>
                  <td style={{ padding: '5px 8px', color: '#64748b', fontSize: 11 }}>{s.triggered_at?.slice(0, 16)}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{s.response_time_sec}s</td>
                  <td style={{ padding: '5px 8px' }}>
                    <span style={{ background: sosColor(s.status), color: '#fff', borderRadius: 4, padding: '1px 7px', fontSize: 11 }}>{s.status}</span>
                  </td>
                  <td style={{ padding: '5px 8px', color: s.escalated_to_emergency ? '#dc2626' : '#16a34a' }}>
                    {s.escalated_to_emergency ? '🚨 Yes' : 'No'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Offline Sync Queue */}
        {overview.offline_sync_queue.length > 0 && (
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Offline Sync Queue (Awaiting Connectivity)</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                  {['Patient', 'Queued Records', 'Type', 'Oldest (h)'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {overview.offline_sync_queue.map((q, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '5px 8px', color: '#e2e8f0' }}>{q.patient_id}</td>
                    <td style={{ padding: '5px 8px', color: '#f59e0b', fontWeight: 700 }}>{q.queued_records}</td>
                    <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{q.type.replace(/_/g, ' ')}</td>
                    <td style={{ padding: '5px 8px', color: q.oldest_record_age_h > 24 ? '#dc2626' : '#94a3b8' }}>{q.oldest_record_age_h}h</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    )
  }

  const renderDiary = () => {
    if (!diary) return null
    return (
      <div>
        {/* Summary cards */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 20 }}>
          {card('Diary Entries', diary.summary.diary_entries, 'last 30 days', '#6366f1')}
          {card('Med Records', diary.summary.medication_records, 'last 14 days', '#16a34a')}
          {card('Symptom Entries', diary.summary.symptom_entries, 'last 21 days', '#f59e0b')}
          {card('SOS Events', diary.summary.sos_events, 'total', '#dc2626')}
        </div>

        {/* Seizure Diary */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Seizure Diary</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                {['Entry ID', 'Patient', 'Date', 'Type', 'Duration', 'Trigger', 'Severity', 'Witnessed'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {diary.seizure_diary.slice(0, 15).map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #0f172a' }}>
                  <td style={{ padding: '5px 8px', color: '#64748b' }}>{e.entry_id}</td>
                  <td style={{ padding: '5px 8px', color: '#e2e8f0' }}>{e.patient_id}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8', fontSize: 11 }}>{e.date}</td>
                  <td style={{ padding: '5px 8px', color: '#c4b5fd', fontSize: 11 }}>{e.seizure_type}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{e.duration_seconds}s</td>
                  <td style={{ padding: '5px 8px', color: '#fbbf24', fontSize: 11 }}>{e.trigger}</td>
                  <td style={{ padding: '5px 8px' }}>
                    <span style={{
                      background: e.severity === 'severe' ? '#dc2626' : e.severity === 'moderate' ? '#d97706' : '#16a34a',
                      color: '#fff', borderRadius: 4, padding: '1px 7px', fontSize: 11,
                    }}>{e.severity}</span>
                  </td>
                  <td style={{ padding: '5px 8px', color: e.witnessed ? '#16a34a' : '#64748b' }}>{e.witnessed ? 'Yes' : 'No'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Medication Log */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Medication Log</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                {['Patient', 'Date', 'Medication', 'Scheduled', 'Taken', 'Late', 'Missed'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {diary.medication_log.slice(0, 15).map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #0f172a' }}>
                  <td style={{ padding: '5px 8px', color: '#e2e8f0' }}>{m.patient_id}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8', fontSize: 11 }}>{m.date}</td>
                  <td style={{ padding: '5px 8px', color: '#6ee7b7', fontSize: 11 }}>{m.medication}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{m.scheduled_time}</td>
                  <td style={{ padding: '5px 8px', color: m.taken ? '#16a34a' : '#dc2626', fontWeight: 700 }}>{m.taken ? '✓' : '✗'}</td>
                  <td style={{ padding: '5px 8px', color: m.taken_late ? '#f59e0b' : '#64748b' }}>{m.taken_late ? 'Late' : '—'}</td>
                  <td style={{ padding: '5px 8px', color: m.missed ? '#dc2626' : '#64748b' }}>{m.missed ? 'Missed' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Device Pairing */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Patient Device Pairing</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {diary.patient_device_pairing.slice(0, 12).map((p, i) => (
              <div key={i} style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px', minWidth: 160 }}>
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>{p.patient_id}</div>
                <div style={{ fontSize: 11, color: p.paired ? '#6ee7b7' : '#64748b', fontWeight: 700 }}>
                  {p.paired ? '🔗 ' : '⭕ '}{p.device}
                </div>
                {p.paired && (
                  <>
                    <div style={{ fontSize: 11, color: '#64748b' }}>Battery: {p.battery_pct}%</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>Signal: {p.signal_quality}</div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const renderDefs = () => {
    if (!defs) return null
    const ao = defs.app_overview
    return (
      <div>
        {/* App Overview */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>📱 {ao.name}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Type</div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>{ao.type}</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Sync</div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>{ao.data_sync}</div>
            </div>
          </div>
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>Core Features</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {ao.core_features.map((f, i) => (
                <div key={i} style={{ background: '#0f172a', borderRadius: 4, padding: '4px 10px', fontSize: 11, color: '#c4b5fd' }}>{f}</div>
              ))}
            </div>
          </div>
        </div>

        {/* SOS Escalation */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>🚨 SOS Escalation Pathway</div>
          <div style={{ display: 'flex', gap: 0, alignItems: 'stretch' }}>
            {defs.sos_escalation.map((s, i) => (
              <div key={i} style={{ flex: 1, background: '#0f172a', padding: '12px 14px', borderLeft: i ? '2px solid #334155' : 'none' }}>
                <div style={{ fontSize: 11, color: '#6366f1', fontWeight: 700, marginBottom: 4 }}>Step {s.step}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{s.trigger}</div>
                <div style={{ fontSize: 11, color: '#fbbf24' }}>{s.action}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Adherence Thresholds */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Medication Adherence Thresholds</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                {['Level', 'Range', 'Action'].map(h => <th key={h} style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 600 }}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {defs.adherence_thresholds.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #0f172a' }}>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={{ background: t.level === 'High' ? '#16a34a' : t.level === 'Moderate' ? '#d97706' : '#dc2626', color: '#fff', borderRadius: 4, padding: '1px 8px', fontSize: 11 }}>{t.level}</span>
                  </td>
                  <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{t.range}</td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Paired Devices */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Supported Paired Devices</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {defs.paired_devices.map((d, i) => (
              <div key={i} style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px', flex: 1, minWidth: 180 }}>
                <div style={{ fontSize: 12, color: '#6ee7b7', fontWeight: 700, marginBottom: 4 }}>📡 {d.device}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>{d.data}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Offline Architecture */}
        <div style={{ background: '#1e293b', borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginBottom: 10 }}>Offline-First Architecture</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {Object.entries(defs.offline_architecture).map(([k, v]) => (
              <div key={k} style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px' }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 3 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: 20, background: '#0f172a', minHeight: '100vh', color: '#e2e8f0' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 800, color: '#f1f5f9' }}>📱 Patient Mobile App</h2>
        <p style={{ margin: '4px 0 0', fontSize: 12, color: '#64748b' }}>
          Seizure diary · Medication adherence · SOS alerts · Offline-first sync · Device pairing
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, borderBottom: '1px solid #334155', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: tab === t.id ? '#6366f1' : 'transparent',
            color: tab === t.id ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: '4px 4px 0 0', padding: '7px 16px',
            cursor: 'pointer', fontSize: 12, fontWeight: tab === t.id ? 700 : 400,
          }}>{t.label}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#64748b', fontSize: 13 }}>Loading patient mobile app data…</div>}
      {err && <div style={{ color: '#dc2626', fontSize: 13 }}>Error: {err}</div>}
      {!loading && !err && (
        <>
          {tab === 'overview' && renderOverview()}
          {tab === 'diary' && renderDiary()}
          {tab === 'definitions' && renderDefs()}
        </>
      )}
    </div>
  )
}
