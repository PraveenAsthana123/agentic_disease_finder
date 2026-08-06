'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sosColor = s => ({ sent: 'warning', resolved: 'success', escalated: 'danger' }[s] || 'secondary');
const adhereBadge = p => p >= 85 ? 'success' : p >= 70 ? 'warning' : 'danger';
const triggerColor = t => ({ manual_button: 'warning', auto_seizure_detection: 'danger', fall_detection: 'info' }[t] || 'secondary');

export default function PatientMobileAppDashboard() {
  const [ov, setOv] = useState(null);
  const [diary, setDiary] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('date');
  const [sortDir, setSortDir] = useState('desc');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/patient-mobile/overview`).then(r => r.json()),
      fetch(`${API}/api/patient-mobile/diary`).then(r => r.json()),
      fetch(`${API}/api/patient-mobile/definitions`).then(r => r.json()),
    ]).then(([o, d, df]) => { setOv(o); setDiary(d); setDefs(df); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Patient Mobile App data…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'diary', label: '📓 Seizure Diary' },
    { id: 'sos', label: '🆘 SOS Events' },
    { id: 'sync', label: '🔄 Offline Sync' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const kpis = ov.kpis || {};
  const adhereTrend = ov.adherence_trend_14d || [];
  const seizureTypes = ov.seizure_type_breakdown || [];
  const triggers = ov.trigger_breakdown || [];
  const sosEvents = ov.sos_recent || [];
  const syncQueue = ov.offline_sync_queue || [];
  const devicePairing = ov.device_pairing_summary || [];
  const diaryEntries = diary?.seizure_diary || [];

  const filteredDiary = diaryEntries
    .filter(e => !search || e.patient_id.toLowerCase().includes(search.toLowerCase()) ||
      e.seizure_type.toLowerCase().includes(search.toLowerCase()) ||
      (e.trigger || '').toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const dir = sortDir === 'asc' ? 1 : -1;
      if (sortBy === 'duration_seconds') return (a.duration_seconds - b.duration_seconds) * dir;
      if (sortBy === 'patient_id') return a.patient_id.localeCompare(b.patient_id) * dir;
      return a.date.localeCompare(b.date) * dir;
    });

  const maxAdhere = Math.max(...adhereTrend.map(d => d.adherence_pct), 1);
  const maxType = Math.max(...seizureTypes.map(s => s.count), 1);
  const maxTrigger = Math.max(...triggers.map(t => t.count), 1);

  return (
    <div className="container-fluid py-3">
      <h4 className="mb-1 fw-bold">📱 Patient Mobile App</h4>
      <p className="text-muted small mb-3">
        iOS + Android — offline-first seizure diary, medication log, SOS alerts, device pairing
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div className="row g-3 mb-4">
            {[
              { label: 'Active Patients', val: kpis.active_patients, cls: 'primary', icon: '👤' },
              { label: 'Seizures Last 7d', val: kpis.seizures_last_7d, cls: 'danger', icon: '⚡' },
              { label: 'Med Adherence', val: `${kpis.medication_adherence_pct}%`, cls: adhereBadge(kpis.medication_adherence_pct), icon: '💊' },
              { label: 'Missed Doses (7d)', val: kpis.missed_doses_7d, cls: 'warning', icon: '❌' },
              { label: 'Open SOS Events', val: kpis.open_sos_events, cls: 'danger', icon: '🆘' },
              { label: 'Escalated SOS', val: kpis.escalated_sos_total, cls: 'dark', icon: '🚨' },
              { label: 'Offline Pending Sync', val: kpis.offline_records_pending_sync, cls: 'secondary', icon: '🔄' },
              { label: 'Avg Diary / Patient', val: kpis.avg_diary_entries_per_patient, cls: 'info', icon: '📓' },
            ].map(k => (
              <div key={k.label} className="col-6 col-md-3">
                <div className={`card border-${k.cls} h-100`}>
                  <div className="card-body py-2 px-3">
                    <div className="text-muted small">{k.icon} {k.label}</div>
                    <div className={`fs-4 fw-bold text-${k.cls}`}>{k.val}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row g-3 mb-4">
            {/* Adherence 14-day trend */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header small fw-semibold">💊 Medication Adherence — 14-day Trend</div>
                <div className="card-body">
                  <div className="d-flex align-items-end gap-1" style={{ height: 120 }}>
                    {adhereTrend.map((d, i) => (
                      <div key={i} className="d-flex flex-column align-items-center flex-fill">
                        <div
                          className={`rounded-top bg-${adhereBadge(d.adherence_pct)}`}
                          style={{ width: '100%', height: `${(d.adherence_pct / 100) * 100}px`, minHeight: 4 }}
                          title={`${d.date}: ${d.adherence_pct}%`}
                        />
                      </div>
                    ))}
                  </div>
                  <div className="d-flex justify-content-between mt-1">
                    <span className="text-muted" style={{ fontSize: 10 }}>{adhereTrend[0]?.date?.slice(5)}</span>
                    <span className="text-muted" style={{ fontSize: 10 }}>{adhereTrend.at(-1)?.date?.slice(5)}</span>
                  </div>
                  <div className="mt-2 text-muted small">Avg: {(adhereTrend.reduce((a, d) => a + d.adherence_pct, 0) / adhereTrend.length).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            {/* Seizure type breakdown */}
            <div className="col-md-3">
              <div className="card h-100">
                <div className="card-header small fw-semibold">⚡ Seizure Types</div>
                <div className="card-body">
                  {seizureTypes.map((s, i) => (
                    <div key={i} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{s.type}</span>
                        <span className="fw-bold">{s.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-danger" style={{ width: `${(s.count / maxType) * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Trigger breakdown */}
            <div className="col-md-3">
              <div className="card h-100">
                <div className="card-header small fw-semibold">🎯 Top Triggers</div>
                <div className="card-body">
                  {triggers.map((t, i) => (
                    <div key={i} className="mb-2">
                      <div className="d-flex justify-content-between small mb-1">
                        <span>{t.trigger}</span>
                        <span className="fw-bold">{t.count}</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-warning" style={{ width: `${(t.count / maxTrigger) * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Device Pairing */}
          <div className="card mb-3">
            <div className="card-header small fw-semibold">🔗 Device Pairing Summary</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Device</th><th>Patients Paired</th></tr>
                </thead>
                <tbody>
                  {devicePairing.map((d, i) => (
                    <tr key={i}>
                      <td>{d.device}</td>
                      <td><span className="badge bg-primary rounded-pill">{d.count}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── Seizure Diary ── */}
      {tab === 'diary' && (
        <>
          <div className="row g-2 mb-3">
            <div className="col-md-6">
              <input
                className="form-control form-control-sm"
                placeholder="Search patient, type, trigger…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <select className="form-select form-select-sm" value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option value="date">Sort by Date</option>
                <option value="patient_id">Sort by Patient</option>
                <option value="duration_seconds">Sort by Duration</option>
              </select>
            </div>
            <div className="col-md-3">
              <select className="form-select form-select-sm" value={sortDir} onChange={e => setSortDir(e.target.value)}>
                <option value="desc">Newest first</option>
                <option value="asc">Oldest first</option>
              </select>
            </div>
          </div>
          <div className="card">
            <div className="card-header small fw-semibold">📓 Seizure Diary — {filteredDiary.length} entries</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Entry ID</th>
                      <th>Patient</th>
                      <th>Date</th>
                      <th>Seizure Type</th>
                      <th>Duration (s)</th>
                      <th>Trigger</th>
                      <th>Severity</th>
                      <th>Witnessed</th>
                      <th>Via</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDiary.slice(0, 50).map((e, i) => (
                      <tr key={i}>
                        <td className="font-monospace small">{e.entry_id}</td>
                        <td>{e.patient_id}</td>
                        <td>{e.date}</td>
                        <td>{e.seizure_type}</td>
                        <td>{e.duration_seconds}s</td>
                        <td>{e.trigger || '—'}</td>
                        <td>
                          <span className={`badge bg-${e.severity === 'severe' ? 'danger' : e.severity === 'moderate' ? 'warning' : 'success'}`}>
                            {e.severity}
                          </span>
                        </td>
                        <td>{e.witnessed ? '✅' : '—'}</td>
                        <td><span className="badge bg-secondary">{e.recorded_via}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {filteredDiary.length > 50 && (
                <div className="p-2 text-muted small text-center">Showing 50 of {filteredDiary.length} entries</div>
              )}
            </div>
          </div>
        </>
      )}

      {/* ── SOS Events ── */}
      {tab === 'sos' && (
        <div className="card">
          <div className="card-header small fw-semibold">🆘 Recent SOS Events — {sosEvents.length} shown</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr>
                  <th>Event ID</th>
                  <th>Patient</th>
                  <th>Triggered At</th>
                  <th>Source</th>
                  <th>Status</th>
                  <th>Response (s)</th>
                  <th>Location Shared</th>
                  <th>Caregiver Notified</th>
                  <th>Escalated</th>
                </tr>
              </thead>
              <tbody>
                {sosEvents.map((ev, i) => (
                  <tr key={i}>
                    <td className="font-monospace small">{ev.event_id}</td>
                    <td>{ev.patient_id}</td>
                    <td className="small">{new Date(ev.triggered_at).toLocaleString()}</td>
                    <td>
                      <span className={`badge bg-${triggerColor(ev.trigger_source)}`}>
                        {ev.trigger_source.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${sosColor(ev.status)}`}>{ev.status}</span>
                    </td>
                    <td>{ev.response_time_sec}s</td>
                    <td>{ev.location_shared ? '📍 Yes' : '—'}</td>
                    <td>{ev.caregiver_notified ? '✅' : '—'}</td>
                    <td>{ev.escalated_to_emergency ? '🚨 Yes' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Offline Sync Queue ── */}
      {tab === 'sync' && (
        <>
          <div className="alert alert-info small mb-3">
            <strong>Offline-first architecture:</strong> All data captured offline is stored in local SQLite,
            then synced to the backend when connectivity is restored.
            {kpis.offline_records_pending_sync} records currently queued.
          </div>
          <div className="card">
            <div className="card-header small fw-semibold">🔄 Offline Sync Queue — {syncQueue.length} patients with pending records</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Patient</th><th>Queued Records</th><th>Oldest Record Age (h)</th><th>Type</th><th>Urgency</th></tr>
                </thead>
                <tbody>
                  {syncQueue.map((s, i) => (
                    <tr key={i}>
                      <td>{s.patient_id}</td>
                      <td><span className="badge bg-secondary">{s.queued_records}</span></td>
                      <td>{s.oldest_record_age_h}h</td>
                      <td><span className="badge bg-info">{s.type}</span></td>
                      <td>
                        <span className={`badge bg-${s.oldest_record_age_h > 24 ? 'danger' : s.oldest_record_age_h > 8 ? 'warning' : 'success'}`}>
                          {s.oldest_record_age_h > 24 ? 'Stale' : s.oldest_record_age_h > 8 ? 'Pending' : 'Recent'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header small fw-semibold">📱 App Overview</div>
              <div className="card-body">
                <p className="small"><strong>Type:</strong> {defs.app_overview?.type}</p>
                <p className="small mb-1"><strong>Modes:</strong></p>
                <ul className="small">
                  {(defs.app_overview?.modes || []).map((m, i) => <li key={i}>{m}</li>)}
                </ul>
                <p className="small mb-1"><strong>Core Features:</strong></p>
                <ul className="small">
                  {(defs.app_overview?.core_features || []).map((f, i) => <li key={i}>{f}</li>)}
                </ul>
                <p className="small"><strong>Data Sync:</strong> {defs.app_overview?.data_sync}</p>
                <p className="small mb-0"><strong>Alert Pathway:</strong> {defs.app_overview?.alert_pathway}</p>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header small fw-semibold">📋 Field Definitions</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Field</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {(defs.fields || []).map((f, i) => (
                      <tr key={i}>
                        <td className="font-monospace small">{f.field}</td>
                        <td className="small">{f.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
