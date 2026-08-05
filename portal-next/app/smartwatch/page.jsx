'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const syncBadge = s => ({ synced: 'success', pending: 'warning', error: 'danger' }[s] || 'secondary');
const battColor = b => b >= 60 ? 'success' : b >= 30 ? 'warning' : 'danger';
const signalBadge = s => s === 'none' ? 'secondary' : 'danger';
const hrColor = hr => hr >= 120 ? 'danger' : hr >= 100 ? 'warning' : 'success';

export default function SmartwatchDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('patient_id');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/smartwatch/overview`).then(r => r.json()),
      fetch(`${API}/api/smartwatch/breakdown`).then(r => r.json()),
      fetch(`${API}/api/smartwatch/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading smartwatch fleet data…</div>;

  const TABS = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'fleet', label: '⌚ Fleet' },
    { id: 'alerts', label: '🚨 Seizure Alerts' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const kpis = ov.kpis || {};
  const devices = bd?.all_devices || [];
  const alertEvents = bd?.alert_events || [];

  const filtered = devices.filter(d =>
    !search || d.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    d.model.toLowerCase().includes(search.toLowerCase())
  ).sort((a, b) => {
    if (sortBy === 'battery_pct') return a.battery_pct - b.battery_pct;
    if (sortBy === 'hr_now') return b.hr_now - a.hr_now;
    if (sortBy === 'spo2_pct') return a.spo2_pct - b.spo2_pct;
    if (sortBy === 'sync_status') return a.sync_status.localeCompare(b.sync_status);
    return a.patient_id.localeCompare(b.patient_id);
  });

  return (
    <div className="container-fluid py-3">
      <h2 className="fw-bold mb-1">⌚ Smartwatch Dashboard</h2>
      <p className="text-muted mb-3">Apple Watch / Wear OS fleet — HR · HRV · SpO₂ · Seizure signals · Sleep</p>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Watches', val: kpis.total_watches, cls: 'primary' },
          { label: 'Synced', val: kpis.synced, cls: 'success' },
          { label: 'Pending', val: kpis.sync_pending, cls: 'warning' },
          { label: 'Errors', val: kpis.sync_errors, cls: 'danger' },
          { label: 'Avg Battery', val: `${kpis.avg_battery_pct}%`, cls: 'info' },
          { label: 'Avg HR', val: `${kpis.avg_hr_bpm} bpm`, cls: 'secondary' },
          { label: 'Avg SpO₂', val: `${kpis.avg_spo2_pct}%`, cls: 'success' },
          { label: 'Avg HRV', val: `${kpis.avg_hrv_sdnn_ms} ms`, cls: 'info' },
          { label: 'Seizure Alerts', val: kpis.seizure_signal_alerts, cls: kpis.seizure_signal_alerts > 0 ? 'danger' : 'success' },
          { label: 'Steps Today', val: (kpis.total_steps_today || 0).toLocaleString(), cls: 'secondary' },
        ].map(k => (
          <div key={k.label} className="col-6 col-md-3 col-lg-2">
            <div className={`card border-${k.cls} text-center py-2`}>
              <div className={`fs-5 fw-bold text-${k.cls}`}>{k.val}</div>
              <div className="small text-muted">{k.label}</div>
            </div>
          </div>
        ))}
      </div>

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

      {/* Overview tab */}
      {tab === 'overview' && (
        <div>
          <div className="row g-3 mb-3">
            {/* HR trend */}
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-semibold">📈 Heart Rate Trend (24 h)</div>
                <div className="card-body p-2">
                  <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.75rem' }}>
                    <thead><tr><th>Hour</th><th>HR (bpm)</th><th>Zone</th></tr></thead>
                    <tbody>
                      {(ov.hr_trend_24h || []).map(r => (
                        <tr key={r.hour}>
                          <td>{String(r.hour).padStart(2, '0')}:00</td>
                          <td className={`fw-semibold text-${hrColor(r.hr_bpm)}`}>{r.hr_bpm}</td>
                          <td>
                            <span className={`badge bg-${hrColor(r.hr_bpm)}`}>
                              {r.hr_bpm >= 120 ? 'Tachycardia' : r.hr_bpm >= 100 ? 'Elevated' : 'Normal'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* HRV trend + SpO2 + sleep */}
            <div className="col-md-6">
              <div className="card mb-3">
                <div className="card-header fw-semibold">💓 HRV Trend — SDNN (7 days)</div>
                <div className="card-body p-2">
                  <table className="table table-sm mb-0">
                    <thead><tr><th>Date</th><th>SDNN (ms)</th><th>Status</th></tr></thead>
                    <tbody>
                      {(ov.hrv_trend_7d || []).map(r => (
                        <tr key={r.day}>
                          <td>{r.day}</td>
                          <td className="fw-semibold">{r.sdnn_ms}</td>
                          <td>
                            <span className={`badge bg-${r.sdnn_ms < 20 ? 'danger' : r.sdnn_ms < 30 ? 'warning' : 'success'}`}>
                              {r.sdnn_ms < 20 ? 'Low' : r.sdnn_ms < 30 ? 'Borderline' : 'Normal'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="card mb-3">
                <div className="card-header fw-semibold">🩸 SpO₂ Distribution</div>
                <div className="card-body">
                  {(ov.spo2_distribution || []).map(b => (
                    <div key={b.range} className="mb-1">
                      <div className="d-flex justify-content-between small">
                        <span>{b.range}</span>
                        <strong>{b.count}</strong>
                      </div>
                      <div className="progress" style={{ height: '6px' }}>
                        <div
                          className={`progress-bar bg-${b.range.includes('normal') ? 'success' : b.range.includes('mild') ? 'warning' : 'danger'}`}
                          style={{ width: `${Math.min(100, b.count * 5)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="card">
                <div className="card-header fw-semibold">😴 Avg Sleep Staging</div>
                <div className="card-body">
                  {(() => {
                    const sl = ov.sleep_stages_avg || {};
                    return (
                      <div className="row text-center">
                        {[
                          { label: 'Total', val: `${sl.total_h}h`, cls: 'primary' },
                          { label: 'REM', val: `${sl.rem_h}h`, cls: 'info' },
                          { label: 'Deep', val: `${sl.deep_h}h`, cls: 'success' },
                          { label: 'Light', val: `${sl.light_h}h`, cls: 'secondary' },
                        ].map(s => (
                          <div key={s.label} className="col-3">
                            <div className={`fs-5 fw-bold text-${s.cls}`}>{s.val}</div>
                            <div className="small text-muted">{s.label}</div>
                          </div>
                        ))}
                      </div>
                    );
                  })()}
                </div>
              </div>
            </div>
          </div>

          {/* Model distribution */}
          <div className="card">
            <div className="card-header fw-semibold">⌚ Device Model Distribution</div>
            <div className="card-body p-2">
              <table className="table table-sm table-hover mb-0">
                <thead><tr><th>Model</th><th>Count</th><th>Share</th></tr></thead>
                <tbody>
                  {(ov.model_distribution || []).map(m => (
                    <tr key={m.model}>
                      <td>{m.model}</td>
                      <td><strong>{m.count}</strong></td>
                      <td>
                        <div className="progress" style={{ height: '8px', minWidth: '80px' }}>
                          <div className="progress-bar bg-primary"
                            style={{ width: `${Math.round(m.count / kpis.total_watches * 100)}%` }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Fleet tab */}
      {tab === 'fleet' && (
        <div>
          <div className="d-flex gap-2 mb-2 flex-wrap">
            <input className="form-control form-control-sm" style={{ maxWidth: 200 }}
              placeholder="Search patient / model…" value={search} onChange={e => setSearch(e.target.value)} />
            <select className="form-select form-select-sm" style={{ maxWidth: 180 }}
              value={sortBy} onChange={e => setSortBy(e.target.value)}>
              <option value="patient_id">Sort: Patient</option>
              <option value="battery_pct">Sort: Battery ↑</option>
              <option value="hr_now">Sort: HR ↓</option>
              <option value="spo2_pct">Sort: SpO₂ ↑</option>
              <option value="sync_status">Sort: Sync</option>
            </select>
            <span className="text-muted small align-self-center">{filtered.length} devices</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-striped">
              <thead className="table-dark">
                <tr>
                  <th>Device ID</th><th>Patient</th><th>Model</th><th>Battery</th>
                  <th>HR (bpm)</th><th>SpO₂</th><th>HRV (ms)</th><th>Steps</th>
                  <th>Sync</th><th>Seizure Signal</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(d => (
                  <tr key={d.device_id}>
                    <td><code className="small">{d.device_id}</code></td>
                    <td><strong>{d.patient_id}</strong></td>
                    <td className="small">{d.model}</td>
                    <td>
                      <span className={`badge bg-${battColor(d.battery_pct)}`}>{d.battery_pct}%</span>
                    </td>
                    <td className={`fw-semibold text-${hrColor(d.hr_now)}`}>{d.hr_now}</td>
                    <td className={`fw-semibold text-${d.spo2_pct < 92 ? 'danger' : d.spo2_pct < 95 ? 'warning' : 'success'}`}>
                      {d.spo2_pct}%
                    </td>
                    <td>{d.hrv_sdnn_ms}</td>
                    <td>{(d.steps_today || 0).toLocaleString()}</td>
                    <td><span className={`badge bg-${syncBadge(d.sync_status)}`}>{d.sync_status}</span></td>
                    <td>
                      <span className={`badge bg-${signalBadge(d.seizure_signal)}`}>
                        {d.seizure_signal === 'none' ? '—' : d.seizure_signal.replace('_', ' ')}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Alerts tab */}
      {tab === 'alerts' && (
        <div>
          <h5 className="mb-3">🚨 Seizure-Signal Alert Events</h5>
          {alertEvents.length === 0 ? (
            <div className="alert alert-success">No seizure-signal alerts in the current window.</div>
          ) : (
            <div className="table-responsive">
              <table className="table table-sm table-hover table-striped">
                <thead className="table-danger">
                  <tr>
                    <th>Event ID</th><th>Device</th><th>Patient</th><th>Signal Type</th>
                    <th>HR (bpm)</th><th>SpO₂</th><th>Timestamp</th><th>Resolved</th>
                  </tr>
                </thead>
                <tbody>
                  {alertEvents.map(ev => (
                    <tr key={ev.event_id}>
                      <td><code className="small">{ev.event_id}</code></td>
                      <td><code className="small">{ev.device_id}</code></td>
                      <td><strong>{ev.patient_id}</strong></td>
                      <td>
                        <span className="badge bg-danger">
                          {ev.signal_type.replace('_', ' ')}
                        </span>
                      </td>
                      <td className={`fw-semibold text-${hrColor(ev.hr_bpm)}`}>{ev.hr_bpm}</td>
                      <td>{ev.spo2_pct}%</td>
                      <td className="small">{ev.timestamp?.replace('T', ' ').replace('Z', ' UTC')}</td>
                      <td>
                        <span className={`badge bg-${ev.resolved ? 'success' : 'warning'}`}>
                          {ev.resolved ? 'Resolved' : 'Active'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Alert type definitions */}
          {defs?.alert_types && (
            <div className="mt-4">
              <h6 className="fw-semibold">Alert Type Definitions</h6>
              <div className="table-responsive">
                <table className="table table-sm table-bordered">
                  <thead className="table-light"><tr><th>Type</th><th>Trigger</th><th>Action</th></tr></thead>
                  <tbody>
                    {defs.alert_types.map(a => (
                      <tr key={a.type}>
                        <td><span className="badge bg-danger">{a.type.replace('_', ' ')}</span></td>
                        <td>{a.trigger}</td>
                        <td>{a.action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          {/* Device overview */}
          {defs.device_overview && (
            <div className="card mb-3">
              <div className="card-header fw-semibold">⌚ Device Overview</div>
              <div className="card-body">
                <dl className="row mb-0">
                  {Object.entries(defs.device_overview).map(([k, v]) => (
                    <div key={k} className="col-md-6 mb-2">
                      <dt className="small text-muted text-capitalize">{k.replace(/_/g, ' ')}</dt>
                      <dd className="mb-0 small">{Array.isArray(v) ? v.join(', ') : v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          )}

          {/* Metric definitions */}
          {defs.metrics && (
            <div className="card mb-3">
              <div className="card-header fw-semibold">📏 Metric Definitions & Epilepsy Flags</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light"><tr><th>Metric</th><th>Normal Range</th><th>Epilepsy Flag</th></tr></thead>
                  <tbody>
                    {defs.metrics.map(m => (
                      <tr key={m.metric}>
                        <td className="fw-semibold">{m.metric}</td>
                        <td className="text-success small">{m.normal}</td>
                        <td className="text-danger small">{m.epilepsy_flag}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Sync modes */}
          {defs.sync_modes && (
            <div className="card">
              <div className="card-header fw-semibold">🔄 Sync Modes</div>
              <div className="card-body">
                {defs.sync_modes.map(s => (
                  <div key={s.mode} className="mb-2">
                    <strong>{s.mode}:</strong> <span className="text-muted small">{s.description}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
