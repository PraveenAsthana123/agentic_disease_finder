'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const battColor = b => b >= 60 ? 'success' : b >= 30 ? 'warning' : 'danger';
const syncBadge = s => ({ synced: 'success', pending: 'warning', error: 'danger' }[s] || 'secondary');
const riskColor = r => ({ none: 'secondary', low: 'success', moderate: 'warning', high: 'danger' }[r] || 'secondary');
const qColor = q => ({ good: 'success', marginal: 'warning', poor: 'danger' }[q] || 'secondary');
const pct = v => `${Math.round(v * 100)}%`;

export default function EmotivInsightDashboard() {
  const [ov, setOv] = useState(null);
  const [sd, setSd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('patient_id');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/emotiv-insight/overview`).then(r => r.json()),
      fetch(`${API}/api/emotiv-insight/sessions`).then(r => r.json()),
      fetch(`${API}/api/emotiv-insight/definitions`).then(r => r.json()),
    ]).then(([o, s, d]) => { setOv(o); setSd(s); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Emotiv Insight fleet data…</div>;

  const TABS = [
    { id: 'overview',  label: '📊 Overview' },
    { id: 'fleet',     label: '🧠 Fleet' },
    { id: 'sessions',  label: '📋 Sessions' },
    { id: 'alerts',    label: '🚨 Alerts' },
    { id: 'defs',      label: '📖 Reference' },
  ];

  const kpis = ov.kpis || {};
  const devices = sd?.all_devices || [];
  const sessions = sd?.sessions || [];
  const alerts = sd?.alert_events || [];

  const filtered = devices.filter(d =>
    !search ||
    d.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    d.device_id.toLowerCase().includes(search.toLowerCase())
  ).sort((a, b) => {
    if (sortBy === 'battery_pct') return a.battery_pct - b.battery_pct;
    if (sortBy === 'seizure_risk') return ['none','low','moderate','high'].indexOf(b.seizure_risk) - ['none','low','moderate','high'].indexOf(a.seizure_risk);
    if (sortBy === 'channels_good') return b.channels_good - a.channels_good;
    if (sortBy === 'sync_status') return a.sync_status.localeCompare(b.sync_status);
    return a.patient_id.localeCompare(b.patient_id);
  });

  return (
    <div className="container-fluid py-3">
      <h2 className="fw-bold mb-1">🧠 Emotiv Insight 2+ Dashboard</h2>
      <p className="text-muted mb-3">5-channel consumer-research EEG — BLE online · offline buffer · seizure-risk edge model</p>

      {/* KPI row */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Devices',        val: kpis.total_devices,       cls: 'primary' },
          { label: 'Synced',         val: kpis.synced,              cls: 'success' },
          { label: 'Pending',        val: kpis.sync_pending,        cls: 'warning' },
          { label: 'Sync Errors',    val: kpis.sync_errors,         cls: 'danger' },
          { label: 'Avg Battery',    val: `${kpis.avg_battery_pct}%`, cls: 'info' },
          { label: 'Low Battery',    val: kpis.low_battery_count,   cls: kpis.low_battery_count > 0 ? 'warning' : 'success' },
          { label: 'Online Now',     val: kpis.online_devices,      cls: 'success' },
          { label: 'Avg Good Ch',    val: `${kpis.avg_good_channels}/5`, cls: 'info' },
          { label: 'Risk Alerts',    val: kpis.seizure_risk_alerts, cls: kpis.seizure_risk_alerts > 0 ? 'danger' : 'success' },
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

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Band power 7-day trend */}
          <div className="col-lg-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">📈 Band Power Trend — 7 Days (mean across fleet)</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.75rem' }}>
                  <thead><tr><th>Date</th><th>δ Delta</th><th>θ Theta</th><th>α Alpha</th><th>β Beta</th><th>γ Gamma</th></tr></thead>
                  <tbody>
                    {(ov.band_trend_7d || []).map(r => (
                      <tr key={r.date}>
                        <td className="small">{r.date}</td>
                        {['delta','theta','alpha','beta','gamma'].map(b => (
                          <td key={b} className="fw-semibold">{pct(r[b])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Right column */}
          <div className="col-lg-6">
            {/* Dominant band distribution */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">🎯 Dominant Band Distribution</div>
              <div className="card-body">
                {(ov.dominant_band_distribution || []).map(row => (
                  <div key={row.band} className="mb-1">
                    <div className="d-flex justify-content-between small">
                      <span className="text-capitalize">{row.band}</span>
                      <strong>{row.count}</strong>
                    </div>
                    <div className="progress" style={{ height: '6px' }}>
                      <div className="progress-bar bg-primary"
                        style={{ width: `${Math.min(100, row.count / kpis.total_devices * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Alpha asymmetry */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">🔀 Alpha Asymmetry (AF4 − AF3)</div>
              <div className="card-body">
                {(ov.alpha_asymmetry_distribution || []).map(row => (
                  <div key={row.range} className="mb-1">
                    <div className="d-flex justify-content-between small">
                      <span>{row.range}</span><strong>{row.count}</strong>
                    </div>
                    <div className="progress" style={{ height: '6px' }}>
                      <div className="progress-bar bg-info"
                        style={{ width: `${Math.min(100, row.count / kpis.total_devices * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Seizure risk summary */}
            <div className="card">
              <div className="card-header fw-semibold">⚠️ Seizure Risk Summary</div>
              <div className="card-body">
                {(ov.seizure_risk_summary || []).map(row => (
                  <div key={row.risk} className="d-flex justify-content-between align-items-center mb-1">
                    <span className={`badge bg-${riskColor(row.risk)} text-capitalize`}>{row.risk}</span>
                    <strong>{row.count}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Fleet tab ── */}
      {tab === 'fleet' && (
        <div>
          <div className="d-flex gap-2 mb-2 flex-wrap">
            <input className="form-control form-control-sm" style={{ maxWidth: 200 }}
              placeholder="Search patient / device…" value={search} onChange={e => setSearch(e.target.value)} />
            <select className="form-select form-select-sm" style={{ maxWidth: 200 }}
              value={sortBy} onChange={e => setSortBy(e.target.value)}>
              <option value="patient_id">Sort: Patient</option>
              <option value="battery_pct">Sort: Battery ↑</option>
              <option value="seizure_risk">Sort: Risk ↓</option>
              <option value="channels_good">Sort: Good Ch ↓</option>
              <option value="sync_status">Sort: Sync</option>
            </select>
            <span className="text-muted small align-self-center">{filtered.length} devices</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-striped" style={{ fontSize: '0.78rem' }}>
              <thead className="table-dark">
                <tr>
                  <th>Device</th><th>Patient</th><th>Battery</th><th>Mode</th>
                  <th>Sync</th><th>Good Ch</th><th>AF3</th><th>AF4</th><th>T7</th><th>T8</th><th>Pz</th>
                  <th>Dom. Band</th><th>α Asym</th><th>Risk</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(d => (
                  <tr key={d.device_id}>
                    <td><code className="small">{d.device_id}</code></td>
                    <td><strong>{d.patient_id}</strong></td>
                    <td><span className={`badge bg-${battColor(d.battery_pct)}`}>{d.battery_pct}%</span></td>
                    <td><span className="badge bg-secondary">{d.mode}</span></td>
                    <td><span className={`badge bg-${syncBadge(d.sync_status)}`}>{d.sync_status}</span></td>
                    <td className="fw-semibold">{d.channels_good}/5</td>
                    {['AF3','AF4','T7','T8','Pz'].map(ch => (
                      <td key={ch}>
                        <span className={`badge bg-${qColor(d.channel_quality?.[ch])}`} style={{ fontSize: '0.65rem' }}>
                          {d.channel_quality?.[ch]?.[0]?.toUpperCase() || '?'}
                        </span>
                      </td>
                    ))}
                    <td className="text-capitalize small">{d.dominant_band}</td>
                    <td className="small">{d.alpha_asymmetry > 0 ? '+' : ''}{d.alpha_asymmetry}</td>
                    <td><span className={`badge bg-${riskColor(d.seizure_risk)}`}>{d.seizure_risk}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Sessions tab ── */}
      {tab === 'sessions' && (
        <div className="table-responsive">
          <table className="table table-sm table-hover table-striped" style={{ fontSize: '0.78rem' }}>
            <thead className="table-dark">
              <tr>
                <th>Session</th><th>Device</th><th>Patient</th><th>Date</th>
                <th>Duration</th><th>Good Ch</th><th>Peak Risk</th>
                <th>Motion Art.</th><th>Quality %</th><th>Mode</th><th>Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map(s => (
                <tr key={s.session_id}>
                  <td><code className="small">{s.session_id}</code></td>
                  <td><code className="small">{s.device_id}</code></td>
                  <td><strong>{s.patient_id}</strong></td>
                  <td className="small">{s.date}</td>
                  <td>{s.duration_min} min</td>
                  <td>{s.channels_good}/5</td>
                  <td><span className={`badge bg-${riskColor(s.seizure_risk_peak)}`}>{s.seizure_risk_peak}</span></td>
                  <td>{s.motion_artifacts}</td>
                  <td>
                    <span className={`badge bg-${s.data_quality_pct >= 90 ? 'success' : s.data_quality_pct >= 75 ? 'warning' : 'danger'}`}>
                      {s.data_quality_pct}%
                    </span>
                  </td>
                  <td><span className="badge bg-secondary">{s.mode}</span></td>
                  <td>
                    <span className={`badge bg-${s.uploaded ? 'success' : 'warning'}`}>
                      {s.uploaded ? '✓' : 'Pending'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Alerts tab ── */}
      {tab === 'alerts' && (
        <div>
          <h5 className="mb-3">🚨 Seizure-Risk Alert Events</h5>
          {alerts.length === 0 ? (
            <div className="alert alert-success">No seizure-risk alerts in the current window.</div>
          ) : (
            <div className="table-responsive">
              <table className="table table-sm table-hover table-striped">
                <thead className="table-danger">
                  <tr>
                    <th>Event</th><th>Device</th><th>Patient</th><th>Risk</th>
                    <th>Dom. Band</th><th>α Asym</th><th>Battery</th>
                    <th>Motion Art.</th><th>Timestamp</th><th>Resolved</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map(ev => (
                    <tr key={ev.event_id}>
                      <td><code className="small">{ev.event_id}</code></td>
                      <td><code className="small">{ev.device_id}</code></td>
                      <td><strong>{ev.patient_id}</strong></td>
                      <td><span className={`badge bg-${riskColor(ev.risk_level)}`}>{ev.risk_level}</span></td>
                      <td className="text-capitalize small">{ev.dominant_band}</td>
                      <td className="small">{ev.alpha_asymmetry > 0 ? '+' : ''}{ev.alpha_asymmetry}</td>
                      <td><span className={`badge bg-${battColor(ev.battery_pct)}`}>{ev.battery_pct}%</span></td>
                      <td><span className={`badge bg-${ev.motion_artifact ? 'warning' : 'success'}`}>{ev.motion_artifact ? 'Yes' : 'No'}</span></td>
                      <td className="small">{ev.timestamp?.replace('T', ' ').replace('Z', ' UTC')}</td>
                      <td><span className={`badge bg-${ev.resolved ? 'success' : 'warning'}`}>{ev.resolved ? 'Resolved' : 'Active'}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── Reference tab ── */}
      {tab === 'defs' && defs && (
        <div className="row g-3">
          {/* Device overview */}
          <div className="col-lg-6">
            <div className="card mb-3">
              <div className="card-header fw-semibold">🧠 Device Specs — Emotiv Insight 2+</div>
              <div className="card-body">
                <dl className="row mb-0">
                  {defs.device_overview && Object.entries(defs.device_overview).map(([k, v]) => (
                    <div key={k} className="col-12 mb-2">
                      <dt className="small text-muted text-capitalize">{k.replace(/_/g, ' ')}</dt>
                      <dd className="mb-0 small">{Array.isArray(v) ? v.join(', ') : v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>

            {/* Channel map */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">📍 Channel Map (10-20 system)</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light"><tr><th>Channel</th><th>Region</th><th>Role</th></tr></thead>
                  <tbody>
                    {(defs.channel_definitions || []).map(c => (
                      <tr key={c.channel}>
                        <td><strong>{c.channel}</strong></td>
                        <td className="small">{c.region}</td>
                        <td className="small text-muted">{c.role}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Alpha asymmetry */}
            {defs.alpha_asymmetry && (
              <div className="card">
                <div className="card-header fw-semibold">🔀 Alpha Asymmetry Index</div>
                <div className="card-body">
                  <dl className="row mb-0">
                    {Object.entries(defs.alpha_asymmetry).map(([k, v]) => (
                      <div key={k} className="col-12 mb-2">
                        <dt className="small text-muted text-capitalize">{k.replace(/_/g, ' ')}</dt>
                        <dd className="mb-0 small">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </div>
            )}
          </div>

          <div className="col-lg-6">
            {/* Band definitions */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">📏 EEG Band Definitions</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light"><tr><th>Band</th><th>Range</th><th>Clinical Significance</th></tr></thead>
                  <tbody>
                    {(defs.band_definitions || []).map(b => (
                      <tr key={b.band}>
                        <td className="fw-semibold text-capitalize">{b.band}</td>
                        <td className="small text-nowrap">{b.range}</td>
                        <td className="small text-muted">{b.significance}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Risk levels */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">⚠️ Seizure Risk Levels</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Level</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.seizure_risk_levels || []).map(r => (
                      <tr key={r.level}>
                        <td><span className={`badge bg-${riskColor(r.level)}`}>{r.level}</span></td>
                        <td className="small">{r.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Contact quality */}
            <div className="card mb-3">
              <div className="card-header fw-semibold">🔌 Contact Quality Grades</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Grade</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.contact_quality || []).map(c => (
                      <tr key={c.quality}>
                        <td><span className={`badge bg-${qColor(c.quality)}`}>{c.quality}</span></td>
                        <td className="small">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Sync modes */}
            <div className="card">
              <div className="card-header fw-semibold">🔄 Sync Modes</div>
              <div className="card-body">
                {(defs.sync_modes || []).map(s => (
                  <div key={s.mode} className="mb-2">
                    <strong className="text-capitalize">{s.mode}:</strong>{' '}
                    <span className="text-muted small">{s.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
